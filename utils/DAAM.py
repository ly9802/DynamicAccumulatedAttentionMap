# -- coding:utf-8 --
# @Time:2024/6/2 8:52
# @Author Steven
from __future__ import print_function
import os
import sys
import cv2
import time
import argparse
import numpy as np
import math
from PIL import Image
import glob
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, CenterCrop
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler, SGD

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("pytorch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("backends cudnn version:", torch.backends.cudnn.version())
print("GPU Type:", torch.cuda.get_device_name(0))
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def reshape_transformation(tensor, height=14, width=14):
    bs,num_tokens,dim=tensor.shape
    import math
    height=int(math.sqrt(num_tokens-1))
    width=height
    result = tensor[:, 1:, :].reshape(bs,height, width, dim)

    result = result.transpose(2, 3).transpose(1, 2)
    return result

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

class ActivationAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers,block_layers, arch_name):
        self.model = model
        self.gradients=[]
        self.block_gradients = [];
        self.activations = []
        self.attentionmap_list=[];
        self.input_list=[]
        self.handles = []
        self.count=0;
        for target_layer in target_layers:
            self.count=self.count+1;
            self.handles.append(target_layer[0].register_forward_hook(self.change_activation))
            self.handles.append(target_layer[1].register_forward_hook(self.prj_gradient))
    def __call__(self, x):
        self.activations = []
        self.gradients = []
        self.attentionmap_list=[];
        self.block_gradients=[];
        self.input_list=[]
        return self.model(x)

    def change_activation(self,module,input,output):
        if not hasattr(output[0], "requires_grad") or not output[0].requires_grad:
            return
        x=input[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attn = module.attn_drop(attn)
        focus = attn[:, :, 0]
        focus = focus.unsqueeze(dim=-1)
        elementwise_product1 = torch.mul(focus, v)
        elementwise_product = elementwise_product1.transpose(1, 2).reshape(B, N, C)
        self.activations.append(elementwise_product.cpu().detach())

    def prj_gradient(self,module,input,output):
        if not hasattr(input[0], "requires_grad") or not input[0].requires_grad:
            print("hello, input doesn't have set reguires_grad to True")
            return

        input[0].register_hook(self.save_grad)

    def save_grad(self,grad):
        weight=grad[:,0,:]
        self.gradients = [weight.cpu().detach()] + self.gradients

    def release(self):
        for handle in self.handles:
            handle.remove()

class DynamicAccumulatedAttentionMap:
    def __init__(self,model,target_layers,block_layers=None,use_cuda=True,
                 reshape_transform=reshape_transformation,arch_name=None, norm=True, gpu_id=0):
        self.model = model.eval()
        self.target_layers = target_layers # list[nn.Module]
        self.cuda = use_cuda
        self.norm=norm
        self.arch_name=arch_name
        self.gpu_id=gpu_id
        self.non_liear_mapping=False

        self.reshape_transform = reshape_transform
        self.activations_and_grads= ActivationAndGradients(self.model,target_layers,block_layers,arch_name) #object


    def __call__(self,input_tensor,target_label=None) :

        return self.forward(input_tensor,target_label)#

    def forward(self,input_tensor,target_label=None) :
        if self.cuda:
            input_tensor=input_tensor.cuda(self.gpu_id)
        target_size = self.get_target_width_height(input_tensor)
        input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
        score_vector = self.activations_and_grads(input_tensor)  # (1,1000)

        if target_label is None:
            predicted_label = np.argmax(score_vector.cpu().data.numpy(), axis=-1)[0]  # scalar 243
            target_categories = np.argmax(score_vector.cpu().data.numpy(), axis=-1) #ndarray [243]
            target_label = [ClassifierOutputTarget(category) for category in target_categories]
            classification_score=score_vector[0,predicted_label]
        else:
            predicted_label=target_label
        feature_matrix_list = self.activations_and_grads.activations
        self.model.zero_grad()
        score = sum([target(output) for target, output in zip(target_label, score_vector)])
        score.backward(retain_graph=True)
        gradients_list = self.activations_and_grads.gradients
        num_layers = len(gradients_list);
        cam_list = self.generate_accumul_cam(gradients_list, feature_matrix_list, target_size)

        return cam_list,predicted_label
    def generate_accumul_cam(self,gradients_list, feature_matrix_list, target_size):

        activations_list = [self.reshape_transform(a) for a in feature_matrix_list]

        weights_list = [g.unsqueeze(dim=-1).unsqueeze(dim=-1) for g in gradients_list]

        activations_list = [a.cpu().data.numpy() for a in activations_list]
        grads_list = [g.cpu().data.numpy() for g in weights_list]
        cam_per_block_list = []
        num_blocks=len(activations_list)
        for i in range(num_blocks):
            gradient = grads_list[i]
            activation_map = activations_list[i]
            gradient=np.maximum(gradient,0)
            weighted_activations = gradient * activation_map

            cam = weighted_activations.sum(axis=1)
            cam=np.maximum(cam, 0)
            if self.norm:
                cam=self.max_min_normalize(cam)
            cam_per_block_list.append(cam);

        dynamic_cam_list=[np.concatenate(cam_per_block_list[0:i],axis=0) for i in range(1,len(cam_per_block_list)+1)]


        accumulated=np.concatenate(cam_per_block_list, axis=0);
        accumul = accumulated.sum(axis=0)
        global_maximum=np.max(accumul.ravel())
        global_minimum=np.min(accumul.ravel())

        cam_list=[]
        total_num=len(dynamic_cam_list)

        for item_no, dynamic_cam in enumerate(dynamic_cam_list):
            accumul=dynamic_cam.sum(axis=0);
            local_minimum=np.min(accumul)
            accumul=np.maximum(accumul,0);
            if self.non_liear_mapping==False:
                normed_cam=self.global_max_min_norm(accumul,global_maximum,global_minimum)
            else:
                normed_cam = self.global_max_min_norm_nonlinearmapping(accumul, global_maximum, global_minimum)
            scaled_cam=self.scale(normed_cam,target_size)

            cam_list.append(scaled_cam);

        return cam_list
    def max_min_normalize(self,cam):

        result=[];
        for img in cam:
            # img (14,14)
            minimum = np.min(img.ravel())
            img = img - minimum
            maximum = np.max(img.ravel()) + 1e-10
            img = img / (maximum)
            result.append(img);
        result = np.float32(result)
        return result

    def global_max_min_norm(self,cam,global_maximum, global_minimum):
        current_maximum=np.max(cam.ravel())
        current_minimum=np.min(cam.ravel())
        img=(cam-global_minimum)/(current_maximum-global_minimum+1e-10)
        img=np.maximum(img,0)

        ratio=(current_maximum-current_minimum)/(global_maximum-global_minimum)
        r=255*ratio
        result=np.uint8(r*img)
        return result

    def global_max_min_norm_nonlinearmapping(self,cam,global_maximum,global_minimum):
        current_maximum = np.max(cam.ravel())
        current_minimum = np.min(cam.ravel())
        img=(cam-0)/(global_maximum-0)
        img=self.sigmod_variant(img,5)
        sigmod_maximum=np.max(img.ravel())
        sigmod_minimum=np.min(img.ravel())
        norm_img=(img-sigmod_minimum)/(sigmod_maximum-sigmod_minimum)
        result=np.uint8(255*norm_img)
        return result

    def max_min_norm(self,cam, r=255):

        img=cam
        minimum = np.min(img.ravel())
        img = img - minimum
        maximum = np.max(img.ravel())+1e-10
        img = img / (maximum)
        result = np.uint8(r * img)
        return result
    def scale(self,cam_ndarray,target_size):

        img = cv2.resize(cam_ndarray, target_size,interpolation=cv2.INTER_LINEAR)
        return img
    def reshape_for_squaremap(self,tensor,height=14, width=14):
        bs, num_tokens, dim = tensor.shape
        import math
        height = int(math.sqrt(num_tokens))
        width = height
        result = tensor.reshape(bs, height, width, dim)

        result = result.transpose(2, 3).transpose(1, 2)
        return result;
    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmod_variant(self,x, apha=1.0):

        y=self.sigmod(apha*x)
        y=y-0.5
        return y

    def correct(self,cam):
        cam[cam>1.0]=1.0;
        return cam

    def get_target_width_height(self,input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height


    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations.release()
        if isinstance(exc_value, IndexError):

            print(f"An exception occurred in FAM with block: {exc_type}. Message: {exc_value}")
            return True


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print("Finish!");
