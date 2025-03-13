# -- coding:utf-8 --
# @Time:2024/6/2 8:52
# @Author Steven Leo 
from __future__ import print_function
import os
import sys
import cv2
import time
import argparse
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
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
import ttach as tta
from typing import Callable, List, Tuple
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from imagenet_index import index2class
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from imagenet_index import index2class
print("pytorch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("backends cudnn version:", torch.backends.cudnn.version())
print("GPU Type:", torch.cuda.get_device_name(0))
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# if you want to import local directory as a module, you should add "sys.append(str local_dirctory)
def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);
def max_min_normal(coefficient):
    if coefficient.max()>coefficient.min():
        normalized=torch.div(coefficient-coefficient.min(),coefficient.max()-coefficient.min())
    else:
        normalized=torch.ones_like(coefficient)
    return normalized
def linear_combine_allchannels(featuremaps,contri_weight):
    activation_maps=featuremaps
    weight_tensor=contri_weight;
    importance_coefficients = max_min_normal(weight_tensor)  # (num_channels,)
    importance_coefficients = importance_coefficients.unsqueeze(dim=-1).unsqueeze(dim=-1)
    activation_maps = torch.mul(activation_maps, importance_coefficients)  # (num_channel,h,w)
    fam = torch.sum(activation_maps, dim=0, keepdim=False)  # (num_channel,h,w)-->(h,w)
    return fam

def reshape_transformation(tensor, height=14, width=14):
    bs,num_tokens,dim=tensor.shape
    # print("input tensor shape:",tensor.shape) # (bs,n_rgions+1,n_channel]
    # print("input batchsize:",bs)
    # print("input num_tokens:",num_tokens-1)
    # print("input num_channels:",dim)
    import math
    height=int(math.sqrt(num_tokens-1))
    width=height
    result = tensor[:, 1:, :].reshape(bs,height, width, dim)
    #去掉cls_token ,result=(batchsize, 14,14,n_channel) 为啥是14, 因为以16 pixel为一个patch, 224*224 pixels, 一共有14*14 patches
    # 每个patch-->vector-->token, 因此, n_regions=14*14=196
    #(bs,197,384)-->(bs,14,14,384)
    # Bring the channels to the first dimension,
    #(bs,14,14,384)-->(bs,14,384,14)-->(bs,384,14,14)
    result = result.transpose(2, 3).transpose(1, 2)
    return result
def preprocess_image(img_ndarray):
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocessing(img_ndarray.copy()).unsqueeze(0)
def imageTotensor(image_path,size=(224,224)):
    height,width=size[0],size[1];
    img_ndarray=cv2.imread(image_path, cv2.IMREAD_COLOR)
    rgb_img = cv2.resize(img_ndarray, (width,height),interpolation=cv2.INTER_LINEAR)
    rgb_img = np.float32(rgb_img) / 255  # (0,255)-->(0,1)
    return preprocess_image(rgb_img)
def extract_feature(tensor):
    #tensor (bs,n_tokens,dim)
    bs,n_tokens,dim=tensor.shape
    return tensor[:,0,:].reshape(bs,1,dim)

def generate_saliency_map(fam_ndarray, size=(224,224),mask=None):
    #ndarray fam_ndarray
    import cv2
    size_upsample = size;
    fam_ndarray = fam_ndarray - np.min(fam_ndarray);
    fam_img = fam_ndarray / (np.max(fam_ndarray)+1e-7) ;
    #fam_img = sigmod_variant(fam_img)
    if mask is not None:
        fam_img=fam_img*mask
    fam_img1 = np.uint8(255 * fam_img);  # ndarray cam_img: 7*7
    # print("The original cam is\n", cam_img1);
    fams = cv2.resize(fam_img1, size_upsample, interpolation=cv2.INTER_LINEAR);
    return fams
def show_image_without_boundingbox(heatmap,img_path, name,save_dir,original=True):
    #ndarray heatmap: (84,84,1) or (224,224,1)
    #name = name.split(".", 1)[0] + str(num);
    name = name.split(".", 1)[0]
    h,w=heatmap.shape
    img_array=cv2.imread(img_path, cv2.IMREAD_COLOR)
    ori_height, ori_width, _ = img_array.shape
    # generate heatmap
    if original==True:
        heatmap_originalsize=cv2.resize(heatmap, (ori_width, ori_height),interpolation=cv2.INTER_LINEAR)
        colormap = cv2.applyColorMap(heatmap_originalsize, cv2.COLORMAP_JET);
        result = colormap * 0.4 + img_array * 0.8
    else:
        img_array_sqaure = cv2.resize(img_array, (w, h),interpolation=cv2.INTER_LINEAR)
        colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET);
        result = colormap * 0.3 + img_array_sqaure * 0.8
    cv2.imwrite(os.path.join(save_dir,name+"_saliencymap.jpg"),result)
    return result


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            print("model output", model_output.shape)
            print("category", self.category)
            return model_output[self.category]
        return model_output[:, self.category]

class ActivationAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers,block_layers, arch_name):
        self.model = model # ViT
        self.gradients=[]
        self.block_gradients = [];
        self.activations = []
        self.attentionmap_list=[];
        self.input_list=[]

        self.handles = []
        self.count=0;# 记录执行顺序, 先执行self.change_activation--->self.store_input--->self.change_attention
        # for target_layer in target_layers:
        #     # Because of https://github.com/pytorch/pytorch/issues/61519,
        #     # we don't use backward hook to record gradients.
        #     #  0: block_no.attn, 1 :block_no.attn.proj, 2 :block_no
        #     self.count=self.count+1;
        #     self.handles.append(target_layer[0].register_forward_hook(self.change_activation))
        #     self.handles.append(target_layer[1].register_forward_hook(self.prj_gradient))
        if "cait" in arch_name: #cait_S24_224
            # for block_layer in block_layers:
            #     self.count=self.count+1;
            #     self.handles.append(block_layer.register_forward_hook(self.cait_activation))
            #     self.handles.append(block_layer.register_backward_hook(self.cait_gradient))
            for target_layer in target_layers:
                self.count = self.count + 1;
                self.handles.append(target_layer[0].register_forward_hook(self.cait_deep_activation))
                self.handles.append(target_layer[1].register_forward_hook(self.cait_prj_gradient))
                #self.handles.append(target_layer[1].register_backward_hook(self.cait_grad2))

        elif "lvvit" in arch_name:
            for target_layer in target_layers:
                self.count = self.count + 1;
                self.handles.append(target_layer[0].register_forward_hook(self.lv_activation))
                self.handles.append(target_layer[1].register_forward_hook(self.prj_gradient))
        elif "FFVT" in arch_name:
            for target_layer in target_layers:
                self.count=self.count+1
                self.handles.append(target_layer[0].register_forward_hook(self.ffvt_activation))
                self.handles.append(target_layer[1].register_forward_hook(self.ffvt_gradient))
        else:
            for target_layer in target_layers:
                # Because of https://github.com/pytorch/pytorch/issues/61519,
                # we don't use backward hook to record gradients.
                #  0: block_no.attn, 1 :block_no.attn.proj, 2 :block_no
                self.count=self.count+1;
                self.handles.append(target_layer[0].register_forward_hook(self.change_activation))
                self.handles.append(target_layer[1].register_forward_hook(self.prj_gradient))
    def __call__(self, x):
        self.activations = []# 清空 下次使用
        self.gradients = []
        self.attentionmap_list=[];
        self.block_gradients=[];
        self.input_list=[]
        return self.model(x)
    def ffvt_activation(self,module,input,output):
        if not hasattr(output[0], "requires_grad") or not output[0].requires_grad:
            # You can only register hooks on tensor requires grad.
            #print("hello")
            return
        x=input[0]
        B,N,C=x.shape
        mixed_query=module.query(x)
        mixed_key=module.key(x)
        mixed_value=module.value(x)

        query_layer=module.transpose_for_scores(mixed_query)
        key_layer=module.transpose_for_scores(mixed_key)
        value_layer=module.transpose_for_scores(mixed_value)
        attention_scores=torch.matmul(query_layer,key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(module.attention_head_size)
        attention_probs = module.softmax(attention_scores)
        weights = attention_probs if module.vis else None
        attn = module.attn_dropout(attention_probs) # (1,12,785,785)
        focus = attn[:, :, 0]  # (bs,num_heads,785)
        focus = focus.unsqueeze(dim=-1)  # (bs,num_heads,785,1)
        elementwise_product1 = torch.mul(focus, value_layer)  # ->(bs,num_heads,785,64)
        elementwise_product = elementwise_product1.transpose(1, 2).reshape(B, N, C) #(bs,785,768)
        self.activations.append(elementwise_product.cpu().detach())  # (bs,785,384)
    def cait_deep_activation(self,module,input,output):
        if not hasattr(output[0], "requires_grad") or not output[0].requires_grad:
            # You can only register hooks on tensor requires grad.
            #print("hello")
            return
        x = input[0]#(bs,197,384)
        B, N, C = x.shape
        q = module.q(x[:, 0]).unsqueeze(1).reshape(B, 1, module.num_heads, C // module.num_heads).permute(0, 2, 1, 3)
        k = module.k(x).reshape(B, N, module.num_heads, C // module.num_heads).permute(0, 2, 1, 3)
        q = q * module.scale
        v = module.v(x).reshape(B, N, module.num_heads, C // module.num_heads).permute(0, 2, 1, 3)
        # (bs,num_heads,197,48)
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = module.attn_drop(attn)#(bs,num_heads,1,197)

        focus = attn[:, :, 0]  # (bs,num_heads,197)
        focus = focus.unsqueeze(dim=-1)  # (bs,num_heads,197,1)
        elementwise_product1 = torch.mul(focus, v)  # ->(bs,num_heads,785,64)
        elementwise_product = elementwise_product1.transpose(1, 2).reshape(B, N, C)  # (bs,197,384)
        self.activations.append(elementwise_product.cpu().detach())  # (bs,197,384)
    def cait_activation(self,module,input,output):
        if not hasattr(output[0], "requires_grad") or not output[0].requires_grad:
            # You can only register hooks on tensor requires grad.
            #print("hello")
            return
        self.attentionmap_list.append(output.cpu().detach()) #(bs,196,384)
    def cait_gradient(self,module,grad_input,grad_ouput):
        #grad: (bs,n,384)
        gradient=grad_ouput[0]; #(bs,n,384)
        weight = torch.sum(gradient, dim=1, keepdim=False)  # (bs,384)
        self.block_gradients=[weight.cpu().detach()]+self.block_gradients

    def lv_activation(self,module,input,output):
        if not hasattr(output[0], "requires_grad") or not output[0].requires_grad:
            return
        x=input[0];
        padding_mask=input[1];
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q (1,4,197,60),k(1,4,197,60), v(1,4,197,60)
        attn = ((q * module.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = module.attn_drop(attn) #(1,4,197,197)

        focus = attn[:, :, 0]  # (bs,num_heads,197)
        focus = focus.unsqueeze(dim=-1)  # (bs,num_heads,197,1)
        elementwise_product1 = torch.mul(focus, v)
        elementwise_product = elementwise_product1.transpose(1, 2).reshape(B, N, C)  # (bs,197,384)
        self.activations.append(elementwise_product.cpu().detach())  # (bs,197,384)
    def change_activation(self,module,input,output):
        #must register at last block.atten module becuase this module has attribute num_heads.
        # input:tuple(tensor (bs,n+1,dim), )
        # output: tensor(bs,n+1,3*dim)
        # call E: Anaconda3/envs/TensorFlow/Lib/site-packages/timm/models/vision_transformer.py  line 184
        #print("How many positional arguments:",len(input)) # 1
        #print(input[1])# 报错, 说明只有一个postional argument 一个位置实参
        #print("How many ouput items:",len(output)) # 1 说明只有输出是一个item
        if not hasattr(output[0], "requires_grad") or not output[0].requires_grad:
            # You can only register hooks on tensor requires grad.
            #print("hello")
            return
        x=input[0]
        B, N, C = x.shape  # C=num_channel, N=num_tokens=num_region+1
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q(bs,num_heads,n+1,per_dim)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attn = module.attn_drop(attn)  # attn :(bs,num_head,197,197)

        focus = attn[:, :, 0]  # (bs,num_heads,197)
        #focus2=attn[:,:,0,:] # (bs,num_heads,197) 两个值是一样的.

        focus = focus.unsqueeze(dim=-1)  # (bs,num_heads,197,1)
        elementwise_product1 = torch.mul(focus, v)  # ->(bs,num_heads,785,64)
        elementwise_product = elementwise_product1.transpose(1, 2).reshape(B, N, C)  # (bs,197,384)
        self.activations.append(elementwise_product.cpu().detach())# (bs,197,384)

    def ffvt_gradient(self,module,input,output):
        if not hasattr(input[0], "requires_grad") or not input[0].requires_grad:
            # You can only register hooks on tensor requires grad.
            print("hello, input doesn't have set reguires_grad to True")
            return
        input[0].register_hook(self.save_grad)
    def cait_prj_gradient(self,module,input,output):
        if not hasattr(input[0], "requires_grad") or not input[0].requires_grad:
            # You can only register hooks on tensor requires grad.
            print("hello, input doesn't have set reguires_grad to True")
            return
        input[0].register_hook(self.cait_grad)
    def cait_grad(self,grad):
        #grad (bs,1,384)
        #print("hello", grad.shape)
        weight=grad[:,0,:];
        self.gradients=[weight.cpu().detach()]+self.gradients
    def cait_grad2(self,modulde,input_grad,output_grad):
        gradient=output_grad[0]; #(bs,1,384)
        print("gradient shape",gradient.shape) #(bs,1,384)
        weight=gradient[:,0,:]
        self.gradients=[weight.cpu().detach()] + self.gradients

    def prj_gradient(self,module,input,output):
        if not hasattr(input[0], "requires_grad") or not input[0].requires_grad:
            # You can only register hooks on tensor requires grad.
            print("hello, input doesn't have set reguires_grad to True")
            return
        #print("hello, this is for norm")
        input[0].register_hook(self.save_grad)

    def save_grad(self,grad):
        # (bs,n+1,384) has been verified
        #weight = torch.sum(grad1, dim=1, keepdim=False)  # (bs,384)
        #print("target gradient shape:",grad.shape)#(bs,n+1,384)
        weight=grad[:,0,:] #bs,384
        self.gradients = [weight.cpu().detach()] + self.gradients

    def release(self):
        for handle in self.handles:
            handle.remove()

class ViT_Accumulation:
    def __init__(self,model,target_layers,block_layers=None,use_cuda=True,reshape_transform=reshape_transformation,arch_name=None, norm=True):
        self.model = model.eval()
        self.target_layers = target_layers # list[nn.Module]
        self.cuda = use_cuda
        self.norm=norm
        self.arch_name=arch_name

        self.reshape_transform = reshape_transform
        self.activations_and_grads= ActivationAndGradients(self.model,target_layers,block_layers, arch_name) #object
        #print("self.activation is object")

    def __call__(self,input_tensor,targets=None,aug_smooth=False,eigen_smooth=False) :
        if aug_smooth:
            return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)
        return self.forward(input_tensor,targets,eigen_smooth)# 先call 这个

    def forward(self,input_tensor,targets=None,eigen_smooth=False) :
        if self.cuda:
            input_tensor=input_tensor.cuda()
        target_size = self.get_target_width_height(input_tensor)
        # target_size (224,224)
        input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
        if "FFVT" in self.arch_name:
            score_vector,_ = self.activations_and_grads(input_tensor) #(1,1000)
        else:
            score_vector = self.activations_and_grads(input_tensor)  # (1,1000)
        # 其实return的是 self.model(input_tensor) 如果model 没有linear classifier, 那么返回的是feature_vector
        # 如果用Dino_vit_small_p8_224 那么outputs 的shape 是(1,384) =(bs, dim) 完全是feature vector的shape
        # 如果self.model 有FC as classifier, 那么返回是 score_vector(1,1000)
        print("Feature Vector shape:",score_vector.shape,"number of class:",score_vector.size(dim=1))#(1,1000)
        predicted_label = np.argmax(score_vector.cpu().data.numpy(), axis=-1)[0]  # scalar 243
        print("Predicted label is :", predicted_label)
        #print("Predication Class Name:", index2class[int(predicted_label)])

        if targets is None:
            # if targets is None, 就是要实例化一系列函数
            target_categories = np.argmax(score_vector.cpu().data.numpy(), axis=-1) #ndarray [243]
            targets = [ClassifierOutputTarget(category) for category in target_categories]
            # int category
            # function 接受的input 是model output score_vector, 然后实例化时给的信息是类别, 那么funciton 的output应该是 类别对应的vector
        score=score_vector[0,predicted_label]
        #attention_map_list=self.activations_and_grads.attentionmap_list
        feature_matrix_list = self.activations_and_grads.activations #element tensor (bs,n+1,384)

        self.model.zero_grad()
        for target, output in zip(targets, score_vector):
            temp = target(output)  # 就是返回label标签对应fclayer的向量
            #print("output is a tensor ", output.shape) #1000
            #print("target(output) is a scalar:", temp)
        # here target 是一个函数, 有属性 category 因为它本来就是用category来实例化的的, 用来挑category对应的component
        # target(score_vector,category)=score 用 category: ndarray:scalar [] 实例化一个函数, 输入的score_vector, 输出是score
        #output 是 每个类别的分数组成的向量, score_vector, target()就是找到类别对应的哪个分数, 之所以这样设计是因为原来label int 是用户指定的, 现在改由机器自己预测了.
        loss = sum([target(output) for target, output in zip(targets, score_vector)])
        # 这里sum就是多个类别的范数相加, 一般是一个类别分数, 所以loss其实就是一个score
        loss.backward(retain_graph=True)
        #score.backward(retain_graph=True) # generate graident you can see console print statement
        # 计算 d(feature_vector)/d(中间变量-非子叶节点) 的偏导数值, 偏导数组成的vector就是梯度
        # 因为在self.activation_and_grads,对象中事先定义了hook函数
        # (在hook函数中, 将形参-(一定是注册的tensor的梯度,注册的tensor为求导对象, 作为分母的部分),将 保存在对象的属性中(起全局变量的作用可依带出函数来),
        # 并注册在non-leaf tensor 上,
        # 以前没注意到顺序问题, 这里我给不同的地方注册了不同的hook 函数,
        # 这样gradient_list的第一个元素是最后一层的梯度,,所以配对相加的时候要用倒序, 不一定
        # block_gradient_list=self.activations_and_grads.block_gradients #[tensor(1,384)...]
        # num_blocks=len(block_gradient_list);

        gradients_list = self.activations_and_grads.gradients # [tensor(1,384) gradient_for_a_layer]
        num_layers = len(gradients_list);# 12
        #reverse_grad_list=[gradients_list[i] for i in range(-1, -num_layers-1,-1)]
        # 这样处理后, gradients_list[-1] 是第一层的梯度

        cam_list = self.generate_accumul_cam(gradients_list, feature_matrix_list, target_size)
        #cam 已经最大最小值normalization好了
        #cam = self.aggregate_multi_layers(cam_per_layer_list, target_size)  # 这个是多个layer 出来的cam 合并
        #cam (224,224)
        # predicted_label: scalar int
        return cam_list,predicted_label
    #####以下的function都是辅助角色

    def generate_accumul_blockcam(self,block_grad_list,attention_map_list,gradients_list,feature_matrix_list,target_size):
        # gradients_list: list[tensor :(bs,dim)]
        # feature_matrix_list: list[tensor: (bs,n,dim]
        # target_size (224,224)
        activations_list=[self.reshape_for_squaremap(a) for a in attention_map_list]
        weights_list = [g.unsqueeze(dim=-1).unsqueeze(dim=-1) for g in block_grad_list]
        activations_list = [a.cpu().data.numpy() for a in activations_list]
        grads_list = [g.cpu().data.numpy() for g in weights_list]
        cam_per_block_list = []
        num = len(activations_list)
        for i in range(num):
            gradient = grads_list[i]  # nadarray (1,dim,1,1)
            activation_map = activations_list[i]  # nadrray (1,dim ,14,14)
            weighted_activations = gradient * activation_map  # (dim*each channel)
            weighted_activations=np.maximum(weighted_activations,0)
            cam = weighted_activations.sum(axis=1)  # (1,dim, 14,14)-->(1,14,14)
            #cam = np.maximum(cam, 0)  # (1,14,14) ReLU
            if self.norm:
                cam = self.max_min_normalize(cam)  # -->(1,14,14) float each element belong to (0,1)
            cam_per_block_list.append(cam);

        # i 最大只能取到11, [0:11]实际上只能取到10
        spatial_map_list=[self.reshape_transform(a) for a in feature_matrix_list]
        coefficients_list=[g.unsqueeze(dim=-1).unsqueeze(dim=-1) for g in gradients_list]
        spatial_map_list=[a.cpu().data.numpy() for a in spatial_map_list]
        coeffi_list=[g.cpu().data.numpy() for g in coefficients_list]
        attentmap_perlayer_list=[];
        num_layers=len(spatial_map_list)
        for i in range(num_layers):
            coefficient=coeffi_list[i];
            spatial_map=spatial_map_list[i]
            linear_combination=coefficient*spatial_map
            heatmap=linear_combination.sum(axis=1)
            heatmap=np.maximum(heatmap,0)
            if self.norm:
                heatmap=self.max_min_normalize(heatmap)
            attentmap_perlayer_list.append(heatmap)
        cam_per_block_list=cam_per_block_list+attentmap_perlayer_list
        total_num=len(cam_per_block_list)
        dynamic_cam_list = [np.concatenate(cam_per_block_list[0:i], axis=0) for i in
                            range(1, total_num + 1)]

        accumulated = np.concatenate(cam_per_block_list, axis=0);  # (12,14,14)
        accumul = accumulated.sum(axis=0)  # (14,14)
        global_maximum = np.max(accumul.ravel())  # 5.79
        global_minimum = np.min(accumul.ravel())  # 0.1
        cam_list = []
        for dynamic_cam in dynamic_cam_list:
            #grade = grade + 1;
            #r=171+grade*((255-171)//num_blocks); # 170 是红黄色的分界值, 因为第一block累积的map最大值必然是1, 所以设为171
            #dynamic_cam=self.sigmod_variant(dynamic_cam)#(n_dynamic,14,14)
            accumul=dynamic_cam.sum(axis=0);#(n_dynamic,14,14)-->(14,14)
            accumul_positive=np.maximum(accumul,0); #ReLU
            normed_cam=self.global_max_min_norm2(accumul_positive,global_maximum,global_minimum)
            scaled_cam=self.scale(normed_cam,target_size)#(14,14)-->(224,224)
            cam_list.append(scaled_cam);
        return cam_list

    def generate_accumul_cam(self,gradients_list, feature_matrix_list, target_size):
        # gradients_list: list[tensor :(bs,dim)]
        # feature_matrix_list: list[tensor: (bs,n+1,dim]
        # target_size (224,224)
        activations_list = [self.reshape_transform(a) for a in feature_matrix_list]
        # activations_list[tensor:(bs,dim,w,h),...] each elemenet is each layer
        weights_list = [g.unsqueeze(dim=-1).unsqueeze(dim=-1) for g in gradients_list]
        # weights_list:list[tensor (bs,1,dim)-->(bs,dim,1)-->(bs, dim, 1,1)]
        activations_list = [a.cpu().data.numpy() for a in activations_list]
        grads_list = [g.cpu().data.numpy() for g in weights_list]
        cam_per_block_list = []
        num_blocks=len(activations_list)
        for i in range(num_blocks):
            gradient = grads_list[i]  # nadarray (1,dim,1,1)
            activation_map = activations_list[i]  # nadrray (1,dim ,14,14)
            gradient=np.maximum(gradient,0) # 只选对的gradient
            weighted_activations = gradient * activation_map  # (dim*each channel)
            #weighted_activations=np.maximum(weighted_activations,0)
            cam = weighted_activations.sum(axis=1)  # (1,dim, 14,14)-->(1,14,14)
            cam=np.maximum(cam, 0)#(1,14,14) ReLU
            if self.norm:# default False
                cam=self.max_min_normalize(cam)#-->(1,14,14) float each element belong to (0,1)
            cam_per_block_list.append(cam);

        dynamic_cam_list=[np.concatenate(cam_per_block_list[0:i],axis=0) for i in range(1,len(cam_per_block_list)+1)]
        # i 最大只能取到11, [0:11]实际上只能取到10

        accumulated=np.concatenate(cam_per_block_list, axis=0); #(12,14,14)
        accumul = accumulated.sum(axis=0)  # (14,14)
        global_maximum=np.max(accumul.ravel()) # 5.79
        global_minimum=np.min(accumul.ravel()) #0.1
        # accumul_cam = np.maximum(accumul, 0)#(14,14)
        # normed_cam=self.max_min_norm(accumul_cam) #(14,14)
        # scaled=self.scale(normed_cam, target_size) #(224,224)
        cam_list=[]
        #grade=-1
        for dynamic_cam in dynamic_cam_list:
            #grade = grade + 1;
            #r=171+grade*((255-171)//num_blocks); # 170 是红黄色的分界值, 因为第一block累积的map最大值必然是1, 所以设为171
            #dynamic_cam=self.sigmod_variant(dynamic_cam)#(n_dynamic,14,14)
            accumul=dynamic_cam.sum(axis=0);#(n_dynamic,14,14)-->(14,14)
            local_minimum=np.min(accumul)

            accumul=np.maximum(accumul,0); #ReLU
            #relative_maximum = np.max(accumul_positive.ravel())
            #relative_minimum = np.min(accumul_positive.ravel())
            #normed_cam=self.max_min_norm(accumul_positive, r);
            normed_cam=self.global_max_min_norm4(accumul,global_maximum,global_minimum)
            scaled_cam=self.scale(normed_cam,target_size)#(14,14)-->(224,224)

            cam_list.append(scaled_cam);

        return cam_list
    def max_min_normalize(self,cam):
        #ndarray cam: (n, 14,14) 第一个维度是多出来的没有意义
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

    def global_max_min_norm(self, cam, global_maximum, global_minimum):
        current_maximum=np.max(cam.ravel())
        current_minimum=np.min(cam.ravel())
        img=(cam-current_minimum)/(current_maximum-current_minimum+1e-10)
        # 第一block的map 肯定最大值是1, 最小值是0, 因为是做了标准的max_min_normal 但是这里最大值, 必然小于全局最大值, 因为累加时,每加一层
        # 这一层也是(0,1) 就是非负, 所以2层block 的累积层 最大不超过2, 最小还是0, ... 有这个规律在, 那么第一个block必然是最弱的,即使最大值也是最弱的
        # 所以设为黄色, 171, 然后累计的结果, 可见, 越加 值越大, 颜色越来越红. 这样, 累加的值, 一开始就是全局的标准.
        r=171+int ((255-171)*((current_maximum-1)/(global_maximum-1)))
        result = np.uint8(r * img)
        return result;
    def global_max_min_norm2(self,cam,global_maximum, global_minimum):
        current_maximum=np.max(cam.ravel())
        current_minimum=np.min(cam.ravel())
        img=(cam-global_minimum)/(current_maximum-global_minimum+1e-10)
        img=np.maximum(img,0)
        #img=self.sigmod_variant(img,10)
        #ratio=(current_maximum-global_minimum)/(global_maximum-global_minimum)
        ratio=(current_maximum-current_minimum)/(global_maximum-global_minimum)
        r=255*ratio
        result=np.uint8(r*img)
        return result

    def global_max_min_norm3(self,cam,global_maximum, global_minimum):
        current_maximum = np.max(cam.ravel())
        current_minimum = np.min(cam.ravel())

        img = (cam - current_minimum) / (global_maximum-global_minimum + 1e-10)
        img = np.maximum(img, 0)

        #img = np.maximum(img, 0)
        img = self.sigmod_variant(img, 100)

        result=np.uint8(255*img)
        return result
    def global_max_min_norm4(self,cam,global_maximum,global_minimum):
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
        # ndarray cam: (14,14) #
        # int range
        img=cam
        minimum = np.min(img.ravel())
        img = img - minimum
        maximum = np.max(img.ravel())+1e-10
        img = img / (maximum)
        result = np.uint8(r * img)
        return result
    def scale(self,cam_ndarray,target_size):
        # cam_ndarray :(14,14)
        img = cv2.resize(cam_ndarray, target_size,interpolation=cv2.INTER_LINEAR)
        return img
    def reshape_for_squaremap(self,tensor,height=14, width=14):
        bs, num_tokens, dim = tensor.shape #(1, 196,384)
        import math
        height = int(math.sqrt(num_tokens))
        width = height
        result = tensor.reshape(bs, height, width, dim)
        # (bs,14,14,384)-->(bs,14,384,14)-->(bs,384,14,14)
        result = result.transpose(2, 3).transpose(1, 2)
        return result;
    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmod_variant(self,x, apha=1.0):
        #y = self.sigmod(apha * (x - 0.5))
        y=self.sigmod(apha*x)
        y=y-0.5
        return y
    def correct(self,cam):
        cam[cam>1.0]=1.0;
        return cam

    def aggregate_multi_layers(self,cam_per_target_layer_list,target_size) :
        # cam_per_target_layer : list [ndarray (1,14,14),.....] 是一个list 每个元素是一个(1,14,14)的attention map
        cam_per_target_layer = np.concatenate(cam_per_target_layer_list, axis=0)# 在第二维度 0 (第一个维度是0) 拼接起来
        # (12,14,14)

        accumulation_map=np.sum(cam_per_target_layer, axis=0)#(14,14)
        # 加起来以后, 这个维度就没有了, keep_dim=False(14,14)
        #accumulation_map=np.maximum(accumulation_map,0); 这个没有必要做 因为 cam_per_target_layer 每个元素都是属于(0,1), 加起来不可能小于0,

        corrected_cam=self.correct(accumulation_map)
        maxi=np.max(corrected_cam.ravel()) #这个值可以到4.7
        nomalized_map=self.max_min_norm(accumulation_map)

        return 0

    def get_target_width_height(self,input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height
    def forward_augmentation_smoothing(self,input_tensor: torch.Tensor,targets: List[torch.nn.Module],eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,targets,eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in FAM with block: {exc_type}. Message: {exc_value}")
            return True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--model_choice",default="deit_tiny_patch16_224",choices=["deit_tiny_patch16_224,deit_small_patch16_224"])
    parser.add_argument('--use-cuda', default=True, action='store_true', help='Use NVIDIA GPU acceleration') #False
    parser.add_argument('--image-path',default='./examples/spider.JPEG',type=str,help='Input image path')
    # both.png ,spider_n01773797.JPEG
    parser.add_argument("--save_dir",default="../../../ExperimentResults/ViT_Saliencymap/",type=str);
    parser.add_argument('--method',default='Gradcam',type=str,choices=["cam,GradCam"])
    parser.add_argument('--visualize_with_bbx', default=False, action='store_true')
    parser.add_argument("--mask", default=False, action='store_true')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_args()
    if args.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'



    print("Finish!");
