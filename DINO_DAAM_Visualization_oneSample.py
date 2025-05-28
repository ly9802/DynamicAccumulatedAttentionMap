# -*- coding: utf-8 -*-
# time: 28/04/2023 01:35
# author: Yi Liao (Steven)
# file: self-supervised Learning.py
from __future__ import print_function
import os
import sys
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, datasets, models
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import InterpolationMode

from torch.optim import lr_scheduler, SGD
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
cudnn.benchmark = True
# if you want to import local directory as a module, you should add "sys.append(str local_dirctory)
from utils import vision_transformer as vits
from utils.imagenet_index import index2class
from utils.ImageNetValImageToClassDict import ImageName2Class_dict
from utils.ImageNetClassIDToNameDict import ClassID2Name_dict
from utils.ImageNetClassIDToNumDict import ClassID2Num_dict
from utils.SelfSupervisedViT_DAAM import DynamicAccumulatedAttentionMap

def define_fps(img_num):

    if img_num==64:
        return 4
    elif img_num==512:
        return 8
    elif img_num==640:
        return 8
    elif img_num==1024:
        return 16
    elif img_num==2048:
        return 16
    elif img_num==12:
        return 1
    elif img_num==14:
        return 1
    elif img_num==24:
        return 2
    else:
        return 1
def write_video(inputpath, outputname, img_num, name,height,width,fps=5):
    """Generate videos
    Args:
        input_path: the path for input Images
        output_name: the output name for the video
        img_num: the number of the input Images
        fps: frames per second: how many images per second
    """

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    fps=define_fps(img_num)
    videoWriter = cv2.VideoWriter(outputname, fourcc, fps, (width, height))
    for i in range(img_num):
        img_no = i + 1

        img_file=str(img_no)+"_"+name+".jpg";
        img_path=os.path.join(inputpath,img_file)
        print("NO.",i,",image:",img_path)

        cv_img=cv2.imread(img_path)
        videoWriter.write(cv_img)
    videoWriter.release()

def change_name(img_dir):
    import shutil
    img_list = os.listdir(img_dir);
    target_dir = os.path.join(img_dir, os.pardir, "video");
    check_path(target_dir)
    for img in img_list:
        t = img.find("Block")
        new_name = img[t + 5:]

        prefix = new_name.split("_", 1)[0]
        img_name = prefix + "_DAAM.jpg"
        old_name = os.path.join(img_dir, img)
        target_name = os.path.join(target_dir, img_name)
        shutil.copy(old_name, target_name)
    return target_dir
def remove_image(image_dir):
    for item in os.listdir(image_dir):
        if item.endswith(".jpg"):
            os.remove(os.path.join(image_dir, item))
def transform_function(resolution=(224,224)):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize((resolution), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    trans_test = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(resolution), transforms.ToTensor(), normalize])
    trans_pil = transforms.Compose([transforms.Resize((resolution))])
    to_pil = transforms.ToPILImage();
    return transform_test
def check_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    img=None
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")
def contribution_weight(product):
    simi_score = torch.sum(product, dim=-1, keepdim=False);
    print("average similarity over all corrected samples in K neighbors:", simi_score)
    if simi_score!= 0:
        contri_weight = torch.div(product, simi_score.abs())
        return contri_weight;
    else:
        contri_weight = product
        return contri_weight;

def max_min_normal(coefficient):
    if coefficient.max()>coefficient.min():
        normalized=torch.div(coefficient-coefficient.min(),coefficient.max()-coefficient.min())
    else:
        normalized=torch.ones_like(coefficient)
    return normalized

def reshape_transform(tensor, height=28, width=28):
    bs, num_tokens, dim = tensor.shape
    import math
    height = int(math.sqrt(num_tokens - 1))
    width = height
    result = tensor[:, 1:, :].reshape(bs,height,width, dim)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def max_min_tensor(feature_map):
    h,w=feature_map.shape
    map=feature_map
    map=map-map.min()
    new_map=torch.div(map,map.max())
    return new_map

def show_image_without_boundingbox(heatmap,img_path, name,save_dir,original=True):

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
        result = colormap * 0.4 + img_array_sqaure * 0.8
    cv2.imwrite(os.path.join(save_dir,name+"_saliencymap.jpg"),result)
    #cv2.imshow("img_bbx.jpg", img_array)
    return result
def image_class_information(image_name):
    classID = ImageName2Class_dict[image_name]
    classNum = ClassID2Num_dict[classID]
    className = ClassID2Name_dict[classID]
    print(classID, ":", classNum, ":", className)
    return classID, classNum,className
def filter_fam(feature_map, threshold=0.90, args=None):
    # featuremap:tensor (bs,28,28) #threshod=0.6
    bs, h_featmap, w_featmap = feature_map.shape
    for i in range(bs):
        feature_map[i]=max_min_tensor(feature_map[i])
    attentions = feature_map.contiguous().view(bs, -1)
    val, idx = torch.sort(attentions, dim=-1,descending=False)  # (num_head,784),  dim default=-1,descending default False
    val /= torch.sum(val, dim=-1, keepdim=True)  # 每个值除以和,就是比例, 从大到小排列 最大不是1, 但是加起来 就是1,
    cumval = torch.cumsum(val, dim=-1)  # 累计和
    th_attn = cumval > threshold  # 累计和>0.4的, 逐元素比, 比0.4大的返回True, otherwise返回False 因为加起来有负值, 所以设0 没什么用
    # idx2=torch.argsort(idx)
    idx2 = torch.argsort(idx, dim=-1, descending=False)  # idx的值按升序排列, 把index返回, 这样就复原,等于torch.sort没做
    for img_item in range(bs):
        th_attn[img_item] = th_attn[img_item][idx2[img_item]]  # 把th_attn 每个元素是true or false 按原来的原先的空间顺序排好,恢复原来的空间顺序, 逐个
    th_attn = th_attn.reshape(bs, w_featmap, h_featmap).float()  # 这里用.float()把bool型转换 数值型.
    # interpolate # nearest
    # th_attn = F.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
    #print(th_attn)
    return th_attn

def contri_weight_generate(feature_vector,memory_bank, train_labels,k=20,T=0.07,num_classes=1000,args=None):

    batch_size=feature_vector.size(dim=0)
    retrieval_one_hot = torch.zeros(k, num_classes).to(memory_bank.device)

    feature_vector_norm=F.normalize(feature_vector,dim=1,p=2,eps=1e-12)
    similarity = torch.mm(feature_vector_norm, memory_bank)

    distances, indices = similarity.topk(k, largest=True, dim=-1, sorted=True)
    selected_memorybank=torch.index_select(memory_bank, dim=-1, index=indices.squeeze(dim=0))
    candidates = train_labels.contiguous().view(1, -1).expand(batch_size, -1)

    retrieved_neighbors = torch.gather(candidates, dim=1,index=indices)


    selected_train_labels = retrieved_neighbors.contiguous().view(-1, 1)

    retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
    retrieval_one_hot.scatter_(1, selected_train_labels, 1)

    distances_transform = distances.clone().div_(T).exp_()

    temp3 = retrieval_one_hot.contiguous().view(batch_size, -1, num_classes)
    temp4 = distances_transform.contiguous().view(batch_size, -1, 1)
    temp5 = torch.mul(temp3, temp4)
    probs = torch.sum(temp5, dim=1, keepdim=False)
    _, predictions = probs.sort(dim=1, descending=True)
    predicted_class = predictions[:, 0].squeeze(dim=0)
    class_label = predicted_class.cpu().item()
    print("DINO predict label:", index2class[class_label])
    mask = torch.eq(retrieved_neighbors, predicted_class)

    num_channels=feature_vector.size(dim=1)
    mask2 = mask.expand(num_channels, -1)
    predictedclass_memorybank = torch.masked_select(selected_memorybank, mask2).contiguous().view(num_channels, -1);
    print("class memory bank", predictedclass_memorybank.shape)

    multiply = torch.mul(feature_vector_norm.unsqueeze(dim=2), predictedclass_memorybank.unsqueeze(dim=0))

    mean_multiply=torch.mean(multiply,dim=-1,keepdim=False)

    contri_weight = contribution_weight(mean_multiply[0])
    contri_weight = max_min_normal(contri_weight)

    return contri_weight,class_label

def generate_args():
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--image-path', default='./inputimage/ILSVRC2012_val_00046307.JPEG', type=str,
                        help='Input image path')
    # ILSVRC2012_val_00046307.JPEG bee eater
    # ILSVRC2012_val_00013393.JPEG kite
    # ILSVRC2012_val_00006969.JPEG crane
    parser.add_argument('--method', default='DAAM', type=str, choices=["DAAM"])
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--norm",default=False,action='store_true')
    parser.add_argument('--target_dir', default="./DAAM/", type=str)

    parser.add_argument('--model_choice', default="dino_small_p8", type=str, help='Architecture', choices=["dino_small_p8"])
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='./pretrainedmodels/dino_deitsmall8_pretrain.pth',
                        type=str, help="Path to pretrained weights to evaluate.")

    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument("--num_chunks",default=1000, type=int, help="how many parts in test images and varies according to GPU capacity") #100
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature used in the voting coefficient')

    parser.add_argument('--use_cuda', default=True, action='store_true',help="Should we store the features on GPU?")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--checkpoint_key", default="teacher", type=str,help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default="./memorybank/DINO/ImageNet2012/",help='Path where to save memorybank')

    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')  # 10
    parser.add_argument("--dist_url", default="env://", type=str,help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--threshold", default=-1, type=float)
    parser.add_argument("--generateVideo", default=True, action='store_true', help='')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args
if __name__ == "__main__":

    torch.cuda.empty_cache()

    args = generate_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.gpu_id = int(args.gpu)

    memorybank_dir = os.path.join(args.dump_features, args.model_choice)
    check_path(memorybank_dir)
    print("memory bank is at{}".format(memorybank_dir))

    train_features = torch.load(os.path.join(memorybank_dir, "trainfeat.pth"))
    test_features = torch.load(os.path.join(memorybank_dir, "testfeat.pth"))
    train_labels = torch.load(os.path.join(memorybank_dir, "trainlabels.pth"))
    test_labels = torch.load(os.path.join(memorybank_dir, "testlabels.pth"))
    print("load feature from memory bank {}".format(memorybank_dir))

    if args.use_cuda:
        train_features = train_features.cuda(args.gpu_id)
        test_features = test_features.cuda(args.gpu_id)
        train_labels = train_labels.cuda(args.gpu_id)
        test_labels = test_labels.cuda(args.gpu_id)

    print("Features are ready!\nStart the k-NN classification.")

    # ============ building network ... ============
    arch_name = "vit_small";
    print(f"Model {args.model_choice} {args.patch_size}x{args.patch_size} built.")
    model = vits.__dict__[arch_name](patch_size=args.patch_size, num_classes=0) #num_classes=0 means linear classfier (FC layer) is not needed
    print("model name is {},patch size is {}".format(args.model_choice, args.patch_size))

    load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, arch_name, args.patch_size)
    if args.use_cuda:
        model.cuda(args.gpu_id)
    model.eval()

    target_class_path = args.target_dir + args.model_choice + "/"
    check_path(target_class_path)


    target_layer_list= [];
    for no, block_no in enumerate(model.blocks[:]):
        target_layer_list.append((block_no.attn, block_no.attn.proj))


    fam_function = DynamicAccumulatedAttentionMap(model=model, target_layers=target_layer_list, reshape_transform=reshape_transform,norm=args.norm)

    img_path=args.image_path
    trans=transform_function()

    image_name=img_path.split("/")[-1]
    prefix = image_name.split(".", 1)[0]
    truth_id, truth_classNum, truth_className = image_class_information(image_name)
    print("Ground truth class id:", truth_id, "class label:", truth_classNum, ",class name:", truth_className);
    img_tensor = trans(read_image(img_path)).unsqueeze(dim=0)
    if args.use_cuda:
        img_tensor=img_tensor.cuda(args.gpu_id)
    feature_vector = fam_function(img_tensor)
    contri_weight,predicted_label =contri_weight_generate(feature_vector,memory_bank=train_features.t(), train_labels=train_labels,k=20,T=args.temperature,num_classes=1000,args=args)

    print("predict label", predicted_label)
    if predicted_label==truth_classNum:
        print("correct prediction!")
    else:
        print("wrong prediction!")

    saliency_map_list= fam_function.generate_fam(feature_vector, contri_weight.unsqueeze(dim=0)) #note gradients , feature_matrix are on CPU
    block_num=0;
    for saliency_map in saliency_map_list:
        block_num=block_num+1;
        file_name = f'{prefix}_{args.method}_{args.model_choice}_DynamicalAccumulBlock{block_num}.jpg';
        show_image_without_boundingbox(saliency_map, img_path=img_path, name=file_name, save_dir=target_class_path,original=False)
    print("model", args.model_choice)
    if args.generateVideo:
        image_dir=target_class_path
        image_dir = change_name(image_dir)
        new_name = "DAAM"
        target_name = os.path.join(image_dir, prefix + ".avi")
        contentList = os.listdir(image_dir)
        first_img = cv2.imread(os.path.join(image_dir, contentList[0]));
        ori_height, ori_width, _ = first_img.shape
        write_video(inputpath=image_dir, outputname=target_name, img_num=len(contentList), name=new_name, height=ori_height, width=ori_width)
        remove_image(image_dir)
    print("Well Done!")
