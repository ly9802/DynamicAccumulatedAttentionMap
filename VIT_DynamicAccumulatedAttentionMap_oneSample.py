# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms, datasets, models
from torchvision.transforms import InterpolationMode

from utils.DAAM import DynamicAccumulatedAttentionMap

from utils.imagenet_index import index2class
from utils.ImageNetValImageToClassDict import ImageName2Class_dict
from utils.ImageNetClassIDToNameDict import ClassID2Name_dict
from utils.ImageNetClassIDToNumDict import ClassID2Num_dict
from utils.ImageNetNumToClassIDDict import Num2ClassID_dict

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
def image_class_information(image_name):
    classID = ImageName2Class_dict[image_name]
    classNum = ClassID2Num_dict[classID]
    className = ClassID2Name_dict[classID]
    print(classID, ":", classNum, ":", className)
    return classID, classNum,className
def load_pretrained_weights(model, pretrained_weights, checkpoint_key="model"):
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
        print("Pretrained weights is not found at {}".format(pretrained_weights))
def show_image_without_boundingbox(heatmap,img_path, name,save_dir,original=True):

    name = name.split(".", 1)[0]
    h,w=heatmap.shape
    img_array=cv2.imread(img_path, cv2.IMREAD_COLOR)
    ori_height, ori_width, _ = img_array.shape

    if original==True:
        heatmap_originalsize=cv2.resize(heatmap, (ori_width, ori_height),interpolation=cv2.INTER_LINEAR)
        colormap = cv2.applyColorMap(heatmap_originalsize, cv2.COLORMAP_JET);
        result = colormap * 0.4 + img_array * 0.8
    else:
        img_array_sqaure = cv2.resize(img_array, (w, h),interpolation=cv2.INTER_LINEAR)
        colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET);
        result = colormap * 0.4 + img_array_sqaure * 0.8

    cv2.imwrite(os.path.join(save_dir,name+"_DAAM.jpg"),result)
    return result
def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);
def transform_function(resolution=224):
    transform_test = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    to_pil = transforms.ToPILImage();
    return transform_test
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
def remove_image(image_dir):
    for item in os.listdir(image_dir):
        if item.endswith(".jpg"):
            os.remove(os.path.join(image_dir, item))
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_choice",default="deit_small_patch16_224",
                        choices=["deit_tiny_patch16_224","deit_small_patch16_224"])
    parser.add_argument('--pretrained_weights', default=None,type=str, help="Path to pretrained weights")
    
    parser.add_argument('--use-cuda', default=True, action='store_true', help='Use NVIDIA GPU acceleration')
    parser.add_argument("--gpu",default="0",type=str)
    parser.add_argument('--image-path',default='./InputImage/ILSVRC2012_val_00046384.JPEG',type=str,help='Input image path')
 
    # ILSVRC2012_val_00002815.JPEG impala
    # ILSVRC2012_val_00046384.JPEG spider
    # ILSVRC2012_val_00012653.JPEG triumphal arch
    
     
    parser.add_argument("--save_dir",default="./DAAM/",type=str);
    parser.add_argument('--norm', default=False, action='store_true', help='')

    parser.add_argument('--method',default='daam_vit',type=str,choices=["daam_vit"])
    parser.add_argument("--generateVideo",default=True,action='store_true', help='')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args
if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.gpu_id=int(args.gpu)
    if args.pretrained_weights==None:
        model = torch.hub.load('facebookresearch/deit:main', args.model_choice, pretrained=True)
    else:
        model = torch.hub.load('facebookresearch/deit:main', args.model_choice, pretrained=False)
        load_pretrained_weights(model, args.pretrained_weights, "model")
    model.eval()
    if args.use_cuda:
        model = model.cuda(args.gpu_id)
    target_layer_list=[(block_no.attn, block_no.attn.proj) for no, block_no in enumerate(model.blocks[:])];
    target_block_list=[];
    print("method:", args.method)
    daam = DynamicAccumulatedAttentionMap(model=model, target_layers=target_layer_list,block_layers=target_block_list,
                                          arch_name=args.model_choice, norm=args.norm,gpu_id=args.gpu_id)
    print("cam_function is a funciton")

    trans = transform_function(resolution=224)
    print("input image:",args.image_path)

    input_tensor = trans(read_image(args.image_path)).unsqueeze(dim=0)
    target_label = None

    image_name = args.image_path.split("/")[-1]
    prefix = image_name.split(".", 1)[0]

    save_path = os.path.join(args.save_dir, args.model_choice, prefix)
    check_path(save_path)

    ClassID, ground_truth_num, ground_truth_className = image_class_information(image_name)
    print("Ground Truth Class ID:{}, Class Label:{}, Class Name:{}".format(ClassID, ground_truth_num,  ground_truth_className));

    cam_list, predicted_label = daam(input_tensor=input_tensor,target_label=target_label)

    block_num = 0;
    for cam_item in cam_list:
        block_num=block_num+1;
        file_name = f'{prefix}_{args.model_choice}_Block{block_num}.jpg';

        show_image_without_boundingbox(cam_item, args.image_path, file_name, save_path, original=False)
    print("model",args.model_choice)
    if args.generateVideo:
        image_dir=save_path
        image_dir = change_name(image_dir)
        new_name = "DAAM"
        target_name = os.path.join(image_dir, prefix + ".avi")
        contentList = os.listdir(image_dir)
        first_img = cv2.imread(os.path.join(image_dir, contentList[0]));
        ori_height, ori_width, _ = first_img.shape
        write_video(inputpath=image_dir, outputname=target_name, img_num=len(contentList), name=new_name,
                    height=ori_height, width=ori_width)
        remove_image(image_dir)

    print("well done!");
