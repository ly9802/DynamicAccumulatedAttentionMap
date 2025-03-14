# -*- coding: utf-8 -*-
# time: 2/04/2023 22:36
# author: Steven Leo
# file: generateVideo.py
from __future__ import print_function
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, datasets, models
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import InterpolationMode


def define_fps(img_num):
    #frames per second: how many images per second
    if img_num==64:
        return 4 # 如果是64张图, 每秒4张, 一共16秒
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
    #videoWriter = cv2.VideoWriter(outputname, fourcc, fps, (1000, 1000))
    # cv_img = cv2.imread(os.path.join(inputpath,"0_"+name+".jpg"));
    # ori_height, ori_width, _ = cv_img.shape
    fps=define_fps(img_num)
    videoWriter = cv2.VideoWriter(outputname, fourcc, fps, (width, height))
    for i in range(img_num):
        img_no = i + 1
        #print(inputpath + 'video' + str(img_no) + '.jpg')
        img_file=str(img_no)+"_"+name+".jpg";
        img_path=os.path.join(inputpath,img_file)
        print("NO.",i,",image:",img_path)
        #cv_img = cv2.imread(inputpath + 'video' + str(img_no) + '.jpg', 1)
        cv_img=cv2.imread(img_path)
        videoWriter.write(cv_img)
    videoWriter.release()

def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);
def change_name(img_dir):
    import shutil
    img_list=os.listdir(img_dir);
    target_dir=os.path.join(img_dir,os.pardir,"video");

    check_path(target_dir)

    for img in img_list:
        t=img.find("Block")
        new_name=img[t+5:]
        print(new_name)
        prefix=new_name.split("_",1)[0]
        img_name=prefix+"_DAAM.jpg"
        old_name=os.path.join(img_dir,img)
        target_name=os.path.join(target_dir,img_name)
        shutil.copy(old_name,target_name)

    return target_dir


if __name__ == "__main__":

    image_dir="./DAAM/deit_small_patch16_224/"
    img_list=os.listdir(image_dir)

    new_name="DAAM"
    print(new_name)

    image_dir=change_name(image_dir)
    target_name = os.path.join(image_dir, new_name + ".avi")
    contentList=os.listdir(image_dir)
    first_img=cv2.imread(os.path.join(image_dir,contentList[0]));
    ori_height, ori_width, _ = first_img.shape
    write_video(inputpath=image_dir, outputname=target_name,img_num=len(contentList), name=new_name, height=ori_height,width=ori_width)

    print("Well Done!")
