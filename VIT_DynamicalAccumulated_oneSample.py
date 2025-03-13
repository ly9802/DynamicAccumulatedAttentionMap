# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms, datasets, models
from torchvision.transforms import InterpolationMode

from DynamicalAccumul import ViT_Accumulation,imageTotensor,ClassifierOutputTarget
from T2TViT.models.t2t_vit import T2t_vit_t_14,T2t_vit_t_19,T2t_vit_t_24
from LVViT.tlt.models import lvvit
#from EvoViT.deit import evo_deit
from CaiT import cait_models
import ml_collections
from FFVT_model import VisionTransformer
from ViTmodel import vit_small_patch16_224,vit_base_patch16_224

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import  preprocess_image #,show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from imagenet_index import index2class
from ImageNetValImageToClassDict import ImageName2Class_dict
from ImageNetClassIDToNameDict import ClassID2Name_dict
from ImageNetClassIDToNumDict import ClassID2Num_dict
from ImageNetNumToClassIDDict import Num2ClassID_dict

def image_class_information(image_name):
    from ImageNetValImageToClassDict import ImageName2Class_dict
    from ImageNetClassIDToNameDict import ClassID2Name_dict
    from ImageNetClassIDToNumDict import ClassID2Num_dict
    from ImageNetNumToClassIDDict import Num2ClassID_dict
    classID = ImageName2Class_dict[image_name]
    classNum = ClassID2Num_dict[classID]
    className = ClassID2Name_dict[classID]
    print(classID, ":", classNum, ":", className)
    return classID, classNum,className
def load_pretrained_weights(model, pretrained_weights, checkpoint_key=None):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        #print("hello")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Pretrained weights is not found at {}".format(pretrained_weights))
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
        result = colormap * 0.4 + img_array_sqaure * 0.8
        #cv2.imwrite(os.path.join(save_dir,name+"_ori.jpg"),img_array_sqaure)
    cv2.imwrite(os.path.join(save_dir,name+"_saliency.jpg"),result)
    return result
def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);
def reshape_transform(tensor, height=14, width=14):
    bs,num_tokens,dim=tensor.shape
    # print("input tensor shape:",tensor.shape) # (bs,n_rgions+1,n_channel]
    # print("input batchsize:",bs)
    # print("input num_tokens(196):",num_tokens-1)
    # print("input num_channels:",dim)
    import math
    height=int(math.sqrt(num_tokens-1))
    width=height
    # tensor.size(dim=0)=batchsize,
    # tensor.size(dim=1)=n_tokens+1
    result = tensor[:, 1:, :].reshape(tensor.size(0),height, width, tensor.size(2))
    #去掉cls_token ,result=(batchsize, 14,14,n_channel) 为啥是14, 因为以16 pixel为一个patch, 224*224 pixels, 一共有14*14 patches
    # 每个patch-->vector-->token, 因此, n_regions=14*14=196
    #(bs,197,384)-->(bs,14,14,384)
    # Bring the channels to the first dimension,
    # like in CNNs.
    #(bs,14,14,384)-->(bs,14,384,14)-->(bs,384,14,14)
    result = result.transpose(2, 3).transpose(1, 2)
    return result
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
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    config.feature_fusion= False
    config.num_token = 12
    return config
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_choice",default="vit-small",
                        choices=["cait_S24_224","FFVT","vit-small","vit-base",
                                 "T2t_vit_t_14", "T2t_vit_t_24",
                            "deit_tiny_patch16_224","deit_small_patch16_224"])
    parser.add_argument('--pretrained_weights', default='../../../PretrainedModels/DINO/dino_deitsmall8_pretrain.pth',
                        type=str, help="Path to pretrained weights to evaluate.")
    # 不用改, 后面根据 model_choice 更改
    parser.add_argument('--use-cuda', default=True, action='store_true', help='Use NVIDIA GPU acceleration') #False
    parser.add_argument('--image-path',default='../../ImageNetSamples/OriginalImage/ILSVRC2012_val_00000714.JPEG',type=str,help='Input image path')

    # ./examples/Common_Yellowthroat_0049_190708.jpg
    #./examples/Common_Yellowthroat_0055_190967.jpg
    # White_Breasted_Nuthatch_0108_86308.jpg 93
    # White_Crowned_Sparrow_0092_125934.jpg 131
    # Red_Headed_Woodpecker_0007_182728.jpg 190
    # ILSVRC2012_val_00023185.JPEG white shark
    # ./examples/Common_Yellowthroat_0055_190967.jpg
    # ./examples/White_Crowned_Sparrow_0092_125934.jpg
    # ILSVRC2012_val_00012619.JPEG beaver
    # ../../../ImageNetSamples/OriginalImage/ILSVRC2012_val_00000714.JPEG
    #../../../dataset/ImageNet2012/val/n02504458/ILSVRC2012_val_00041799.JPEG elephant
    #../../../dataset/ImageNet2012/val/n02111889/ILSVRC2012_val_00009061.JPEG  Samoyede
    #../../../dataset/ImageNet2012/val/n02701002/ILSVRC2012_val_00003703.JPEG ambulance
    #../../../dataset/ImageNet2012/val/n03770679/ILSVRC2012_val_00007374.JPEG minivan
    #../../../dataset/ImageNet2012/val/n01773797/ILSVRC2012_val_00047551.JPEG
    #../../ImageNetSamples/OriginalImage/ILSVRC2012_val_00013393.JPEG kite
    #../../ImageNetSamples/OriginalImage/ILSVRC2012_val_00000714.JPEG
    # ILSVRC2012_val_00049426.JPEG meerkat
    # ILSVRC2012_val_00049074.JPEG tou ying yi  projector
    # ILSVRC2012_val_00007946
    # ILSVRC2012_val_00031095.JPEG giant panada
    # both.png ,spider_n01773797.JPEG
    # ILSVRC2012_val_00007143.JPEG fly
    # ILSVRC2012_val_00046384.JPEG spider n01773797
    # ILSVRC2012_val_00005835.JPEG #bowl
    # ILSVRC2012_val_00048482.JPEG  frog
    # ILSVRC2012_val_00002815.JPEG  lingyang n02422699
    # ILSVRC2012_val_00006969.JPEG   鹤
    # ILSVRC2012_val_00012653.JPEG   凯旋门
    # ILSVRC2012_val_00013393.JPEG   老鹰
    # ILSVRC2012_val_00020075.JPEG   油轮
    # ILSVRC2012_val_00021086.JPEG   虫
    # ILSVRC2012_val_00046307.JPEG   鸟
    # ILSVRC2012_val_00003766.JPEG  you
    # ILSVRC2012_val_00035908.JPEG white wolf
    # ILSVRC2012_val_00016446.JPEG carbonara
    # ILSVRC2012_val_00004210.JPEG cockroach
    # ILSVRC2012_val_00044474.JPEG lawn
    # ILSVRC2012_val_00042876.JPEG butterfly
    # ILSVRC2012_val_00000714.JPEG train
    # ILSVRC2012_val_00002315.JPEG English springer
    # ILSVRC2012_val_00048310.JPEG Leonberg dog
    # ILSVRC2012_val_00003315.JPEG dogsled; dog sled; dog sleigh
    # ILSVRC2012_val_00005470.JPEG ice bear; polar bear
    # ILSVRC2012_val_00028670.JPEG king penguin
    # ILSVRC2012_val_00048284.JPEG  basketball
    # ILSVRC2012_val_00045377.JPEG  volleyball
    # ILSVRC2012_val_00003781.JPEG  hanta
    # ILSVRC2012_val_00002140.JPEG  monkey
    # ILSVRC2012_val_00008749.JPEG snake
    # ILSVRC2012_val_00034442.JPEG panda
    # ILSVRC2012_val_00037442.JPEG  dadan dog

    # ILSVRC2012_val_00012619.JPEG fusu
    # ILSVRC2012_val_00035908.JPEG white wolf
    # ILSVRC2012_val_00049074.JPEG projector

    parser.add_argument("--save_dir",default="../../ExperimentResults/PR_sameImage/",type=str);
    parser.add_argument('--norm', default=False, action='store_true', help='')
    parser.add_argument('--aug_smooth', default=False,action='store_true',help='')
    parser.add_argument('--eigen_smooth',default=False,action='store_true',help='')
    parser.add_argument('--method',default='daam_vit',type=str,
                        choices=["daam_vit","gradcam","gradcam++","scorecam","xgradcam","ablationcam","layercam","fullgrad","eigencam","eigengradcam"])
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args
if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    methods_dict = {"gradcam": GradCAM,"scorecam": ScoreCAM, "gradcam++": GradCAMPlusPlus,
               "ablationcam": AblationCAM,"xgradcam": XGradCAM,"eigencam": EigenCAM,
                "eigengradcam": EigenGradCAM,"layercam": LayerCAM,"fullgrad": FullGrad,
                "daam_vit":GradCAM_ViT_Accumulation}

    if args.method not in list(methods_dict.keys()):
        raise Exception(f"method should be one of {list(methods_dict.keys())}")
    checkpoint_key = None;
    # if args.model_choice=="dino_vits8":
    #     model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')  # dino_vits8 没有linear classifier 所以不能用
    # elif args.model_choice=="deit_small_patch16_224":
    #     model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
    # elif args.model_choice=="deit_tiny_patch16_224":
    #     model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    # else:
    #     model = torch.hub.load('facebookresearch/deit:main','deit_tiny_patch16_224', pretrained=True)
    if args.model_choice=="deit_small_patch16_224":
        model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=False)
        args.pretrained_weights="../../../PretrainedModels/DeiT/deit_small_patch16_224-cd65a155.pth"
        checkpoint_key = "model"
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key)
    elif args.model_choice=="deit_tiny_patch16_224":
        model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        args.pretrained_weights = "../../../PretrainedModels/DeiT/deit_tiny_patch16_224-a1311bcf.pth"
        checkpoint_key = "model"
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key)
    elif args.model_choice=="T2t_vit_t_14":
        model = T2t_vit_t_14(pretrained=False)# depth 14, embed_dim=384, img_size=224
        args.pretrained_weights = "../../../PretrainedModels/T2TViT/81.7_T2T_ViTt_14.pth.tar"
        checkpoint_key="state_dict_ema"
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key)
    elif args.model_choice=="T2t_vit_t_24":
        model=T2t_vit_t_24(pretrained=False) # depth 24, embed_dim=384, img_size=224
        args.pretrained_weights = "../../../PretrainedModels/T2TViT/82.6_T2T_ViTt_24.pth.tar"
        checkpoint_key = "state_dict_ema"
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key)
    elif args.model_choice=="lvvit_t":
        model=create_model("lvvit_t",pretrained=False,drop_rate=0,drop_path_rate=None,drop_block_rate=None,
                           global_pool=None,bn_tf=False,bn_momentum=None,bn_eps=None,scriptable=False,
                           checkpoint_path="../../../PretrainedModels/LVViT/lvvit_t.pth", img_size=224)
        args.pretrained_weights ="../../../PretrainedModels/LVViT/lvvit_t.pth"
        checkpoint_key = "state_dict"
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key)
    elif args.model_choice=="cait_S24_224":
        #model = create_model("cait_S24_224", pretrained=False)
        model=cait_models.cait_S24_224(pretrained=True)
        args.pretrained_weights = "../../../PretrainedModels/CaiT/S24_224.pth"
        checkpoint_key = "model"
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key)
    elif args.model_choice=="FFVT":
        config=get_b16_config()
        config.feature_fusion = True
        config.num_token = 12
        model = VisionTransformer(config,448, zero_head=True,num_classes=200,vis=True,smoothing_value=0.0,dataset="CUB")
        checkpoint_key = "model"
        args.pretrained_weights ="../../../PretrainedModels/FFVT/CUB.bin"
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key)
    elif args.model_choice=="vit-small":
        model=vit_small_patch16_224(pretrained=True)
        checkpoint_key = "model"
        args.pretrained_weights = "../../../PretrainedModels/ViT/vit_small_p16_224-15ec54c9.pth"
    elif args.model_choice=="vit-base":
        model = vit_base_patch16_224(pretrained=True)
        checkpoint_key = "model"
        args.pretrained_weights = "../../../PretrainedModels/ViT/jx_vit_base_p16_224-80ecf9dd.pth"
    else:
        model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
        args.pretrained_weights = "../../../PretrainedModels/DeiT/deit_small_patch16_224-cd65a155.pth"
        checkpoint_key="model"
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key)
    #deit_small_patch8_224


    model.eval()
    if args.use_cuda:
        model = model.cuda()
    target_layers=[];
    target_block_list=[];
    # 是所有层都设一个钩子函数, 所以不是自由一层, 这和以前不一样
    # for no, block_no in enumerate(model.blocks[:]):
    #     #print(no)
    #     target_layers.append((block_no.attn, block_no.attn.proj))
    if "deit" in args.model_choice:
        for no, block_no in enumerate(model.blocks[:]):
            target_layers.append((block_no.attn, block_no.attn.proj))
    elif "T2t" in args.model_choice:
        for no, block_no in enumerate(model.blocks[:]):
            target_layers.append((block_no.attn, block_no.attn.proj))
    elif "cait" in args.model_choice:
        # for no, block_no in enumerate(model.blocks[:]):
        #     target_block_list.append(block_no)# 前面24block没有clss_token所以直接用output做attention map的
        #     # 那么这个就不需要我的方法, 同时梯度的求法也不同.
        for no, block_no in enumerate(model.blocks_token_only[:]):
            target_layers.append((block_no.attn, block_no.attn.proj))
    elif "FFVT" in args.model_choice:
        for no,block_no in enumerate(model.transformer.encoder.layer[:]):
            target_layers.append((block_no.attn, block_no.attn.out))
            #print("well")
    else:
        for no, block_no in enumerate(model.blocks[:]):
            target_layers.append((block_no.attn, block_no.attn.proj))
    #num=-1
    #block_no=model.blocks[num]
    #(norm1->atten->norm2->MLP) including attention moduel and MLP
    # define target_layers
    # ViT model has attributes : .blocks : list 可以用[-1]访问, head(classifer),norm(layerNorm)最后一个norm, Path_embed,pos_drop(Dropout(p=0)
    # 是不是module 含不含可训练的parameter, 有()提示.
    if args.method not in methods_dict:
        raise Exception(f"Method {args.method} not implemented")
    if args.method == "ablationcam":
        cam_function = methods_dict[args.method](model=model,target_layers=target_layers,use_cuda=args.use_cuda,reshape_transform=reshape_transform,ablation_layer=AblationLayerVit())
    elif args.method=="daam_vit":
        print("method:", args.method, methods_dict[args.method])  # "grad-cam"
        #cam_function = GradCAM_ViT_Accumulation(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        cam_function = GradCAM_ViT_Accumulation(model=model, target_layers=target_layers,block_layers=target_block_list,
                                                arch_name=args.model_choice, norm=args.norm)
        print("cam_function is a funciton")
    else:
        print("method:",args.method, methods_dict[args.method])# "grad-cam"
        cam_function = methods_dict[args.method](model=model,target_layers=target_layers,use_cuda=args.use_cuda,reshape_transform=reshape_transform)
        print("cam_function is a funciton")
    if "FFVT" in args.model_choice:
        trans=transform_function(448)
    else:
        trans = transform_function(resolution=224)
    print("input image:",args.image_path)

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1] #ndarray[h,w,3]
    # rgb_img = cv2.resize(rgb_img, (224, 224))
    # rgb_img = np.float32(rgb_img) / 255 #(0,255)-->(0,1)
    #input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #for ImageNet
    #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #input_tensor = imageTotensor(args.image_path) # 这个方式读取,会有点问题, 造成attention很小
    input_tensor = trans(read_image(args.image_path)).unsqueeze(dim=0)  # 这个读就判断对了
    targets = None
    # If None, returns the map for the highest scoring category. # Otherwise, targets the requested category.
    save_path=os.path.join(args.save_dir,args.model_choice)
    check_path(save_path)

    image_name = args.image_path.split("/")[-1]
    prefix = image_name.split(".", 1)[0]
    is_CUB=False
    if is_CUB==True:
        pass
        ClassID="CUB200";
        ground_truth_num=199;
        ground_truth_className=prefix
    else:

        ClassID, ground_truth_num, ground_truth_className = image_class_information(image_name)
        print("Ground truth class id:", ClassID, "class label:", ground_truth_num, ",class name:", ground_truth_className);


    cam_list, predicted_label = cam_function(input_tensor=input_tensor,targets=targets,eigen_smooth=args.eigen_smooth,aug_smooth=args.aug_smooth)
    #(224,224) : each element is integer.  prediected_label: int64

    block_num = 0;
    for cam_item in cam_list:
        block_num=block_num+1;
        file_name = f'{prefix}_{args.method}_{args.model_choice}_DAAMBlock{block_num}.jpg';
        file_path = os.path.join(args.save_dir, file_name)
        show_image_without_boundingbox(cam_item, args.image_path, file_name, save_path, original=False)
    print("model",args.model_choice)
    print("well done!");
