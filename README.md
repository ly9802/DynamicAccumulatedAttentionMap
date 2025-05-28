# Dynamic Accumulated Attention Map
This repository contains source code necessary to reproduce the explanation map for ViT model's prediction in image classification task.
# Method Overview
To generate the attention flow for ViT model's prediction, an image is fed into a ViT model to obtain the decison-making [cls] token. During the calculation of the [cls] token, the semantic spatial map is stored by the proposed decomposition module. Then the semantic spactial map is combined linearly with the importance coefficients derived from the classification score, forming the attention map for a ViT block. According the residual structure in ViT, the attention flow is constructed by acculmulating attention maps from the first ViT block to the last ViT block. The method is depicted by the following, 

![Framwork](./.img/FrameworkDAAM.jpg)

# Dynamic Accumulated Attention Map for Self-Supervised ViT
To generate the attention flow for self-supervised ViT models, the first step is to download the pretrained models. In this work, we can download the pretrained self-supervised ViT models from [DINO][https://github.com/facebookresearch/dino] and [XCiT][https://github.com/facebookresearch/xcit]. The second step is to generate the corresponding memory bank by following the algorithm in 2018 CVPR unsupervised learning paper ["Unsupervised Feature Learning via Non-parameteric Instance Discrimination"] [http://arxiv.org/pdf/1805.01978]  
