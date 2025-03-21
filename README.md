# Dynamic Accumulated Attention Map
This repository contains source code necessary to reproduce the explanation map for ViT model's prediction in image classification task.
# Method Overview
To generate the attention flow for ViT model's prediction, an image is fed into a ViT model to obtain the decison-making [cls] token. During the calculation of the [cls] token, the semantic spatial map is stored by the proposed decomposition module. Then the semantic spactial map is combined linearly with the importance coefficients derived from the classification score, forming the attention map for a ViT block. According the residual structure in ViT, the attention flow is constructed by acculmulating attention maps from the first ViT block to the last ViT block. The method is depicted by the following, 

![Framwork](./.img/FrameworkDAAM.jpg)
