# Knowledge Distillation from A Stronger Teacher (DIST)  
Official implementation of paper "Knowledge Distillation from A Stronger Teacher" (DIST).  
By Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu.

:fire: **DIST: state-of-the-art KD performance with an extremely simple implementation (simply replace the KLDivLoss in vanilla KD).**

## Updates  
Code is coming soon.

## Results  
### ImageNet

* Baseline settings (`R34-R101` and `R50-MBV1`):  
    |Student|Teacher|DIST|Log|
    |:--:|:--:|:--:|:--:|
    |ResNet-18 (69.76)|ResNet-34 (73.31)|72.07|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/baseline_Res34-Res18.txt)|
    |MobileNet V1 (70.13)|ResNet-50 (76.16)|73.24|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/baseline_Res50-MBV1.txt)|

* Stronger teachers (`R18` and `R34` students with various ResNet teachers):  
    |Student|Teacher|KD (T=4)|DIST|Log|
    |:--:|:--:|:--:|:--:|:--:|
    |ResNet-18 (69.76)|ResNet-34 (73.31)|71.21|72.07||
    |ResNet-18 (69.76)|ResNet-50 (76.13)|71.35|72.12||
    |ResNet-18 (69.76)|ResNet-101 (77.37)|71.09|72.08||
    |ResNet-18 (69.76)|ResNet-152 (78.31)|71.12|72.24||
    |ResNet-34 (73.31)|ResNet-50 (76.13)|74.73|75.06||
    |ResNet-34 (73.31)|ResNet-101 (77.37)|74.89|75.36||
    |ResNet-34 (73.31)|ResNet-152 (78.31)|74.87|75.42||
    
* Stronger training strategies:  
    `ResNet-50-SB`: stronger ResNet-50 trained by [TIMM](https://github.com/rwightman/pytorch-image-models) ([ResNet strikes back](https://arxiv.org/abs/2110.00476)) .
    |Student|Teacher|KD (T=4)|DIST|Log|
    |:--:|:--:|:--:|:--:|:--:|
    |ResNet-18 (73.4)|ResNet-50-SB (80.1)|72.6|74.5|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_Res50SB-Res18.txt)|
    |ResNet-34 (76.8)|ResNet-50-SB (80.1)|77.2|77.8|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_Res50SB-Res34.txt)|
    |MobileNet V2 (73.6)|ResNet-50-SB (80.1)|71.7|74.4|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_Res50SB-MBV2.txt)|
    |EfficientNet-B0 (78.0)|ResNet-50-SB (80.1)|77.4|78.6|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_Res50SB-EfficientNetB0.txt)|
    |ResNet-50 (78.5)|Swin-L (86.3)|80.0|80.2|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_SwinL-Res50.txt)|
    |Swin-T (81.3)|Swin-L (86.3)|81.5|82.3|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_SwinL-SwinT.txt)|


### COCO Detection  
|Student|Teacher|DIST|DIST+mimic|Config|Log|
|:--:|:--:|:--:|:--:|:--:|:--:|
|Faster RCNN-R50 (38.4)|Cascade Mask RCNN-X101 (45.6)|40.4|41.8|||
|RetinaNet-R50 (37.4)|RetinaNet-X101 (41.0)|39.8|40.1|||


### Cityscapes Segmentation  
|Student|Teacher|DIST|Log|
|:--:|:--:|:--:|:--:|
|DeepLabV3-R18 (74.21)|DeepLabV3-R101 (78.07)|77.10|
|PSPNet-R18 (72.55)|DeepLabV3-R101 (78.07)|76.31|