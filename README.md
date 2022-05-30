# Knowledge Distillation from A Stronger Teacher (DIST)  
Official implementation of paper "[Knowledge Distillation from A Stronger Teacher](https://arxiv.org/abs/2205.10536)" (DIST).  
By Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu.

:fire: **DIST: state-of-the-art KD method with an extremely simple implementation (simply replace the KLDivLoss in vanilla KD).**

## Updates  
### May 27, 2022  
Code for ImageNet classification is available.  

## Getting started  
### Clone training code  
```shell
git clone https://github.com/hunto/DIST_KD.git --recurse-submodules
cd DIST_KD
```

**The loss function of DIST is in** [classification/lib/models/losses/dist_kd.py](https://github.com/hunto/image_classification_sota/blob/main/lib/models/losses/dist_kd.py).

* classification: prepare your environment and datasets following the `README.md` in `classification`.  
* object detection: coming soon.
* semantic segmentation: coming soon.

## Reproducing our results  
### ImageNet

```
cd classification
sh tools/dist_train.sh 8 ${CONFIG} ${MODEL} --teacher-model ${T_MODEL} --experiment ${EXP_NAME}
```

* Baseline settings (`R34-R101` and `R50-MBV1`):  
    ```
    CONFIG=configs/strategies/distill/resnet_dist.yaml
    ```
    |Student|Teacher|DIST|MODEL|T_MODEL|Log|
    |:--:|:--:|:--:|:--:|:--:|:--:|
    |ResNet-18 (69.76)|ResNet-34 (73.31)|72.07|`tv_resnet18`|`tv_resnet34`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/baseline_Res34-Res18.txt)|
    |MobileNet V1 (70.13)|ResNet-50 (76.16)|73.24|`mobilenet_v1`|`tv_resnet50`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/baseline_Res50-MBV1.txt)|


* Stronger teachers (`R18` and `R34` students with various ResNet teachers):  
    |Student|Teacher|KD (T=4)|DIST|
    |:--:|:--:|:--:|:--:|
    |ResNet-18 (69.76)|ResNet-34 (73.31)|71.21|72.07|
    |ResNet-18 (69.76)|ResNet-50 (76.13)|71.35|72.12|
    |ResNet-18 (69.76)|ResNet-101 (77.37)|71.09|72.08|
    |ResNet-18 (69.76)|ResNet-152 (78.31)|71.12|72.24|
    |ResNet-34 (73.31)|ResNet-50 (76.13)|74.73|75.06|
    |ResNet-34 (73.31)|ResNet-101 (77.37)|74.89|75.36|
    |ResNet-34 (73.31)|ResNet-152 (78.31)|74.87|75.42|
    
* Stronger training strategies:  
    ```
    CONFIG=configs/strategies/distill/dist_b2.yaml
    ```
    `ResNet-50-SB`: stronger ResNet-50 trained by [TIMM](https://github.com/rwightman/pytorch-image-models) ([ResNet strikes back](https://arxiv.org/abs/2110.00476)) .
    |Student|Teacher|KD (T=4)|DIST|MODEL|T_MODEL|Log|
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    |ResNet-18 (73.4)|ResNet-50-SB (80.1)|72.6|74.5|`tv_resnet18`|`timm_resnet50`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_Res50SB-Res18.txt)|
    |ResNet-34 (76.8)|ResNet-50-SB (80.1)|77.2|77.8|`tv_resnet34`|`timm_resnet50`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_Res50SB-Res34.txt)|
    |MobileNet V2 (73.6)|ResNet-50-SB (80.1)|71.7|74.4|`tv_mobilenet_v2`|`timm_resnet50`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_Res50SB-MBV2.txt)|
    |EfficientNet-B0 (78.0)|ResNet-50-SB (80.1)|77.4|78.6|<details>`timm_tf_efficientnet_b0`|`timm_resnet50`</details>|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_Res50SB-EfficientNetB0.txt)|
    |ResNet-50 (78.5)|Swin-L (86.3)|80.0|80.2|`tv_resnet50`|<details>`timm_swin_large_patch4_window7_224`</details>|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_SwinL-Res50.txt)|
    |Swin-T (81.3)|Swin-L (86.3)|81.5|82.3|-|-|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_SwinL-SwinT.txt)|

    * `Swin-L` student:
    We implement our DIST on the official code of [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).


### COCO Detection  

The training code is in [MasKD/mmrazor](https://github.com/hunto/MasKD/tree/main/mmrazor). An example to train `cascade_mask_rcnn_x101-fpn_r50`:  
```shell
sh tools/mmdet/dist_train_mmdet.sh configs/distill/dist/dist_cascade_mask_rcnn_x101-fpn_x50_coco.py 8 work_dirs/dist_cmr_x101-fpn_x50
```

|Student|Teacher|DIST|DIST+mimic|Config|Log|
|:--:|:--:|:--:|:--:|:--:|:--:|
|Faster RCNN-R50 (38.4)|Cascade Mask RCNN-X101 (45.6)|40.4|41.8|[[DIST]](https://github.com/hunto/MasKD/blob/main/mmrazor/configs/distill/dist/dist_cascade_mask_rcnn_x101-fpn_x50_coco.py) [[DIST+Mimic]](https://github.com/hunto/MasKD/blob/main/mmrazor/configs/distill/dist/dist+mimic_cascade_mask_rcnn_x101-fpn_x50_coco.py)|[[DIST]](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/det_DIST_fpn-r50_cascade-rcnn-x101.txt) [[DIST+Mimic]](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/det_DIST+mimic_fpn-r50_cascade-rcnn-x101.txt)|
|RetinaNet-R50 (37.4)|RetinaNet-X101 (41.0)|39.8|40.1|[[DIST]](https://github.com/hunto/MasKD/blob/main/mmrazor/configs/distill/dist/dist_retinanet_x101-retinanet-r50_coco.py) [[DIST+Mimic]](https://github.com/hunto/MasKD/blob/main/mmrazor/configs/distill/dist/dist%2Bmimic_retinanet_x101-retinanet-r50_coco.py)|[[DIST]](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/det_DIST_retinanet-r50_retinanet-x101.txt) [[DIST+Mimic]](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/det_DIST+mimic_retinanet-r50_retinanet-x101.txt)|


### Cityscapes Segmentation  
|Student|Teacher|DIST|Log|
|:--:|:--:|:--:|:--:|
|DeepLabV3-R18 (74.21)|DeepLabV3-R101 (78.07)|77.10|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/seg_DIST_deeplabv3_resnet101_resnet18_log.txtv)|
|PSPNet-R18 (72.55)|DeepLabV3-R101 (78.07)|76.31|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/seg_DIST_psp_resnet101_resnet18_log.txt)|


## License  
This project is released under the [Apache 2.0 license](LICENSE).

## Citation  
```
@article{huang2022knowledge,
  title = {Knowledge Distillation from A Stronger Teacher},
  author = {Huang, Tao and You, Shan and Wang, Fei and Qian, Chen and Xu, Chang},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  journal = {arXiv preprint arXiv:2205.10536},
  year = {2022}
}
```