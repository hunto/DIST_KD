# Knowledge Distillation from A Stronger Teacher (DIST)  
Official implementation of paper "[Knowledge Distillation from A Stronger Teacher](https://arxiv.org/abs/2205.10536)" (DIST), NeurIPS 2022.  
By Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu.

:fire: **DIST: a simple and effective KD method.**

## Updates  

* **December 27, 2022**: Update CIFAR-100 distillation code and logs.

* **September 20, 2022**: Release code for semantic segmentation task.

* **September 15, 2022**: DIST was accepted by NeurIPS 2022!

* **May 30, 2022**: Code for object detection is available.

* **May 27, 2022**: Code for ImageNet classification is available.  

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
    |Student|Teacher|DIST|MODEL|T_MODEL|Log|Ckpt|
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    |ResNet-18 (69.76)|ResNet-34 (73.31)|72.07|`tv_resnet18`|`tv_resnet34`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/baseline_Res34-Res18.txt)|[ckpt](https://drive.google.com/file/d/1_nzcAwxZApLU496iypsdeNhXYzPA4ZF4/view?usp=sharing)|
    |MobileNet V1 (70.13)|ResNet-50 (76.16)|73.24|`mobilenet_v1`|`tv_resnet50`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/baseline_Res50-MBV1.txt)|[ckpt](https://drive.google.com/file/d/1uSzFbcY6uudQgfDBxHataBPO1xX8J_yW/view?usp=sharing)|


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
    |ResNet-50 (78.5)|Swin-L (86.3)|80.0|80.2|`tv_resnet50`|<details>`timm_swin_large_patch4_window7_224`</details>|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_SwinL-Res50.txt) [ckpt](https://drive.google.com/file/d/1iZFP53i4Yw7lqvfV707aTBddN_waEU1r/view?usp=sharing)|
    |Swin-T (81.3)|Swin-L (86.3)|81.5|82.3|-|-|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/stronger_SwinL-SwinT.txt)|

    * `Swin-L` student:
    We implement our DIST on the official code of [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).


### CIFAR-100

Download and extract the [teacher checkpoints](https://github.com/hunto/DIST_KD/releases/download/v0.0.2/cifar_ckpts.zip) to your disk, then specify the path of the corresponding checkpoint `pth` file using `--teacher-ckpt`:

```
cd classification
sh tools/dist_train.sh 1 configs/strategies/distill/dist_cifar.yaml ${MODEL} --teacher-model ${T_MODEL} --experiment ${EXP_NAME} --teacher-ckpt ${CKPT}
```

**NOTE**: For `MobileNetV2`, `ShuffleNetV1`, and `ShuffleNetV2`, `lr` and `warmup-lr` should be `0.01`:
```
sh tools/dist_train.sh 1 configs/strategies/distill/dist_cifar.yaml ${MODEL} --teacher-model ${T_MODEL} --experiment ${EXP_NAME} --teacher-ckpt ${CKPT} --lr 0.01 --warmup-lr 0.01
```

|Student|Teacher|DIST|MODEL|T_MODEL|Log|
|:--:|:--:|:--:|:--:|:--:|:--:|
|WRN-40-1 (71.98)|WRN-40-2 (75.61)|74.43±0.24|`cifar_wrn_40_1`|`cifar_wrn_40_2`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.2/log_cifar100_wrn_40_1-wrn_40_2.zip)|
|ResNet-20 (69.06)|ResNet-56 (72.34)|71.75±0.30|`cifar_resnet20`|`cifar_resnet56`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.2/log_cifar100_res56-res20.zip)|
|ResNet-8x4 (72.50)|ResNet-32x4 (79.42)|76.31±0.19|`cifar_resnet8x4`|`cifar_resnet32x4`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.2/log_cifar100_res32x4-res8x4.zip)|
|MobileNetV2 (64.60)|ResNet-50 (79.34)|68.66±0.23|`cifar_mobile_half`|`cifar_ResNet50`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.2/log_cifar100_res50-mbv2.zip)|
|ShuffleNetV1 (70.50)|ResNet-32x4 (79.42)|76.34±0.18|`cifar_ShuffleV1`|`cifar_resnet32x4`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.2/log_cifar100_res32x4-shufflev1.zip)|
|ShuffleNetV2 (71.82)|ResNet-32x4 (79.42)|77.35±0.25|`cifar_ShuffleV2`|`cifar_resnet32x4`|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.2/log_cifar100_res32x4-shufflev2.zip)|



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
Detailed instructions of reproducing our results are in `segmentation` folder ([README](./segmentation/README.md)).
|Student|Teacher|DIST|Log|
|:--:|:--:|:--:|:--:|
|DeepLabV3-R18 (74.21)|DeepLabV3-R101 (78.07)|77.10|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/seg_DIST_deeplabv3_resnet101_resnet18_log.txt)|
|PSPNet-R18 (72.55)|DeepLabV3-R101 (78.07)|76.31|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/seg_DIST_psp_resnet101_resnet18_log.txt)|


## License  
This project is released under the [Apache 2.0 license](LICENSE).

## Citation  
```
@article{huang2022knowledge,
  title={Knowledge Distillation from A Stronger Teacher},
  author={Huang, Tao and You, Shan and Wang, Fei and Qian, Chen and Xu, Chang},
  journal={arXiv preprint arXiv:2205.10536},
  year={2022}
}
```
