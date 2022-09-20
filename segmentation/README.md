#  Cross-Image Relational Knowledge Distillation for Semantic Segmentation

This repository contains the segmentation code of DIST (Knowledge Distillation from A Stronger Teacher).


## Preparation

```shell
git clone https://github.com/hunto/DIST_KD.git --recurse-submodules
cd DIST_KD/segmentation
```

### Dataset  
Put the Cityscapes dataset into `./data/cityscapes` folder.

### Pretrained checkpoints  
Download the required checkpoints into `./ckpts` folder.
Backbones pretrained on ImageNet:
* [resnet101-imagenet.pth](https://drive.google.com/file/d/1V8-E4wm2VMsfnNiczSIDoSM7JJBMARkP/view?usp=sharing) 
* [resnet18-imagenet.pth](https://drive.google.com/file/d/1_i0n3ZePtQuh66uQIftiSwN7QAUlFb8_/view?usp=sharing) 
* [mobilenetv2-imagenet.pth](https://drive.google.com/file/d/12EDZjDSCuIpxPv-dkk1vrxA7ka0b0Yjv/view?usp=sharing) 

Teacher backbones:
* [deeplabv3_resnet101_citys_best_model.pth](https://drive.google.com/file/d/1zUdhYPYCDCclWU3Wo7GbbTlM8ibQ_UC1/view?usp=sharing)

## Performance on Cityscapes

Student models are trained on 8 * NVIDIA Tesla V100 GPUs.

|Role|Network|Method|val mIoU|train script|log|ckpt|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Teacher|DeepLabV3-ResNet101|-|78.07|[sh](./train_scripts/train_baseline/deeplabv3_res101.sh)|-|[Google Drive](https://drive.google.com/file/d/1zUdhYPYCDCclWU3Wo7GbbTlM8ibQ_UC1/view?usp=sharing)|
|Student|DeepLabV3-ResNet18|DIST|77.10|[sh](./train_scripts/train_kd/deeplabv3_res18.sh)|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/seg_DIST_deeplabv3_resnet101_resnet18_log.txt)|[Google Drive](https://drive.google.com/file/d/1tqZ9W0t35PDlld81QpEusDjyGnj_kyKX/view?usp=sharing)|
|Student|PSPNet-ResNet18|DIST|76.31|[sh](./train_scripts/train_kd/pspnet_res18.sh)|[log](https://github.com/hunto/DIST_KD/releases/download/v0.0.1/seg_DIST_psp_resnet101_resnet18_log.txt)|[Google Drive](https://drive.google.com/file/d/1U9xuOnjJg-RIRZfWBUylI-DdFhP44QdF/view?usp=sharing)|

## Evaluate pre-trained models on Cityscapes val and test sets

### Evaluate the pre-trained models on val set
```
python -m torch.distributed.launch --nproc_per_node=8 eval.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store log files] \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --pretrained [your checkpoint path]/deeplabv3_resnet101_citys_best_model.pth
```

### Generate the resulting images on test set
```
python -m torch.distributed.launch --nproc_per_node=4 test.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store resulting images] \
    --gpu-id 0,1,2,3 \
    --save-pred \
    --pretrained [your checkpoint path]/deeplabv3_resnet101_citys_best_model.pth
```
You can submit the resulting images to the [Cityscapes test server](https://www.cityscapes-dataset.com/submit/).

## Acknowledgement

The code is mostly based on the code in [CIRKD](https://github.com/winycg/CIRKD.git).

