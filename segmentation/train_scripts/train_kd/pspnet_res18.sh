python -m torch.distributed.launch --nproc_per_node=8 \
    train_kd.py \
    --teacher-model deeplabv3 \
    --student-model psp \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --data data/cityscapes/ \
    --save-dir work_dirs/dist_dv3-r101_psp_r18 \
    --log-dir work_dirs/dist_dv3-r101_psp_r18 \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --teacher-pretrained ckpts/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base ckpts/resnet18-imagenet.pth