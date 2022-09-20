python -m torch.distributed.launch --nproc_per_node=4 \
    test.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --data data/cityscapes/ \
    --save-dir work_dirs/test_images \
    --gpu-id 0,1,2,3 \
    --save-pred \
    --pretrained [your_ckpt_path]
