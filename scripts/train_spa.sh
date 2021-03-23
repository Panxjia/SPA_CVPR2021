#!/bin/sh

cd ../exper/

python train_cam_spa.py \
    --arch=vgg_spa \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=30 \
    --gpus=0 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --train_list=../data/CUB_200_2011/list/train.txt \
    --num_classes=200 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --seed=0 \
    --snapshot_dir=../snapshots/vgg_16_ram_s20_.5_bg_.1_gap.2 \
    --log_dir=../log/vgg_16_ram_s20_.5_bg_.1_gap_.2 \
    --onehot=False \
    --decay_point=80 \
    --ram \
    --ram_start=20 \
    --ra_loss_weight=0.5 \
    --ram_th_bg=0.1 \
    --ram_bg_fg_gap=0.2 \

