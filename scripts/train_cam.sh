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
    --num_classes=200 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --seed=0 \
    --snapshot_dir=../snapshots/vgg_16_baseline \
    --log_dir=../log/vgg_16_baseline \
    --onehot=False \
    --decay_point=80 \
