#!/bin/sh

cd ../exper/

python train_cam_spa.py \
    --arch=vgg_spa \
    --epoch=20 \
    --lr=0.001 \
    --batch_size=30 \
    --gpus=0,1,2,3 \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/train \
    --train_list=../data/ILSVRC/list/train_list.txt \
    --num_classes=1000 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --seed=0 \
    --snapshot_dir=../snapshots/vgg_16_baseline_ilsvrc \
    --log_dir=../log/vgg_16_baseline_ilsvrc \
    --onehot=False \
    --decay_point=80 \
