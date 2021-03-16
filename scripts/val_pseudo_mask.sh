#!/bin/sh

cd ../exper/

python val_pseudo_mask.py \
    --arch=inception3_spa \
    --gpus=0 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --test_list=../data/CUB_200_2011/list/test.txt \
    --test_box=../data/CUB_200_2011/list/test_boxes.txt \
    --num_classes=200 \
    --snapshot_dir=../snapshots/inceptionV3_baseline \
    --onehot=False \
    --debug_dir=../debug/eval_var \
    --restore_from=cub_epoch_100.pth.tar \
    --evaluate_var \
