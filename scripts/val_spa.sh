#!/bin/sh

cd ../exper/

python val_spa.py \
    --arch=vgg_spa \
    --gpus=0 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_16_baseline \
    --onehot=False \
    --debug_dir=../debug/vgg_16_baseline_scg_fo_.1_so_.2_pos_.1_neg_.05_f45hscsum \
    --restore_from=cub_epoch_100.pth.tar \
    --threshold=0.1,0.25,0.3,0.35 \
    --scg \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.1 \
    --scg_sosc_th=0.5 \
    --scg_so_weight=2 \
    --scg_fg_th=0.1 \
    --scg_bg_th=0.05 \
    --scg_order=2 \
 
