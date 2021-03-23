#!/bin/sh

cd ../exper/

python  val_spa.py \
    --arch=vgg_spa \
    --gpus=0 \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/val \
    --test_list=../data/ILSVRC/list/val_list.txt \
    --test_box=../data/ILSVRC/list/val_bboxes.txt \
    --num_classes=1000 \
    --snapshot_dir=../snapshots/vgg_16_baseline_ilsvrc_20e_10_15d \
    --onehot=False \
    --debug_dir=../debug/vgg_16_baseline_scg_fo_.2_so_.2_pos_.1_neg_.05_f45hscsum_test \
    --restore_from=ilsvrc_epoch_20.pth.tar \
    --threshold=0.2,0.5 \
    --scg \
    --scg_com \
    --scg_blocks=4,5 \
    --scg_fosc_th=0.2 \
    --scg_sosc_th=0.2 \
    --scg_so_weight=1. \
    --scg_fg_th=0.1 \
    --scg_bg_th=0.05 \
    --scg_order=2 \
 
