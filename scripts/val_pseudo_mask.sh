#!/bin/sh

cd ../exper/

python val_pseudo_mask.py \
    --arch=inception3_spa \
    --gpus=0 \
    --dataset=ilsvrc \
    --img_dir=../data/ILSVRC/val \
    --test_list=../data/ILSVRC/list/val_mask_list.txt \
    --test_box=../data/ILSVRC/list/val_bboxes.txt \
    --num_classes=1000 \
    --snapshot_dir=../snapshots/inceptionv3_baseline_ilsvrc_20e_10_15d_large_5e-3 \
    --onehot=False \
    --debug_dir=../debug/eval_var \
    --restore_from=ilsvrc_epoch_20.pth.tar \
    --evaluate_var \
