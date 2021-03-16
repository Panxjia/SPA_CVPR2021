#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 21:41
# @Author  : Xingjia Pan
# @File    : extract_mask_list.py
# @Software: PyCharm
import os
import numpy as np

full_val_list_file = 'val_list.txt'
with open(full_val_list_file, 'r') as fvl:
    full_val_list = fvl.readlines()

mask_val_list =[]
img_mask_val_list_file = 'image_mask_ids.txt'
with open(img_mask_val_list_file, 'r') as imvl:
    img_mask_val_list = imvl.readlines()
for i in range(len(img_mask_val_list)):
    img_mask_val_list[i] = img_mask_val_list[i].strip()

for i in range(len(full_val_list)):
    line = full_val_list[i]
    img_name = line.strip().split(' ')[0]
    img_base = img_name.split('.')[0]
    if img_base in img_mask_val_list:
        mask_val_list.append(line)

mask_val_list_file = 'val_mask_list.txt'
with open(mask_val_list_file, 'w') as mvl:
    for line in mask_val_list:
        mvl.write(line.strip() + '\n')
