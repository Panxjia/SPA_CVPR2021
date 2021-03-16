#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/9 18:46
# @Author  : Xingjia Pan
# @File    : eval_scm.py
# @Software: PyCharm
import numpy as np
import cv2

def norm_atten_map(attention_map):
    min_val = np.min(attention_map)
    max_val = np.max(attention_map)
    atten_norm = (attention_map - min_val) / (max_val - min_val+1e-10)
    return atten_norm

def get_topk_boxes_hier(logits, feat_map, topk=(1, )):
    logits = logits.squeeze().data.cpu().numpy()
    feat_map = feat_map.data.cpu().numpy()

    maxk = max(topk)

    species_cls = np.argsort(logits)[::-1][:maxk]

    maxk_maps = []
    for i in range(maxk):
        feat_map_i = feat_map[0, species_cls[i], :, :]
        cam_map_i = norm_atten_map(feat_map_i)  # normalize cam map
        maxk_maps.append(cam_map_i.copy())
    return maxk_maps

def get_topk_boxes_hier_sim(top_cams, hsc_maps, fg_th=0.1, bg_th=0.05):
    if isinstance(hsc_maps, tuple) or isinstance(hsc_maps, list):
        pass
    else:
        hsc_maps = [hsc_maps]
    cleaned_hsc_maps =[]
    for i in range(len(hsc_maps)):
        if hsc_maps[i] is not None:
            cleaned_hsc_maps.append(hsc_maps[i])

    maxk_maps = []
    for i in range(len(top_cams)):
        aff_map_cls = 0
        for aff_map in cleaned_hsc_maps:
            cam_map_cls = top_cams[i]
            aff_map = aff_map.squeeze().data.cpu().numpy()
            wh_aff = aff_map.shape[0]
            h_aff, w_aff = int(np.sqrt(wh_aff)), int(np.sqrt(wh_aff))
            cam_map_cls = cv2.resize(cam_map_cls, dsize=(w_aff, h_aff))
            cam_map_cls_vector = cam_map_cls.reshape(-1)
            #positive
            cam_map_cls_id = np.arange(wh_aff).astype(np.int)
            cam_map_cls_th_ind_pos = cam_map_cls_id[cam_map_cls_vector >= fg_th]
            aff_map_sel_pos = aff_map[:,cam_map_cls_th_ind_pos]
            aff_map_sel_pos = (aff_map_sel_pos - np.min(aff_map_sel_pos,axis=0, keepdims=True))/(
                    np.max(aff_map_sel_pos, axis=0, keepdims=True) - np.min(aff_map_sel_pos, axis=0, keepdims=True) + 1e-10)
            cam_map_cls_val_pos = cam_map_cls_vector[cam_map_cls_th_ind_pos].reshape(1,-1)
            # aff_map_sel_pos = np.sum(aff_map_sel_pos * cam_map_cls_val_pos, axis=1).reshape(h_aff, w_aff)
            if aff_map_sel_pos.shape[1] > 0:
                aff_map_sel_pos = np.sum(aff_map_sel_pos, axis=1).reshape(h_aff, w_aff)
                aff_map_sel_pos = (aff_map_sel_pos - np.min(aff_map_sel_pos))/( np.max(aff_map_sel_pos) - np.min(aff_map_sel_pos)  + 1e-10)
            else:
                aff_map_sel_pos = 0
            #negtive
            cam_map_cls_th_ind_neg = cam_map_cls_id[cam_map_cls_vector <= bg_th]
            aff_map_sel_neg = aff_map[:, cam_map_cls_th_ind_neg]
            aff_map_sel_neg = (aff_map_sel_neg - np.min(aff_map_sel_neg,axis=0, keepdims=True))/(
                    np.max(aff_map_sel_neg, axis=0, keepdims=True) - np.min(aff_map_sel_neg, axis=0, keepdims=True)+ 1e-10)
            cam_map_cls_val_neg = cam_map_cls_vector[cam_map_cls_th_ind_neg].reshape(1, -1)
            # aff_map_sel_neg = np.sum(aff_map_sel_neg * (1-cam_map_cls_val_neg), axis=1).reshape(h_aff, w_aff)
            if aff_map_sel_neg.shape[1] > 0:
                aff_map_sel_neg = np.sum(aff_map_sel_neg, axis=1).reshape(h_aff, w_aff)
                aff_map_sel_neg = (aff_map_sel_neg - np.min(aff_map_sel_neg))/(np.max(aff_map_sel_neg)-np.min(aff_map_sel_neg) + 1e-10)
            else:
                aff_map_sel_neg = 0
            aff_map_cls_i = aff_map_sel_pos - aff_map_sel_neg
            # aff_map_cls_i = aff_map_sel_pos
            aff_map_cls_i = aff_map_cls_i * (aff_map_cls_i>=0)
            aff_map_cls_i = (aff_map_cls_i-np.min(aff_map_cls_i))/(np.max(aff_map_cls_i) - np.min(aff_map_cls_i)+1e-10)
            aff_map_cls= np.maximum(aff_map_cls, aff_map_cls_i)
        # aff_map_cls = (aff_map_cls - np.min(aff_map_cls)) / (np.max(aff_map_cls) + 1e-10)
        maxk_maps.append(aff_map_cls.copy())
    return maxk_maps

##

def get_scm_map(logits, feat_maps, hsc_maps, topk=(1,5), fg_th=0.1, bg_th=0.05, im_file=None):
    cam_maps = get_topk_boxes_hier(logits, feat_maps, topk=topk)
    scm_maps = get_topk_boxes_hier_sim(cam_maps, hsc_maps, fg_th=fg_th, bg_th=bg_th)
    if im_file is None:
        im = cv2.imread(im_file)
        h, w, _ = np.shape(im)
        for i in range(len(scm_maps)):
            resized_scm_map_i = cv2.resize(scm_maps[i], dsize=(w, h))
            scm_maps[i] = resized_scm_map_i
    return scm_maps

# introduction
#scm_maps = get_scm_map(cls_logits,loc_map,aff_maps[-2]+aff_maps[-1],fg_th=args.nl_fg_th, bg_th=args.nl_bg_th,
#                                           im_file=img_path[0])
