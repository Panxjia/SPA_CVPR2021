#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 19:56
# @Author  : Xingjia Pan
# @File    : val_pseudo_mask.py
# @Software: PyCharm

import sys

sys.path.append('../')
import argparse
import os
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn.functional as F

from utils import AverageMeter
from utils import evaluate
from utils.loader import data_loader
from utils.restore import restore
from utils.localization import get_topk_boxes_hier, get_topk_boxes_hier_scg
from utils.vistools import save_im_heatmap_box, save_im_sim, save_sim_heatmap_box, vis_feature, vis_var
from models import *
import cv2
from collections import defaultdict

LR = 0.001
EPOCH = 200
DISP_INTERVAL = 50

# default settings
ROOT_DIR = os.getcwd()

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CVPR2021-SPA')
        self.parser.add_argument("--root_dir", type=str, default='')
        self.parser.add_argument("--img_dir", type=str, default='')
        self.parser.add_argument("--test_list", type=str, default='')
        self.parser.add_argument("--test_box", type=str, default='')
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--input_size", type=int, default=256)
        self.parser.add_argument("--crop_size", type=int, default=224)
        self.parser.add_argument("--dataset", type=str, default='ilsvrc')
        self.parser.add_argument("--num_classes", type=int, default=200)
        self.parser.add_argument("--arch", type=str, default='vgg_v0')
        self.parser.add_argument("--threshold", type=str, default='0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45')
        self.parser.add_argument("--lr", type=float, default=LR)
        self.parser.add_argument("--decay_points", type=str, default='none')
        self.parser.add_argument("--epoch", type=int, default=EPOCH)
        self.parser.add_argument("--tencrop", type=str, default='True')
        self.parser.add_argument("--onehot", type=str, default='False')
        self.parser.add_argument("--gpus", type=str, default='0', help='-1 for cpu, split gpu id by comma')
        self.parser.add_argument("--num_workers", type=int, default=12)
        self.parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
        self.parser.add_argument("--snapshot_dir", type=str, default='')
        self.parser.add_argument("--resume", type=str, default='True')
        self.parser.add_argument("--restore_from", type=str, default='')
        self.parser.add_argument("--global_counter", type=int, default=0)
        self.parser.add_argument("--current_epoch", type=int, default=0)
        self.parser.add_argument("--debug", action='store_true', help='.')
        self.parser.add_argument("--debug_detail", action='store_true', help='.')
        self.parser.add_argument("--vis_feat", action='store_true', help='.')
        self.parser.add_argument("--vis_var", action='store_true', help='.')
        self.parser.add_argument("--evaluate_var", action='store_true', help='.')
        self.parser.add_argument("--debug_dir", type=str, default='../debug', help='save visualization results.')
        self.parser.add_argument("--vis_dir", type=str, default='../vis_dir', help='save visualization results.')
        self.parser.add_argument("--scg", action='store_true', help='switch on the self-correlation generating module.')
        self.parser.add_argument("--scg_blocks", type=str, default='2,3,4,5', help='2 for feat2, etc.')
        self.parser.add_argument("--scg_com", action='store_true',
                                 help='switch on using both first-order and high-order self-correlation.')
        self.parser.add_argument("--scg_fo", action='store_true',
                                 help='switch on using first-order self-correlation only.')
        self.parser.add_argument("--scg_fosc_th", type=float, default=0.1,
                                 help='the suppress threshold for first-order self-correlation.')
        self.parser.add_argument("--scg_sosc_th", type=float, default=0.1,
                                 help='the suppress threshold for second-order self-correlation.')
        self.parser.add_argument("--scg_order", type=int, default=2,
                                 help='the order of similarity of HSC.')
        self.parser.add_argument("--scg_so_weight", type=float, default=1,
                                 help='the weight for second order affinity matrix.')
        self.parser.add_argument("--scg_fg_th", type=float, default=0.01,
                                 help='the threshold for the object in scg module.')
        self.parser.add_argument("--scg_bg_th", type=float, default=0.01,
                                 help='the threshold for the background in scg module.')
        self.parser.add_argument("--iou_th", type=float, default=0.5,
                                 help='the threshold for iou.')
        self.parser.add_argument("--ram_th_bg", type=float, default=0.2, help='the variance threshold for back ground.')
        self.parser.add_argument("--ram_bg_fg_gap", type=float, default=0.5,
                                 help='the gap between background and object in ram.')

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus_str=opt.gpus
        opt.gpus = list(map(int, opt.gpus.split(',')))
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0]>=0 else [-1]
        opt.threshold = list(map(float, opt.threshold.split(',')))
        return opt

def get_model(args):
    model = eval(args.arch).model(num_classes=args.num_classes, args=args)

    model = torch.nn.DataParallel(model, args.gpus)
    model.cuda()

    if args.resume == 'True':
        restore(args, model, None, istrain=False)

    return model

def eval_loc(cls_logits, cls_map, img_path, label, gt_boxes, topk=(1,5), threshold=None, mode='union', iou_th=0.5):
    top_boxes, top_maps, gt_known_box, gt_known_map = get_topk_boxes_hier(cls_logits[0], cls_map, img_path,
                                                                          label, topk=topk, threshold=threshold,
                                                                          mode=mode)
    top1_box, top5_boxes = top_boxes

    # update result record
    (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label.data.long().numpy(), gt_boxes,
                                         topk=(1, 5), iou_th=iou_th)
    locerr_gt_known,_ = evaluate.locerr((gt_known_box,), label.data.long().numpy(), gt_boxes, topk=(1,), iou_th=iou_th)


    return locerr_1, locerr_5, locerr_gt_known[0], top_maps,  top5_boxes, gt_known_map, top1_wrong_detail

def eval_loc_scg(cls_logits, top_cams, gt_known_cams, aff_maps, img_path, label, gt_boxes,
             topk=(1,5), threshold=None, mode='union',fg_th=0.1, bg_th=0.01, iou_th=0.5, sc_maps_fo=None):
    top_boxes, top_maps = get_topk_boxes_hier_scg(cls_logits[0], top_cams, aff_maps, img_path, topk=topk,
                                                  threshold=threshold, mode=mode,fg_th=fg_th, bg_th=bg_th,
                                                  sc_maps_fo=sc_maps_fo)
    top1_box, top5_boxes = top_boxes

    # update result record
    (locerr_1, locerr_5), top1_wrong_detail = evaluate.locerr((top1_box, top5_boxes), label.data.long().numpy(), gt_boxes,
                                         topk=(1, 5), iou_th=iou_th)

    gt_known_boxes, gt_known_maps = get_topk_boxes_hier_scg(cls_logits[0], gt_known_cams, aff_maps, img_path, topk=(1,),
                                                  threshold=threshold, mode=mode,  gt_labels= label, fg_th=fg_th,
                                                            bg_th=bg_th, sc_maps_fo=sc_maps_fo)

    # update result record
    locerr_gt_known, _ = evaluate.locerr(gt_known_boxes, label.data.long().numpy(), gt_boxes, topk=(1,), iou_th=iou_th)

    return locerr_1, locerr_5, locerr_gt_known[0], top_maps, top5_boxes, top1_wrong_detail


def normalize_feat(feat):
    feat = feat.data.cpu().numpy()
    min_val = np.min(feat)
    max_val = np.max(feat)
    norm_feat = (feat - min_val) / (max_val - min_val + 1e-15)

    return norm_feat

def cal_iou(mask1, mask2, method='iou'):
    """
    support:
    1. box1 and box2 are the same shape: [N, 4]
    2.
    :param box1:
    :param box2:
    :return:
    """
    mask1 = np.asarray(mask1, dtype=float)
    mask2 = np.asarray(mask2, dtype=float)
    intera = np.sum(mask1 * mask2)

    if method == 'iog':
        iou_val = intera / (np.sum(mask2) + 1e-5)
    elif method == 'iob':
        iou_val = intera / (np.sum(mask1) + 1e-5)
    else:
        iou_val = intera / (np.sum(mask2) + np.sum(mask1) - intera + 1e-5)
    return iou_val

def extract_bbox_from_map(boolen_map):
    assert boolen_map.ndim == 2, 'Invalid input shape'
    rows = np.any(boolen_map, axis=1)
    cols = np.any(boolen_map, axis=0)
    if rows.max() == False or cols.max() == False:
        return 0, 0, 0, 0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

def val(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    gt_boxes = []
    if args.dataset == 'ilsvrc':
        img_name = []
        with open(args.test_box, 'r') as f:
            for x in f.readlines():
                x = x.strip().split(' ')
                if len(x[1:]) % 4 == 0:
                    gt_boxes.append(list(map(float, x[1:])))
                    img_name.append(os.path.join(args.img_dir, x[0].replace('.xml', '.JPEG')))
                else:
                    print('Wrong gt bboxes.')
    elif args.dataset == 'cub':
        with open(args.test_box, 'r') as f:
            gt_boxes = [list(map(float, x.strip().split(' ')[2:])) for x in f.readlines()]
        gt_boxes = [(box[0], box[1], box[0]+box[2]-1, box[1]+box[3]-1) for box in gt_boxes]
    else:
        print('Wrong dataset.')
    # meters
    top1_clsacc = AverageMeter()
    top5_clsacc = AverageMeter()
    top1_clsacc.reset()
    top5_clsacc.reset()

    loc_err = {}
    for th in args.threshold:
        loc_err['top1_locerr_{}'.format(th)] = AverageMeter()
        loc_err['top1_locerr_{}'.format(th)].reset()
        loc_err['top5_locerr_{}'.format(th)] = AverageMeter()
        loc_err['top5_locerr_{}'.format(th)].reset()
        loc_err['gt_known_locerr_{}'.format(th)] = AverageMeter()
        loc_err['gt_known_locerr_{}'.format(th)].reset()
        for err in ['right', 'cls_wrong', 'mins_wrong', 'part_wrong', 'more_wrong','other']:
            loc_err['top1_locerr_{}_{}'.format(err, th)] = AverageMeter()
            loc_err['top1_locerr_{}_{}'.format(err, th)].reset()
        if args.scg:
            loc_err['top1_locerr_scg_{}'.format(th)] = AverageMeter()
            loc_err['top1_locerr_scg_{}'.format(th)].reset()
            loc_err['top5_locerr_scg_{}'.format(th)] = AverageMeter()
            loc_err['top5_locerr_scg_{}'.format(th)].reset()
            loc_err['gt_known_locerr_scg_{}'.format(th)] = AverageMeter()
            loc_err['gt_known_locerr_scg_{}'.format(th)].reset()
            for err in ['right', 'cls_wrong', 'mins_wrong', 'part_wrong', 'more_wrong', 'other']:
                loc_err['top1_locerr_scg_{}_{}'.format(err, th)] = AverageMeter()
                loc_err['top1_locerr_scg_{}_{}'.format(err, th)].reset()
    # get model
    model = get_model(args)
    model.eval()
    # get data
    valcls_loader, valloc_loader = data_loader(args, test_path=True, train=False)
    assert len(valcls_loader) == len(valloc_loader), \
        'Error! Different size for two dataset: loc({}), cls({})'.format(len(valloc_loader), len(valcls_loader))

    # testing
    if args.debug:
        # show_idxs = np.arange(20)
        np.random.seed(2333)
        show_idxs = np.arange(len(valcls_loader))
        np.random.shuffle(show_idxs)
        show_idxs = show_idxs[:]

    # evaluation classification task

    if not os.path.exists(args.debug_dir):
        os.makedirs(args.debug_dir)

    fg_ious = defaultdict(list)
    fg_iogs = defaultdict(list)
    fg_iobs = defaultdict(list)
    bg_ious = defaultdict(list)
    bg_iogs = defaultdict(list)
    bg_iobs = defaultdict(list)
    threshold = np.linspace(0, 1., 11)
    for idx, (dat_cls, dat_loc ) in tqdm(enumerate(zip(valcls_loader, valloc_loader))):
        # parse data
        img_path, img, label_in = dat_cls
        img_seg_path = img_path[0].replace('images', 'segmentations')
        img_seg_path = img_seg_path.replace('.jpg', '.png')
        if args.tencrop == 'True':
            bs, ncrops, c, h, w = img.size()
            img = img.view(-1, c, h, w)

        # forward pass
        args.device = torch.device('cuda') if args.gpus[0]>=0 else torch.device('cpu')
        img = img.to(args.device)

        if args.vis_feat:
            if idx in show_idxs:
                _, img_loc, label = dat_loc
                _ = model(img_loc)
                vis_feature(model.module.feat4, img_path[0], args.vis_dir, layer='feat4')
                vis_feature(model.module.feat5, img_path[0], args.vis_dir, layer='feat5')
                vis_feature(model.module.cls_map, img_path[0], args.vis_dir, layer='cls_map')
            continue
        if args.vis_var:
            if idx in show_idxs:
                _, img_loc, label = dat_loc
                logits, _,  = model(img_loc)
                cls_logits = F.softmax(logits,dim=1)
                var_logits = torch.var(cls_logits,dim=1).squeeze()
                logits_cls = logits[0,label.long(),...]
                vis_var(var_logits, logits_cls, img_path[0], args.vis_dir, net='vgg_s10_loc_.4_.7_fpn_l4_var_cls')
            continue
        if args.evaluate_var:
            _, img_loc, label = dat_loc
            with torch.no_grad():
                logits, _, _, = model(img_loc)
            cls_logits = F.softmax(logits, dim=1)
            var_logits = torch.var(cls_logits, dim=1).squeeze()
            norm_var_logits = normalize_feat(var_logits)

            for th in threshold:
                bg_mask = (norm_var_logits < th).astype(float)
                fg_mask = (norm_var_logits > th).astype(float)
                im = cv2.imread(img_path[0])
                h, w, _ = np.shape(im)
                bg_mask = cv2.resize(bg_mask, dsize=(w, h))
                bg_mask[bg_mask > 0] = 1
                fg_mask = cv2.resize(fg_mask, dsize=(w, h))
                fg_mask[fg_mask > 0] = 1
                gt_mask = cv2.imread(img_seg_path)
                gt_mask = np.max(gt_mask, axis=-1)
                gt_mask[gt_mask > 0] = 1
                gt_masks = [gt_mask]
                # gt_masks = []
                # gt_boxes_cur = gt_boxes[idx]
                # gt_box_cnt = len(gt_boxes_cur) // 4
                # for i in range(gt_box_cnt):
                #     gt_box = list(map(int, gt_boxes_cur[i * 4:(i + 1) * 4]))
                #     gt_mask_mode = np.zeros((h, w))
                #     gt_mask_mode[gt_box[1]:gt_box[3],gt_box[0]:gt_box[2]] =1
                #     gt_masks.append(gt_mask_mode)
                if th >= 0.5:
                    fg_mask_img = fg_mask[:,:,np.newaxis].repeat(3, axis=-1)
                    img_fg_mask = im * 0.7 + fg_mask_img * 255 * 0.3
                    cv2.imwrite(os.path.join(args.debug_dir, 'img_fg_mask_{}_{}.jpg'.format(th, idx)), img_fg_mask)

                max_iou = 0
                max_iog = 0
                max_iob = 0
                for gt_mask in gt_masks:
                    iou_i = cal_iou(fg_mask, gt_mask)
                    if iou_i > max_iou:
                        max_iou = iou_i
                    iog_i = cal_iou(fg_mask, gt_mask, method='iog')
                    if iog_i > max_iog:
                        max_iog = iog_i
                    iob_i = cal_iou(fg_mask, gt_mask, method='iob')
                    if iob_i > max_iob:
                        max_iob = iob_i
                fg_ious[th].append(max_iou)
                fg_iogs[th].append(max_iog)
                fg_iobs[th].append(max_iob)

                max_iou = 0
                max_iog = 0
                max_iob = 0
                for gt_mask in gt_masks:
                    gt_mask_bg = 1-gt_mask
                    iou_i = cal_iou(bg_mask, gt_mask_bg)
                    if iou_i > max_iou:
                        max_iou = iou_i
                    iog_i = cal_iou(bg_mask, gt_mask_bg, method='iog')
                    if iog_i > max_iog:
                        max_iog = iog_i
                    iob_i = cal_iou(bg_mask, gt_mask_bg, method='iob')
                    if iob_i > max_iob:
                        max_iob = iob_i
                bg_ious[th].append(max_iou)
                bg_iogs[th].append(max_iog)
                bg_iobs[th].append(max_iob)

    eval_res_file = 'inception_baseline_cub_eval_pseudo_mask.txt'
    with open(os.path.join(args.debug_dir, eval_res_file), 'a') as erf:
        erf.write('threshold \t fg_iou_mean\t fg_iog_mean \t fg_iob_mean \n')
        for key in fg_ious.keys():
            erf.write('th_{} \t {:.2f} \t {:.2f} \t {:.2f} \n'.
                      format(str(key),
                             np.mean(np.array(fg_ious[key])),
                             np.mean(np.array(fg_iogs[key])),
                             np.mean(np.array(fg_iobs[key]))))
        erf.write('threshold \t bg_iou_mean \t bg_iog_mean \t bg_iob_mean \n')
        for key in fg_ious.keys():
            erf.write('th_{} \t {:.2f} \t {:.2f} \t {:.2f} \n'.
                      format(str(key),
                             np.mean(np.array(bg_ious[key])),
                             np.mean(np.array(bg_iogs[key])),
                             np.mean(np.array(bg_iobs[key]))))





if __name__ == '__main__':
    args = opts().parse()
    val(args)
