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
        self.parser.add_argument("--dataset", type=str, default='imagenet')
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





def val(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    if args.dataset == 'ilsvrc':
        gt_boxes = []
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

    for idx, (dat_cls, dat_loc ) in tqdm(enumerate(zip(valcls_loader, valloc_loader))):
        # parse data
        img_path, img, label_in = dat_cls
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
                logits, _, _, _,_ = model(img_loc)
                cls_logits = F.softmax(logits,dim=1)
                var_logits = torch.var(cls_logits,dim=1).squeeze()
                logits_cls = logits[0,label.long(),...]
                vis_var(var_logits, logits_cls, img_path[0], args.vis_dir, net='vgg_s10_loc_.4_.7_fpn_l4_var_cls')
            continue
        with torch.no_grad():
            logits, _ , _= model(img)
            cls_logits = torch.mean(torch.mean(logits, dim=2), dim=2)
            cls_logits = F.softmax(cls_logits, dim=1)
            if args.tencrop == 'True':
                cls_logits = cls_logits.view(1, ncrops, -1).mean(1)

            prec1_1, prec5_1 = evaluate.accuracy(cls_logits.cpu().data, label_in.long(), topk=(1, 5))
            top1_clsacc.update(prec1_1[0].numpy(), img.size()[0])
            top5_clsacc.update(prec5_1[0].numpy(), img.size()[0])



        _, img_loc, label = dat_loc
        with torch.no_grad():
            logits, sc_maps_fo, sc_maps_so  = model(img_loc, scg_flag=args.scg)
            loc_map = F.relu(logits)

        for th in args.threshold:
            locerr_1, locerr_5, gt_known_locerr, top_maps, top5_boxes, gt_known_maps, top1_wrong_detail = \
                eval_loc(cls_logits, loc_map, img_path[0], label, gt_boxes[idx], topk=(1, 5), threshold=th,
                         mode='union', iou_th=args.iou_th)
            loc_err['top1_locerr_{}'.format(th)].update(locerr_1, img_loc.size()[0])
            loc_err['top5_locerr_{}'.format(th)].update(locerr_5, img_loc.size()[0])
            loc_err['gt_known_locerr_{}'.format(th)].update(gt_known_locerr, img_loc.size()[0])

            cls_wrong, multi_instances, region_part, region_more, region_wrong = top1_wrong_detail
            right = 1 - (cls_wrong + multi_instances + region_part + region_more + region_wrong)
            loc_err['top1_locerr_right_{}'.format(th)].update(right, img_loc.size()[0])
            loc_err['top1_locerr_cls_wrong_{}'.format(th)].update(cls_wrong, img_loc.size()[0])
            loc_err['top1_locerr_mins_wrong_{}'.format(th)].update(multi_instances, img_loc.size()[0])
            loc_err['top1_locerr_part_wrong_{}'.format(th)].update(region_part, img_loc.size()[0])
            loc_err['top1_locerr_more_wrong_{}'.format(th)].update(region_more, img_loc.size()[0])
            loc_err['top1_locerr_other_{}'.format(th)].update(region_wrong, img_loc.size()[0])
            if args.debug and idx in show_idxs and (th == args.threshold[0]):
                top1_wrong_detail_dir = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}'.format(cls_wrong,
                                                                                            multi_instances,
                                                                                            region_part,
                                                                                            region_more,
                                                                                            region_wrong)
                debug_dir = os.path.join(args.debug_dir,top1_wrong_detail_dir) if args.debug_detail else args.debug_dir
                save_im_heatmap_box(img_path[0], top_maps, top5_boxes, debug_dir,
                                    gt_label=label.data.long().numpy(), gt_box=gt_boxes[idx],
                                    epoch=args.current_epoch,threshold=th)

            if args.scg:
                sc_maps = []
                if args.scg_com:
                    for sc_map_fo_i, sc_map_so_i in zip(sc_maps_fo, sc_maps_so):
                        if (sc_map_fo_i is not None) and (sc_map_so_i is not None):
                            sc_map_i = torch.max(sc_map_fo_i, args.scg_so_weight * sc_map_so_i)
                            sc_map_i = sc_map_i / (torch.sum(sc_map_i, dim=1, keepdim=True) + 1e-10)
                            sc_maps.append(sc_map_i)
                elif args.scg_fo:
                    sc_maps = sc_maps_fo
                else:
                    sc_maps = sc_maps_so
                locerr_1_scg, locerr_5_scg ,gt_known_locerr_scg, top_maps_scg, top5_boxes_scg, top1_wrong_detail_scg = \
                    eval_loc_scg(cls_logits, top_maps, gt_known_maps, sc_maps[-1]+sc_maps[-2], img_path[0], label,
                                                     gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
                                                     fg_th=args.scg_fg_th, bg_th=args.scg_bg_th,iou_th=args.iou_th,
                                                      sc_maps_fo= None)
                loc_err['top1_locerr_scg_{}'.format(th)].update(locerr_1_scg, img_loc.size()[0])
                loc_err['top5_locerr_scg_{}'.format(th)].update(locerr_5_scg, img_loc.size()[0])
                loc_err['gt_known_locerr_scg_{}'.format(th)].update(gt_known_locerr_scg, img_loc.size()[0])

                cls_wrong_scg, multi_instances_scg, region_part_scg, region_more_scg, region_wrong_scg = top1_wrong_detail_scg
                right_scg = 1- (cls_wrong_scg + multi_instances_scg + region_part_scg + region_more_scg + region_wrong_scg)
                loc_err['top1_locerr_scg_right_{}'.format(th)].update(right_scg, img_loc.size()[0])
                loc_err['top1_locerr_scg_cls_wrong_{}'.format(th)].update(cls_wrong_scg, img_loc.size()[0])
                loc_err['top1_locerr_scg_mins_wrong_{}'.format(th)].update(multi_instances_scg, img_loc.size()[0])
                loc_err['top1_locerr_scg_part_wrong_{}'.format(th)].update(region_part_scg, img_loc.size()[0])
                loc_err['top1_locerr_scg_more_wrong_{}'.format(th)].update(region_more_scg, img_loc.size()[0])
                loc_err['top1_locerr_scg_other_{}'.format(th)].update(region_wrong_scg, img_loc.size()[0])

                if args.debug and idx in show_idxs and (th == args.threshold[0]):
                    top1_wrong_detail_dir = 'cls_{}-mins_{}-rpart_{}-rmore_{}-rwrong_{}_scg'.format(cls_wrong_scg,
                                                                                                multi_instances_scg,
                                                                                                region_part_scg,
                                                                                                region_more_scg,
                                                                                                region_wrong_scg)
                    debug_dir = os.path.join(args.debug_dir,
                                             top1_wrong_detail_dir) if args.debug_detail else args.debug_dir
                    save_im_heatmap_box(img_path[0], top_maps_scg, top5_boxes_scg, debug_dir,
                                        gt_label=label.data.long().numpy(),gt_box=gt_boxes[idx],
                                        epoch=args.current_epoch, threshold=th, suffix='scg')

                    save_im_sim(img_path[0], sc_maps_fo, debug_dir, gt_label=label.data.long().numpy(),
                                epoch=args.current_epoch, suffix='fo')
                    save_im_sim(img_path[0], sc_maps_so, debug_dir, gt_label=label.data.long().numpy(),
                                epoch=args.current_epoch, suffix='so')
                    save_im_sim(img_path[0], sc_maps_fo[-2]+sc_maps_fo[-1], debug_dir, gt_label=label.data.long().numpy(),
                                epoch=args.current_epoch, suffix='fo_45')
                    # save_im_sim(img_path[0], aff_maps_so[-2] + aff_maps_so[-1], debug_dir,
                    #             gt_label=label.data.long().numpy(),
                    #             epoch=args.current_epoch, suffix='so_45')
                    # # save_im_sim(img_path[0], aff_maps, debug_dir, gt_label=label.data.long().numpy(),
                    # #             epoch=args.current_epoch, suffix='com')
                    save_sim_heatmap_box(img_path[0], top_maps,  debug_dir, gt_label=label.data.long().numpy(),
                                         sim_map=sc_maps_fo[-2] + sc_maps_fo[-1], epoch=args.current_epoch, threshold=th,
                                         suffix='aff_fo_f45_cam', fg_th=args.scg_fg_th, bg_th=args.scg_bg_th)
                    # save_sim_heatmap_box(img_path[0], top_maps, debug_dir, gt_label=label.data.long().numpy(),
                    #                      sim_map=aff_maps_so[-2] + aff_maps_so[-1], epoch=args.current_epoch, threshold=th,
                    #                      suffix='aff_so_f5_cam', fg_th=args.scg_fg_th, bg_th=args.scg_bg_th)
                    # save_sim_heatmap_box(img_path[0], df_top_maps, debug_dir, gt_label=label.data.long().numpy(),
                    #                      sim_map=aff_maps_so[-2], epoch=args.current_epoch, threshold=th,
                    #                      suffix='aff_so_f4_cam',fg_th=args.nl_fg_th, bg_th=args.nl_bg_th)
                    # save_sim_heatmap_box(img_path[0], df_top_maps, debug_dir, gt_label=label.data.long().numpy(),
                    #                      sim_map=aff_maps_so[-1], epoch=args.current_epoch, threshold=th,
                    #                      suffix='aff_so_f5_cam', fg_th=args.nl_fg_th, bg_th=args.nl_bg_th)
                    # save_sim_heatmap_box(img_path[0], df_top_maps, debug_dir, gt_label=label.data.long().numpy(),
                    #                      sim_map=aff_maps[-2:],
                    #                      epoch=args.current_epoch, threshold=th, suffix='aff_com_cam',fg_th=args.nl_fg_th, bg_th=args.nl_bg_th)


    print('== cls err')
    print('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
    for th in args.threshold:
        print('=========== threshold: {} ==========='.format(th))
        print('== loc err')
        print('CAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_{}'.format(th)].avg,
                                                         loc_err['top5_locerr_{}'.format(th)].avg))
        print('CAM-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_right_{}'.format(th)].sum,
                                                              loc_err['top1_locerr_cls_wrong_{}'.format(th)].sum,
                                                              loc_err['top1_locerr_mins_wrong_{}'.format(th)].sum,
                                                              loc_err['top1_locerr_part_wrong_{}'.format(th)].sum,
                                                              loc_err['top1_locerr_more_wrong_{}'.format(th)].sum,
                                                              loc_err['top1_locerr_other_{}'.format(th)].sum))
        if args.scg:
            print('SCG-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_scg_{}'.format(th)].avg,
                                                            loc_err['top5_locerr_scg_{}'.format(th)].avg))
            print('SCG-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_scg_right_{}'.format(th)].sum,
                                                                  loc_err[
                                                                      'top1_locerr_scg_cls_wrong_{}'.format(th)].sum,
                                                                  loc_err[
                                                                      'top1_locerr_scg_mins_wrong_{}'.format(th)].sum,
                                                                  loc_err[
                                                                      'top1_locerr_scg_part_wrong_{}'.format(th)].sum,
                                                                  loc_err[
                                                                      'top1_locerr_scg_more_wrong_{}'.format(th)].sum,
                                                                  loc_err['top1_locerr_scg_other_{}'.format(th)].sum))
        print('== Gt-Known loc err')
        print('CAM-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_{}'.format(th)].avg ))
        if args.scg:
            print('SCG-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_{}'.format(th)].avg))

    setting = args.debug_dir.split('/')[-1]
    results_log_name = '{}_results.log'.format(setting)
    result_log = os.path.join(args.snapshot_dir, results_log_name)
    with open(result_log,'a') as fw:
        fw.write('== cls err ')
        fw.write('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
        for th in args.threshold:
            fw.write('=========== threshold: {} ===========\n'.format(th))
            fw.write('== loc err ')
            fw.write('CAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_{}'.format(th)].avg,
                                                             loc_err['top5_locerr_{}'.format(th)].avg))
            fw.write('CAM-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_right_{}'.format(th)].sum,
                                                                  loc_err['top1_locerr_cls_wrong_{}'.format(th)].sum,
                                                                  loc_err['top1_locerr_mins_wrong_{}'.format(th)].sum,
                                                                  loc_err['top1_locerr_part_wrong_{}'.format(th)].sum,
                                                                  loc_err['top1_locerr_more_wrong_{}'.format(th)].sum,
                                                                  loc_err['top1_locerr_other_{}'.format(th)].sum))
            if args.scg:
                fw.write('SCG-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_scg_{}'.format(th)].avg,
                                                                   loc_err['top5_locerr_scg_{}'.format(th)].avg))
                fw.write('SCG-Top1_err: {} {} {} {} {} {}\n'.format(loc_err['top1_locerr_scg_right_{}'.format(th)].sum,
                                                                      loc_err[
                                                                          'top1_locerr_scg_cls_wrong_{}'.format(th)].sum,
                                                                      loc_err[
                                                                          'top1_locerr_scg_mins_wrong_{}'.format(th)].sum,
                                                                      loc_err[
                                                                          'top1_locerr_scg_part_wrong_{}'.format(th)].sum,
                                                                      loc_err[
                                                                          'top1_locerr_scg_more_wrong_{}'.format(th)].sum,
                                                                      loc_err['top1_locerr_scg_other_{}'.format(th)].sum))
            fw.write('== Gt-Known loc err ')
            fw.write('CAM-Top1: {:.2f} \n'.format(loc_err['top1_locerr_{}'.format(th)].avg))
            if args.scg:
                fw.write('SCG-Top1: {:.2f} \n'.format(loc_err['gt_known_locerr_scg_{}'.format(th)].avg))

if __name__ == '__main__':
    args = opts().parse()
    val(args)
