import numpy as np
import cv2
import os
import torch
import random

idx2catename = {
    'voc20': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],

    'coco80': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
               'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
               'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']}


def save_im_heatmap_box(im_file, top_maps, topk_boxes, save_dir, gt_label=None, gt_box=None,
                        epoch=100, threshold=-1, suffix=''):
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)

    draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
    draw_hm = 255 * np.ones((h + 15, w, 3), np.uint8)
    cam_to_save = [draw_hm.copy()]
    draw_im[:h, :, :] = im
    cv2.putText(draw_im, 'original image: {}'.format(threshold), (0, h + 12), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save = [draw_im.copy()]
    for cls_box, cam_map_cls in zip(topk_boxes, top_maps):
        draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_im[:h, :, :] = im
        cam_map_cls = cv2.resize(cam_map_cls, dsize=(w, h))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_cls), cv2.COLORMAP_JET)

        draw_im[:h, :, :] = heatmap * 0.7 + draw_im[:h, :, :] * 0.3
        draw_hm[:h, :, :] = heatmap

        # cv2.rectangle(draw_im, (cls_box[1], cls_box[2]), (cls_box[3], cls_box[4]), color=(255, 0, 0), thickness=2)
        if gt_box is not None:
            box_cnt = len(gt_box) // 4
            gt_box = np.asarray(gt_box, int)
            loc_flag = False
            for i in range(box_cnt):
                gt_bbox = gt_box[i*4:(i+1)*4]
                # cv2.rectangle(draw_im, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), color=(0, 0, 255), thickness=2)
                loc_flag = loc_flag or (cal_iou(cls_box[1:], gt_bbox) > 0.5)
            loc_str = 'LOC_TRUE' if loc_flag else 'LOC_FALSE'
        if gt_label is not None:
            cls_str = 'CLS_TRUE' if int(cls_box[0]) == int(gt_label) else 'CLS_FALSE'
        else:
            cls_str = 'classified as {}'.format(cls_box[0])
        cv2.putText(draw_im, cls_str+'|{}'.format(loc_str), (0, h + 12), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        im_to_save.append(draw_im.copy())
        cam_to_save.append(draw_hm.copy())
    im_to_save = np.concatenate(im_to_save, axis=1)
    cam_to_save = np.concatenate(cam_to_save, axis=1)
    im_to_save = np.concatenate((im_to_save, cam_to_save), axis=0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = str(gt_label[0])+'_'+str(epoch)+'_'+im_file.split('/')[-1]
    save_name = save_name.replace('.', 'th-{}_{}.'.format(threshold, suffix))
    cv2.imwrite(os.path.join(save_dir, save_name), im_to_save)

def save_sim_heatmap_box(im_file, top_maps, save_dir, gt_label=None, sim_map=None,
                        epoch=100, threshold=-1, suffix='', fg_th=0.1, bg_th=0.05):
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)

    # draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
    # draw_hm = 255 * np.ones((h + 15, w, 3), np.uint8)
    # draw_im[:h, :, :] = im
    # cv2.putText(draw_im, 'original image: {}'.format(threshold), (0, h + 12), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX,
    #             fontScale=0.5)
    final_to_save = []
    if isinstance(sim_map, tuple) or isinstance(sim_map, list):
        pass
    else:
        sim_map = [sim_map]
    for sim_map_i in sim_map:
        if sim_map_i is None:
            continue
        draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_hm = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_im[:h, :, :] = im
        cv2.putText(draw_im, 'original image: {}'.format(threshold), (0, h + 12), color=(0, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5)
        sim_map_i = sim_map_i.squeeze().data.cpu().numpy()
        cam_to_save = [draw_hm.copy()]
        cam_pos_to_save = [draw_hm.copy()]
        cam_neg_to_save = [draw_hm.copy()]
        im_to_save = [draw_im.copy()]
        wh_sim = sim_map_i.shape[0]
        h_sim, w_sim = int(np.sqrt(wh_sim)), int(np.sqrt(wh_sim))
        for cam_map_cls in  top_maps:
            draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
            draw_im[:h, :, :] = im
            cam_map_cls = cv2.resize(cam_map_cls, dsize=(w_sim, h_sim))
            cam_map_cls_vector = cam_map_cls.reshape(-1)
            #positive
            cam_map_cls_id = np.arange(wh_sim).astype(np.int)
            cam_map_cls_th_ind_pos = cam_map_cls_id[cam_map_cls_vector > fg_th]
            sim_map_sel_pos = sim_map_i[:,cam_map_cls_th_ind_pos]
            sim_map_sel_pos = (sim_map_sel_pos - np.min(sim_map_sel_pos,axis=0, keepdims=True))/(
                    np.max(sim_map_sel_pos, axis=0, keepdims=True) - np.min(sim_map_sel_pos, axis=0, keepdims=True) + 1e-10)
            cam_map_cls_val_pos = cam_map_cls_vector[cam_map_cls_th_ind_pos].reshape(1,-1)
            # sim_map_sel_pos = np.sum(sim_map_sel_pos * cam_map_cls_val_pos, axis=1).reshape(h_sim, w_sim)
            sim_map_sel_pos = np.sum(sim_map_sel_pos, axis=1).reshape(h_sim, w_sim)
            sim_map_sel_pos = (sim_map_sel_pos - np.min(sim_map_sel_pos))/(np.max(sim_map_sel_pos) - np.min(sim_map_sel_pos) + 1e-10)

            #negtive
            cam_map_cls_th_ind_neg = cam_map_cls_id[cam_map_cls_vector < bg_th]
            sim_map_sel_neg = sim_map_i[:, cam_map_cls_th_ind_neg]
            sim_map_sel_neg = (sim_map_sel_neg - np.min(sim_map_sel_neg,axis=0, keepdims=True))/(
                    np.max(sim_map_sel_neg, axis=0, keepdims=True)-np.min(sim_map_sel_neg, axis=0, keepdims=True) + 1e-10)
            cam_map_cls_val_neg = cam_map_cls_vector[cam_map_cls_th_ind_neg].reshape(1, -1)
            # sim_map_sel_neg = np.sum(sim_map_sel_neg * (1-cam_map_cls_val_neg), axis=1).reshape(h_sim, w_sim)
            sim_map_sel_neg = np.sum(sim_map_sel_neg, axis=1).reshape(h_sim, w_sim)
            sim_map_sel_neg = (sim_map_sel_neg - np.min(sim_map_sel_neg))/(np.max(sim_map_sel_neg) - np.min(sim_map_sel_neg) + 1e-10)

            #
            sim_map_sel = sim_map_sel_pos - sim_map_sel_neg
            # sim_map_sel = sim_map_sel_pos
            sim_map_sel = sim_map_sel * (sim_map_sel>0)
            sim_map_sel = (sim_map_sel-np.min(sim_map_sel))/(np.max(sim_map_sel) - np.min(sim_map_sel)+1e-10)
            sim_map_sel = cv2.resize(sim_map_sel, dsize=(w, h))
            sim_map_sel_pos = cv2.resize(sim_map_sel_pos, dsize=(w, h))
            sim_map_sel_neg = cv2.resize(sim_map_sel_neg, dsize=(w, h))
            heatmap = cv2.applyColorMap(np.uint8(255 * sim_map_sel), cv2.COLORMAP_JET)
            heatmap_pos = cv2.applyColorMap(np.uint8(255 * sim_map_sel_pos), cv2.COLORMAP_JET)
            heatmap_neg = cv2.applyColorMap(np.uint8(255 * sim_map_sel_neg), cv2.COLORMAP_JET)
            draw_im[:h, :, :] = heatmap * 0.7 + draw_im[:h, :, :] * 0.3
            draw_hm[:h, :, :] = heatmap

            im_to_save.append(draw_im.copy())
            cam_to_save.append(draw_hm.copy())
            draw_hm[:h, :, :] = heatmap_pos
            cam_pos_to_save.append(draw_hm.copy())
            draw_hm[:h, :, :] = heatmap_neg
            cam_neg_to_save.append(draw_hm.copy())
        im_to_save = np.concatenate(im_to_save, axis=1)
        cam_to_save = np.concatenate(cam_to_save, axis=1)
        cam_pos_to_save = np.concatenate(cam_pos_to_save, axis=1)
        cam_neg_to_save = np.concatenate(cam_neg_to_save, axis=1)
        final_to_save.append(im_to_save)
        final_to_save.append(cam_to_save)
        final_to_save.append(cam_pos_to_save)
        final_to_save.append(cam_neg_to_save)
    final_to_save = np.concatenate(final_to_save, axis=0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = str(gt_label[0])+'_'+str(epoch)+'_'+im_file.split('/')[-1]
    save_name = save_name.replace('.', 'th-{}_{}.'.format(threshold, suffix))
    cv2.imwrite(os.path.join(save_dir, save_name), final_to_save)

def save_im_sim(im_file, aff_maps, save_dir, suffix='', gt_label=None, epoch=100):

    if isinstance(aff_maps, tuple) or isinstance(aff_maps, list):
        pass
    else:
        aff_maps = [aff_maps]
    break_flag = True
    for aff_i in aff_maps:
        if aff_i is not None:
            break_flag = False
            break
    if break_flag:
        return

    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)
    draw_aff = []
    draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
    for i in range(len(aff_maps)):
        if aff_maps[i] is not None:
            draw_aff.append(255 * np.ones((h + 15, w, 3), np.uint8))
    draw_aff = np.concatenate(draw_aff, axis=0)
    aff_to_save = [draw_aff.copy()]
    draw_im[:h, :, :] = im
    cv2.putText(draw_im, 'original image', (0, h + 12), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save = [draw_im.copy()]
    points = [[h//4, w//4],[h//4*3, w//4*3],[h//2, w//2],[h//4*3, w//4],[h//4, w//4*3]]
    for point in points:
        draw_im = 255 * np.ones((h + 15, w, 3), np.uint8)
        draw_im[:h, :, :] = im
        draw_aff_j = []
        for i, aff_i in enumerate(aff_maps):
            if aff_i is None:
                continue
            aff_i = aff_i.squeeze().data.cpu().numpy()
            draw_aff_ij = 255 * np.ones((h + 15, w, 3), np.uint8)
            h_w_aff = aff_i.shape[0]
            h_aff, w_aff = int(np.sqrt(h_w_aff)), int(np.sqrt(h_w_aff))
            h_aff_i, w_aff_i = int(point[0] * h_aff/h), int(point[1] * w_aff/w)
            aff_map_i = aff_i[:,h_aff_i*w_aff + w_aff_i].reshape(h_aff, w_aff)
            aff_map_i = (aff_map_i - np.min(aff_map_i))/(np.max(aff_map_i) - np.min(aff_map_i) + 1e-10)
            aff_map_i = cv2.resize(aff_map_i, dsize=(w, h))
            aff_map_i = cv2.applyColorMap(np.uint8(255 * aff_map_i), cv2.COLORMAP_JET)

            ptStart_h = (point[1]-5, point[0])
            ptEnd_h = (point[1]+5, point[0])
            point_color = (0, 255, 0)  # BGR
            thickness = 2
            lineType = 4
            cv2.line(draw_im, ptStart_h, ptEnd_h, point_color, thickness, lineType)

            ptStart_v = (point[1], point[0]-5)
            ptEnd_v = (point[1], point[0]+5)
            cv2.line(draw_im, ptStart_v, ptEnd_v, point_color, thickness, lineType)

            draw_aff_ij[:h, :, :] = aff_map_i
            cv2.putText(draw_aff_ij, '{} layer'.format(i+2), (0, h + 12), color=(0, 0, 0),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            draw_aff_j.append(draw_aff_ij)
        draw_aff_j = np.concatenate(draw_aff_j, axis=0)
        im_to_save.append(draw_im.copy())
        aff_to_save.append(draw_aff_j.copy())

    im_to_save = np.concatenate(im_to_save, axis=1)
    aff_to_save = np.concatenate(aff_to_save, axis=1)
    im_to_save = np.concatenate((im_to_save, aff_to_save), axis=0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = str(gt_label[0])+'_'+str(epoch)+'_'+im_file.split('/')[-1]
    save_name = save_name.replace('.', '_sim_{}.'.format(suffix))
    cv2.imwrite(os.path.join(save_dir, save_name), im_to_save)

def cal_iou(box1, box2):
    """
    support:
    1. box1 and box2 are the same shape: [N, 4]
    2.
    :param box1:
    :param box2:
    :return:
    """
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    iou_val = i_area / (box1_area + box2_area - i_area)

    return iou_val

def vis_feature(feat, img_path, vis_path, col=4, row=4, layer='feat3'):
    ## normalize feature
    feat = feat[0,...]
    c, fh, fw = feat.size()
    feat = feat.view(c, -1)
    min_val, _ = torch.min(feat, dim=-1, keepdim=True)
    max_val, _ = torch.max(feat, dim=-1, keepdim=True)
    norm_feat = (feat - min_val) / (max_val - min_val+1e-10)
    norm_feat = norm_feat.view(c, fh, fw).contiguous().permute(1,2,0)
    norm_feat = norm_feat.data.cpu().numpy()

    im = cv2.imread(img_path)
    h, w, _ = np.shape(im)
    resized_feat = cv2.resize(norm_feat, (w, h))

    # draw images
    feat_ind = 0
    fig_id = 0

    while feat_ind < 20:
        im_to_save = []
        for i in range(row):
            draw_im = 255 * np.ones((h + 15, w+5, 3), np.uint8)
            draw_im[:h, :w, :] = im
            cv2.putText(draw_im, 'original image', (0, h + 12), color=(0, 0, 0),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5)
            im_to_save_row = [draw_im.copy()]
            for j in range(col):
                draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
                draw_im[:h, :w, :] = im

                heatmap = cv2.applyColorMap(np.uint8(255 * resized_feat[:,:,feat_ind]), cv2.COLORMAP_JET)
                draw_im[:h, :w, :] = heatmap * 1. + draw_im[:h, :w, :] * 0.0

                im_to_save_row.append(draw_im.copy())
                feat_ind += 1
            im_to_save_row = np.concatenate(im_to_save_row, axis=1)
            im_to_save.append(im_to_save_row)
        im_to_save = np.concatenate(im_to_save, axis=0)
        vis_path = os.path.join(vis_path,'vis_feat')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        save_name = 'vgg_' + img_path.split('/')[-1]
        save_name = save_name.replace('.','_{}_{}.'.format(layer, fig_id))
        cv2.imwrite(os.path.join(vis_path, save_name), im_to_save)
        fig_id +=1

def vis_var(feat, cls_logits, img_path, vis_path, net='vgg_fpn_l3'):

    cls_logits = cls_logits.squeeze()

    norm_var_no_white = norm_tensor(feat)
    norm_var_no_white =  1- norm_var_no_white
    norm_var_no_white[norm_var_no_white<0.05] = 0
    norm_var_no_white = norm_atten_map(norm_var_no_white)
    # norm_var_no_white = (norm_var_no_white < 0.4).astype(norm_var_no_white.dtype)
    norm_cls_no_white = norm_tensor(cls_logits)
    norm_cls_no_white[norm_cls_no_white<0.2] = 0
    norm_cls_no_white = norm_atten_map(norm_cls_no_white)

    # norm_cls_no_white = (norm_cls_no_white>0.7).astype(norm_cls_no_white.dtype)

    white_feat = whitening_tensor(feat)
    white_cls_logits = whitening_tensor(cls_logits)
    norm_var = norm_tensor(white_feat)
    norm_var[norm_var < 0.8] = 0
    norm_var = norm_atten_map(norm_var)
    # norm_var = (norm_var<0.4).astype(norm_var.dtype)
    norm_cls = norm_tensor(white_cls_logits)
    norm_cls =1 - norm_cls
    norm_cls[norm_cls < 0.8] = 0
    norm_cls= norm_atten_map(norm_cls)
    # norm_cls = (norm_cls>0.2).astype(norm_cls.dtype)

    im = cv2.imread(img_path)
    h, w, _ = np.shape(im)
    resized_var_no_white = cv2.resize(norm_var_no_white, (w, h))
    resized_cls_no_white = cv2.resize(norm_cls_no_white, (w, h))
    resized_var = cv2.resize(norm_var, (w, h))
    resized_cls= cv2.resize(norm_cls, (w, h))

    draw_im = 255 * np.ones((h + 15, w+5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    cv2.putText(draw_im, 'original image', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save = [draw_im.copy()]

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_var_no_white), cv2.COLORMAP_BONE)
    draw_im[:h, :w, :] = heatmap * 1.0 + draw_im[:h, :w, :] * 0
    cv2.putText(draw_im, 'var_nw', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())
    im_to_save.append(draw_im.copy())

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_cls_no_white), cv2.COLORMAP_BONE)
    draw_im[:h, :w, :] = heatmap * 1. + draw_im[:h, :w, :] * 0
    cv2.putText(draw_im, 'cls_nw', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    resized_var_cls_no_white = (resized_var_no_white + resized_cls_no_white)*0.5
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_var_cls_no_white), cv2.COLORMAP_JET)
    draw_im[:h, :w, :] = heatmap * 0.5 + draw_im[:h, :w, :] * 0.5
    cv2.putText(draw_im, 'var_cls_nw', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())


    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_var), cv2.COLORMAP_JET)
    draw_im[:h, :w, :] = heatmap * 1. + draw_im[:h, :w, :] * 0
    cv2.putText(draw_im, 'var', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_cls), cv2.COLORMAP_BONE)
    draw_im[:h, :w, :] = heatmap * 1 + draw_im[:h, :w, :] * 0
    cv2.putText(draw_im, 'cls', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())

    draw_im = 255 * np.ones((h + 15, w + 5, 3), np.uint8)
    draw_im[:h, :w, :] = im
    resized_var_cls = (resized_var + resized_cls) * 0.5
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_var_cls), cv2.COLORMAP_JET)
    draw_im[:h, :w, :] = heatmap * 0.5 + draw_im[:h, :w, :] * 0.5
    cv2.putText(draw_im, 'var_cls', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save.append(draw_im.copy())

    im_to_save = np.concatenate(im_to_save, axis=1)

    vis_path = os.path.join(vis_path, 'vis_var/{}'.format(net))
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    save_name = 'vgg_' + img_path.split('/')[-1]
    cv2.imwrite(os.path.join(vis_path, save_name), im_to_save)

def norm_tensor(feat):
    min_val = torch.min(feat)
    max_val = torch.max(feat)
    norm_feat = (feat - min_val) / (max_val - min_val + 1e-20)
    norm_feat = norm_feat.data.cpu().numpy()
    return norm_feat

def whitening_tensor(feat):
    mean = torch.mean(feat)
    var = torch.std(feat)
    norm_feat = (feat-mean)/(var+1e-15)
    return norm_feat

def norm_atten_map(attention_map):
    min_val = np.min(attention_map)
    max_val = np.max(attention_map)
    atten_norm = (attention_map - min_val) / (max_val - min_val+1e-10)
    return atten_norm

def save_im_gcam_ggrads(im_file, grads, save_dir, layers=None, topk=5):
    im = cv2.imread(im_file)
    h, w, _ = np.shape(im)

    draw_im = 255 * np.ones((int(topk)*2*(h + 15), w, 3), np.uint8)
    for i in range(topk):
        start = i * 2 * (h+15)
        draw_im[start:start+h, :, :] = im
        cv2.putText(draw_im, 'original image', (0, start+h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    im_to_save = [draw_im.copy()]

    gcam_com = {}
    ggrad_com = {}
    for layer in layers:
        draw_im = 255 * np.ones((int(topk) * 2 * (h + 15), w, 3), np.uint8)
        for i in range(topk):
            gcam = grads['gcam_{}'.format(layer)][0,i,:,:].data.cpu().numpy()
            ggrad = grads['g2_{}'.format(layer)][0,i,:,:].data.cpu().numpy()
            gcam = norm_atten_map(gcam)
            ggrad = norm_atten_map(ggrad)
            gcam = cv2.resize(gcam, dsize=(w, h))
            ggrad = cv2.resize(ggrad, dsize=(w, h))
            gcam_rgb = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
            ggrad_rgb = cv2.applyColorMap(np.uint8(255 * ggrad), cv2.COLORMAP_JET)
            start = i * 2 * (h + 15)
            draw_im[start:start+h] = gcam_rgb*0.7 + im*0.3
            cv2.putText(draw_im, 'gcam: {}'.format(layer), (0, start+h + 12), color=(0, 0, 0),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5)
            draw_im[start+h+15:start + 2*h+15] = ggrad_rgb
            cv2.putText(draw_im, 'ggrads: {}'.format(layer), (0, start+2 * h + 15 + 12), color=(0, 0, 0),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.5)
            if 'gcam_top_{}'.format(i) not in gcam_com:
                gcam_com['gcam_top_{}'.format(i)] = gcam
            else:
                gcam_com['gcam_top_{}'.format(i)]= np.maximum(gcam_com['gcam_top_{}'.format(i)],gcam)
            if 'ggrad_top_{}'.format(i) not in ggrad_com:
                ggrad_com['ggrad_top_{}'.format(i)] = ggrad
            else:
                # ggrad_com['ggrad_top_{}'.format(i)] += ggrad
                ggrad_com['ggrad_top_{}'.format(i)]=np.maximum(ggrad_com['ggrad_top_{}'.format(i)],ggrad)
        im_to_save.append(draw_im.copy())


    draw_im = 255 * np.ones((int(topk) * 2 * (h + 15), w, 3), np.uint8)

    for i in range(topk):
        gcam_com_i = gcam_com['gcam_top_{}'.format(i)]
        ggrad_com_i = ggrad_com['ggrad_top_{}'.format(i)]
        # gcam_com_i = norm_atten_map(gcam_com_i)
        gcam_rgb_com = cv2.applyColorMap(np.uint8(255 * gcam_com_i), cv2.COLORMAP_JET)

        # ggrad_com_i = norm_atten_map(ggrad_com_i)
        ggrad_rgb_com = cv2.applyColorMap(np.uint8(255 * ggrad_com_i), cv2.COLORMAP_JET)
        start = i * 2 * (h + 15)
        draw_im[start:start + h] = gcam_rgb_com * 0.7 + im * 0.3
        cv2.putText(draw_im, 'gcam_com', (0, start + h + 12), color=(0, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5)
        draw_im[start + h + 15:start + 2 * h + 15] = ggrad_rgb_com
        cv2.putText(draw_im, 'ggrads_com', (0, start + 2 * h + 15 + 12), color=(0, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5)
    im_to_save.append(draw_im.copy())

    im_to_save = np.concatenate(im_to_save, axis=1)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = im_file.split('/')[-1].replace('.', '_gcam_ggrads.')
    cv2.imwrite(os.path.join(save_dir, save_name), im_to_save)


class SAVE_ATTEN(object):
    def __init__(self, save_dir='save_bins', dataset=None):
        # type: (object, object) -> object
        self.save_dir = save_dir
        if dataset is not None:
            self.idx2cate = self._get_idx2cate_dict(datasetname=dataset)
        else:
            self.idx2cate = None

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_top_5_pred_labels(self, preds, org_paths, global_step):
        img_num = np.shape(preds)[0]
        for idx in range(img_num):
            img_name = org_paths[idx].strip().split('/')[-1]
            if '.JPEG' in img_name:
                img_id = img_name[:-5]
            elif '.png' in img_name or '.jpg' in img_name:
                img_id = img_name[:-4]

            out = img_id + ' ' + ' '.join(map(str, preds[idx, :])) + '\n'
            out_file = os.path.join(self.save_dir, 'pred_labels.txt')

            if global_step == 0 and idx == 0 and os.path.exists(out_file):
                os.remove(out_file)
            with open(out_file, 'a') as f:
                f.write(out)

    def save_masked_img_batch(self, path_batch, atten_batch, label_batch):

        # img_num = np.shape(atten_batch)[0]
        img_num = atten_batch.size()[0]
        # fid = open('imagenet_val_shape.txt', 'a')
        # print(np.shape(img_batch), np.shape(label_batch), np.shape(org_size_batch), np.shape(atten_batch))
        for idx in range(img_num):
            atten = atten_batch[idx]
            atten = atten.cpu().data.numpy()
            label = label_batch[idx]
            label = int(label)
            self._save_masked_img(path_batch[idx], atten, label)

    def _get_idx2cate_dict(self, datasetname=None):
        if datasetname not in idx2catename.keys():
            print('The given %s dataset category names are not available. The supported are: %s' \
                  % (str(datasetname), ','.join(idx2catename.keys())))
            return None
        else:
            return {idx: cate_name for idx, cate_name in enumerate(idx2catename[datasetname])}

    def _save_masked_img(self, img_path, atten, label):
        '''
        save masked images with only one ground truth label
        :param path:
        :param img:
        :param atten:
        :param org_size:
        :param label:
        :param scores:
        :param step:
        :param args:
        :return:
        '''
        if not os.path.isfile(img_path):
            raise 'Image not exist:%s' % (img_path)
        img = cv2.imread(img_path)
        org_size = np.shape(img)
        w = org_size[0]
        h = org_size[1]

        attention_map = atten[label, :, :]
        atten_norm = attention_map
        print(np.shape(attention_map), 'Max:', np.max(attention_map), 'Min:', np.min(attention_map))
        # min_val = np.min(attention_map)
        # max_val = np.max(attention_map)
        # atten_norm = (attention_map - min_val)/(max_val - min_val)
        atten_norm = cv2.resize(atten_norm, dsize=(h, w))
        atten_norm = atten_norm * 255
        heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)

        img_id = img_path.strip().split('/')[-1]
        img_id = img_id.strip().split('.')[0]
        save_dir = os.path.join(self.save_dir, img_id + '.png')
        cv2.imwrite(save_dir, img)

    def get_img_id(self, path):
        img_id = path.strip().split('/')[-1]
        return img_id.strip().split('.')[0]

    def save_top_5_atten_maps(self, atten_fuse_batch, top_indices_batch, org_paths, topk=5):
        '''
        Save top-5 localization maps for generating bboxes
        :param atten_fuse_batch: normalized last layer feature maps of size (batch_size, C, W, H), type: numpy array
        :param top_indices_batch: ranked predicted labels of size (batch_size, C), type: numpy array
        :param org_paths:
        :param args:
        :return:
        '''
        img_num = np.shape(atten_fuse_batch)[0]
        for idx in range(img_num):
            img_id = org_paths[idx].strip().split('/')[-1].split('.')[0]
            im = cv2.imread(org_paths[idx])
            img_shape = im.shape[:2]
            for k in range(topk):
                atten_pos = top_indices_batch[idx, k]
                atten_map = atten_fuse_batch[idx, atten_pos, :, :]
                # heat_map = cv2.resize(atten_map, dsize=(224, 224))
                heat_map = cv2.resize(atten_map, dsize=(img_shape[1], img_shape[0]))
                heat_map = heat_map * 255
                save_path = os.path.join(self.save_dir, 'heat_maps', 'top%d' % (k + 1))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(save_path, img_id + '.png')
                cv2.imwrite(save_path, heat_map)

    # def save_heatmap_segmentation(self, img_path, atten, gt_label, save_dir=None, size=(224,224), maskedimg=False):
    #     assert np.ndim(atten) == 4
    #
    #     labels_idx = np.where(gt_label[0]==1)[0] if np.ndim(gt_label)==2 else np.where(gt_label==1)[0]
    #
    #     if save_dir is None:
    #         save_dir = self.save_dir
    #         if not os.path.exists(save_dir):
    #             os.mkdir(save_dir)
    #
    #     if isinstance(img_path, list) or isinstance(img_path, tuple):
    #         batch_size = len(img_path)
    #         for i in range(batch_size):
    #             img, size = self.read_img(img_path[i], size=size)
    #             atten_img = atten[i]     #get attention maps for the i-th img of the batch
    #             img_name = self.get_img_id(img_path[i])
    #             img_dir = os.path.join(save_dir, img_name)
    #             if not os.path.exists(img_dir):
    #                 os.mkdir(img_dir)
    #             for k in labels_idx:
    #                 atten_map_k = atten_img[k,:,:]
    #                 atten_map_k = cv2.resize(atten_map_k, dsize=size)
    #                 if maskedimg:
    #                     img_to_save = self._add_msk2img(img, atten_map_k)
    #                 else:
    #                     img_to_save = self.normalize_map(atten_map_k)*255.0
    #
    #                 save_path = os.path.join(img_dir, '%d.png'%(k))
    #                 cv2.imwrite(save_path, img_to_save)

    def normalize_map(self, atten_map):
        min_val = np.min(atten_map)
        max_val = np.max(atten_map)
        atten_norm = (atten_map - min_val) / (max_val - min_val)

        return atten_norm

    def _add_msk2img(self, img, msk, isnorm=True):
        if np.ndim(img) == 3:
            assert np.shape(img)[0:2] == np.shape(msk)
        else:
            assert np.shape(img) == np.shape(msk)

        if isnorm:
            min_val = np.min(msk)
            max_val = np.max(msk)
            atten_norm = (msk - min_val) / (max_val - min_val)
        atten_norm = atten_norm * 255
        heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
        w_img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)

        return w_img

    def _draw_text(self, pic, txt, pos='topleft'):
        font = cv2.FONT_HERSHEY_SIMPLEX  # multiple line
        txt = txt.strip().split('\n')
        stat_y = 30
        for t in txt:
            pic = cv2.putText(pic, t, (10, stat_y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            stat_y += 30

        return pic

    def _mark_score_on_picture(self, pic, score_vec, label_idx):
        score = score_vec[label_idx]
        txt = '%.3f' % (score)
        pic = self._draw_text(pic, txt, pos='topleft')
        return pic

    def get_heatmap_idxes(self, gt_label):

        labels_idx = []
        if np.ndim(gt_label) == 1:
            labels_idx = np.expand_dims(gt_label, axis=1).astype(np.int)
        elif np.ndim(gt_label) == 2:
            for row in gt_label:
                idxes = np.where(row[0] == 1)[0] if np.ndim(row) == 2 else np.where(row == 1)[0]
                labels_idx.append(idxes.tolist())
        else:
            labels_idx = None

        return labels_idx

    def get_map_k(self, atten, k, size=(224, 224)):
        atten_map_k = atten[k, :, :]
        # print np.max(atten_map_k), np.min(atten_map_k)
        atten_map_k = cv2.resize(atten_map_k, dsize=size)
        return atten_map_k

    def read_img(self, img_path, size=(224, 224)):
        img = cv2.imread(img_path)
        if img is None:
            print("Image does not exist. %s" % (img_path))
            exit(0)

        if size == (0, 0):
            size = np.shape(img)[:2]
        else:
            img = cv2.resize(img, size)
        return img, size[::-1]

    def get_masked_img(self, img_path, atten, gt_label,
                       size=(224, 224), maps_in_dir=False, save_dir=None, only_map=False):

        assert np.ndim(atten) == 4

        save_dir = save_dir if save_dir is not None else self.save_dir

        if isinstance(img_path, list) or isinstance(img_path, tuple):
            batch_size = len(img_path)
            label_indexes = self.get_heatmap_idxes(gt_label)
            for i in range(batch_size):
                img, size = self.read_img(img_path[i], size)
                img_name = img_path[i].split('/')[-1]
                img_name = img_name.strip().split('.')[0]
                if maps_in_dir:
                    img_save_dir = os.path.join(save_dir, img_name)
                    os.mkdir(img_save_dir)

                for k in label_indexes[i]:
                    atten_map_k = self.get_map_k(atten[i], k, size)
                    msked_img = self._add_msk2img(img, atten_map_k)

                    suffix = str(k + 1)
                    if only_map:
                        save_img = (self.normalize_map(atten_map_k) * 255).astype(np.int)
                    else:
                        save_img = msked_img

                    if maps_in_dir:
                        cv2.imwrite(os.path.join(img_save_dir, suffix + '.png'), save_img)
                    else:
                        cv2.imwrite(os.path.join(save_dir, img_name + '_' + suffix + '.png'), save_img)

                    #     if score_vec is not None and labels_idx is not None:
                    #         msked_img = self._mark_score_on_picture(msked_img, score_vec, labels_idx[k])
                    #     if labels_idx is not None:
                    #         suffix = self.idx2cate.get(labels_idx[k], k)

    # def get_masked_img_ml(self, img_path, atten, save_dir=None, size=(224,224),
    #                       gt_label=None, score_vec=None):
    #     assert np.ndim(atten) == 4
    #
    #     if gt_label is not None and self.idx2cate is not None:
    #         labels_idx = np.where(gt_label[0]==1)[0] if np.ndim(gt_label)==2 else np.where(gt_label==1)[0]
    #     else:
    #         labels_idx = None
    #
    #
    #     if save_dir is not None:
    #         self.save_dir = save_dir
    #     if isinstance(img_path, list) or isinstance(img_path, tuple):
    #         batch_size = len(img_path)
    #         for i in range(batch_size):
    #             img = cv2.imread(img_path[i])
    #             if img is None:
    #                 print "Image does not exist. %s" %(img_path[i])
    #                 exit(0)
    #
    #             else:
    #                 atten_img = atten[i]     #get attention maps for the i-th img
    #                 img_name = img_path[i].split('/')[-1]
    #                 for k in range(np.shape(atten_img)[0]):
    #                     if size == (0,0):
    #                         w, h, _ = np.shape(img)
    #                         # h, w, _ = np.shape(img)
    #                     else:
    #                         h, w = size
    #                         img = cv2.resize(img, dsize=(h, w))
    #                     atten_map_k = atten_img[k,:,:]
    #                     # print np.max(atten_map_k), np.min(atten_map_k)
    #                     atten_map_k = cv2.resize(atten_map_k, dsize=(h,w))
    #                     msked_img = self._add_msk2img(img, atten_map_k)
    #                     if score_vec is not None and labels_idx is not None:
    #                         msked_img = self._mark_score_on_picture(msked_img, score_vec, labels_idx[k])
    #                     if labels_idx is not None:
    #                         suffix = self.idx2cate.get(labels_idx[k], k)
    #                     else:
    #                         suffix = str(k)
    #                     if '.' in img_name:
    #                         img_name = img_name.strip().split('.')[0]
    #                     cv2.imwrite(os.path.join(self.save_dir, img_name + '_' + suffix + '.png'), msked_img)
    #
    #
    # def get_masked_img(self, img_path, atten, save_dir=None,  size=(224,224), combine=True):
    #     '''
    #
    #     :param img_path:
    #     :param atten:
    #     :param size: if it is (0,0) use original image size, otherwise use the specified size.
    #     :param combine:
    #     :return:
    #     '''
    #
    #     if save_dir is not None:
    #         self.save_dir = save_dir
    #     if isinstance(img_path, list) or isinstance(img_path, tuple):
    #         batch_size = len(img_path)
    #
    #         for i in range(batch_size):
    #             atten_norm = atten[i]
    #             min_val = np.min(atten_norm)
    #             max_val = np.max(atten_norm)
    #             atten_norm = (atten_norm - min_val)/(max_val - min_val)
    #             # print np.max(atten_norm), np.min(atten_norm)
    #             img = cv2.imread(img_path[i])
    #             if img is None:
    #                 print "Image does not exist. %s" %(img_path[i])
    #                 exit(0)
    #
    #             if size == (0,0):
    #                 w, h, _ = np.shape(img)
    #                 # h, w, _ = np.shape(img)
    #             else:
    #                 h, w = size
    #                 img = cv2.resize(img, dsize=(h, w))
    #
    #             atten_norm = cv2.resize(atten_norm, dsize=(h,w))
    #             # atten_norm = cv2.resize(atten_norm, dsize=(w,h))
    #             atten_norm = atten_norm* 255
    #             heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
    #             img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)
    #
    #
    #             # font = cv2.FONT_HERSHEY_SIMPLEX
    #             # cv2.putText(img,'OpenCV \n hello',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
    #
    #             img_name = img_path[i].split('/')[-1]
    #             print os.path.join(self.save_dir, img_name)
    #             cv2.imwrite(os.path.join(self.save_dir, img_name), img)

    def get_atten_map(self, img_path, atten, save_dir=None, size=(321, 321)):
        '''
        :param img_path:
        :param atten:
        :param size: if it is (0,0) use original image size, otherwise use the specified size.
        :param combine:
        :return:
        '''

        if save_dir is not None:
            self.save_dir = save_dir
        if isinstance(img_path, list) or isinstance(img_path, tuple):
            batch_size = len(img_path)

            for i in range(batch_size):
                atten_norm = atten[i]
                min_val = np.min(atten_norm)
                max_val = np.max(atten_norm)
                atten_norm = (atten_norm - min_val) / (max_val - min_val)
                # print np.max(atten_norm), np.min(atten_norm)
                h, w = size

                atten_norm = cv2.resize(atten_norm, dsize=(h, w))
                # atten_norm = cv2.resize(atten_norm, dsize=(w,h))
                atten_norm = atten_norm * 255

                img_name = img_path[i].split('/')[-1]
                img_name = img_name.replace('jpg', 'png')
                cv2.imwrite(os.path.join(self.save_dir, img_name), atten_norm)


###############################################################################
########  copy from save_mask.py
from PIL import Image
import numpy as np

# Colour map.
label_colours = [(0, 0, 0),
                 # 0=background
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def decode_labels(mask):
    """Decode batch of segmentation masks.

    Args:
      label_batch: result of inference after taking argmax.

    Returns:
      An batch of RGB images of the same size
    """
    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < 21:
                pixels[k_, j_] = label_colours[k]
            if k == 255:
                pixels[k_, j_] = (255, 255, 255)
    return np.array(img)


#####################################################################
######### copy from save_det_heatmap.py
idx2catename = {
    'voc20': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']}


def get_imgId(path_str):
    return path_str.strip().split('/')[-1].strip().split('.')[0]


def norm_atten_map(attention_map):
    min_val = np.min(attention_map)
    max_val = np.max(attention_map)
    atten_norm = (attention_map - min_val) / (max_val - min_val+1e-10)
    return atten_norm



def add_colormap2img(img, atten_norm):
    heat_map = cv2.applycolormap(atten_norm.astype(np.uint8), cv2.colormap_jet)
    img = cv2.addweighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)
    return img


def save_atten(imgpath, atten, num_classes=20, base_dir='../save_bins/', idx_base=0):
    atten = np.squeeze(atten)
    for cls_idx in range(num_classes):
        cat_dir = os.path.join(base_dir, idx2catename['voc20'][cls_idx])
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)
        cat_map = atten[cls_idx + idx_base]
        # read rgb image
        img = cv2.imread(imgpath)
        h, w, _ = np.shape(img)

        # reshape image
        cat_map = cv2.resize(cat_map, dsize=(w, h))
        cat_map = norm_atten_map(cat_map)

        # save heatmap
        save_path = os.path.join(cat_dir, get_imgId(imgpath) + '.png')
        cv2.imwrite(save_path, cat_map)
        # cv2.imwrite(save_path, add_colormap2img(img, cat_map))


def save_cls_scores(img_path, scores, base_dir='../save_bins/'):
    scores = np.squeeze(scores).tolist()
    score_str = map(lambda x: '%.4f' % (x), scores)

    with open(os.path.join(base_dir, 'scores.txt'), 'a') as fw:
        out_str = get_imgId(img_path) + ' ' + ' '.join(score_str) + '\n'
        fw.write(out_str)
