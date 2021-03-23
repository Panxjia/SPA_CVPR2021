'''
----------------------------------------
Given an heatmap, give out the bbox.

0. Get the DT-ed images
1. detect all contour in the bboxes
2. merge based on some rules.
3. output the bbox
----------------------------------------
'''

import os, sys
import numpy as np
import math
import cv2

def get_bbox(img, g_ths):
    '''
    :param img: single channel heatmap, np.ndarray
    :param g_ths: list of binarization threshold, [th_1, th_2, ..., th_n]
    :return: bboxes [N, (x, y, w, h)]
    '''
    H, W = img.shape
    bboxes = []
    for th in g_ths:
        _, binary = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        # Distance Transform
        binary = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        binary[binary > 255.0] = 255.0
        binary = binary.astype(np.uint8)
        _, binary = cv2.threshold(binary, 3, 255, cv2.THRESH_BINARY)
        contours, hie = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bbox = cv2.boundingRect(contour)
            x, y, w, h = bbox
            x = min(max(x, 0), W-5)
            y = min(max(y, 0), H-5)
            w = min(max(w, 0), W-x-5)
            h = min(max(h, 0), H-y-5)
            bboxes.append([x, y, w, h])

    return bboxes

def big_overlap(bbox1, bbox2):
    '''
    :param bbox1: [x1, y1, w1, h1]
    :param bbox2: [x2, y2, w2, h2]
    :return: bool, whether overlap > max(area(bbox1), area(bbox2)) * 0.5
    '''
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    t = max(w1 * h1, w2 * h2) * 0.5
    x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    y_overlap = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    overlap = x_overlap * y_overlap
    return overlap > t

def merge_bbox(bboxes):
    '''
    :param bboxes: [N, (x, y, w, h)]
    :return: [M, (x, y, w, h)], output of nms
    '''
    bboxes = sorted(bboxes, key=lambda s:s[2]*s[3], reverse=True)
    # print(bboxes)
    not_keep = []
    for i in range(len(bboxes)):
        if i in not_keep:
            continue
        for j in range(i+1, len(bboxes)):
            if j in not_keep:
                continue
            if big_overlap(bboxes[i], bboxes[j]):
                not_keep.append(j)

    bboxes_nms = []
    for i in range(len(bboxes)):
        if i not in not_keep:
            bboxes_nms.append(bboxes[i])
    return bboxes_nms

def show_bboxes(image, bboxes):
    """
    :param image: colorful or grey image
    :param bboxes: [[x,y,w,h], ...]
    :return:
    """
    h, w = image.shape
    draw_im = 255 * np.ones((h + 15, w), np.uint8)
    draw_im[:h, :] = image
    cv2.putText(draw_im, 'bboxes results', (0, h + 12), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5)
    for cls_box in bboxes:
        x_0 = cls_box[0]
        x_1 = cls_box[0] + cls_box[2]
        y_0 = cls_box[1]
        y_1 = cls_box[1] + cls_box[3]
        cv2.rectangle(draw_im, (x_0, y_0), (x_1, y_1), color=(255, 0, 0), thickness=2)

    cv2.imwrite('detect_results.jpg', draw_im)

if __name__ == "__main__":
    image_path = "./heatmap_6.jpg"
    image = cv2.imread(image_path, 0)
    g_ths = [20, 100, 110]
    bboxes = get_bbox(image, g_ths)
    bboxes = merge_bbox(bboxes)
    show_bboxes(image, bboxes)
    print(bboxes)