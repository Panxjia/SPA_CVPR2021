#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/13 20:20
# @Author  : Xingjia Pan
# @File    : extract_xml.py
# @Software: PyCharm

from xml.etree import ElementTree as ET
import os
import os.path  as osp
import sys


def parse_xml(xml_dir, bbox_file, save_dir):
    xml_files = os.listdir(xml_dir)
    annotations = []
    xml_files = sorted(xml_files)
    for i in range(len(xml_files)):
        anno_line = []
        xml_file = osp.join(xml_dir, xml_files[i])
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = xml_files[i]
        anno_line.append(filename)
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            anno_line.append(xmin)
            anno_line.append(ymin)
            anno_line.append(xmax)
            anno_line.append(ymax)
        annotations.append(anno_line)

    with open(osp.join(save_dir, bbox_file), 'w') as sb:
        for anno in annotations:
            sb.write(' '.join(anno) + '\n')







if __name__ == '__main__':
    data_dir = sys.argv[1]
    parse_xml(data_dir, 'val_bboxes.txt', './')