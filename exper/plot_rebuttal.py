#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 17:34
# @Author  : Xingjia Pan
# @File    : plot_rebuttal.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import os, sys


bg_ious = [0.02,0.45, 0.58, 0.65, 0.71, 0.76, 0.78, 0.80, 0.82, 0.83, 0.84]
bg_iogs = [0.02, 0.46, 0.58, 0.67, 0.73, 0.79, 0.84, 0.86, 0.89, 0.91, 1.0]
bg_iops = [1.0, 0.99, 0.98, 0.98, 0.96, 0.95, 0.93, 0.92, 0.91, 0.91, 0.84]

x = np.linspace(0, 1., 11)
plt.plot(x, bg_ious, color='r', label='bg_iou')
plt.plot(x, bg_iogs, color='g' ,label='bg_iog')
plt.plot(x, bg_iops, color='b', label='bg_iop')
plt.xlabel('tau')
plt.show()
plt.savefig('eval_var_bg.png')