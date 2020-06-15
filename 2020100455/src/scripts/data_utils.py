# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 18:19
# @Author  : guardianV
# @Email   : Tang_dq35@163.com
# @File    : data_utils.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np

import matplotlib.pyplot as plt

source_path = '/root/workspace/datasets/re/SanWen/preprocessed.txt'
train_path = '/root/workspace/datasets/re/SanWen/preprocessed-train.txt'
test_path = '/root/workspace/datasets/re/SanWen/preprocessed-test.txt'


train = ''
test = ''
with open(source_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if torch.rand(1) > 0.9:
            test += line
            print(1)
        else:
            train += line
            print(0)
    with open(train_path, 'w') as train_f:
        train_f.write(train)
    with open(test_path, 'w') as test_f:
        test_f.write(test)
