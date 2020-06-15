# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 20:33
# @Author  : guardianV
# @Email   : Tang_dq35@163.com
# @File    : fast_text_classifier.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np

import matplotlib.pyplot as plt
import fasttext

# model = fasttext.train_supervised(input="/usr/tdq/THUCNews/news.train", epoch=15, lr=0.5, wordNgrams=2)
# model.save_model("/usr/tdq/fasttext/model_news.bin")
model = fasttext.load_model("/usr/tdq/fasttext/model_news.bin")
print(model.predict(" ".join("专家预计中国2020年GDP将突破百万亿"), k=1))
print(model.test("/usr/tdq/dataset/THUCNews/news.test"))
