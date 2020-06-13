import os
import numpy as np

import torch
from torch.utils.data import Dataset
_DATASET_DIR = '../dataset/imageTensors.pth'


class MyDataset(Dataset):
    def __init__(self, phase='train'):
        assert(phase=='train' or phase=='val' or phase=='test')
        self.tensors = torch.load(_DATASET_DIR)
        self.classNum, self.perNum, self.channels, self.imgSize, _ = self.tensors.size()
        assert(self.classNum==50 and self.perNum==300 and self.channels==3 and self.imgSize==84)
        self.labels = torch.arange(self.classNum)
        self.startIdx = 0 
        self.endIdx = 200
        if phase=='val':
            self.startIdx = 200
            self.endIdx = 250
        if phase=='test':
            self.startIdx = 250
            self.endIdx = 300

    def __getitem__(self, index):
        idx_label = index / (self.endIdx - self.startIdx)
        label = self.labels[int(idx_label)]
        idx_img = index % (self.endIdx - self.startIdx)
        img = self.tensors[label, idx_img + self.startIdx, :, :, :]
        return img, label

    def __len__(self):
        return self.classNum * (self.endIdx - self.startIdx)
