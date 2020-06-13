import os
import random
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
        self.rot_labels = torch.arange(4)
        self.startIdx = 0 
        self.endIdx = 200
        if phase=='val':
            self.startIdx = 200
            self.endIdx = 250
        if phase=='test':
            self.startIdx = 250
            self.endIdx = 300
    
    def rotate90(self, img):
        sz = img.size(-1)
        rimg = torch.empty_like(img)
        for idx in range(sz):
            rimg[:,:, self.imgSize-idx-1] = img[:,idx,:]
        return rimg

    def __getitem__(self, index):
        idx_label = index / (self.endIdx - self.startIdx)
        label = self.labels[int(idx_label)]
        idx_img = index % (self.endIdx - self.startIdx)
        img0 = self.tensors[label, idx_img + self.startIdx, :, :, :]
        img1 = self.rotate90(img0)
        img2 = self.rotate90(img1)
        img3 = self.rotate90(img2)
        img = torch.stack([img0, img1, img2, img3], 0)
        label_rot = self.rot_labels
        return img, label, label_rot

    def __len__(self):
        return self.classNum * (self.endIdx - self.startIdx)
