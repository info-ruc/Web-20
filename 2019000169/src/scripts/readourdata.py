from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
from tqdm import tqdm
import h5py
import numpy as np
import copy

import os
import warnings
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


dataset='train'
class OurDataset(Dataset):
    def __init__(self, npoints=2048, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.split = split
        f = h5py.File(split+'.h5','r')
        self.complete_data= np.array(f['complete_data'])
        self.incomplete_data=np.array(f['incomplete_data'])
        self.cls_label=np.array(f['cls_label'])
        self.seg_label=np.array(f['seg_label'])
        f.close()  
   


    def __getitem__(self,idx):
        complete=self.complete_data[idx]
        incomplete=self.incomplete_data[idx]
        cls=self.cls_label[idx]
        seg=self.seg_label[idx]
        
 

        return  complete,incomplete, cls, seg
       
    def __len__(self):
   
        return len(self.cls_label)


'''TRAIN_DATASET=OurDataset(npoints=2048, split=dataset, normal_channel=False)
trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=10,shuffle=False, num_workers=4) 
for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        complete_data,incomplete_data, cls_label, seg_label=data
        print(complete_data.data.numpy().shape)
'''


