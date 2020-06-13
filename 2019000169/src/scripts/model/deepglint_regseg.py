import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet import STN3d, STNkd, feature_transform_reguliarzer,PointNetEncoder
import numpy as np
from chamferdist import ChamferDistance


class get_model(nn.Module):
    def __init__(self, part_num=2, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.encoder=PointNetEncoder(channel)
        self.convs1=torch.nn.Conv1d(2048,1024,1)
        self.convs2=torch.nn.Conv1d(1024,512,1)
        self.convs3=torch.nn.Conv1d(512,256,1)
        self.convs4=torch.nn.Conv1d(256,128,1)
        self.convs5=torch.nn.Conv1d(128,part_num,1)
    
        self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(512)
        self.bns3 = nn.BatchNorm1d(256)
        self.bns4 = nn.BatchNorm1d(128)
        
        self.regconvs1=torch.nn.Conv1d(2048,1024,1)
        self.regconvs2=torch.nn.Conv1d(1024,512,1)
        self.regconvs3=torch.nn.Conv1d(512,256,1)
        self.regconvs4=torch.nn.Conv1d(256,128,1)
        self.regconvs5=torch.nn.Conv1d(128,3,1)
    
        self.regbns1 = nn.BatchNorm1d(1024)
        self.regbns2 = nn.BatchNorm1d(512)
        self.regbns3 = nn.BatchNorm1d(256)
        self.regbns4 = nn.BatchNorm1d(128)
       

    def forward(self, point_cloud1, point_cloud2):
        B, D, N1 = point_cloud1.size()
        _, _, N2 = point_cloud2.size()
        
       
        
        out1,features1=self.encoder(point_cloud1)
        out2,features2=self.encoder(point_cloud2)
        out=out1-out2
        net=out.view(-1, 1024, 1).repeat(1, 1, N1)
        net=torch.cat([net, features1], 1)
        
       
        net = F.relu(self.bns1(self.convs1(net)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = F.relu(self.bns4(self.convs4(net)))
        net = self.convs5(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N1, self.part_num) # [B, N, 50]
        
        reg=out.view(-1, 1024, 1).repeat(1, 1, N2)
        reg=torch.cat([reg, features2], 1)
        reg = F.relu(self.regbns1(self.regconvs1(reg)))
        reg = F.relu(self.regbns2(self.regconvs2(reg)))
        reg = F.relu(self.regbns3(self.regconvs3(reg)))
        reg = F.relu(self.regbns4(self.regconvs4(reg)))
        reg = self.regconvs5(reg)
        reg = reg.transpose(2, 1).contiguous()
        reg = reg.view(B, N2, 3) # [B, N, 50]
        
        
    

        return net, reg


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, reg,point1,pred, target):
        loss = F.nll_loss(pred, target)
       # mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        chamferDist = ChamferDistance()
        dist1, dist2, idx1, idx2 = chamferDist(reg, point1)
        reg_loss=torch.mean(torch.mean(dist1,1)+torch.mean(dist2,1))
        total_loss = loss +reg_loss#+ mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
        

'''point1 = torch.randn([4,3,2048])
point2 = torch.randn([4,3,1848])
model = get_model()
net,reg= model(point1,point2)
pred=net.contiguous().view(-1, 2)
target=net.view(-1, 1)
loss=get_loss()
l=loss(reg,point1,pred,target)
print('net',net.shape)
print(reg.shape)
print(l.shape)'''


