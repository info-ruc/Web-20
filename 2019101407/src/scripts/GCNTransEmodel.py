import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))# 卷积核 W
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 参数初始化
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        # 用于显示(例如建立model后，print(model)得到如下内容)
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNTransE(nn.Module):
    def __init__(self,entNum,relNum,relFeatureDim,entFeatureDim,hiddenDim,outputDim,useGPU = False,GPUnum = 0):
        super(GCNTransE, self).__init__()
        self.ent_embeddinglayer = nn.Embedding(entNum,entFeatureDim)
        self.rel_embeddinglayer = nn.Embedding(relNum,relFeatureDim)
        nn.init.xavier_normal_(self.ent_embeddinglayer.weight)
        nn.init.xavier_normal_(self.rel_embeddinglayer.weight)
        self.gclay1 = GraphConvolution(entFeatureDim,hiddenDim)
        self.gclay2 = GraphConvolution(hiddenDim,outputDim)
        self.entidxs = torch.LongTensor(list(range(entNum)))
        self.relidxs = torch.LongTensor(list(range(relNum)))
        if useGPU:
            self.entidxs = self.entidxs.cuda(GPUnum)
            self.relidxs = self.relidxs.cuda(GPUnum)

    def forward(self,adj):
        x = self.ent_embeddinglayer(self.entidxs)
        x = F.relu(self.gclay1(x, adj))
        x = self.gclay2(x, adj)
        ent_emb = F.normalize(x,p=2,dim=-1)
        rel_emb = F.normalize(self.rel_embeddinglayer(self.relidxs),p=2,dim=-1)
        return ent_emb,rel_emb