import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num
        chanel_num = 1
        filter_num = 100
        filter_sizes = [3,4,5] # 通过设计不同 kernel_size 的 filter 获取不同宽度的视野。

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.embedding = self.embedding.from_pretrained(args.vectors, freeze=True)
        # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(0.5) # 防止模型过拟合
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1) # 在第二维增加一个维度
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # 同时进行卷积、relu激活，不是连续
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x] # 同时最大池化,torch.nn.functional.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
        x = torch.cat(x, 1) # 维度1，横向
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

