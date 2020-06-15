# -*- coding: utf-8 -*-
# @Time    : 2020/3/21 15:18
# @Author  : guardianV
# @Email   : Tang_dq35@163.com
# @File    : model.py.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
from pytorch_pretrained_bert.modeling import BertModel, BertForSequenceClassification


class ListDataset(Dataset):
    def __init__(self, x, y, head):
        self.x = x
        self.y = y
        self.head = head
        # self.tail = tail

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.head[idx]


class Config(object):
    def __init__(self):
        self.model_name = 'bert'        # 模型名称
        self.train_path = '../dataset/preprocessed-train.txt'    # 训练集路径
        self.test_path = '../dataset/preprocessed-test.txt'      # 测试集路径
        self.label_list = ["unknown", "Create", "Use", "Near", "Social", "Located", "Ownership", "General-Special", "Family", "Part-Whole"]  # 标签列表
        self.num_labels = len(self.label_list)  # 类别数量
        self.device = torch.device('cuda')  # 是否使用GPU训练
        if self.device:
            torch.cuda.set_device(3)    # 使用的GPU序号
        self.num_epoches = 10   # 训练集迭代次数
        self.batch_size = 32    # 批训练数据大小
        self.lr = 5e-5          # 学习率
        self.bert_path = "./chinese-wwm-bert"   # 预训练模型路径
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)  # 预训练模型对应的文本切分器
        self.hidden_size = 768  # 隐藏层大小



class SingleModel(nn.Module):
    def __init__(self, config):
        super(SingleModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.cls_extractor = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_extractor = nn.Linear(config.hidden_size, config.hidden_size)
        # self.tail_extractor = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.config = config

    def forward(self, **inputs):
        batch_head_pos = inputs["batch_head_pos"]
        encoded, cls = self.bert(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], token_type_ids=inputs["token_type_ids"], output_all_encoded_layers=False)
        head_mean = torch.zeros(len(batch_head_pos[0]), self.config.hidden_size).to(self.config.device)
        for i in range(len(batch_head_pos[0])):
            head = encoded[i][batch_head_pos[0][i]-1:batch_head_pos[1][i]-1]
            head_mean[i] = torch.mean(head, 0, False)
            # print(self.config.tokenizer.decode(inputs["input_ids"][i]))
            # print(self.config.tokenizer.decode(inputs["input_ids"][i][batch_head_pos[0][i]:batch_head_pos[1][i]]))
        cls_out = self.cls_extractor(cls)
        head_out = self.head_extractor(head_mean)
        out = self.out(torch.cat((cls_out, head_out), 1))
        return out


class PairModel(nn.Module):
    def __init__(self, config):
        super(PairModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.out = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, **inputs):
        encoded, cls = self.bert(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], token_type_ids=inputs["token_type_ids"], output_all_encoded_layers=False)
        out = self.out(cls)
        return out


# class BertQA(nn.Module):
#     def __init__(self, config):
#         super(BertQA, self).__init__()
#         self.num_labels = 2
#         self.bert = BertModel.from_pretrained(config.bert_path)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.num_labels)
#         # self.apply(self.init_bert_weights)
#
#     def forward(self, **inputs):
#         sequence_output, pooled_output = self.bert(inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"], output_all_encoded_layers=False)
#         logits = self.classifier(sequence_output)  # (B, T, 2)
#         start_logits, end_logits = logits.split(1, dim=-1)# ((B, T, 1),(B, T, 1))
#         start_logits = start_logits.squeeze(-1) # (B, T)
#         end_logits = end_logits.squeeze(-1) # (B, T)
#         return start_logits, end_logits
