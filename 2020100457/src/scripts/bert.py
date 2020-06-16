import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertModel
from transformers import AutoTokenizer


class Config(object):
    def __init__(self):
        self.model_name = 'bert'        # 模型名称
        self.train_path = '../dataset/train.tsv'    # 训练集路径
        self.dev_path = '../dataset/dev.tsv'    # 训练集路径
        self.test_path = '../dataset/test.tsv'      # 测试集路径
        self.model_path = '../../extra/bert_snapshot/'
        self.label_list = [0,1]  # 标签列表
        self.num_classes = len(self.label_list) # 类别数量
        self.device = torch.device('cuda')  # 是否使用GPU训练
        if self.device:
            torch.cuda.set_device(3)    # 使用的GPU序号
        self.num_epoches = 5    # 训练集迭代次数
        self.batch_size = 64    # 批训练数据大小
        self.lr = 5e-5          # 学习率
        self.bert_path = "../../extra/pretrained/bert-base-chinese"   # 预训练模型路径
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)  # 预训练模型对应的文本切分器
        self.hidden_size = 768  # 隐藏层大小

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, **inputs):
        _, pooled = self.bert(**inputs)
        out = self.fc(pooled)
        return out

