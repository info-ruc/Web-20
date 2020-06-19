#!/usr/bin/env python
# coding:utf-8
import torch
import torch.nn as nn


class ClassificationLoss(torch.nn.Module):
    def __init__(self, label_size, class_weight=None,
                 loss_type="BCEWithLogitsLoss"):
        super(ClassificationLoss, self).__init__()
        self.label_size = label_size
        self.loss_type = loss_type
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, target,
                use_hierar=False,
                is_multi=False,
                *argvs):
        device = logits.device
        if use_hierar:
            if not is_multi:
                target = torch.eye(self.label_size)[target].to(device)
            hierar_penalty, hierar_paras, hierar_relations = argvs[0:3]
            # BCE loss + recursive正则项 来做正则化
            return self.criterion(logits, target) + \
                   hierar_penalty * self.cal_recursive_regularize(hierar_paras,
                                                                  hierar_relations,
                                                                  device)


    def cal_recursive_regularize(self, paras, hierar_relations, device="cpu"):
        """ Only support hierarchical text classification with BCELoss
        references: http://www.cse.ust.hk/~yqsong/papers/2018-WWW-Text-GraphCNN.pdf
                    http://www.cs.cmu.edu/~sgopal1/papers/KDD13.pdf
        """
        recursive_loss = 0.0
        for i in range(len(paras)):
            if i not in hierar_relations:
                continue
            children_ids = hierar_relations[i]
            if not children_ids:
                continue
            children_ids_list = torch.tensor(children_ids, dtype=torch.long).to(
                device)
            # 取node的参数
            children_paras = torch.index_select(paras, 0, children_ids_list)
            parent_para = torch.index_select(paras, 0,
                                             torch.tensor(i).to(device))
            parent_para = parent_para.repeat(children_ids_list.size()[0], 1)
            # 计算所有子节点参数到父节点参数的欧几里得距离之和
            # 这样优化可以使得鼓励层次结构中邻近的类共享类似的模型参数，既可以降低模型复杂度，一定程度上防止过拟合
            # 同时也可以将类标签之间的层次依赖关系合并到参数的正则化结构中。
            diff_paras = parent_para - children_paras
            diff_paras = diff_paras.view(diff_paras.size()[0], -1)
            recursive_loss += 1.0 / 2 * torch.norm(diff_paras, p=2) ** 2
        return recursive_loss
