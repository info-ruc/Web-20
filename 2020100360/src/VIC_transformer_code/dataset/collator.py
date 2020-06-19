#!/usr/bin/env python
#coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

"""Collator for NeuralClassifier"""

import torch

from dataset.classification_dataset import ClassificationDataset as cDataset


class Collator(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        raise NotImplementedError


class ClassificationCollator(Collator):
    def __init__(self, conf, label_size):
        super(ClassificationCollator, self).__init__(conf.device)
        self.classification_type = conf.task_info.label_type
        min_seq = 1
        self.min_token_max_len = min_seq
        self.label_size = label_size

    def _get_multi_hot_label(self, doc_labels):
        """For multi-label classification
        Generate multi-hot for input labels
        e.g. input: [[0,1], [2]]
             output: [[1,1,0], [0,0,1]]
        """
        batch_size = len(doc_labels)
        max_label_num = max([len(x) for x in doc_labels])
        doc_labels_extend = \
            [[doc_labels[i][0] for x in range(max_label_num)] for i in range(batch_size)]
        for i in range(0, batch_size):
            doc_labels_extend[i][0 : len(doc_labels[i])] = doc_labels[i]
        y = torch.Tensor(doc_labels_extend).long()
        y_onehot = torch.zeros(batch_size, self.label_size).scatter_(1, y, 1)
        return y_onehot

    def _append_label(self, doc_labels, sample):
        doc_labels.append(sample[cDataset.DOC_LABEL])


    def __call__(self, batch):
        def _append_vocab(ori_vocabs, vocabs, max_len):
            padding = [cDataset.VOCAB_PADDING] * (max_len - len(ori_vocabs))
            vocabs.append(ori_vocabs + padding)

        doc_labels = []
        doc_token = []
        doc_token_len = []
        doc_token_max_len = self.min_token_max_len

        for _, value in enumerate(batch):
            doc_token_max_len = max(doc_token_max_len,
                                    len(value[cDataset.DOC_TOKEN]))

        for _, value in enumerate(batch):
            self._append_label(doc_labels, value)
            _append_vocab(value[cDataset.DOC_TOKEN], doc_token,
                          doc_token_max_len)
            doc_token_len.append(len(value[cDataset.DOC_TOKEN]))

        tensor_doc_labels = self._get_multi_hot_label(doc_labels)

        if self.classification_type == "single_label":
            tensor_doc_labels = torch.tensor(doc_labels)
            doc_label_list = [[x] for x in doc_labels]
        elif self.classification_type == "multi_label":
            tensor_doc_labels = self._get_multi_hot_label(doc_labels)
            doc_label_list = doc_labels

        batch_map = {
            cDataset.DOC_LABEL: tensor_doc_labels,
            cDataset.DOC_LABEL_LIST: doc_label_list,
            cDataset.DOC_TOKEN: torch.tensor(doc_token),
            cDataset.DOC_TOKEN_MASK: torch.tensor(doc_token).gt(0).float(),
            cDataset.DOC_TOKEN_LEN: torch.tensor(
                doc_token_len, dtype=torch.float32),
            cDataset.DOC_TOKEN_MAX_LEN:
                torch.tensor([doc_token_max_len], dtype=torch.float32)
        }
        return batch_map

