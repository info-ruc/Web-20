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


from dataset.dataset import DatasetBase


class ClassificationDataset(DatasetBase):
    DOC_LABEL = "doc_label"
    DOC_LABEL_LIST = "doc_label_list"
    DOC_TOKEN = "doc_token"
    DOC_TOKEN_LEN = "doc_token_len"
    DOC_TOKEN_MASK = "doc_token_mask"
    DOC_TOKEN_MAX_LEN = "doc_token_max_len"

    def __init__(self, config, json_files, generate_dict=False,
                 mode="eval"):
        super(ClassificationDataset, self).__init__(
            config, json_files, generate_dict=generate_dict, mode=mode)

    def _init_dict(self):
        self.dict_names = \
            [self.DOC_LABEL, self.DOC_TOKEN]

        self.dict_files = []
        for dict_name in self.dict_names:
            self.dict_files.append(
                self.config.data.dict_dir + "/" + dict_name + ".dict")
        self.label_dict_file = self.dict_files[0]

        # By default keep all labels
        self.min_count = [0, self.config.feature.min_token_count]

        # By default keep all labels
        self.max_dict_size = [self.BIG_VALUE, self.config.feature.max_token_dict_size]

        self.max_sequence_length = [self.config.feature.max_token_len]

        # When generating dict, the following map store vocab count.
        # Then clear dict and load vocab of word index
        self.label_map = dict()
        self.token_map = dict()
        self.dicts = [self.label_map, self.token_map]

        # Save sorted dict according to the count
        self.label_count_list = []
        self.token_count_list = []
        self.count_list = [self.label_count_list, self.token_count_list]

        self.id_to_label_map = dict()
        self.id_to_token_map = dict()
        self.id_to_vocab_dict_list = [self.id_to_label_map, self.id_to_token_map]

    def _insert_vocab(self, json_obj, mode="all"):
        """Insert vocab to dict
        """
        if mode == "all" or mode == "label":
            doc_labels = json_obj[self.DOC_LABEL]
            self._insert_sequence_vocab(doc_labels, self.label_map)
        if mode == "all" or mode == "other":
            doc_tokens = json_obj[self.DOC_TOKEN][0:self.config.feature.max_token_len]
            self._insert_sequence_tokens(
                doc_tokens, self.token_map)

    def _get_vocab_id_list(self, json_obj):
        """Use dict to convert all vocabs to ids
        """
        doc_labels = json_obj[self.DOC_LABEL]
        doc_tokens = \
            json_obj[self.DOC_TOKEN][0:self.config.feature.max_token_len]

        token_ids = \
            self._token_to_id(doc_tokens, self.token_map)
        return {self.DOC_LABEL: self._label_to_id(doc_labels, self.label_map) if self.model_mode != "infer" else [0],
                self.DOC_TOKEN: token_ids}
