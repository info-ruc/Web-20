# -*- coding: utf-8 -*-
# @Time    : 2020/4/7 11:09
# @Author  : guardianV
# @Email   : Tang_dq35@163.com
# @File    : train.py
# @Software: PyCharm
import json
import os
import time

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from pytorch_pretrained_bert import BertAdam
from torch.optim import Adam, SGD
import re
import matplotlib.pyplot as plt
from bert_model import Config, ListDataset, SingleModel, PairModel


def labels2onehots(num,labels):
    onehots = []
    for label in labels:
        onehot = [0 for i in range(num)]
        onehot[label] = 1
        onehots.append(onehot)
    return onehots


def single_train(config):
    start_time = time.asctime(time.localtime(time.time()))
    print(start_time)
    x_trian = []
    y_train = []
    head_train = []
    relations = ["unknown","Create","Use","Near","Social","Located","Ownership","General-Special","Family","Part-Whole"]
    with open(config.train_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # label = config.label_list.index(line.split(" ")[0].strip("__label__"))
            # sentence = "".join(line.lower().strip("\n").split(" ")[1:])
            _,head,tail,label,sentence =line.split("\t")
            if(sentence):
                head_pos = (re.search('\[E11\]', sentence).span()[1], re.search('\[E12\]', sentence).span()[0])
                # tail_pos = (re.search('\[E21\]', line).span()[1], re.search('\[E22\]', line).span()[0])
                sentence = sentence.replace('[E21]', '')
                sentence = sentence.replace('[E22]', '')
                x_trian.append(sentence)
                y_train.append(relations.index(label))
                head_train.append(head_pos)
    print('handle data over.')
    torch_dataset = ListDataset(x=x_trian, y=y_train, head=head_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=config.batch_size,  # 批大小
        shuffle=True,  # 是否随机打乱顺序
        num_workers=4,  # 多线程读取数据的线程数
    )
    # model = Model(config).to(config.device)
    model = SingleModel(config).to(config.device)
    optimizer = BertAdam(model.parameters(),
                         lr=config.lr,
                         warmup=0.05,
                         t_total=len(torch_dataset) * config.num_epoches)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_li = []
    print_loss = 0
    for epoch in range(config.num_epoches):
        model.train()
        for step, (batch_texts, batch_span, batch_head_pos) in enumerate(loader):
            max_len = max([len(i) for i in batch_texts])
            x = config.tokenizer.batch_encode_plus(batch_texts, add_special_tokens=True,
                                                   return_tensors="pt", max_length=max_len, pad_to_max_length=True)
            x["input_ids"] = x["input_ids"].to(config.device)
            x["attention_mask"] = torch.abs(torch.ones(x["token_type_ids"].size(), dtype=torch.long).to(config.device)-x["token_type_ids"].to(config.device))
            x["token_type_ids"] = x["token_type_ids"].to(config.device)
            out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], token_type_ids=x["token_type_ids"], batch_head_pos=batch_head_pos)
            # print(loss)
            # print(torch.argmax(start[0]), torch.argmax(end[0]))
            optimizer.zero_grad()
            loss = loss_func(out, batch_span.to(config.device)).to(config.device)
            print_loss += loss
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                print("epoch:", epoch, "step:", step, "loss", print_loss/10)
                # print(config.tokenizer.decode(x["input_ids"][1]))
                # print(x["input_ids"][1])
                # print(x["attention_mask"][1])
                # print(x["token_type_ids"][1])
                # print(batch_question_doc[0][0])
                # print(torch.argmax(start[0]), torch.argmax(end[0]))
                # print(config.tokenizer.decode(x["input_ids"][0][torch.argmax(start[0]):torch.argmax(end[0])]))
                # print('real', batch_span[0][0], batch_span[1][0])
                # print(config.tokenizer.decode(x["input_ids"][0][batch_span[0][0]: batch_span[1][0]]))
                loss_li.append(print_loss / 50)
                print_loss = 0
    model_path = '/usr/tdq/models/re/aliBert-Sanwen-10'
    torch.save(model, model_path)
    end_time = time.asctime(time.localtime(time.time()))
    print("start time:{}, end time:{}".format(start_time, end_time))
    return model_path


def single_test_one(config, model, text):

    x = config.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt")
    x["input_ids"] = x["input_ids"].to(config.device)
    x["attention_mask"] = x["attention_mask"].to(config.device)
    x["token_type_ids"] = x["token_type_ids"].to(config.device)
    x["batch_head_pos"] = [[re.search('\[E11\]', text).span()[1]], [re.search('\[E12\]', text).span()[0]]]
    text_tokens = config.tokenizer.decode(x["input_ids"][0]).split(" ")
    model.eval()
    out = model(**x)
    label = torch.argmax(out)
    return label


def single_test(config, model_path):
    model = torch.load(model_path).to(config.device)
    model.eval()
    x_test = []
    y_test = []
    T = 0
    F = 0
    with open(config.test_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            _, head, tail, label, sentence = line.split("\t")
            sentence = sentence.replace('[E11]', '').replace('[E12]', '').replace('[E21]', '').replace('[E22]', '')
            x_test.append(sentence)
            y_test.append(label)
    pred_mat = torch.zeros((config.num_labels, config.num_labels), dtype=torch.int)
    print(pred_mat.size())
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        label = single_test_one(config, model, x)
        pred_mat[label][config.label_list.index(y)] += 1
        if label == config.label_list.index(y):
            T += 1
        else:
            F += 1
        print("\r{} of {}".format(i, len(x_test)), end="")
    print(pred_mat)
    print("true:{}, false:{}".format(T, F))


def pair_train(config):
    start_time = time.asctime(time.localtime(time.time()))
    print(start_time)
    x_trian = []
    y_train = []
    head_train = []
    relations = ["unknown","Create","Use","Near","Social","Located","Ownership","General-Special","Family","Part-Whole"]
    with open(config.train_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # label = config.label_list.index(line.split(" ")[0].strip("__label__"))
            # sentence = "".join(line.lower().strip("\n").split(" ")[1:])
            _, head, tail, label, sentence = line.split("\t")
            sentence = sentence.strip("\n")
            if(sentence):
                # tail_pos = (re.search('\[E21\]', line).span()[1], re.search('\[E22\]', line).span()[0])
                sentence = sentence.replace('[E11]', '').replace('[E12]', '').replace('[E21]', '').replace('[E22]', '')
                x_trian.append(sentence)
                y_train.append(relations.index(label))
                head_train.append(head)
    print('handle data over.')
    torch_dataset = ListDataset(x=x_trian, y=y_train, head=head_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=config.batch_size,  # 批大小
        shuffle=True,  # 是否随机打乱顺序
        num_workers=4,  # 多线程读取数据的线程数
    )
    # model = Model(config).to(config.device)
    model = PairModel(config).to(config.device)
    optimizer = BertAdam(model.parameters(),
                         lr=config.lr,
                         warmup=0.05,
                         t_total=len(torch_dataset) * config.num_epoches)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_li = []
    print_loss = 0
    for epoch in range(config.num_epoches):
        model.train()
        for step, (batch_texts, batch_span, batch_head_pos) in enumerate(loader):
            max_len = max([len(i) for i in batch_texts])
            x = config.tokenizer.batch_encode_plus(zip(batch_texts, batch_head_pos), add_special_tokens=True,
                                                   return_tensors="pt", max_length=max_len, pad_to_max_length=True)
            x["input_ids"] = x["input_ids"].to(config.device)
            # print(config.tokenizer.decode(x["input_ids"][0]))
            x["attention_mask"] = torch.abs(torch.ones(x["token_type_ids"].size(), dtype=torch.long).to(config.device)-x["token_type_ids"].to(config.device))
            x["token_type_ids"] = x["token_type_ids"].to(config.device)
            out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], token_type_ids=x["token_type_ids"])
            # print(loss)
            # print(torch.argmax(start[0]), torch.argmax(end[0]))
            optimizer.zero_grad()
            loss = loss_func(out, batch_span.to(config.device)).to(config.device)
            print_loss += loss
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                print("epoch:", epoch, "step:", step, "loss", print_loss/10)
                # print(config.tokenizer.decode(x["input_ids"][1]))
                # print(x["input_ids"][1])
                # print(x["attention_mask"][1])
                # print(x["token_type_ids"][1])
                # print(batch_question_doc[0][0])
                # print(torch.argmax(start[0]), torch.argmax(end[0]))
                # print(config.tokenizer.decode(x["input_ids"][0][torch.argmax(start[0]):torch.argmax(end[0])]))
                # print('real', batch_span[0][0], batch_span[1][0])
                # print(config.tokenizer.decode(x["input_ids"][0][batch_span[0][0]: batch_span[1][0]]))
                loss_li.append(print_loss / 50)
                print_loss = 0
    model_path = '/usr/tdq/models/re/aliBert-Sanwen-10-pair'
    torch.save(model, model_path)
    end_time = time.asctime(time.localtime(time.time()))
    print("start time:{}, end time:{}".format(start_time, end_time))
    return model_path


def pair_test_one(config, model, text, head):

    x = config.tokenizer.encode_plus(text, head, add_special_tokens=True, return_tensors="pt")
    x["input_ids"] = x["input_ids"].to(config.device)
    # print(config.tokenizer.decode(x["input_ids"][0]))
    x["attention_mask"] = x["attention_mask"].to(config.device)
    x["token_type_ids"] = x["token_type_ids"].to(config.device)
    text_tokens = config.tokenizer.decode(x["input_ids"][0]).split(" ")
    model.eval()
    out = model(**x)
    label = torch.argmax(out)
    return label


def pair_test(config, model_path):
    model = torch.load(model_path).to(config.device)
    model.eval()
    x_test = []
    y_test = []
    heads = []
    T = 0
    F = 0
    with open(config.test_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            _, head, tail, label, sentence = line.split("\t")
            sentence = sentence.replace('[E11]', '').replace('[E12]', '').replace('[E21]', '').replace('[E22]', '')
            x_test.append(sentence)
            y_test.append(label)
            heads.append(head)
    pred_mat = torch.zeros((config.num_labels, config.num_labels), dtype=torch.int)
    print(pred_mat.size())
    for i, (x, y, head) in enumerate(zip(x_test, y_test, heads)):
        label = pair_test_one(config, model, x, head)
        pred_mat[label][config.label_list.index(y)] += 1
        if label == config.label_list.index(y):
            T += 1
        else:
            F += 1
        print("\r{} of {}".format(i, len(x_test)), end="")
    print(pred_mat)
    print("true:{}, false:{}".format(T, F))


if __name__ == '__main__':
    config = Config()
    relations = ["unknown", "Create", "Use", "Near", "Social", "Located", "Ownership", "General-Special", "Family",
                 "Part-Whole"]
    model_path = '/usr/tdq/models/re/aliBert-Sanwen-10-pair'
    # single_train(config)
    # pair_train(config)
    # label = single_test_one(config, torch.load(model_path).to(config.device), '[E11]帕米尔高原[E12]再西边就是哈萨克斯坦的地界了')
    # label = pair_test_one(config, torch.load(model_path).to(config.device), '帕米尔高原再西边就是哈萨克斯坦的地界了', '帕米尔高原')
    # print(relations[label])
    # single_test(config, model_path)
    pair_test(config, model_path)
