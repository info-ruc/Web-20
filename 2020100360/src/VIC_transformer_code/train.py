#!/usr/bin/env python
#coding:utf-8
import os
import shutil
import sys
import time
import numpy

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset.classification_dataset import ClassificationDataset
from dataset.collator import ClassificationCollator
import eval_util as myeval
from model.transformer import Transformer
from model.loss import ClassificationLoss
from model.model_util import get_optimizer, get_hierar_relations

# 全局声明
ClassificationDataset, ClassificationCollator, ClassificationLoss, Transformer

def get_data_loader(dataset_name, collate_name, conf):
    """Get data loader: Train, Validate, Test
    """
    train_dataset = globals()[dataset_name](
        conf, conf.data.train_json_files, generate_dict=True)
    collate_fn = globals()[collate_name](conf, len(train_dataset.label_map))

    train_data_loader = DataLoader(
        train_dataset, batch_size=conf.train.batch_size, shuffle=True,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    validate_dataset = globals()[dataset_name](
        conf, conf.data.validate_json_files)
    validate_data_loader = DataLoader(
        validate_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    return train_data_loader, validate_data_loader, test_data_loader

class ClassificationTrainer(object):
    def __init__(self, label_map, conf, loss_fn):
        self.label_map = label_map
        self.conf = conf
        self.loss_fn = loss_fn
        if self.conf.task_info.hierarchical:
            self.hierar_relations = get_hierar_relations(
                    self.conf.task_info.hierar_taxonomy, label_map)

    def train(self, data_loader, model, optimizer, stage, epoch):
        model.update_lr(optimizer, epoch)
        model.train()
        return self.run(data_loader, model, optimizer, stage, epoch, "train")

    def eval(self, data_loader, model, optimizer, stage, epoch):
        model.eval()
        return self.run(data_loader, model, optimizer, stage, epoch)

    def run(self, data_loader, model, optimizer, stage,
            epoch, mode="eval"):
        is_multi = False
        # multi-label classifcation
        if self.conf.task_info.label_type == "multi_label":
            is_multi = True
        predict_probs = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        for batch in data_loader:
            logits = model(batch)
            # hierarchical classification
            if self.conf.task_info.hierarchical:
                linear_paras = model.linear.weight
                is_hierar = True
                used_argvs = (self.conf.task_info.hierar_penalty, linear_paras, self.hierar_relations)
                loss = self.loss_fn(
                    logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    is_hierar,
                    is_multi,
                    *used_argvs)
            else:  # flat classification
                loss = self.loss_fn(
                    logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                continue
            total_loss += loss.item()
            if not is_multi:
                result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
            else:
                result = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(result)
            standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])
        if mode == "eval":
            total_loss = total_loss / num_batch
            new_labels = []
            for i in range(len(standard_labels)):
                temp = []
                for j in range(252):
                    if j in standard_labels[i]:
                        temp.append(1)
                    else:
                        temp.append(0)
                new_labels.append(temp)
            evl_metrics = myeval.EvaluationMetrics(252, 20)
            evl_metrics.clear()
            evl_results = evl_metrics.accumulate(predict_probs, new_labels)
            evl_results = evl_metrics.get()
            hit_at_one = evl_results["avg_hit_at_one"]
            perr = evl_results["avg_perr"]
            gap = evl_results["gap"]
            m_ap = numpy.mean(evl_results["aps"])
            print(("{0:s} performance at epoch {1:d} is Avg_Hit@1: {2:.3f} | Avg_PERR: {3:.3f} | MAP: {4:.3f} | GAP: {5:.3f}").format(stage, epoch, hit_at_one, perr, m_ap, gap))
            # 金标准
            return gap


def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    best_performance = checkpoint["best_performance"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return best_performance


def save_checkpoint(state, file_prefix):
    file_name = file_prefix + "_" + str(state["epoch"])
    torch.save(state, file_name)


def train(conf):
    if not os.path.exists(conf.checkpoint_dir):
        os.makedirs(conf.checkpoint_dir)
    # load data
    dataset_name = "ClassificationDataset"
    collate_name = "ClassificationCollator"
    train_data_loader, validate_data_loader, test_data_loader = \
        get_data_loader(dataset_name, collate_name, conf)
    empty_dataset = globals()[dataset_name](conf, [], mode="train")
    # build model and set cuda or cpu
    model = Transformer(empty_dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    # training setting
    loss_fn = ClassificationLoss(label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)
    optimizer = get_optimizer(conf, model)
    trainer = ClassificationTrainer(empty_dataset.label_map, conf, loss_fn)

    best_epoch = -1
    best_performance = 0
    model_file_prefix = conf.checkpoint_dir + "/Transformer"
    for epoch in range(conf.train.start_epoch,
                       conf.train.start_epoch + conf.train.num_epochs):
        start_time = time.time()
        trainer.train(train_data_loader, model, optimizer, "Train", epoch)
        trainer.eval(train_data_loader, model, optimizer, "Train", epoch)
        performance = trainer.eval(
            validate_data_loader, model, optimizer, "Validate", epoch)
        trainer.eval(test_data_loader, model, optimizer, "test", epoch)
        if performance > best_performance:  # record the best model
            best_epoch = epoch
            best_performance = performance
        save_checkpoint({
            'epoch': epoch,
            'model_name': "Transformer",
            'state_dict': model.state_dict(),
            'best_performance': best_performance,
            'optimizer': optimizer.state_dict(),
        }, model_file_prefix)
        time_used = time.time() - start_time
        print("Epoch %d cost time: %d second" % (epoch, time_used))

    # best model on validateion set
    best_epoch_file_name = model_file_prefix + "_" + str(best_epoch)
    best_file_name = model_file_prefix + "_best"
    shutil.copyfile(best_epoch_file_name, best_file_name)

    load_checkpoint(model_file_prefix + "_" + str(best_epoch), conf, model,
                    optimizer)
    trainer.eval(test_data_loader, model, optimizer, "Best test", best_epoch)


if __name__ == '__main__':
    config = Config(config_file=sys.argv[1]) #config其实就是一个字典，要用的时候再去取东西
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
    torch.manual_seed(2020) #随机初始化种子，用于神经网络的初始化，用于cpu
    torch.cuda.manual_seed(2020) #随机初始化种子，用于神经网络的初始化，用于gpu
    train(config)
