# -*- encoding:utf-8 -*-
"""
  This script provides BERT exmaple for classification.
"""
import os
import random
import sys
import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
#from tqdm import notebook, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
import torch.nn as nn
import argparse
from multiprocessing import Process, Pool
import matplotlib.pyplot as plt


class DataPrecessForSingleSentence(object):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, max_workers=10):
        self.bert_tokenizer = bert_tokenizer
        # 创建多线程池
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, sentences, max_seq_len=30):
        """
        通过多线程（因为notebook中多进程使用存在一些问题）的方式对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        
        入参:
            sentences   : 传入的text。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。  
            
        """
        # 切词
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 获取定长序列及其mask
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments

    def trunate_and_pad(self, seq, max_seq_len):
        """
        1. 因为本类处理的是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        
        入参: 
            seq         : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度
        
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
           
        """
        # 对超长序列进行截断
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + seq + ['[SEP]']
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment

'''
class BertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels=2): # Change number of labels here.
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert_model')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size+10, num_labels)
        #self.apply(self.init_bert_weights)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        #out = torch.cat((pooled_output, metadata), 1)
        
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True'''


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--output_lossfig_path", default="./models/loss.png", type=str,
                        help="Path of the output model.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Device use.")

    args = parser.parse_args()

    def set_seed(seed=7):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    set_seed(args.seed)
    
    # 读取数据
    train = pd.read_csv('../data5k/train.tsv', encoding='utf-8', sep='\t')
    dev = pd.read_csv('../data5k/dev.tsv', encoding='utf-8', sep='\t')
    test = pd.read_csv('../data5k/test.tsv', encoding='utf-8', sep='\t')

    # Load bert vocabulary and tokenizer
    bert_config = BertConfig('bert_model/bert_config.json')
    BERT_MODEL_PATH = 'bert_model'
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=False)

    # 产生输入数据
    processor = DataPrecessForSingleSentence(bert_tokenizer=bert_tokenizer)
    
    # train dataset
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=train['text_a'].tolist(), max_seq_len=args.seq_length)
    labels = train['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloder = DataLoader(dataset=train_data, sampler=train_sampler, batch_size=args.batch_size)

    # dev dataset
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=dev['text_a'].tolist(), max_seq_len=args.seq_length)
    labels = dev['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    dev_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloder = DataLoader(dataset=dev_data, sampler=dev_sampler, batch_size=args.batch_size)

    # test dataset
    seqs, seq_masks, seq_segments = processor.get_input(
        sentences=test['text_a'].tolist(), max_seq_len=args.seq_length)
    labels = test['label'].tolist()
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)
    test_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloder = DataLoader(dataset=test_data, sampler=test_sampler, batch_size=args.batch_size)

    # build classification model
    model = BertForSequenceClassification(bert_config, 2)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    model = model.to(device)

    # evaluation function
    def evaluate(args, is_test, metrics='Acc'):
        if is_test:
            dataset = test_dataloder
            instances_num = test.shape[0]
            print("The number of evaluation instances: ", instances_num)
        else:
            dataset = dev_dataloder
            instances_num = dev.shape[0]
            print("The number of evaluation instances: ", instances_num)
        
        correct = 0
        model.eval()
        # Confusion matrix.
        confusion = torch.zeros(2, 2, dtype=torch.long)
        
        for i, batch_data in enumerate(dataset):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data        
            with torch.no_grad():
                logits = model(
                    batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            pred = logits.softmax(dim=1).argmax(dim=1)
            gold = batch_labels
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()
            
        if is_test:
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")

        for i in range(confusion.size()[0]):
            p = confusion[i,i].item()/confusion[i,:].sum().item()
            r = confusion[i,i].item()/confusion[:,i].sum().item()
            f1 = 2*p*r / (p+r)
            if i == 1:
                label_1_f1 = f1
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/instances_num, correct, instances_num))
        if metrics == 'Acc':
            return correct/instances_num
        elif metrics == 'f1':
            return label_1_f1
        else:
            return correct/instances_num

    # training phase
    print("Start training.")
    instances_num = train.shape[0]
    batch_size = args.batch_size
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)


    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            0.01
        },
        {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        warmup=args.warmup,
                        t_total=train_steps)

    # 存储每一个batch的loss
    all_loss = []
    all_acc = []
    total_loss = 0.0
    result = 0.0
    best_result = 0.0

    for epoch in range(1, args.epochs_num+1):
        model.train()
        for step, batch_data in enumerate(train_dataloder):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
            # 对标签进行onehot编码
            one_hot = torch.zeros(batch_labels.size(0), 2).long()
            '''one_hot_batch_labels = one_hot.scatter_(
                dim=1,
                index=torch.unsqueeze(batch_labels, dim=1),
                src=torch.ones(batch_labels.size(0), 2).long())

            
            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            logits = logits.softmax(dim=1)
            loss_function = CrossEntropyLoss()
            loss = loss_function(logits, batch_labels)'''
            loss = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels)
            loss.backward()
            total_loss += loss.item()
            if (step + 1) % 100 == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, step+1, total_loss / 100))
                sys.stdout.flush()
                total_loss = 0.
            #print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, step+1, loss))
            optimizer.step()
            optimizer.zero_grad()
        
        all_loss.append(total_loss)
        total_loss = 0.
        print("Start evaluation on dev dataset.")
        result = evaluate(args, False)
        all_acc.append(result)
        if result > best_result:
            best_result = result
            torch.save(model, open(args.output_model_path,"wb"))
            #save_model(model, args.output_model_path)
        else:
            continue

        print("Start evaluation on test dataset.")
        evaluate(args, True)
    
    print('all_loss:', all_loss)
    print('all_acc:', all_acc)

    # Evaluation phase.
    print("Final evaluation on the test dataset.")
    model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, True)

    '''
    print(loss_collect)
    plt.figure(figsize=(12,8))
    plt.plot(range(len(loss_collect)), loss_collect,'g.')
    plt.grid(True)
    plt.savefig(args.output_lossfig_path)'''

if __name__ == "__main__":
    main()