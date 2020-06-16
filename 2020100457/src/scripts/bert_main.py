import sys
import os
import argparse

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertAdam

from data_util import ListDataset, load_dataset
from bert import Config, Model

parser = argparse.ArgumentParser(description='TextCNN text classifier')
parser.add_argument('-test', type=bool, default=False,
                    help='Train or Test, default False: Train')
parser.add_argument('-snapshot', type=str, default=None,
                    help='filepath of model snapshot [default: None]')
args = parser.parse_args()

def train(config):
    train_loader = load_dataset(config.train_path, config)
    dev_loader = load_dataset(config.dev_path, config)

    model = Model(config).to(config.device)
    optimizer = BertAdam(model.parameters(),
                         lr=config.lr,
                         warmup=0.05,
                         t_total=len(train_loader) * config.num_epoches)
    loss_func = torch.nn.CrossEntropyLoss()
    print_loss = 0
    best_acc = 0
    model.train()
    for epoch in range(config.num_epoches):
        for step, (batch_texts, batch_span) in enumerate(train_loader):
            max_len = max([len(i) for i in batch_texts])
            x = config.tokenizer.batch_encode_plus(batch_texts, add_special_tokens=True,
                                                   return_tensors="pt", max_length=max_len, pad_to_max_length=True)
            x["input_ids"] = x["input_ids"].to(config.device)
            x["attention_mask"] = x["attention_mask"].to(config.device)
            x["token_type_ids"] = x["token_type_ids"].to(config.device)
            batch_span = batch_span.to(config.device)

            out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], token_type_ids=x["token_type_ids"])
            optimizer.zero_grad()
            loss = loss_func(out, batch_span)
            loss.backward()
            optimizer.step()

            if step % 1  == 0:
                corrects = (torch.max(out, 1)[1].view(batch_span.size()).data == batch_span.data).sum()
                train_acc = 100.0 * corrects / config.batch_size
                # print("epoch:", epoch, "step:", step, "loss:", print_loss.item() / 50)
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(step,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             config.batch_size))
            if step % 50 == 0:
                dev_acc = eval(dev_loader, model, config)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    print(
                        'Saving best model, acc: {:.4f}%\n'.format(best_acc))
                    save(model, config.model_path, 'best', step)

def eval(loader, model, config):
    length = 7102
    model.eval()
    corrects, avg_loss = 0, 0
    loss_func = torch.nn.CrossEntropyLoss()
    for step, (batch_texts, batch_span) in enumerate(loader):
        max_len = max([len(i) for i in batch_texts])
        x = config.tokenizer.batch_encode_plus(batch_texts, add_special_tokens=True,
                                                return_tensors="pt", max_length=max_len, pad_to_max_length=True)
        x["input_ids"] = x["input_ids"].to(config.device)
        x["attention_mask"] = x["attention_mask"].to(config.device)
        x["token_type_ids"] = x["token_type_ids"].to(config.device)
        batch_span = batch_span.to(config.device)

        out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], token_type_ids=x["token_type_ids"])
        loss = loss_func(out, batch_span)
        avg_loss += loss.item()
        corrects += (torch.max(out, 1)[1].view(batch_span.size()).data == batch_span.data).sum()
    avg_loss /= length
    accuracy = 100.0 * corrects / length
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           length))
    return accuracy

def test(config):
    model = Model(config).to(config.device)
    if args.snapshot:
        print('\nLoading model from {}...\n'.format(args.snapshot))
        model.load_state_dict(torch.load(args.snapshot))
    dev_loader = load_dataset(config.dev_path, config)
    eval(dev_loader, model, config)


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)



               

if __name__ == '__main__':
    config = Config()
    if test:
        test(config)
    else:
        train(config)
   

            
