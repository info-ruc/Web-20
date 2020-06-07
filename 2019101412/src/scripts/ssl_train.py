import os
import argparse

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from network import ssl_ConvNet
from dataloader import ssl_MyDataset

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)    
    args = parser.parse_args()
    
    trainset = ssl_MyDataset('train')
    train_loader = DataLoader(dataset=trainset, num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    valset = ssl_MyDataset('val')
    val_loader = DataLoader(dataset=valset, num_workers=4, batch_size=args.batch_size, pin_memory=True)

    testset = ssl_MyDataset('test')
    test_loader = DataLoader(dataset=testset, num_workers=4, batch_size=args.batch_size, pin_memory=True)    
    
    model = ssl_ConvNet()   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)        
    model = model.cuda()

    best_acc = 0.0
    for epoch in range(args.max_epoch):
        lr_scheduler.step()
        model.train()
        for i, batch in enumerate(train_loader):
            imgs, labels, rot_labels = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            optimizer.zero_grad()
            logits, rot_logits = model(imgs)
            loss1 = F.cross_entropy(logits, labels)
            loss2 = F.cross_entropy(rot_logits, rot_labels.view(-1))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            acc = count_acc(logits, labels)
            print('epoch {}, train {}/{}, loss={:.4f}, acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

        model.eval()
        tmp_acc = 0.0
        tmp_num = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                imgs, labels = batch[0].cuda(), batch[1].cuda()
                logits, _ = model(imgs)
                acc = count_acc(logits, labels)
                tmp_acc += acc
                tmp_num += 1.0
                print('epoch {}, val {}/{}, acc={:.4f}'
                      .format(epoch, i, len(val_loader), acc))    
            tmp_acc /= tmp_num
            print('val acc = {}'.format(tmp_acc))
            if tmp_acc > best_acc:
                print('saving model...')
                torch.save(model.state_dict(), "best.pth")           
    

    model.load_state_dict(torch.load("best.pth"))
    model.eval()
    tmp_acc = 0.0
    tmp_num = 0.0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            imgs, labels = batch[0].cuda(), batch[1].cuda()
            logits, _ = model(imgs)
            acc = count_acc(logits, labels)
            tmp_acc += acc
            tmp_num += 1.0
            print('epoch {}, test {}/{}, acc={:.4f}'
                  .format(epoch, i, len(test_loader), acc))    
        tmp_acc /= tmp_num
        print('test acc = {}'.format(tmp_acc))
