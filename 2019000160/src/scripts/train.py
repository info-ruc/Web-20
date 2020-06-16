import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import log, euclidean_metric, count_acc
from data_utils import CategoriesSampler, BirdsDataset


def get_dataloaders(args):
    train_set = BirdsDataset(root=args.data_root, mode='train')
    train_sampler = CategoriesSampler(train_set.label, args.num_iter, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
    val_set = BirdsDataset(root=args.data_root, mode='val')
    val_sampler = CategoriesSampler(val_set.label, args.val_episodes, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def get_networks(args):
    if args.cnn == 'WideResNet':
        import network.wideresnet as wideresnet
        base_net = wideresnet.WideResNet().cuda()
        pretrained_dict = torch.load(args.pretrained)['state_dict']
        model_dict = base_net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        base_net.load_state_dict(model_dict)
    else:
        print('No implementation!')
        exit()

    return base_net


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--cnn', default='WideResNet', type=str)
parser.add_argument('--max_epoch', default=50, type=int)
parser.add_argument('--num_iter', default=50, type=int)
parser.add_argument('--step_size', default=10, type=int)
parser.add_argument('--init_lr', default=0.001, type=float)
parser.add_argument('--data_root', default='../dataset', type=str)
parser.add_argument('--output_dir', default='./exp_5shot/proto_exp1', type=str)
parser.add_argument('--shot', default=5, type=int)
parser.add_argument('--query', default=15, type=int)
parser.add_argument('--way', default=5, type=int)
parser.add_argument('--val_episodes', default=300, type=int)
parser.add_argument('--lambda_align', default=0, type=float)
parser.add_argument('--T', default=16, type=float)
parser.add_argument('--pretrained', default='./pretrain/wrn28_10_places.pth.tar', type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
log_file_path = os.path.join(args.output_dir, 'train_log.txt')
log(log_file_path, str(vars(args)))

# get dataloaders
train_loader, val_loader = get_dataloaders(args)
# get networks
base_net = get_networks(args)
# set optimizer
param_groups = [{"params":base_net.parameters()}]
optimizer = torch.optim.SGD(param_groups, lr=args.init_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

# preparations
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')

n_epi = args.way * (args.shot + args.query) # 100
n_shot = args.way * args.shot # 25

label = torch.arange(args.way).repeat(args.query) # 75
label = label.type(torch.cuda.LongTensor)

# train
print('start training...')
len_train_loader = len(train_loader)
print('len_train_loader:', len_train_loader)
best_acc = 0.0
for epoch in range(1, args.max_epoch + 1):
    base_net.train()
    for i, batch in enumerate(train_loader, 1):
        inputs_ori, inputs_aug = batch[0], batch[1]
        # [100, 3, 80, 80], [100, 3, 80, 80]
        inputs_aug = inputs_aug[:n_shot] # [25, 3, 80, 80]
        inputs = torch.cat([inputs_ori, inputs_aug], 0).cuda()
        # [125, 3, 80, 80]

        (fea_ori, fea_aug) = torch.split(base_net(inputs), n_epi, 0)
        # [100, 640], [25, 640]
        fea_shot, fea_query = fea_ori[:n_shot], fea_ori[n_shot:]
        # [25, 640], [75, 640]

        proto = fea_shot.reshape(args.shot, args.way, -1).mean(dim=0) # [5, 640]
        proto_aug = fea_aug.reshape(args.shot, args.way, -1).mean(dim=0) # [5, 640]
        logits = euclidean_metric(fea_query, proto) # [75, 5]
        logits_aug = euclidean_metric(fea_query, proto_aug) # [75, 5]

        # fsl loss
        fsl_loss = ce_loss(logits, label)
        fsl_acc = count_acc(logits, label)

        # align
        if args.lambda_align > 0:
            probs1 = F.softmax(logits.detach(), 1)
            probs2 = F.softmax(logits_aug.detach(), 1)
            log_probs1 = F.log_softmax(logits / args.T, 1)
            log_probs2 = F.log_softmax(logits_aug / args.T, 1)
            align_loss = args.T * (kl_loss(log_probs2, probs1) + kl_loss(log_probs1, probs2))
            total_loss = fsl_loss + args.lambda_align * align_loss
        else:
            align_loss = torch.tensor(0)
            total_loss = fsl_loss
    
        if i % 10 == 0:
            print('iter:', i, 'align_loss:', align_loss.item(), 'fsl_loss:', fsl_loss.item())
            print('iter:', i, 'total_loss:', total_loss.item(), 'fsl_acc:', fsl_acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    lr_scheduler.step()

    # validation
    base_net.eval()
    ave_acc = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader, 1):
            inputs = batch[0].cuda() # [100, 3, 80, 80]
            features = base_net(inputs) # [100, 640]

            fea_shot, fea_query = features[:n_shot], features[n_shot:]
            # [25, 640], [75, 640]
            proto = fea_shot.reshape(args.shot, args.way, -1).mean(dim=0) # [5, 640]
            logits = euclidean_metric(fea_query, proto)
            acc = count_acc(logits, label)
            ave_acc.append(acc)
    ave_acc = np.mean(np.array(ave_acc))
    print('epoch {}: {:.2f}({:.2f})'.format(epoch, ave_acc * 100, acc * 100))
    if ave_acc > best_acc:
        best_acc = ave_acc
        torch.save({'base_net':base_net.state_dict()}, os.path.join(args.output_dir, 'best_model.pth.tar'))
    log_str = "epoch: {:05d}, accuracy: {:.5f}".format(epoch, ave_acc)
    log(log_file_path, log_str)
