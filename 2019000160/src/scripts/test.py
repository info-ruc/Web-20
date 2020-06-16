import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import log, euclidean_metric, count_acc, setup_seed
from data_utils import CategoriesSampler, BirdsDataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--output_dir', default='./exp_5shot/proto_exp1', type=str)
parser.add_argument('--load', default='./exp_5shot/proto_exp1/best_model.pth.tar', type=str)
parser.add_argument('--cnn', default='WideResNet', type=str)
parser.add_argument('--batch', default=2000, type=int)
parser.add_argument('--way', default=5, type=int)
parser.add_argument('--shot', default=5, type=int)
parser.add_argument('--query', default=15, type=int)
parser.add_argument('--data_root', default='../dataset', type=str)
parser.add_argument('--test_seed', default=111, type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
log_file_path = os.path.join(args.output_dir, 'test_log.txt')
log(log_file_path, str(vars(args)))
setup_seed(args.test_seed)

test_set = BirdsDataset(root=args.data_root, mode='test')
test_sampler = CategoriesSampler(test_set.label, args.batch, args.way, args.shot + args.query)
test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler, num_workers=4, pin_memory=True)

if args.cnn == 'WideResNet':
    import network.wideresnet as wideresnet
    base_net = wideresnet.WideResNet().cuda()
else:
    print('No implementation!')
    exit()
saved_models = torch.load(args.load)
base_net.load_state_dict(saved_models['base_net'])
base_net.eval()

n_shot = args.way * args.shot # 25
label = torch.arange(args.way).repeat(args.query) #75
label = label.type(torch.cuda.LongTensor)
test_accuracies = []
with torch.no_grad():
    for i, batch in enumerate(test_loader, 1):
        inputs = batch[0].cuda() # [100, 3, 80, 80]
        features = base_net(inputs) # [100, 640]
        
        fea_shot, fea_query = features[:n_shot], features[n_shot:]
        # [25, 640], [75, 640]
        proto = fea_shot.reshape(args.shot, args.way, -1).mean(dim=0) # [5, 640]
        logits = euclidean_metric(fea_query, proto)
        acc = count_acc(logits, label)
        test_accuracies.append(acc)
        
        if i % 50 == 0:
            avg = np.mean(np.array(test_accuracies))
            std = np.std(np.array(test_accuracies))
            ci95 = 1.96 * std / np.sqrt(i + 1)
            log_str = 'batch {}: Accuracy: {:.4f} +- {:.4f} % ({:.4f} %)'.format(i, avg, ci95, acc)
            log(log_file_path, log_str)
