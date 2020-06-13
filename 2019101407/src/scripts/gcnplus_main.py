import torch
import torch.nn as nn
import torch.nn.functional as F
from readdate import *
from GCNTransEmodel import GCNTransE
import time
import torch.optim as optim
from utils import *
import pickle
if __name__ == '__main__':
    all_start_time = time.time()
    train_ill, test_ill, adj, entid_1, entid_2, triples, relnum  = load_data("../dataset/zh_en/")
    gpunum = 0
    save_emb_name = "G_zhen_emb.pkl"
    batch_size = 4096
    model = GCNTransE(
        entNum = adj.shape[0],
        relNum = relnum,
        relFeatureDim = 100,
        entFeatureDim = 100,
        hiddenDim = 100,
        outputDim = 100,
        useGPU = True,
        GPUnum = gpunum)
    print(model)
    model = model.cuda(gpunum)
    adj = adj.cuda(gpunum)
    criterion = nn.MarginRankingLoss(margin = 1)
    optimizer = optim.Adam(model.parameters(),lr = 5e-4)
    for name,v in model.named_parameters():
        print(name)

    for epoch in range(200):
        np.random.shuffle(train_ill)
        if epoch % 20 == 0:
            model.eval()
            ent_emb,rel_emb = model(adj)
            ent_emb = ent_emb.detach().cpu()
            candidate_dict = negsample_func(ent_emb, entid_1, entid_2, useGPU=True, GPUnum=gpunum, bs=512, k=2048)

        batch_num = int(len(triples) / batch_size)
        np.random.shuffle(triples)
        start_time = time.time()
        for i in range(batch_num):
            batch_triples = triples[i * batch_size: (i+1) * batch_size]
            ph,pr,pt,nh,nr,nt,time1 = generate_batchtrainging_data(candidate_dict,batch_triples,useGPU=True,GPUnum=gpunum,neg_num=4)
            model.train()
            optimizer.zero_grad()
            ent_emb, rel_emb = model(adj)
            pos_score = F.pairwise_distance(ent_emb[ph] + rel_emb[pr] , ent_emb[pt],p=2,keepdim=True)
            neg_score = F.pairwise_distance(ent_emb[nh] + rel_emb[nr] , ent_emb[nt],p=2,keepdim=True)
            y = -torch.ones(pos_score.shape).cuda(gpunum)
            loss = criterion(pos_score ** 2 ,neg_score ** 2 ,y)
            loss.backward()
            optimizer.step()
        print("epoch {}  loss : {:.6f} traintime {:.3f} traindatagtime {:.3f}".format(epoch,loss.item(),time.time()-start_time,time1))

        if (epoch + 1) % 20 == 0 and epoch != 0:
            model.eval()
            ent_emb, rel_emb = model(adj)
            # get_hits(ent_emb.detach().cpu().tolist(),test_ill)
            pickle.dump(ent_emb,open(save_emb_name,"wb"))
    print("all using time {:.3f}".format(time.time()-all_start_time))
    model.eval()
    ent_emb, rel_emb = model(adj)
    get_hits(ent_emb.detach().cpu().tolist(), test_ill)









