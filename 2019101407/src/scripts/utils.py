import numpy as np
import time
import torch
import scipy
import scipy.spatial
import torch.nn as nn
import torch.nn.functional as F



def negsample_func(emb,ent_ids1,ent_ids2,useGPU = False,GPUnum = 0,bs = 512,k = 512):
    ent1_emb = emb[torch.LongTensor(ent_ids1)]
    ent2_emb = emb[torch.LongTensor(ent_ids2)]
    ent1_emb = F.normalize(ent1_emb, p=2)
    ent2_emb = F.normalize(ent2_emb, p=2)
    ent1_to_ent2 = []
    ent2_to_ent1 = []
    ent1_num = ent1_emb.shape[0]
    ent2_num = ent2_emb.shape[0]
    for i in range(0, ent1_num, bs):
        tt = ent1_emb[i:min(i + bs, ent1_num)].cuda(GPUnum)
        res = tt.mm(ent2_emb.t().cuda(GPUnum))
        _, index_mat = res.topk(k, largest=True)
        ent1_to_ent2.append(index_mat.cpu())
    ent1_to_ent2 = torch.cat(ent1_to_ent2, 0)
    for i in range(0, ent2_num, bs):
        tt = ent2_emb[i:min(i + bs, ent2_num)].cuda(GPUnum)
        res = tt.mm(ent1_emb.t().cuda(GPUnum))
        _, index_mat = res.topk(k, largest=True)
        ent2_to_ent1.append(index_mat.cpu())
    ent2_to_ent1 = torch.cat(ent2_to_ent1, 0)
    print("ent1_to_ent2 shape:",ent1_to_ent2.shape,"ent2_to_ent1_shape:",ent2_to_ent1.shape)

    ent1_to_ent2 = ent1_to_ent2.tolist()
    ent2_to_ent1 = ent2_to_ent1.tolist()

    candidate_dict = dict()
    for i in range(len(ent_ids1)):
        e = ent_ids1[i]
        candidate_dict[e] = []
        for c in ent1_to_ent2[i]:
            candidate_dict[e].append(ent_ids2[c])
    for i in range(len(ent_ids2)):
        e = ent_ids2[i]
        candidate_dict[e] = []
        for c in ent2_to_ent1[i]:
            candidate_dict[e].append(ent_ids1[c])
    print("candidate_dict:",len(candidate_dict))
    return candidate_dict



# def generate_trainging_data(train_ill,ent_ids1,ent_ids2,useGPU = False,GPUnum = 0,neg_num = 5):
#     start_time = time.time()
#     pos_ents1 = []
#     pos_ents2 = []
#     neg_ents1 = []
#     neg_ents2 = []
#     for i in range(len(train_ill)):
#         pe1 = train_ill[i][0]
#         pe2 = train_ill[i][1]
#         for j in range(neg_num):
#             if np.random.rand() < 0.5:
#                 ne1 = ent_ids1[np.random.randint(len(ent_ids1))]
#                 ne2 = pe2
#             else:
#                 ne1 = pe1
#                 ne2 = ent_ids2[np.random.randint(len(ent_ids2))]
#             neg_ents1.append(ne1)
#             neg_ents2.append(ne2)
#             pos_ents1.append(pe1)
#             pos_ents2.append(pe2)
#
#     assert len(pos_ents1) == len(pos_ents2)
#     assert len(neg_ents1) == len(pos_ents2)
#     assert len(neg_ents1) == len(neg_ents2)
#
#     using_time = time.time()-start_time
#     if useGPU:
#         return torch.LongTensor(pos_ents1).cuda(GPUnum),\
#                torch.LongTensor(pos_ents2).cuda(GPUnum),\
#                torch.LongTensor(neg_ents1).cuda(GPUnum),\
#                torch.LongTensor(neg_ents2).cuda(GPUnum),\
#                using_time
#
#     else:
#         return torch.LongTensor(pos_ents1),\
#                torch.LongTensor(pos_ents2),\
#                torch.LongTensor(neg_ents1),\
#                torch.LongTensor(neg_ents2),\
#                using_time


# def generate_trainging_data(candidate_dict,train_ill,ent_ids1,ent_ids2,useGPU = False,GPUnum = 0,neg_num = 5):
#     #plus version
#     start_time = time.time()
#
#     pos_ents1 = []
#     pos_ents2 = []
#     neg_ents1 = []
#     neg_ents2 = []
#     max_candidate_num = len(candidate_dict[list(candidate_dict.keys())[0]])
#     for i in range(len(train_ill)):
#         pe1 = train_ill[i][0]
#         pe2 = train_ill[i][1]
#         for j in range(neg_num):
#             if np.random.rand() < 0.5:
#                 ne1 = candidate_dict[pe2][np.random.randint(max_candidate_num)]
#                 ne2 = pe2
#             else:
#                 ne1 = pe1
#                 ne2 = candidate_dict[pe1][np.random.randint(max_candidate_num)]
#             neg_ents1.append(ne1)
#             neg_ents2.append(ne2)
#             pos_ents1.append(pe1)
#             pos_ents2.append(pe2)
#
#     assert len(pos_ents1) == len(pos_ents2)
#     assert len(neg_ents1) == len(pos_ents2)
#     assert len(neg_ents1) == len(neg_ents2)
#
#     using_time = time.time()-start_time
#     if useGPU:
#         return torch.LongTensor(pos_ents1).cuda(GPUnum),\
#                torch.LongTensor(pos_ents2).cuda(GPUnum),\
#                torch.LongTensor(neg_ents1).cuda(GPUnum),\
#                torch.LongTensor(neg_ents2).cuda(GPUnum),\
#                using_time
#
#     else:
#         return torch.LongTensor(pos_ents1),\
#                torch.LongTensor(pos_ents2),\
#                torch.LongTensor(neg_ents1),\
#                torch.LongTensor(neg_ents2),\
#                using_time


def generate_batchtrainging_data(candidate_dict,batch_triples, useGPU = False,GPUnum = 0,neg_num = 5):
    #plusKE version
    start_time = time.time()
    phs = []
    prs = []
    pts = []
    nhs = []
    nrs = []
    nts = []
    max_candidate_num = len(candidate_dict[list(candidate_dict.keys())[0]])
    for i in range(len(batch_triples)):
        ph = batch_triples[i][0]
        pr = batch_triples[i][1]
        pt = batch_triples[i][2]
        for j in range(neg_num):
            if np.random.rand() < 0.5:
                nh = candidate_dict[ph][np.random.randint(max_candidate_num)]
                nr = pr
                nt = pt
            else:
                nh = ph
                nr = pr
                nt = candidate_dict[pt][np.random.randint(max_candidate_num)]
            phs.append(ph);prs.append(pr);pts.append(pt)
            nhs.append(nh);nrs.append(nr);nts.append(nt)

    using_time = time.time()-start_time
    if useGPU:
        return torch.LongTensor(phs).cuda(GPUnum),\
               torch.LongTensor(prs).cuda(GPUnum),\
               torch.LongTensor(pts).cuda(GPUnum),\
               torch.LongTensor(nhs).cuda(GPUnum),\
               torch.LongTensor(nrs).cuda(GPUnum),\
               torch.LongTensor(nts).cuda(GPUnum),\
               using_time

    else:
        return torch.LongTensor(phs),\
               torch.LongTensor(prs),\
               torch.LongTensor(pts),\
               torch.LongTensor(nhs),\
               torch.LongTensor(nrs),\
               torch.LongTensor(nts),\
               using_time


def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))