import numpy as np
import torch
import pickle
import scipy
import scipy.spatial
dataset_path = "/home/txb/webHW/dataset/zh_en/"
save_Gemb_path = "./G_zhen_emb.pkl"


def read_idtuple_file(file_path):
    print('loading a idtuple file...   ' + file_path)
    ret = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            th = line.strip('\n').split('\t')
            x = []
            for i in range(len(th)):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')#cosine
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


entid2nameemb = pickle.load(open(dataset_path + "name_embedding","rb"))
name_emb = [entid2nameemb[i] for i in range(len(entid2nameemb))]
structure_emb = pickle.load(open(save_Gemb_path,"rb")).detach().cpu().tolist()
test_ill = read_idtuple_file(dataset_path + "ref_pairs")
print("name embedding:",np.array(name_emb).shape)
print("structure embedding:",np.array(structure_emb).shape)

get_hits(name_emb,test_ill)
get_hits(structure_emb,test_ill)
emb = np.concatenate((name_emb,structure_emb),axis=1)
print("concat embedding shape:",emb.shape)
get_hits(emb.tolist(),test_ill)

