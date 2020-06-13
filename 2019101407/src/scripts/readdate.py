import numpy as np
import scipy.sparse as sp
import torch


def read_data(data_path):
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
    def read_id2object(file_paths):
        id2object = {}
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                print('loading a (id2object)file...  ' + file_path)
                for line in f:
                    th = line.strip('\n').split('\t')
                    id2object[int(th[0])] = th[1]
        return id2object
    def read_idobj_tuple_file(file_path):
        print('loading a idx_obj file...   ' + file_path)
        ret = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                th = line.strip('\n').split('\t')
                ret.append( ( int(th[0]),th[1] ) )
        return ret

    print("load data from... :", data_path)
    #读取编码和实体/关系的转换
    index2entity = read_id2object([data_path + "ent_ids_1",data_path + "ent_ids_2"])
    index2rel = read_id2object([data_path + "rel_ids_1",data_path + "rel_ids_2"])
    entity2index = {e:idx for idx,e in index2entity.items()}
    rel2index = {r:idx for idx,r in index2rel.items()}

    #读取三元组
    rel_triples_1 = read_idtuple_file(data_path + 'triples_1')
    rel_triples_2 = read_idtuple_file(data_path + 'triples_2')
    index_with_entity_1 = read_idobj_tuple_file(data_path + 'ent_ids_1')
    index_with_entity_2 = read_idobj_tuple_file(data_path + 'ent_ids_2')
    entid_1 = [entid for entid, _ in index_with_entity_1]
    entid_2 = [entid for entid, _ in index_with_entity_2]

    #读取训练集和测试集数据
    train_ill = read_idtuple_file(data_path + 'sup_pairs')
    test_ill = read_idtuple_file(data_path + 'ref_pairs')
    ent_ill = []
    ent_ill.extend(train_ill)
    ent_ill.extend(test_ill)
    return ent_ill, train_ill, test_ill, index2rel, index2entity, rel2index, entity2index,rel_triples_1, rel_triples_2,entid_1,entid_2

# def generate_adj(triples):
#     #生成邻接矩阵
#     coo2value = dict()
#     for h,r,t in triples:
#         if (h,t) not in coo2value:
#             coo2value[(h,t)] = 0
#         if (t,h) not in coo2value:
#             coo2value[(t,h)] = 0
#         if (h,h) not in coo2value:
#             coo2value[(h,h)] = 0
#         if (t,t) not in coo2value:
#             coo2value[(t,t)] = 0
#         coo2value[(h,t)] += 1
#         coo2value[(h,h)] += 1
#         coo2value[(t,h)] += 1
#         coo2value[(t,t)] += 1
#     values = []
#     rows = []
#     cols = []
#     for h,t in coo2value.keys():

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def generate_adj(triples,ent_num):
    coo2value = dict()
    for h,r,t in triples:
        coo2value[(h,t)] = 1
        coo2value[(t,h)] = 1
    values = []
    rows = []
    cols = []
    for r,c in coo2value:
        v = coo2value[(r,c)]
        values.append(v)
        rows.append(r)
        cols.append(c)
    adj = sp.coo_matrix((np.array(values),(np.array(rows),np.array(cols))),
                        shape = (ent_num,ent_num),dtype=np.float32)

    print("adj shape:",adj.shape)
    adj = normalize(adj + sp.eye(adj.shape[0]) + sp.eye(adj.shape[0]))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def swap_triples(triples,train_ill):
    swapdict = dict()
    for e1,e2 in train_ill:
        swapdict[e1] = e2
        swapdict[e2] = e1
    new_triples = []
    for h,r,t in triples:
        new_triples.append((h,r,t))
        if h in swapdict:
            new_triples.append((swapdict[h],r,t))
        if t in swapdict:
            new_triples.append((h,r,swapdict[t]))
    return new_triples



def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(e, KG):
    r2f = func(KG)
    r2if = ifunc(KG)
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, e))



def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_normalized = adj_normalized + sp.eye(adj.shape[0])
    return adj_normalized


def load_data(file_path):
    #读取并处理数据
    ent_ill, train_ill, test_ill, \
    index2rel, index2entity, rel2index, entity2index, \
    rel_triples_1, rel_triples_2,entid_1,entid_2 = read_data(file_path)

    triples = []
    triples.extend(rel_triples_1)
    triples.extend(rel_triples_2)
    adj = get_weighted_adj(len(index2entity), triples)
    triples = swap_triples(triples,train_ill)
    print("ent_ill:",len(ent_ill))
    print("train_ill:",len(train_ill))
    print("test_ill:",len(test_ill))
    print("index2entity:",len(index2entity),"entity2index:",len(entity2index))
    print("index2rel:",len(index2rel),"rel2index:",len(rel2index))
    print("triples:",len(triples),"rel_tri1:",len(rel_triples_1),"rel_tri2:",len(rel_triples_2))
    print("entid_1s",len(entid_1),"entid_2s",len(entid_2))
    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj) #sparse.FloatTensor
    relnum = len(index2rel)


    return train_ill,test_ill,adj,entid_1,entid_2,triples,relnum
