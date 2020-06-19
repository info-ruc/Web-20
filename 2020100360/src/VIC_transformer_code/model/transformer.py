#!/usr/bin/env python
# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.model_util import init_tensor
from dataset.classification_dataset import ClassificationDataset as cDataset

# embedding 词嵌入
class Embedding(torch.nn.Module):
    def __init__(self, dict_map, embedding_dim, name, config, padding_idx=None,
                 dropout=0, init_type="uniform", low=0, high=1,
                 mean=0, std=1, activation_type="linear",
                 model_mode="train"):
        super(Embedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mode = "flat"
        self.embedding = torch.nn.Embedding(
            len(dict_map), embedding_dim, padding_idx=padding_idx)
        embedding_lookup_table = init_tensor(
            tensor=torch.empty(len(dict_map), embedding_dim), low=low, high=high, mean=mean, std=std,
            activation_type="linear")
        if padding_idx is not None:
            embedding_lookup_table[padding_idx] = 0.0
        self.embedding.weight.data.copy_(embedding_lookup_table)

    def forward(self, vocab_ids, offset=None):
        embedding = self.embedding(vocab_ids)
        return self.dropout(embedding)


class PositionEmbedding(torch.nn.Module):
    ''' Reference: attention is all you need '''

    def __init__(self, seq_max_len, embedding_dim, padding_idx):
        super(PositionEmbedding, self).__init__()

        self.position_enc = nn.Embedding.from_pretrained(
            self.get_sinusoid_encoding_table(seq_max_len + 1,
                                             embedding_dim,
                                             padding_idx=padding_idx),
            freeze=True)

    def forward(self, src_pos):
        return self.position_enc(src_pos)

    # 静态方法
    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array(
            [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.FloatTensor(sinusoid_table)


# self-attention
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature  # temperature其实是scaler，归一化的时候用的
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # QK' torch.bmm表示矩阵乘法
        attn = torch.bmm(q, k.transpose(1, 2))
        # 归一化
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # q和k的dimension肯定得是一样的
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # 维度操作
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


# classifier
class Classifier(torch.nn.Module):
    def __init__(self, dataset, config):
        super(Classifier, self).__init__()
        self.config = config
        assert len(self.config.feature.feature_names) == 1
        assert self.config.feature.feature_names[0] == "token" or \
               self.config.feature.feature_names[0] == "char"
        # 词嵌入
        self.token_embedding = \
            Embedding(dataset.token_map, config.embedding.dimension,
                        cDataset.DOC_TOKEN, config, dataset.VOCAB_PADDING,
                        dropout=self.config.embedding.dropout,
                        init_type="uniform",
                        low=-self.config.embedding.uniform_bound,
                        high=self.config.embedding.uniform_bound,
                        std=self.config.embedding.random_stddev,
                        activation_type="linear", model_mode=dataset.model_mode)

    def get_embedding(self, batch, pad_shape=None, pad_value=0):
        token_id = batch[cDataset.DOC_TOKEN].to(self.config.device)
        if pad_shape is not None:
            token_id = torch.nn.functional.pad(
                token_id, pad_shape, mode='constant', value=pad_value)
        embedding = self.token_embedding(token_id)
        length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
        mask = batch[cDataset.DOC_TOKEN_MASK].to(self.config.device)
        return embedding, length, mask


# Transformer
# Feed Forward Network 编码器和解码器中的每一层除了注意子层外，还包含一个全连接的前馈网络，
# 这个FFN包含两个线性变换，中间有一个ReLU激活。这个线性变换在不同的位置都表现地一样，但是在不同的层之间使用不同的参数。
# 起到变形的作用
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x  # 同时为了更好的优化深度网络，这里可以看到还使用了 residual 残差连接(ADD)
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # Layer Normalization也是归一化数据的一种方式，可以加快模型收敛速度
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # 使用mask来避免attention放在用于对齐的padding填充位置上
        enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

# 这并不是一个完整的transformer，因为不需要seq2seq的翻译，
class Transformer(Classifier):
    def __init__(self, dataset, config):
        super(Transformer, self).__init__(dataset, config)
        # 位置编码（pad表示对不足的序列做padding）
        self.pad = dataset.token_map[dataset.VOCAB_PADDING]
        seq_max_len = config.feature.max_token_len
        self.position_enc = PositionEmbedding(seq_max_len,
                                              config.embedding.dimension,
                                              self.pad)
        # 构造n_layers层的编码层，inner表示feedforward中间层的维度
        # head表示注意力头数，d_k, d_v表示注意力向量中key和value的维度，之前config文件中的dimension表示词嵌入的维度
        self.layer_stack = nn.ModuleList([
            EncoderLayer(config.embedding.dimension,
                            config.Transformer.d_inner,
                            config.Transformer.n_head,
                            config.Transformer.d_k,
                            config.Transformer.d_v,
                            dropout=config.Transformer.dropout)
            for _ in range(config.Transformer.n_layers)])

        hidden_size = config.embedding.dimension
        self.linear = torch.nn.Linear(hidden_size, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        for i in range(0, len(self.layer_stack)):
            params.append({'params': self.layer_stack[i].parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        if epoch == 1:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        if epoch % 10 == 0:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = param_group["lr"] * self.config.train.decay_rate

    def forward(self, batch):
        def _get_non_pad_mask(seq, pad):
            assert seq.dim() == 2
            return seq.ne(pad).type(torch.float).unsqueeze(-1)

        def _get_attn_key_pad_mask(seq_k, seq_q, pad):
            ''' For masking out the padding part of key sequence. '''
            # Expand to fit the shape of key query attention matrix.
            len_q = seq_q.size(1)
            padding_mask = seq_k.eq(pad)
            padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
            return padding_mask

        src_seq = batch[cDataset.DOC_TOKEN].to(self.config.device)
        embedding = self.token_embedding(src_seq)

        # Prepare masks
        slf_attn_mask = _get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad=self.pad)
        non_pad_mask = _get_non_pad_mask(src_seq, self.pad)

        batch_lens = (src_seq != self.pad).sum(dim=-1)
        src_pos = torch.zeros_like(src_seq, dtype=torch.long)
        for row, length in enumerate(batch_lens):
            src_pos[row][:length] = torch.arange(1, length + 1)

        # 词嵌入和位置编码
        enc_output = embedding + self.position_enc(src_pos)
        # 依次进行n_layer层编码
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output,
                                        non_pad_mask=non_pad_mask,
                                        slf_attn_mask=slf_attn_mask)
        enc_output = torch.mean(enc_output, 1)

        return self.dropout(self.linear(enc_output))
