import torch

class rnnbase(torch.nn.Module):
    """
    RNN with packed sequence and dropout, only one layer
    Args:
        input_size: The number of expected features in the input x
    """
    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm=False):
        super(rnnbase, self).__init__()
        self.mode = mode
        self.enable_layer_norm = enable_layer_norm

        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size = input_size,
                                        hidden_size = hidden_size,
                                        num_layers = 1,
                                        bidirectional = bidirectional)
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size,
                                       hidden_size = hidden_size,
                                       num_layers = 1,
                                       bidirectional = bidirectional)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)

        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        :return:
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, v, mask): #v: (seq_len, batch, input_size); mask: (batch, seq_len)
        # layer normalization
        if self.enable_layer_norm:
            seq_len, batch, input_size = v.shape
            v = v.view(-1, input_size)
            v = self.layer_norm(v)
            v = v.view(seq_len, batch, input_size)

        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        v_sort = v.index_select(1, idx_sort)
        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)

        o_pack_dropout, _ = self.hidden.forward(v_pack_dropout)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        o_unsort = o.index_select(1, idx_unsort)

        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)
        o_last = o_unsort.gather(0, len_idx)
        o_last = o_last.squeeze(0)

        return o_unsort, o_last

class Word2vecEmbedding(torch.nn.Module):
    """
    input: the word one-hot vector sequence
    output: every time step output, the last time step output
    """
    def __init__(self, embedding_matrix):
        """
        load the pretrained embedding_matrix
        :param embedding_matrix:
        """
        super(Word2vecEmbedding, self).__init__()
        self.embedding_layer = torch.nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding_layer.weight.requires_grad = True

    def compute_mask(self, v, padding_idx=0):
        """
        compute mask on given tensor v
        :param v:
        :param padding_idx:
        :return:
        input (batch_size, sequence_length)
        output (batch_size, sequence_length)
        """
        mask = torch.ne(v, padding_idx).float()
        return mask

    def forward(self, x):
        mask = self.compute_mask(x)

        tmp_emb = self.embedding_layer.forward(x)
        out_emb = tmp_emb.transpose(0, 1)

        return out_emb, mask

class Classifier(torch.nn.Module):
    """
    a simple full-connect layer to use it to produce the probability distribution
    initialization:
    :parameter
    in_dim: input dimention
    class_num: the number of classes

    compute:
    :input
    batch_size, hidden_dim
    """
    def __init__(self, in_dim, class_num):
        super(Classifier, self).__init__()
        self.layer1 = torch.nn.Linear(in_dim, class_num)

    def forward(self, x):
        x = self.layer1(x)
        return x
