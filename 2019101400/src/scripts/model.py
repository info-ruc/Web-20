import torch
from base_model import rnnbase, Word2vecEmbedding, Classifier
import pdb

class my_model(torch.nn.Module):
    def __init__(self, embedding_matrix):
        super(my_model, self).__init__()

        hidden_size = 150
        dropout_p = 0.4
        emb_dropout_p = 0.1
        hidden_mode = 'LSTM'
        num_class = 5

        word_embedding_size = 300
        encoder_bidirection = True
        encoder_bidirection_num = 2 if encoder_bidirection else 1

        self.hidden_size = hidden_size

        self.embedding = Word2vecEmbedding(embedding_matrix)

        self.encoder = rnnbase(mode=hidden_mode,
                                input_size=word_embedding_size,
                                hidden_size=hidden_size,
                                bidirectional=encoder_bidirection,
                                dropout_p=emb_dropout_p)

        encode_out_size = hidden_size * encoder_bidirection_num

        self.attention = Classifier(encode_out_size, 1)

        self.classifier = Classifier(encode_out_size, num_class)

    def forward(self, context):
        """
        baseline: only use the simple classifier
        :param context:
        :param ground_truth:
        :return:
        """
#        pdb.set_trace()
        # get embedding: (seq_len, batch, embedding_size)
        context_vec, context_mask = self.embedding.forward(context)

        # encode: (seq_len, batch, embedding_size)
        context_encode, last_state = self.encoder.forward(context_vec, context_mask)

        # attention: (seq_len, batch)
        attention = self.attention.forward(context_encode)
        attention = attention.reshape([context_encode.shape[0], 1, context_encode.shape[1]])
        attention = attention.permute(2, 1, 0) # (batch, 1, seq_len)
        context_encode = context_encode.permute(1, 0, 2) # (batch, seq_len, embedding_dim)
        sentence_encode = torch.bmm(attention, context_encode)
        sentence_encode = sentence_encode.squeeze() # (batch, embedding_dim)

        # probability distribution: (batch, num_class)
        prob = self.classifier(sentence_encode)

        return prob
