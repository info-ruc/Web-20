from pyfasttext import FastText
import numpy as np
import pandas as pd
import logging
import pdb
import os

#w2v_path = '/data1/hjw/backup/cc.zh.300.bin'

#model = FastText(w2v_path)
#vocab = model.vocab

class myVocab(object):
    """
    input: the pretrained data vocab_path, the used vocab list
    Implements the main reading embedding
    """
    def __init__(self, vocab_path, vocab):
        self.pad_token = '<blank>'
        self.unk_token = '<unk>'
        self.model = FastText(vocab_path)
        self.vocab = ['<blank>', '<unk>'] + vocab
        self.token2id = {}
        self.id2token = {}
        self.embed_dim = 300    #this is deployed temporarily
        if not os.path.exists('embeddings.npy'):
            self.embeddings = np.random.rand(self.size(), self.embed_dim)
        else:
            self.embeddings = np.load('embeddings.npy')
        self.logger = logging.getLogger("sentiment")

        i = 0
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[i] = np.zeros([self.embed_dim])
            self.token2id[token] = i
            self.id2token[i] = token
            i += 1
        for token in vocab:
            self.token2id[token] = i
            self.id2token[i] = token
            i += 1
        '''
        cnt = 28000
        len_vocab = len(self.vocab)
        df = pd.DataFrame(self.vocab, columns=['vocab_word'])
        if not os.path.exists('vocab.csv'):
            df.to_csv('vocab.csv', index=False)
 
        for word in vocab[28000:]:
            if word in self.model.words:
                self.embeddings[cnt] = self.model[word]
            cnt += 1
            if cnt % 100 == 0:
                self.logger.info("%d / %d" % (cnt, len_vocab))
            if cnt % 2000 == 0:
                self.logger.info("%d / %s" % (cnt, word))
                self.logger.info(self.embeddings[cnt-1])
                np.save('embeddings.npy', self.embeddings)


        np.save('embeddings.npy', self.embeddings)
        '''
    def get_id(self, token):
        return self.token2id[token]


    def size(self):
        """
        :return: the length of vocab
        """
        return len(self.vocab)

    def convert2ids(self, wordlist):
        """
        input a list of word and output their ids
        :param wordlist:
        :return:
        """
        token_id = [self.get_id(label) for label in wordlist]
        return token_id

