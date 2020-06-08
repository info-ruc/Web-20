import pickle as pk
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import sys
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pdb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.autograd as autograd
from sklearn.model_selection import KFold
import time
from collections import Counter
import itertools

path_data = "../dataset/origin_new.txt"
path_cache_data = "../dataset/origin.pk"

if os.path.exists(path_cache_data):
    with open(path_cache_data, 'rb') as f:
        data_list = pk.load(f, encoding='latin1')
else:
    data_list = []
    sentences = []
    with open(path_data, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sentences.append(line)
            print(line)
    for sentence in sentences:
        data_ = sentence.strip().split('\t')
        if len(data_) > 1:
            data_[1] = jieba.lcut(data_[1])
            data_list.append(data_)
    with open(path_cache_data, 'wb') as f:
        pk.dump(data_list, f, protocol=2)

embedding_dim = 300
num_filters = 100
filter_sizes = [3, 4, 5]
batch_size = 200
num_epochs = 10

np.random.seed(0)
torch.manual_seed(0)

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
#    sequence_length = max(len(x) for x in sentences)
#    pdb.set_trace()
    sequence_length = 100
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) > sequence_length:
            new_sentence = sentence[:sequence_length]
        else:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def change_data_form(data):
    """
    data_form: [label(str), text(str)]
    """
    text = []
    y = []
    for data_ in data:
        text.append(data_[1])
        y.append(data_[0])
    y = [[0, 1] if label=='1' else [1, 0] for label in y]
    sentences_padded = pad_sentences(text)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences_padded])
    y = np.array(y)
    # y = y.argmax(axis=1)
    return [x, y, vocabulary, vocabulary_inv]

class TextCNN(nn.Module):
    
    def __init__(self, max_sent_len, embedding_dim, filter_sizes, num_filters, vocab_size, num_classes, use_cuda):
        '''
        :param embedding_dim:
        :param filer_sizes: list  -> e.g. [3, 4, 5]
        :param num_filters: size filter
        :param vocab_size:
        :param target_size:
        '''
        super(TextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # do we really need this?
        # (non-static word embedding)
        self.word_embeddings.weight.requires_grad = True

        conv_blocks = []
        for filter_size in filter_sizes:
            maxpool_kernel_size = max_sent_len - filter_size + 1
            conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=filter_size)
            # TODO: Sequential 
            component = nn.Sequential(
                conv1,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size))

            if use_cuda:
                component = component.cuda()
            conv_blocks.append(component)

        # TODO: ModuleList 
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, sentence_len)
        x = self.word_embeddings(x)
        # x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
        x = x.transpose(1, 2)   # switch 2nd and 3rd axis
        x_list = [conv_block(x) for conv_block in self.conv_blocks]

        # x_list.shape: [(num_filters, filter_size_3), (num_filters, filter_size_4), ...]
        out = torch.cat(x_list, 2)      # concatenate along filter_sizes
        out = out.view(out.size(0), -1)
        # feature_extracted = out
        out = F.dropout(out, p=0.5, training=self.training)
        return F.softmax(self.fc(out), dim=1)

def evaluate(model, test_loader, use_cuda):
    preds_ = []
    y_test_ = []
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = autograd.Variable(inputs), autograd.Variable(labels)
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        preds = model(inputs)
        if use_cuda:
            preds = preds.cuda()
        preds = list(torch.argmax(preds,dim=1).cpu().numpy())
        y_test = list(torch.argmax(labels.cpu(),dim=1).numpy())
        preds_ = preds_ + preds
        y_test_ = y_test_ + y_test
    f1 = f1_score(y_test_, preds_, average='macro')
    precision = precision_score(y_test_, preds_, average='macro')
    recall = recall_score(y_test_, preds_, average='macro')
    return {"f1":f1,"precision":precision,"recall":recall}



#def evaluate(model, test_loader, use_cuda):
#    inputs = autograd.Variable(x_test)
#    preds = model(inputs)
#    preds = torch.max(preds, 1)[1]
#    y_test = torch.max(y_test, 1)[1]
#    if use_cuda:
#        preds = preds.cuda()
#    preds = preds.cpu().numpy()
#    y_test = y_test.cpu().numpy()
#    f1 = f1_score(y_test, preds, average='macro')
#    precision = precision_score(y_test, preds, average='macro')
#    recall = recall_score(y_test, preds, average='macro')
#    return {"f1":f1,"precision":precision,"recall":recall}


def train_test_one_split(train_index, test_index, use_cuda_,vocab_size, max_sent_len, num_classes):
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]

    # numpy array to torch tensor
    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).long()
    y_test = torch.from_numpy(y_test).float()    
#    pdb.set_trace()
    dataset_train = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    dataset_test = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

#    x_test = torch.from_numpy(x_test).long()
#    y_test = torch.from_numpy(y_test).float()
#    if use_cuda_:
#        x_test = x_test.cuda()
#        y_test = y_test.cuda()

    model = TextCNN(max_sent_len=max_sent_len,
                    embedding_dim=embedding_dim,
                    filter_sizes=filter_sizes,
                    num_filters=num_filters,
                    vocab_size=vocab_size,
                    num_classes=num_classes,
                    use_cuda=use_cuda_)

    if use_cuda_:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    loss_fn = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()       # set the model to training mode
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = autograd.Variable(inputs), autograd.Variable(labels)
            if use_cuda_:
                inputs, labels = inputs.cuda(), labels.cuda()

            preds = model(inputs)
            if use_cuda_:
                preds = preds.cuda()

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()        # set the model to evaluation mode
#        pdb.set_trace()
        metric_result = evaluate(model, test_loader, use_cuda_)
#        pdb.set_trace()
        print('[epoch: {:d}] train_loss: {:.3f}'.format(epoch, loss.item()))
        print(metric_result)

    model.eval()        # set the model to evaluation mode
    metric_result = evaluate(model, test_loader, use_cuda_)
    return metric_result

def avg_metric_result(metric_result):
    f1 = 0
    precision = 0
    recall = 0
    for i in range(len(metric_result)):
        f1 += metric_result[i]['f1']
        precision += metric_result[i]['precision']
        recall += metric_result[i]['recall']
    f1 = f1*1.0/len(metric_result)
    precision = precision*1.0/len(metric_result)
    recall = recall*1.0/len(metric_result)
    return [f1, precision, recall]

data = change_data_form(data_list)
#print(data)
X, Y, word_to_ix, ix_to_word = data
vocab_size = len(word_to_ix)

#print(vocab_size);
max_sent_len = X.shape[1]
num_classes = Y.shape[1]
cv_folds = 5    # 5-fold cross validation
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
metric_list = []
tic = time.time()
use_cuda_ = True
for cv, (train_index, test_index) in enumerate(kf.split(X)):
    metric_result = train_test_one_split(train_index, test_index, use_cuda_, vocab_size, max_sent_len, num_classes)
#    pdb.set_trace()
    print('cv = {}    train size = {}    test size = {}\n'.format(cv, len(train_index), len(test_index)))
    metric_list.append(metric_result)
f1, precision, recall = avg_metric_result(metric_list)
print('\navg f1 = {:.3f}   avg pre = {:.3f}   avg recall = {:.3f}   (total time: {:.1f}s)\n'.format(f1, precision, recall, time.time()-tic))
