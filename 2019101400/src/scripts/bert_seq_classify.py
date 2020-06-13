from transformers import AlbertTokenizer, BertTokenizer, AlbertModel, BertModel
import torch
import pickle
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.autograd as autograd
import pandas as pd
import pdb

batch_size = 27
use_cuda = True
num_epochs = 200
num_class = 5
max_length = 128
model_type = 'bert'
data_path = "data/comments.csv"

train_text = []
train_label = []
dev_text = []
dev_label = []

class Classify_model(nn.Module):

    def __init__(self, model, num_class1):
        super(Classify_model, self).__init__()
        self.model = model
        self.hidden_dim = 768
        self.num_class1 = num_class1
        self.fc1 = nn.Linear(self.hidden_dim, self.num_class1)

    def load_pretrained(require):
        self.model = AlbertModel.from_pretrained(require)
    
    def forward(self, x):
        hidden_state = self.model(x)[1] #
        class1 = self.fc1(hidden_state)

        return F.softmax(class1,dim=1)

data = pd.read_csv(data_path)
train, val = train_test_split(data, test_size=0.1)

if model_type == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model_ = BertModel.from_pretrained('bert-base-chinese')
elif model_type == 'albert':
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    model_ = AlbertModel.from_pretrained('albert-base-v2')
model = Classify_model(model_,num_class)
for i in range(len(train)):
    train_text.append(torch.tensor(tokenizer.encode(train.iloc[i]['short'], max_length=max_length, pad_to_max_length=True)).unsqueeze(0))
    train_label.append(train.iloc[i]['score'])
for i in range(len(val)):
    dev_text.append(torch.tensor(tokenizer.encode(val.iloc[i]['short'], max_length=max_length, pad_to_max_length=True)).unsqueeze(0))
    dev_label.append(val.iloc[i]['score'])

#pdb.set_trace()
train_text = torch.cat(train_text, dim=0)
train_label = torch.from_numpy(np.array(train_label,dtype='int32')).long()
dev_text = torch.cat(dev_text, dim=0)
dev_label = torch.from_numpy(np.array(dev_label,dtype='int32')).long()


dataset_train = data_utils.TensorDataset(train_text, train_label)
train_loader = data_utils.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
dataset_dev = data_utils.TensorDataset(dev_text, dev_label)
dev_loader = data_utils.DataLoader(dataset_dev, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

if use_cuda:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

loss_fn = nn.CrossEntropyLoss()

def evaluate(model, test_loader, use_cuda):
    preds_ = []
    y_test_ = []
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = autograd.Variable(inputs), autograd.Variable(labels)
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        preds = model(inputs)
#        pdb.set_trace()
        preds = list(torch.argmax(preds,dim=1).cpu().numpy())
        y_test = list(labels.cpu().numpy())
        preds_ = preds_ + preds
        y_test_ = y_test_ + y_test
    f1 = f1_score(y_test_, preds_, average='macro')
    precision = precision_score(y_test_, preds_, average='macro')
    recall = recall_score(y_test_, preds_, average='macro')
    return {'bert':{"f1":f1,"0.34*recall+0.66*f1":0.34*recall+0.66*f1,"recall":recall}
    }

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
#        pdb.set_trace()
        outputs = model(inputs)
        loss = loss_fn(outputs,labels)
#        loss = outputs[1]
#        pdb.set_trace()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
#    pdb.set_trace()
    metric_result = evaluate(model, dev_loader, use_cuda=True)
    print('[epoch: {:d}] train_loss: {:.3f}'.format(epoch, loss.item()))
    print(metric_result)
