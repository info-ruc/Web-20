import numpy as np
import torch
from model import my_model
import pandas as pd
from vocab import myVocab
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import jieba
import logging
import pdb

w2v_path = '/data1/hjw/backup/cc.zh.300.bin'
data_path = "data/comments.csv"

batch_size = 128
max_len = 500
epochs = 200

logger = logging.getLogger("sentiment")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def to_long_tensor(np_array):
    """
    convert to long torch tensor
    :param np_array:
    :return:
    """
    return torch.from_numpy(np_array).type(torch.long)

def cut(str1):
    str2 = ''.join(str1.strip('"').split())
    return list(jieba.cut(str2))

def create_set(data):
    k0 = set()
    for data1 in data:
        k0 = k0 | set(data1)
    return k0

def get_mini_batch(data, batch_size, shuffle = True, max_len=300, pad_id=0):
    """
    get one mini_batch data
    :param data:
    :param batch_size:
    :return:
    """
#    self.max_len = max_len # set with the pre-research

    def _dynamic_padding(batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
#        pdb.set_trace()
        pad_len = min(max_len, max(list(batch_data['length'])))
        batch_data['token_ids'] = [(ids + [pad_id] * (pad_len - len(ids)))[: pad_len]
                                           for ids in batch_data['token']]
        return batch_data, pad_len

    def _one_mini_batch(data, indices, pad_id):
        """
        get one batch data
        :param data:
        :param indices:
        :param pad_id:
        :return:
        """
        batch_data = [data.iloc[i] for i in indices]
        batch_data = pd.DataFrame(batch_data)
        batch_data, padded_len = _dynamic_padding(batch_data, pad_id)
        return batch_data

    data_size = len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for batch_start in np.arange(0, data_size, batch_size):
        batch_indices = indices[batch_start: batch_start + batch_size]
        yield _one_mini_batch(data, batch_indices, pad_id)

#model1 = my_model(vocab.embeddings)
#print('Model is loaded successfully.')
device = torch.device('cuda')
if torch.cuda.is_available():
    logger.info("CUDA is avaliable, you can enable CUDA")
else:
    logger.info("CUDA is not avaliable, please unable CUDA in config file")

data = pd.read_csv(data_path)
train, val = train_test_split(data, test_size=0.1)

train['token_list'] = train['short'].apply(cut)
train['length'] = train['token_list'].apply(lambda x: len(x))
val['token_list'] = val['short'].apply(cut)
val['length'] = val['token_list'].apply(lambda x: len(x))

logger.info('load sentiment datasets successfully')

k_train = create_set(train['token_list'])
k_val = create_set(val['token_list'])

vocab_list = k_train | k_val
vocab_list = list(vocab_list)

vocab = myVocab(w2v_path, vocab_list)
logger.info("create vocab successfully")

def transform(l1):
    global vocab
    l2 = []
    for word in l1:
        l2.append(vocab.get_id(word))
    return l2

train['token'] = train['token_list'].apply(transform)
val['token'] = val['token_list'].apply(transform)

logger.info("construct model")
model = my_model(vocab.embeddings)
model = model.to(device)
loss = torch.nn.CrossEntropyLoss()
pad_id = vocab.get_id(vocab.pad_token)

val_pad_len = min(max_len, max(list(val['length'])))
val['token_ids'] = [(ids + [pad_id] * (val_pad_len - len(ids)))[: val_pad_len]
                           for ids in val['token']]

logger.info('setting optimizer')
optimizer_lr = 0.001
optimizer_param = filter(lambda p: p.requires_grad, model.parameters()) # filter the parameters required to grad
optimizer = torch.optim.Adam(optimizer_param)

def train_on_model(model, criterion, optimizer, batch_data, epoch, device, batch_cnt, writer, batch_size, part_name, eval_data):
    batch_cnt = batch_cnt
    sum_loss = 0.
    eval_p = eval_data['token_ids']
    eval_p = list(eval_p)
    eval_p = torch.from_numpy(np.array(eval_p)).type(torch.long).to(device)
    eval_length = eval_data['length']
    eval_label = eval_data[part_name]
    eval_label = list(eval_label)
    eval_label = torch.from_numpy(np.array(eval_label)).type(torch.long)
    for i, batch in enumerate(batch_data):
        p = batch['token_ids']
        p = list(p)
#        pdb.set_trace()
        p_length = batch['length'] # load data
        p_length = list(p_length)
        label = batch[part_name]
        label = list(label)

        p1 = torch.from_numpy(np.array(p)).type(torch.long).to(device)  # [batch_size * sequence_length]
        p1_length = torch.from_numpy(np.array(p_length)).type(torch.long).to(device)

        label1 = torch.from_numpy(np.array(label)).type(torch.long).to(device)

        optimizer.zero_grad()
        label_prop = model.forward(p1)
#        pdb.set_trace()

        loss = criterion.forward(label_prop, label1)  # ans_range_prop [batch_size, answer_len, prob]
        loss.backward()  # label_one_hot  [batch_size, answer_len]

        optimizer.step()

        batch_loss = loss.item()
        sum_loss += batch_loss * label1.shape[0]

        if i % 10 == 0:
            model.eval()
#            pdb.set_trace()
            label_prop1 = label_prop.cpu().detach().numpy()
            label_pre1 = np.argmax(label_prop1, axis=1)
#            pdb.set_trace()
            batch_f1 = f1_score(label_pre1, label1.cpu(), average="macro")
            val_prob = model.forward(eval_p)
            val_prob1 = val_prob.cpu().detach().numpy()
            val_pre1 = np.argmax(val_prob1, axis=1)
            eval_loss = criterion.forward(val_prob, eval_label.to(device))
            eval_loss_show = eval_loss.item()
            eval_f1 = f1_score(val_pre1, eval_label, average="macro")
            eval_pre = precision_score(val_pre1, eval_label, average="macro")
            eval_recall = recall_score(val_pre1, eval_label, average="macro")
            logger.info('epoch=%d, batch=%d/%d, loss=%.5f, batch_f1=%.5f, val_loss=%.5f, val_f1=%.5f, val_pre=%.5f, val_recall=%.5f' % (epoch, i, batch_cnt, batch_loss, batch_f1, eval_loss_show, eval_f1, eval_pre, eval_recall))
            writer.add_scalar('scalar/batch_loss', batch_loss, i + (epoch - 1) * batch_size)
            writer.add_scalar('scalar/eval_loss_show', eval_loss_show, i + (epoch - 1) * batch_size)
            model.train()
        del batch, label_prop, loss
    return sum_loss

writer = SummaryWriter(log_dir='/data1/hjw/text_excavation/base_writter')

#train_batches = get_mini_batch(train, batch_size, shuffle=True, max_len=max_len)
for epoch in range(1, epochs + 1):
    logger.info('Training the model for epoch {}'.format(epoch))
    #train
    model.train()

#    total_num, total_loss = 0, 0     #train for an epoch
#    log_every_n_batch, n_batch_loss = 50, 0
    train_batches = get_mini_batch(train, batch_size, shuffle=True, max_len=max_len, pad_id=pad_id)
    batch_cnt = int(len(train)/batch_size)
    sum_loss = train_on_model(model=model, criterion=loss, optimizer=optimizer, batch_data=train_batches,
                              epoch = epoch, device = device, batch_cnt = batch_cnt, writer=writer, batch_size = batch_size, part_name='score', eval_data=val)
    logger.info('epoch=%d, sum_loss = %.5f'%(epoch, sum_loss))
writer.close()
