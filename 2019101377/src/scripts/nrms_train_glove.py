import sys
#sys.path.append("../../")
from newsrec_utils import prepare_hparams
from bert_nrms2 import Glove_NRMSModel
from news_iterator import NewsIterator
import papermill as pm
from tempfile import TemporaryDirectory
import tensorflow as tf
import os

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

# tmpdir = TemporaryDirectory()#/tmp/tmp_4m0jy0u
# data_path = tmpdir.name
data_path='./data'
print(data_path)
# yaml_file = os.path.join(data_path, r'./data/nrms.yaml')
# train_file = os.path.join(data_path, r'./data/MINDsmall_train/train_ms.txt')
# valid_file = os.path.join(data_path, r'./data/MINDsmall_train/valid_ms.txt')
# wordEmb_file = os.path.join(data_path, r'./data/embedding.npy')
yaml_file = './nrms.yaml'
train_file = './data/glove/train_ms_glove.txt'
valid_file = './data/glove/valid_ms_glove.txt'
wordEmb_file = './data/glove/embedding.npy'
# if not os.path.exists(yaml_file):
#     download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/newsrec/', data_path, 'nrms.zip')

epochs=10
seed=42

hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=epochs)
print(hparams)

iterator = NewsIterator
#model = NRMSModel(hparams, iterator, seed=seed)

sess = tf.keras.backend.get_session()

# model = BERT_NRMSModel(hparams, iterator, seed=seed)
model = Glove_NRMSModel(hparams, iterator, initepoch=1, restore_path='', save_path='./models/glove_nrms',seed=seed)
writer = tf.summary.FileWriter("tb", sess.graph)

#print(model.run_eval(valid_file))
# res_syn = model.run_eval(valid_file)
# print(res_syn)
# 
model.fit(train_file, valid_file)
# print('training ok...')
res_syn = model.run_eval(valid_file)
# 
print(res_syn)
#pm.record("res_syn", res_syn)