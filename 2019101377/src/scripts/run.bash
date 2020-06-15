#数据下载 https://msnews.github.io/ download training set and valid set into ./data

unzip  MINDsmall_dev.zip
unzip  MINDsmall_train.zip




mkdir ./models
mkdir ./data
cd data && mkdir glove

cd data/glove
git clone http://github.com/stanfordnlp/glove
cd glove && make
cp ../../glove_word.sh ./
chmod +x ./glove_word.sh


# 环境准备 python>=3.7
pip install tensorflow-gpu==1.4.0

#数据处理：
python data_process.py

#训练glove编码的模型:
CUDA_VISIBLE_DEVICES=2  nohup python nrms_train_glove.py > log_glove_nrms.txt 2 >&1 &
#训练bert编码的模型:
CUDA_VISIBLE_DEVICES=0  nohup python nrms_train.py > log_bert_nrms.txt 2 >&1 &
