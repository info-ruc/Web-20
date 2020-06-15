from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import csv
import numpy as np
import os



BERT_MODEL_HUB = 'https://tfhub.dev/google/small_bert/bert_uncased_L-12_H-128_A-2/1'

def precess_csv(tokenizer,data_path):
    
    # w_train=open('train_MIND.txt','w')
    # w_dev=open('valid_MIND.txt','w')
    csvFile = open(data_path+"news.tsv", "r")
    reader = csv.reader(csvFile,delimiter='\t')
    w=open(data_path+'news_token.txt','w')
    w2=open(data_path+'news_token_features.txt','w')
    news_id={}
    for item in reader:
        tokens=tokenizer.tokenize(item[3])
        w.write(item[0]+'\t'+"\t".join(tokens)+'\n')
        tokens=["[CLS]"]+tokens
        tokens.append("[SEP]")
        features=tokenizer.convert_tokens_to_ids(tokens)
        w2.write(item[0]+'\t'+"\t".join([str(x) for x in features])+'\n')
        news_id[item[0]]=features

    return news_id

def read_features(data_path):
    news_id={}
    news_len={}
    max_length=10
    f=open(data_path+'news_token_features.txt','r')
    for line in f:
        line=line.strip().split('\t')
        features=line[1:]
        if len(features)>max_length:
            features=features[:max_length]
            news_len[line[0]]=len(features)
        else:
            news_len[line[0]]=len(features)
            features=features+['0']*(max_length - len(features))
        news_id[line[0]]=features
    return news_id,news_len

def process_data_train(news_id,news_len,mode="train"):
    
    # if mode=='valid':
    #     csvFile = open("MINDsmall_dev/behaviors.tsv", "r")
    #     reader = csv.reader(csvFile,delimiter='\t')
    #     w=open('valid_ms.txt','w')
    #elif mode=="train":
    csvFile = open("./data/MINDsmall_train/behaviors.tsv", "r")
    reader = csv.reader(csvFile,delimiter='\t')
    w=open('./data/train_ms.txt','w')
    imp_id=0
    max_his=0
    min_his=100000
    max_can=0
    min_can=10000
    select_his=50
    candidate_each_row=4
    max_length=10
    max_his_length=50
    imp_id=0
    for item in reader:
        #print(item[2])
        #assert 1==0
        user_id=int(item[0][1:])#可能要改成int型
        #如果是test可能要改成一个candidate一条数据
        history=item[2].split(' ')
        candidate=item[3].split(' ')
        # his_len=len(history)
        # can_len=len(candidate)
        # if his_len>max_his:
        #     max_his=his_len
        # if his_len<min_his:
        #     min_his=his_len
        # if can_len>max_can:
        #     max_can=can_len
        # if can_len<min_can:
        #     min_can=can_len
        # if can_len==299:
        #     print(item[3])

        label_list=[x[-1] for x in candidate]
        i=0
        candidate_pad=0
        #print('???',history)
        print('row id ',imp_id,156965)
        #print('???',label_list)
        hislen=[news_len[x] for x in history if x !='']
        history=[news_id[x] for x in history if x !='']

        allhis=0
        if len(history)>max_his_length:
            history=history[-max_his_length:]
            allhis=max_length
            hislen=hislen[-max_his_length:]

        else:
            allhis=len(history)
            hislen=hislen+[0]*(max_his_length-len(history))
            history=history+[['0']*max_length]*(max_his_length-len(history))
            #0好像不太好

        neg_list=[]
        pos_list=[]
        for i in range(len(label_list)):
            if label_list[i]=='1':
                pos_list.append(i)
            else:
                assert label_list[i]=='0'
                neg_list.append(i)

        for item in pos_list:
            i=0
            while i<len(neg_list):
                if i+candidate_each_row>len(neg_list):
                    candidate_pad=i+candidate_each_row-len(neg_list)
                    #w.write(' '.join(label_list[i:i+candidate_each_row]+[str(2)]*candidate_pad)+' ')
                    w.write(' '.join(['1','0','0','0','0'])+' ')
                    w.write('Impression:'+str(imp_id)+' User:'+str(user_id)+' ')
                    canlen=[]
                    #print('???',news_id[candidate[item][:-2]])
                    w.write('CandidateNews'+str(0)+':'+','.join(news_id[candidate[item][:-2]])+' ')
                    canlen.append(news_len[candidate[item][:-2]])
                    cid=1
                    for x in range(i,len(neg_list)):
                        w.write('CandidateNews'+str(cid)+':'+','.join(news_id[candidate[neg_list[x]][:-2]])+' ')
                        canlen.append(news_len[candidate[neg_list[x]][:-2]])
                        cid+=1
                    allcan=len(neg_list)-i+1
                    for x in range(len(neg_list),i+candidate_each_row):
                        w.write('CandidateNews'+str(cid)+':'+','.join(['0']*max_length)+' ')
                        canlen.append(0)
                        cid+=1 
                    for x in range(len(history)):
                        w.write('ClickedNews'+str(x)+':'+','.join(history[x])+' ')
                    
                    w.write('Hislen:'+','.join([str(x) for x in hislen ])+' ')
                    w.write('Canlen:'+','.join([str(x) for x in canlen])+' ')
                    w.write('Allhis:'+str(allhis)+' ')
                    w.write('Allcan:'+str(allcan)+' ')
                    w.write('\n')
                else:
                    w.write(' '.join(['1','0','0','0','0'])+' ')
                    w.write('Impression:'+str(imp_id)+' User:'+str(user_id)+' ')
                    canlen=[]
                    w.write('CandidateNews'+str(0)+':'+','.join(news_id[candidate[item][:-2]])+' ')
                    canlen.append(news_len[candidate[item][:-2]])
                    cid=1
                    for x in range(i,i+candidate_each_row):
                        w.write('CandidateNews'+str(cid)+':'+','.join(news_id[candidate[neg_list[x]][:-2]])+' ')
                        canlen.append(news_len[candidate[neg_list[x]][:-2]])
                        cid+=1
                    allcan=candidate_each_row+1
                    # for x in range(len(label_list),i+candidate):
                    #     w.write('CandidateNews'+str(cid)+':'+','.join(['0']*max_length)+' ')
                    #     cid+=1
                    for x in range(len(history)):
                        w.write('ClickedNews'+str(x)+':'+','.join(history[x])+' ')

                    w.write('Hislen:'+','.join([str(x) for x in hislen ])+' ')
                    w.write('Canlen:'+','.join([str(x) for x in canlen])+' ')
                    w.write('Allhis:'+str(allhis)+' ')
                    w.write('Allcan:'+str(allcan)+' ')
                    w.write('\n')
                i+=candidate_each_row
            
        imp_id+=1
        #print('!!!',imp_id)
    # print('max_his: ',max_his,' min_his: ',min_his)
    # print('max_can: ',max_can,' min_can: ',min_can)

def process_data_valid(news_id,news_len,mode="train"):
    
    # if mode=='valid':
    #     csvFile = open("MINDsmall_dev/behaviors.tsv", "r")
    #     reader = csv.reader(csvFile,delimiter='\t')
    #     w=open('valid_ms.txt','w')
    #elif mode=="train":
    csvFile = open("./data/MINDsmall_dev/behaviors.tsv", "r")
    reader = csv.reader(csvFile,delimiter='\t')
    w=open('./data/valid_ms.txt','w')
    imp_id=0
    max_his=0
    min_his=100000
    max_can=0
    min_can=10000
    select_his=50
    candidate_each_row=4
    max_length=10
    max_his_length=50
    for item in reader:
        #print(item[2])
        #assert 1==0
        user_id=int(item[0][1:])#可能要改成int型
        #如果是test可能要改成一个candidate一条数据
        history=item[2].split(' ')
        candidate=item[3].split(' ')
        print('???',imp_id,73152)

        label_list=[x[-1] for x in candidate]
        i=0
        candidate_pad=0
        hislen=[news_len[x] for x in history if x!='']
        history=[news_id[x] for x in history if x!='']

        allhis=0
        if len(history)>max_his_length:
            history=history[-max_his_length:]
            allhis=max_length
            hislen=hislen[-max_his_length:]

        else:
            allhis=len(history)
            hislen=hislen+[0]*(max_his_length-len(history))
            history=history+[['0']*max_length]*(max_his_length-len(history))

        for i in range(len(label_list)):
               
            w.write(label_list[i]+' ')
            w.write('Impression:'+str(imp_id)+' User:'+str(user_id)+' ')
            cid=0
            canlen=[]
            w.write('CandidateNews'+str(cid)+':'+','.join(news_id[candidate[i][:-2]])+' ')
            canlen.append(news_len[candidate[i][:-2]])
            for x in range(len(history)):
                w.write('ClickedNews'+str(x)+':'+','.join(history[x])+' ')

            w.write('Hislen:'+','.join([str(x) for x in hislen ])+' ')
            w.write('Canlen:'+','.join([str(x) for x in canlen])+' ')
            w.write('Allhis:'+str(allhis)+' ')
            w.write('Allcan:'+str(1)+' ')
               
            w.write('\n')
        imp_id+=1


def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

def get_glove_corpus():
	f1=open('./data/MINDsmall_train/news_token.txt','r')
	f2=open('./data/MINDsmall_dev/news_token.txt','r')
	w=open('./data/glove/corpus.txt','w')
	for line in f1:
		w.write(' '.join(line.strip().split('\t')[1:])+'\n')
	for line in f2:
		w.write(' '.join(line.strip().split('\t')[1:])+'\n')
	w.close()

def generate():
	word_dict={}
	word_id={}
	
	with open('./data/glove/vocab.txt', 'r') as f:
		words = [x.rstrip().split(' ')[0] for x in f.readlines()]
	with open('./data/glove/vectors.txt', 'r') as f:
		vectors = {}
		for line in f:
			vals = line.rstrip().split(' ')
			vectors[vals[0]] = [float(x) for x in vals[1:]]
	#myw=open('./wordvec.txt','w')
	#vocab_size = len(words)
	vocab = {w: idx for idx, w in enumerate(words)}
	ivocab = {idx: w for idx, w in enumerate(words)}

	vocab_size = len(vectors)
	vector_dim = len(vectors[ivocab[0]])
	W = np.zeros((vocab_size, vector_dim))
	#word_vector=np.zeros((vocab_size, vector_dim))
	for word, v in vectors.items():
		if word == '<unk>':
			#continue
			W[len(words), :] = v
		else:
			W[vocab[word], :] = v
		#word_dict[word]=v

	# vocab['<start>']=len(word_dict)
	# vocab['<end>']=len(word_dict)+1
	# word_dict['<start>']=np.random.randn(50)
	# word_dict['<end>']=np.random.randn(50)
	# pickle.dump(vocab,open('word_id','wb'))
	# pickle.dump(word_dict,open('word_dict','wb'))
	np.save('./data/glove/embedding.npy',W)
	f1=open('./data/MINDsmall_train/news_token.txt','r')
	w1=open('./data/glove/news_token_features_train.txt','w')
	count=0
	for line in f1:
		line=line.strip().split('\t')
		w1.write(line[0]+'\t')
		for item in line[1:]:
			if item in vocab:
				w1.write(str(vocab[item])+'\t')
			else:
				w1.write(str(len(words))+'\t')
				print('unk: ',item)
				count+=1
		w1.write('\n')


	f2=open('./data/MINDsmall_dev/news_token.txt','r')
	w2=open('./data/glove/news_token_features_valid.txt','w')
	for line in f2:
		line=line.strip().split('\t')
		w2.write(line[0]+'\t')
		for item in line[1:]:
			if item in vocab:
				w2.write(str(vocab[item])+'\t')
			else:
				w2.write(str(len(words))+'\t')
				print('unk: ',item)
				count+=1
		w2.write('\n')
	print('???',count)

def read_glove_features(mode='valid'):
    news_id={}
    news_len={}
    max_length=10
    print('mode: ',mode)
    if mode=='train':

        f=open('./data/glove/news_token_features_train.txt','r')
    else:
        f=open('./data/glove/news_token_features_valid.txt','r')
    
    for line in f:
        line=line.strip().split('\t')
        features=line[1:]
        if len(features)>max_length:
            features=features[:max_length]
            news_len[line[0]]=len(features)
        else:
            news_len[line[0]]=len(features)
            features=features+['0']*(max_length - len(features))
        news_id[line[0]]=features
    return news_id,news_len

def process_glove_data_train(news_id,news_len,mode="train"):
    
    # if mode=='valid':
    #     csvFile = open("MINDsmall_dev/behaviors.tsv", "r")
    #     reader = csv.reader(csvFile,delimiter='\t')
    #     w=open('valid_ms.txt','w')
    #elif mode=="train":
    csvFile = open("./data/MINDsmall_train/behaviors.tsv", "r")
    reader = csv.reader(csvFile,delimiter='\t')
    w=open('./data/glove/train_ms_glove.txt','w')
    imp_id=0
    max_his=0
    min_his=100000
    max_can=0
    min_can=10000
    select_his=50
    candidate_each_row=4
    max_length=10
    max_his_length=50
    imp_id=0
    for item in reader:
        #print(item[2])
        #assert 1==0
        user_id=int(item[0][1:])#可能要改成int型
        #如果是test可能要改成一个candidate一条数据
        history=item[2].split(' ')
        candidate=item[3].split(' ')

        label_list=[x[-1] for x in candidate]
        i=0
        candidate_pad=0
        #print('???',history)
        print('???',imp_id,156965)
        #print('???',label_list)
        hislen=[news_len[x] for x in history if x !='']
        history=[news_id[x] for x in history if x !='']

        allhis=0
        if len(history)>max_his_length:
            history=history[-max_his_length:]
            allhis=max_length
            hislen=hislen[-max_his_length:]

        else:
            allhis=len(history)
            hislen=hislen+[0]*(max_his_length-len(history))
            history=history+[['0']*max_length]*(max_his_length-len(history))
            #0好像不太好

        neg_list=[]
        pos_list=[]
        for i in range(len(label_list)):
            if label_list[i]=='1':
                pos_list.append(i)
            else:
                assert label_list[i]=='0'
                neg_list.append(i)

        for item in pos_list:
            i=0
            while i<len(neg_list):
                if i+candidate_each_row>len(neg_list):
                    candidate_pad=i+candidate_each_row-len(neg_list)
                    #w.write(' '.join(label_list[i:i+candidate_each_row]+[str(2)]*candidate_pad)+' ')
                    w.write(' '.join(['1','0','0','0','0'])+' ')
                    w.write('Impression:'+str(imp_id)+' User:'+str(user_id)+' ')
                    canlen=[]
                    #print('???',news_id[candidate[item][:-2]])
                    w.write('CandidateNews'+str(0)+':'+','.join(news_id[candidate[item][:-2]])+' ')
                    canlen.append(news_len[candidate[item][:-2]])
                    cid=1
                    for x in range(i,len(neg_list)):
                        w.write('CandidateNews'+str(cid)+':'+','.join(news_id[candidate[neg_list[x]][:-2]])+' ')
                        canlen.append(news_len[candidate[neg_list[x]][:-2]])
                        cid+=1
                    allcan=len(neg_list)-i+1
                    for x in range(len(neg_list),i+candidate_each_row):
                        w.write('CandidateNews'+str(cid)+':'+','.join(['0']*max_length)+' ')
                        canlen.append(0)
                        cid+=1 
                    for x in range(len(history)):
                        w.write('ClickedNews'+str(x)+':'+','.join(history[x])+' ')
                    
                    w.write('Hislen:'+','.join([str(x) for x in hislen ])+' ')
                    w.write('Canlen:'+','.join([str(x) for x in canlen])+' ')
                    w.write('Allhis:'+str(allhis)+' ')
                    w.write('Allcan:'+str(allcan)+' ')
                    w.write('\n')
                else:
                    w.write(' '.join(['1','0','0','0','0'])+' ')
                    w.write('Impression:'+str(imp_id)+' User:'+str(user_id)+' ')
                    canlen=[]
                    w.write('CandidateNews'+str(0)+':'+','.join(news_id[candidate[item][:-2]])+' ')
                    canlen.append(news_len[candidate[item][:-2]])
                    cid=1
                    for x in range(i,i+candidate_each_row):
                        w.write('CandidateNews'+str(cid)+':'+','.join(news_id[candidate[neg_list[x]][:-2]])+' ')
                        canlen.append(news_len[candidate[neg_list[x]][:-2]])
                        cid+=1
                    allcan=candidate_each_row+1
                    # for x in range(len(label_list),i+candidate):
                    #     w.write('CandidateNews'+str(cid)+':'+','.join(['0']*max_length)+' ')
                    #     cid+=1
                    for x in range(len(history)):
                        w.write('ClickedNews'+str(x)+':'+','.join(history[x])+' ')

                    w.write('Hislen:'+','.join([str(x) for x in hislen ])+' ')
                    w.write('Canlen:'+','.join([str(x) for x in canlen])+' ')
                    w.write('Allhis:'+str(allhis)+' ')
                    w.write('Allcan:'+str(allcan)+' ')
                    w.write('\n')
                i+=candidate_each_row
            
        imp_id+=1
        #print('!!!',imp_id)
    # print('max_his: ',max_his,' min_his: ',min_his)
    # print('max_can: ',max_can,' min_can: ',min_can)

def process_glove_data_valid(news_id,news_len,mode="train"):
    
    # if mode=='valid':
    #     csvFile = open("MINDsmall_dev/behaviors.tsv", "r")
    #     reader = csv.reader(csvFile,delimiter='\t')
    #     w=open('valid_ms.txt','w')
    #elif mode=="train":
    csvFile = open("./data/MINDsmall_dev/behaviors.tsv", "r")
    reader = csv.reader(csvFile,delimiter='\t')
    w=open('./data/glove/valid_ms_glove.txt','w')
    imp_id=0
    max_his=0
    min_his=100000
    max_can=0
    min_can=10000
    select_his=50
    candidate_each_row=4
    max_length=10
    max_his_length=50
    for item in reader:
        #print(item[2])
        #assert 1==0
        user_id=int(item[0][1:])#可能要改成int型
        #如果是test可能要改成一个candidate一条数据
        history=item[2].split(' ')
        candidate=item[3].split(' ')
        print('???',imp_id,73152)

        label_list=[x[-1] for x in candidate]
        i=0
        candidate_pad=0
        hislen=[news_len[x] for x in history if x!='']
        history=[news_id[x] for x in history if x!='']

        allhis=0
        if len(history)>max_his_length:
            history=history[-max_his_length:]
            allhis=max_length
            hislen=hislen[-max_his_length:]

        else:
            allhis=len(history)
            hislen=hislen+[0]*(max_his_length-len(history))
            history=history+[['0']*max_length]*(max_his_length-len(history))

        for i in range(len(label_list)):
               
            w.write(label_list[i]+' ')
            w.write('Impression:'+str(imp_id)+' User:'+str(user_id)+' ')
            cid=0
            canlen=[]
            w.write('CandidateNews'+str(cid)+':'+','.join(news_id[candidate[i][:-2]])+' ')
            canlen.append(news_len[candidate[i][:-2]])
            for x in range(len(history)):
                w.write('ClickedNews'+str(x)+':'+','.join(history[x])+' ')

            w.write('Hislen:'+','.join([str(x) for x in hislen ])+' ')
            w.write('Canlen:'+','.join([str(x) for x in canlen])+' ')
            w.write('Allhis:'+str(allhis)+' ')
            w.write('Allcan:'+str(1)+' ')
               
            w.write('\n')
        imp_id+=1

# tokenizer = create_tokenizer_from_hub_module()
# print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))
# process_data("train")
# process_data("valid")








if __name__ == '__main__':
    tokenizer = create_tokenizer_from_hub_module()
    print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))
    print(tokenizer.tokenize("[CLS] This here's an example of using the BERT tokenizer [SEP]"))
    precess_csv(tokenizer,'./data/MINDsmall_train/')
    a,b=read_features('./data/MINDsmall_train/')
    process_data_train(a,b)
    precess_csv(tokenizer,'./data/MINDsmall_dev/')
    a,b=read_features('./data/MINDsmall_dev/')
    process_data_valid(a,b)
    get_glove_corpus()
    os.system('./data/glove/demo.sh')
    generate()
    a,b=read_glove_features(mode='train')
    process_glove_data_train(a,b)
    a,b=read_glove_features(mode='valid')
    process_glove_data_valid(a,b)
    












