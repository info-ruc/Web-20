from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
#from bert_layer import BertLayer


import tensorflow.keras.backend as K

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from base_model import BaseModel
from layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel"]


class Glove_NRMSModel(BaseModel):
    

    def __init__(self, hparams, iterator_creator, initepoch=1, restore_path='', save_path='./models/glove_nrms', seed=None):
        

        #BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.word_size,self.word_emb_dim=self.word2vec_embedding.shape[0],self.word2vec_embedding.shape[1]
        #self.bert_module = hub.Module(BERT_MODEL_HUB,trainable=True)
        self.hparam = hparams

        super().__init__(hparams, iterator_creator,initepoch, restore_path, save_path, seed=seed)

    def _init_embedding(self, file_path):
        
        return np.load(file_path).astype(np.float32)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["impression_index_batch"],
            batch_data["user_index_batch"],
            batch_data["clicked_news_batch"],
            batch_data["candidate_news_batch"],
            #batch_data["input_ids"],
            batch_data["c_input_masks"],
            batch_data["c_segments"],
            batch_data["h_input_masks"],
            batch_data["h_segments"],
            batch_data["c_length"],
            batch_data["h_length"],
            batch_data["all_his"],
            batch_data["all_can"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_graph(self):
        
        hparams = self.hparams
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )
        h_input_masks = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )
        h_segments = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )
        h_length = keras.Input(
            shape=(hparams.his_size, 1), dtype="int32"
        )
        all_his = keras.Input(shape=(1,), dtype="int32")


        # his_input_title_reshape=layers.Reshape((hparams.doc_size,))(
        #     his_input_title
        # )
        # h_input_masks_reshape=layers.Reshape((hparams.doc_size,))(
        #     h_input_masks
        # )
        # h_segments_reshape=layers.Reshape((hparams.doc_size,))(
        #     h_segments
        # )
        his_input_title_reshape=K.reshape( his_input_title, (-1, hparams.doc_size)) 
        h_input_masks_reshape=K.reshape( h_input_masks, (-1, hparams.doc_size)) 
        h_segments_reshape=K.reshape( h_segments, (-1, hparams.doc_size)) 
        h_length_reshape=K.reshape( h_length, (-1, 1))      

        #click_title_presents = layers.TimeDistributed(titleencoder)([his_input_title,h_input_masks,h_segments])
        click_title_presents1 = titleencoder([his_input_title_reshape,h_input_masks_reshape,h_segments_reshape,h_length_reshape])
        print('???1: ',click_title_presents1)
        click_title_presents=K.reshape(click_title_presents1,(-1,hparams.his_size,click_title_presents1.shape[-1]))
        print('???2: ',click_title_presents)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y,all_his)

        


        model = keras.Model([his_input_title,h_input_masks,h_segments,h_length,all_his], user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):
        
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.doc_size,), dtype="int32")
        embedded_sequences_title = embedding_layer(sequences_input_title)
        input_ids=sequences_input_title
        input_mask=keras.Input(shape=(hparams.doc_size,), dtype="int32")
        segment_ids=keras.Input(shape=(hparams.doc_size,), dtype="int32")
        input_len=keras.Input(shape=(1,), dtype="int32")
        # bert_inputs = dict(
        #       input_ids=input_ids,
        #       input_mask=input_mask,
        #       segment_ids=segment_ids)
        bert_inputs = [input_ids, input_mask, segment_ids]
        # bert_path='https://tfhub.dev/google/small_bert/bert_uncased_L-12_H-128_A-2/1'
        # mybert = hub.Module(
        #     bert_path,
        #     trainable=True,
        #     #name="{}_module".format(self.name)
        # )
        # bert_inputs = dict(
        #     input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        # )
        # bert_output = mybert(inputs=bert_inputs, signature="tokens", as_dict=True)['sequence_output']
        
        #print('???sequences_input_title: ',sequences_input_title)
        #bert_inputs = [sequences_input_title[:,x,:] for x in range(3)]
        #bert_inputs = sequences_input_title[:,0]
        #print("???bert_inputs: ",bert_inputs)
        # bert_output = BertLayer(n_fine_tune_layers=6)(bert_inputs)
        # embedded_sequences_title=bert_output
        
        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y,input_len,input_len])
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y,input_len)

        #self.test1=keras.Model([sequences_input_title,input_mask,segment_ids], bert_output, name="test1")
        #self.test1=K.function([sequences_input_title,input_mask,segment_ids], [bert_output])

        model = keras.Model([sequences_input_title,input_mask,segment_ids,input_len], pred_title, name="news_encoder")
        # model = keras.Model([sequences_input_title], pred_title, name="news_encoder")
        #model = keras.Model([sequences_input_title], bert_inputs, name="news_encoder")
        return model

    def _build_nrms(self):
        
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.doc_size), dtype="int32"
        )

        # input_ids = keras.Input(
        #     shape=(hparams.npratio + 1, hparams.doc_size), dtype="int32"
        # )
        c_input_masks = keras.Input(
            shape=(hparams.npratio + 1, hparams.doc_size), dtype="int32"
        )
        c_segments = keras.Input(
            shape=(hparams.npratio + 1, hparams.doc_size), dtype="int32"
        )
        c_length = keras.Input(
            shape=(hparams.npratio + 1, 1), dtype="int32"
        )
        h_input_masks = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )
        h_segments = keras.Input(
            shape=(hparams.his_size, hparams.doc_size), dtype="int32"
        )
        h_length=keras.Input(
            shape=(hparams.his_size, 1), dtype="int32"
        )
        one_input_masks = keras.Input(
            shape=(1, hparams.doc_size), dtype="int32"
        )
        one_segments = keras.Input(
            shape=(1, hparams.doc_size), dtype="int32"
        )
        one_length = keras.Input(
            shape=(1, 1), dtype="int32"
        )


        pred_input_title_one = keras.Input(shape=(1, hparams.doc_size,), dtype="int32")
        pred_title_one_reshape =K.reshape(pred_input_title_one,(-1,hparams.doc_size)) 

        imp_indexes = keras.Input(shape=(1,), dtype="int32")
        user_indexes = keras.Input(shape=(1,), dtype="int32")
        all_his = keras.Input(shape=(1,), dtype="int32")
        all_can = keras.Input(shape=(1,), dtype="int32")

        embedding_layer = layers.Embedding(
            #hparams.word_size,
            #hparams.word_emb_dim,
            self.word_size,
            self.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer)
        
        newsencoder = titleencoder


        #=userencoder
        #merge_input=
        
        #news_present = layers.TimeDistributed(newsencoder)([pred_input_title,c_input_masks,c_segments])
        # pred_input_title_reshape=layers.Reshape((hparams.npratio + 1,hparams.doc_size,))(
        #     pred_input_title
        # )
        # c_input_masks_reshape=layers.Reshape((hparams.npratio + 1,1,hparams.doc_size))(
        #     c_input_masks
        # )
        # c_segments_reshape=layers.Reshape((hparams.npratio + 1,1,hparams.doc_size))(
        #     c_segments
        # )
        pred_input_title_reshape=K.reshape( pred_input_title, (-1, hparams.doc_size))
        print('???pred_input_title_reshape: ',pred_input_title_reshape)
        c_input_masks_reshape=K.reshape(c_input_masks, (-1, hparams.doc_size))
        c_segments_reshape=K.reshape( c_segments, (-1, hparams.doc_size))
        c_length_reshape=K.reshape( c_length, (-1, 1))
            
        
        # news_present = newsencoder([pred_input_title_reshape,c_input_masks_reshape,c_segments_reshape])
        # print(news_present)
        # news_present=layers.Reshape((hparams.npratio + 1,news_present.shape[-1]))(
        #     news_present
        # )
        # print(news_present)
        # merge_input=keras.layers.concatenate([pred_input_title_reshape,c_input_masks_reshape],axis=2)
        # print('???merge_input: ',merge_input)
        # merge_input=keras.layers.concatenate([merge_input,c_segments_reshape],axis=2)
        # print('???merge_input2: ',merge_input)
        # merge_input=keras.Input(shape=(5,hparams.doc_size,), dtype="int32")
        # print('???merge_input3: ',merge_input)

        news_present1 = newsencoder([pred_input_title_reshape,c_input_masks_reshape,c_segments_reshape,c_length_reshape])
        print("???news_present1: ",news_present1)
        news_present=K.reshape( news_present1, (-1, hparams.npratio + 1,news_present1.shape[-1]))

        #news_present = layers.TimeDistributed(newsencoder)(merge_input)
        print("???news_present2: ",news_present)

        userencoder = self._build_userencoder(titleencoder)
        user_present = userencoder([his_input_title,h_input_masks,h_segments,h_length,all_his])

        one_input_masks_reshape=K.reshape(one_input_masks,(-1,hparams.doc_size))
        one_segments_reshape=K.reshape(one_segments,(-1,hparams.doc_size))
        one_length_reshape=K.reshape(one_length,(-1,1))

        news_present_one = newsencoder([pred_title_one_reshape,one_input_masks_reshape,one_segments_reshape,one_length_reshape])

        preds = layers.Dot(axes=-1)([news_present, user_present])

        mask = K.one_hot(indices=all_can[:, 0], num_classes=K.shape(preds)[1])
        mask = 1 - K.cumsum(mask, axis=1)
        preds=preds - (1 - mask) * 1e12

        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [imp_indexes, user_indexes, his_input_title, pred_input_title,c_input_masks,c_segments,h_input_masks,h_segments,c_length,h_length,all_his,all_can], preds,name="model"
        )
        scorer = keras.Model(
            [imp_indexes, user_indexes, his_input_title, pred_input_title_one,one_input_masks,one_segments,h_input_masks,h_segments,one_length,h_length,all_his,all_can], pred_one,name="score"
        )

        # self.test1=keras.Model([imp_indexes, user_indexes, his_input_title, pred_input_title,c_input_masks,c_segments,h_input_masks,h_segments], news_present, name="test")

        return model, scorer




