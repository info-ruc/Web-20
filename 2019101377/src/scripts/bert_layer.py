import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
#import keras
import tensorflow.keras.backend as K
#from tensorflow.keras import layers


from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime



class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        # self.output_size = 768
        self.output_size = 128
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #bert_path='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
        #https://tfhub.dev/s?q=bert
        bert_path='https://tfhub.dev/google/small_bert/bert_uncased_L-12_H-128_A-2/1'
        #bert_path='/home/shuqilu/Recommenders/data/bert_model'
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            #name="{}_module".format(self.name)
            #name=""
        )
        trainable_vars = self.bert.variables
        
        # Remove unused layers
        #print([var.name for var in trainable_vars ])
        trainable_vars = [var for var in trainable_vars if not ("/cls/" in var.name or "/pooler/" in var.name )]
        #print('???',trainable_vars)
        
        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
        
        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        
        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)
        
        self.initialize_module()
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[#"pooled_output",
            "sequence_output"
        ]
        print('???result',result)
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

    def initialize_module(self):
        sess = tf.keras.backend.get_session()
      
        vars_initialized = sess.run([tf.is_variable_initialized(var) 
                                   for var in self.bert.variables])

        uninitialized = []
        for var, is_initialized in zip(self.bert.variables, vars_initialized):
            if not is_initialized:
                uninitialized.append(var)
        #print('???uninitialized: ',uninitialized)
        if len(uninitialized):
            sess.run(tf.variables_initializer(uninitialized))
            #others=[var for var in uninitialized if var.name!='module/bert/embeddings/LayerNorm/beta']

            # saver_vgg = tf.train.Saver(uninitialized)
            # saver_vgg.restore(sess, '/home/shuqilu/Recommenders/data/bert_model.ckpt')
            #sess.run(tf.variables_initializer(uninitialized))
        # print('...........................initialized....................................')
        # for var in uninitialized:
        #     if 'word_embeddings' in var.name:
        #         print(sess.run(var))
        #不知道为什么initial了之后还是pre-train的参数，难道我对initial的原理理解地有一点问题？








