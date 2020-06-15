# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from os.path import join
import abc
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from reco_utils.recommender.deeprec.deeprec_utils import cal_metric
from utils import cal_metric


__all__ = ["BaseModel"]


class BaseModel:
    

    def __init__(self, hparams, iterator_creator, initepoch=1, restore_path='', save_path='./models/glove_nrms',seed=None):
        
        self.seed = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.train_iterator = iterator_creator(hparams, hparams.npratio)
        self.test_iterator = iterator_creator(hparams, 0)

        self.hparams = hparams
        self.model, self.scorer = self._build_graph()

        self.loss = self._get_loss()
        self.train_optimizer = self._get_opt()


        # self.initepoch=2
        # self.restore_path='/home/shuqilu/Recommenders/models/bert_nrms1'
        # self.save_path='/home/shuqilu/Recommenders/models/bert_nrms'

        self.initepoch=initepoch
        self.restore_path=restore_path
        self.save_path=save_path

        self.sess=tf.keras.backend.get_session()
        self.saver=tf.train.Saver()
        variables = tf.contrib.framework.get_variables_to_restore()
        print('????variables',variables)
        if self.restore_path!='':
            saver2=tf.train.Saver(variables)
            # saver2.restore(self.sess,'/home/shuqilu/Recommenders/models/bert_nrms1')
            saver2.restore(self.sess,self.restore_path)

        self.model.compile(loss=self.loss, optimizer=self.train_optimizer)
        

        
        
        # set GPU use with demand growth
        gpu_options = tf.GPUOptions(allow_growth=True)


    @abc.abstractmethod
    def _build_graph(self):
        
        pass

    @abc.abstractmethod
    def _get_input_label_from_iter(self, batch_data):
       
        pass

    


    def _get_loss(self):
        
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif self.hparams.loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _get_opt(self):
        
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adam":
            train_opt = keras.optimizers.Adam(lr=lr)

        return train_opt

    def _get_pred(self, logit, task):
        
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def train(self, train_batch_data):
        
        train_input, train_label = self._get_input_label_from_iter(train_batch_data)
        # print(len(train_input),type(train_label))
        # print(train_input[0].shape,train_input[1].shape,train_input[2].shape,train_input[3].shape)
        # print(train_input[4].shape,train_input[5].shape,train_input[6].shape)
        # for item in train_input:
        #     print(item.shape,)
        # print([item.shape for item in train_input])
        # print(train_label.shape)
        # print(self.test1([train_input[0],train_input[1],train_input[2],train_input[3],train_input[4],train_input[5],train_input[6],train_input[7]]))
        #print(train_input[3])
        # temp_x=self.test1.predict_on_batch([np.asarray([[1045,1045,1045,0,0,0,0,0,0,0]], dtype=np.int32),np.asarray([[1,1,1,0,0,0,0,0,0,0]], dtype=np.int32),np.asarray([[0,0,0,0,0,0,0,0,0,0]], dtype=np.int32)])
        # print('???test1: ',temp_x)
        #print(temp_x[0].shape)

        #(64, 1) (64, 1) (64, 50, 10) (64, 5, 10)
        rslt = self.model.train_on_batch(train_input, train_label)
        return rslt

    def eval(self, eval_batch_data):
        
        eval_input, eval_label = self._get_input_label_from_iter(eval_batch_data)
        #print('eval...')

        # print([item.shape for item in eval_input])
        # print(eval_label.shape)
        #print('eval: ',len(eval_input),type(eval_label))
        #print(eval_input[0].shape,eval_input[1].shape,eval_input[2].shape,eval_input[3].shape)
        # print(eval_input[4].shape,eval_input[5].shape,eval_input[6].shape,eval_input[7].shape)
        imp_index = eval_input[0]
        pred_rslt = self.scorer.predict_on_batch(eval_input)

        return pred_rslt, eval_label, imp_index

    def fit(self, train_file, valid_file, test_file=None):
        
        print('train...')
        for epoch in range(self.initepoch, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            train_start = time.time()

            for batch_data_input in self.train_iterator.load_data_from_file(train_file):
                step_result = self.train(batch_data_input)
                step_data_loss = step_result

                epoch_loss += step_data_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, epoch_loss, step_data_loss
                        )
                    )

            # self.saver.save(self.sess,'/home/shuqilu/Recommenders/models/bert_nrms'+str(epoch))
            self.saver.save(self.sess,self.save_path+str(epoch))

            train_end = time.time()
            train_time = train_end - train_start

            eval_start = time.time()

            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step)]
                ]
            )

            eval_res = self.run_eval(valid_file)
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            if test_file is not None:
                test_res = self.run_eval(test_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )

        return self

    def group_labels(self, labels, preds, group_keys):
        

        all_keys = list(set(group_keys))
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_labels, all_preds

    def run_eval(self, filename):
        
        preds = []
        labels = []
        imp_indexes = []
        print('eval...')

        for batch_data_input in self.test_iterator.load_data_from_file(filename):
            step_pred, step_labels, step_imp_index = self.eval(batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
            imp_indexes.extend(np.reshape(step_imp_index, -1))

        group_labels, group_preds = self.group_labels(labels, preds, imp_indexes)
        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        return res
