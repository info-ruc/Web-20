import pandas as pd
import numpy as np
import config

import tensorflow as tf
import dataLoader

from model_v2 import Model

import random

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import math

 
model = Model()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    train_lstm1, train_lstm2, test_lstm1, \
    test_lstm2, trainy, testy = dataLoader.main()
    #train_cnnmap, test_cnnmap = dataLoader.main()


    print(train_lstm2.shape)
    print(test_lstm2.shape)
    lrmodel = LinearRegression()
    bpnnmodel = MLPRegressor(hidden_layer_sizes=(20, 50, 10), activation='relu',
                         solver='adam', learning_rate='constant', max_iter=50000, early_stopping=True)
    svmmodel = SVR(C=100,gamma=0.001)
    rf = RandomForestRegressor(n_estimators=200,max_depth=5)
    gbdt = GradientBoostingRegressor(n_estimators=200,max_depth=5)


    basicmodel_train = train_lstm1[:,-config.input_size:]
    basicmodel_test = test_lstm1[:, -config.input_size:]

    #lrmodel.fit(basicmodel_train,trainy)
    # bpnnmodel.fit(basicmodel_train,trainy)
    svmmodel.fit(basicmodel_train,trainy)
    rf.fit(basicmodel_train,trainy)
    gbdt.fit(basicmodel_train,trainy)

    #lrpred = lrmodel.predict(basicmodel_test)
    # bpnnpred = bpnnmodel.predict(basicmodel_test)
    svmpred = svmmodel.predict(basicmodel_test)
    rfpred = rf.predict(basicmodel_test)
    gbdtpred = gbdt.predict(basicmodel_test)

    #lrcost = math.sqrt(np.array([(lrpred[i] - testy[i]) ** 2 for i in range(len(basicmodel_test))]).mean())
    # bpnncost = math.sqrt(np.array([(bpnnpred[i] - testy[i]) ** 2 for i in range(len(basicmodel_test))]).mean())
    svmcost = math.sqrt(np.array([(svmpred[i] - testy[i]) ** 2 for i in range(len(basicmodel_test))]).mean())
    rfcost = math.sqrt(np.array([(rfpred[i] - testy[i]) ** 2 for i in range(len(basicmodel_test))]).mean())
    gbdtcost = math.sqrt(np.array([(gbdtpred[i] - testy[i]) ** 2 for i in range(len(basicmodel_test))]).mean())

    #print("lr:",lrcost)
    # print('bpnn:',bpnncost)
    print('svmcost:',svmcost)
    print(','.join([str(svmpred[i]) for i in range(len(svmpred))]))
    print('rfcost:', rfcost)
    print(','.join([str(rfpred[i]) for i in range(len(rfpred))]))
    print('gbdtcost:', gbdtcost)
    print(','.join([str(gbdtpred[i]) for i in range(len(gbdtpred))]))

    #pd.DataFrame(train_lstm1).to_csv('data/sss.csv')
    best = 0xffff

    for i in range(config.eposide):
        num_batch = train_lstm1.shape[0] // config.batch_size
        rlist = list(range(train_lstm1.shape[0]))
        random.shuffle(rlist)
        rg = 0
        for j in range(num_batch):
            lstm1 = train_lstm1[rlist[rg:rg + config.batch_size],:-config.input_size].reshape([config.batch_size, config.timestep1-1, config.input_size])
            lstm2 = train_lstm2[rlist[rg:rg + config.batch_size],:].reshape([config.batch_size, config.timestep2-1, config.input_size])
            #cnn = train_cnnmap[rlist[rg:rg + config.batch_size],:,:,:]
            linear = train_lstm1[rlist[rg:rg + config.batch_size],-config.input_size:].reshape([config.batch_size,config.input_size])
            y = trainy[rlist[rg:rg + config.batch_size]].reshape(-1,1)
            # seq,res = trainx.reshape(),trainy
            rg += config.batch_size
            feed_dict = {
                    model.lstm_input1: lstm1,
                    model.lstm_input2: lstm2,
                    # create initial state
                    model.linear:linear,
                    #model.cnn_input: cnn,
                    model.ys:y,
                    model.keep_prob: 0.5
                }
            _, cost,pred= sess.run(
                [model.train_op, model.cost,model.pred],
                feed_dict=feed_dict)

        if i > 0 and i % 1 == 0:

            feed_dict = {
                model.lstm_input1: test_lstm1[:,:-config.input_size].reshape((-1,config.timestep1-1, config.input_size)),
                model.lstm_input2: test_lstm2[:,:].reshape((-1,config.timestep2-1, config.input_size)),
                # create initial state
                model.linear : test_lstm1[:,-config.input_size:].reshape(-1,config.input_size),
                #model.cnn_input: test_cnnmap,
                model.ys: testy.reshape(-1,1),
                model.keep_prob: 1
            }
            test_cost, pred = sess.run(
                [model.accuracy, model.pred],
                feed_dict=feed_dict)
            best = min(best, test_cost)

            pred = pred.reshape(-1)

            print(str(i), 'pred cost', round(test_cost, 4))
            print('best cost', round(best, 4))
            print(','.join([str(testy[i]) for i in range(len(testy))]))
            print(','.join([str(pred[i]) for i in range(len(pred))]))
            # print(testy[:4000],pred)
            # plt.plot(range(4000),testy[0:4000].reshape([-1]), 'r')
            # plt.plot(range(4000),pred.flatten()[:4000], 'b')
            # plt.draw()
            # plt.pause(0.3)











