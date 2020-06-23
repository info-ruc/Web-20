import pandas as pd
import numpy as np
import random
import config

from sklearn.preprocessing import StandardScaler

def getLstmData(data,train_index,test_index,site='陵东街',start=0):

    train_lstm_direction1 = []
    train_lstm_direction2 = []
    train_target = []

    test_lstm_direction1 = []
    test_lstm_direction2 = []
    test_target = []

    group_data = data.groupby('mname')
    # data = data[data['mname'] == site].sort_values('mtime')
    # data = data[config.reorder_col + ['month', 'day'] + ['label']].values.tolist()
    for (site, data) in group_data:
        if site not in config.sitelist:
            continue
        data = data.sort_values('mtime')
        data = data[config.reorder_col + ['month', 'day'] + ['label']].values.tolist()
        for i in range(144,len(data)):
            lstm1 = []
            lstm2 = []
            month = data[i][-3]

            if month not in config.month:
                continue
            day = data[i][-2]
            target = data[i][-1]
            for pre in range(config.timestep1,0,-1):
                lstm1.extend(data[i-pre][:-3])
            for pre in range(config.timestep2,1,-1):
                lstm2.extend(data[i-pre * 24][:-3])
            if month in config.train_month:
            # if random.random() > 0.2:
                train_lstm_direction1.append(lstm1)
                train_lstm_direction2.append(lstm2)
                train_target.append(target)
                train_index.append(i-72+start)
            else:
                test_lstm_direction1.append(lstm1)
                test_lstm_direction2.append(lstm2)
                test_target.append(target)
                test_index.append(i-72+start)

    return train_lstm_direction1,train_lstm_direction2,train_target,\
           test_lstm_direction1,test_lstm_direction2,test_target,train_index,test_index


def getLocMap(data):
    locdf = pd.read_excel('data/monitor_loc.xlsx')

    print(locdf['mname'])
    locmap = dict(zip(locdf['mname'].values.tolist(),locdf[['x','y']].values.tolist()))
    print(locmap)
    data_len = len(data) // len(config.sitelist)
    print(data_len)
    #cnnmap = np.zeros((len(list(range(72,data_len))),len(config.reorder_col),21,21),dtype=np.float32)
    cnnmap = np.zeros((len(list(range(72, data_len))), 5, 21, 21), dtype=np.float32)
    sitelist = config.sitelist
    for site in sitelist:
        x = locmap[site][0]
        y = locmap[site][1]
        sitedata = data[data['mname'] == site].sort_values('mtime')
        sitedata = sitedata[config.reorder_col].values.tolist()
        for i in range(72,data_len):
            #for j in range(len(sitedata[i])):
            for j in [0,-1,-2,-3,-4]:
                cnnmap[i-72,j,x,y] = sitedata[i-1][j]
    return locmap,cnnmap


def getCNNData(locmap,cnnmap,center = '陵东街'):
    x = locmap[center][0]
    y = locmap[center][1]
    return cnnmap[:,:,x-config.cnnwidth//2:x+config.cnnwidth//2,y-config.cnnheight//2:y+config.cnnheight//2]


def ensemble_lstm(bc_df,ad_df):
    train_lstm_data1 = []
    train_lstm_data2 = []
    train_y = []

    test_lstm_data1 = []
    test_lstm_data2 = []
    test_y = []

    train_index = []
    test_index = []

    for data in [bc_df,ad_df]:
        train_lstm_direction1, train_lstm_direction2, train_target, \
        test_lstm_direction1, test_lstm_direction2, test_target,train_index,test_index = \
            getLstmData(data,train_index,test_index,start =len(train_index))

        train_lstm_data1.extend(train_lstm_direction1)
        train_lstm_data2.extend(train_lstm_direction2)
        train_y.extend(train_target)
        test_lstm_data1.extend(test_lstm_direction1)
        test_lstm_data2.extend(test_lstm_direction2)
        test_y.extend(test_target)




    return train_index,test_index,\
           np.array(train_lstm_data1),\
           np.array(train_lstm_data2),\
           np.array(test_lstm_data1),\
           np.array(test_lstm_data2),\
           np.array(train_y),\
           np.array(test_y)


def main():

    ad_df = pd.read_csv('data/ad_data.csv', encoding='gb2312')
    bc_df = pd.read_csv('data/bc_data.csv',encoding='gb2312')


    bc_df['type'] = 1
    ad_df['type'] = 2

    alldf = pd.concat([bc_df,ad_df],axis=0)

    alldf['label'] = alldf['PM2.5']

    for i in config.scaler:
        ss = StandardScaler()
        alldf[[i]] = ss.fit_transform(alldf[[i]])
        #test_lstm1[:, i:i+config.input_size * config.timestep1:config.input_size] = ss.transform(test_lstm1[:,i:i+config.input_size * config.timestep1:config.input_size])




    bc_df = alldf[alldf['type']==1].drop('type',axis=1)
    ad_df = alldf[alldf['type'] == 2].drop('type', axis=1)

    train_index, test_index, train_lstm1, train_lstm2, test_lstm1, test_lstm2, trainy, testy = ensemble_lstm(bc_df,ad_df)

    # locmap,cnnmap1 = getLocMap(bc_df)
    # _,cnnmap2= getLocMap(ad_df)
    #
    # cnnmap = np.concatenate((cnnmap1,cnnmap2),axis=0)
    # sitecnnmap = getCNNData(locmap,cnnmap)
    # train_cnnmap = sitecnnmap[train_index,:,:,:]
    # test_cnnmap = sitecnnmap[test_index,:,:,:]

    return np.array(train_lstm1),\
           np.array(train_lstm2),\
           np.array(test_lstm1),\
           np.array(test_lstm2),\
           np.array(trainy),\
           np.array(testy),

        
        
        










