import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
import sys


def MergeTable():
    datas = []
    # 遍历所有污染物和气象数据
    for i in range(1, 9):
        acpath = 'data/%daq.xls' % i
        aircontaminant = ['mname', 'mtime', 'SO2', 'N0', 'NO2', 'NOx', 'O3', 'CO', 'PM10', 'PM2.5']
        ac = pd.read_excel(acpath, header=0, names=aircontaminant)
        atpath = 'data/%datmosphere.xls' % i
        atmosphere = ['mname', 'mtime', 'pressure', 'temperature', 'humidity', 'winddirection', 'windspeed',
                      'precipitation', 'visibility']
        at = pd.read_excel(atpath, header=0, names=atmosphere)
        # 同一时间段的污染物和气象数据进行合并
        data = pd.merge(ac, at)
        # print(data)
        datas.append(data)
        # datas=sorted(datas)
        # print('------------datas---------------')
        # print(datas)
    # 所有合并数据进行连接，成为一张表
    # print('------------alldata---------------')
    alldata = pd.concat(datas, ignore_index=True)
    # alldata=sorted(alldata)
    alldata = alldata.drop_duplicates()
    return alldata


def Calculation(alldata):
    # 按照监测点分组并求和
    # ratings_by_mname=alldata.groupby('mname').size()
    # print(ratings_by_mname)
    # 将Excel里面的时间从字符串转为datetime
    alldata['mtime'] = pd.to_datetime(alldata['mtime'])
    # 将Excel里面的时间从字符串转为date
    # alldata['mtime'] = pd.to_datetime(alldata['mtime']).dt.date
    # print(alldata['mtime'])

    # ratings_by_date=alldata.groupby(alldata['mtime']).size()
    # print(ratings_by_date)
    # print(alldata)
    # 以日期为小组，每个监测点的日监测数据数目
    # a=pd.pivot_table(alldata,index=['mtime','mname'],values=['PM2.5'],aggfunc=[len])
    # a.plot(title='oh my god')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.show()
    # print(a)
    # 取出2016-10-30~2017-10-30的数据
    # 这句不对，应为不能传入布尔值计算，只能传入单个值，分开写可以
    # select_data=alldata.loc[alldata['mtime']>='2016-10-30 00:00:00' and  alldata['mtime']<='2017-10-30 23:00:00']
    select_data = alldata.loc[alldata['mtime'] >= '20161030 000000']
    select_data = select_data.loc[select_data['mtime'] <= '2017-10-30 23:00:00']
    # 相当于建立一个数据列表，拿数据与列表进行比对，但是计算量复杂
    # select_data = alldata.loc[alldata['mtime'].isin(pd.date_range('2016-10-30 00:00:00', '2017-10-30 23:00:00',freq='H'))]
    # select_data=select_data.reindex()
    # 增加一个新的从0开始赋值的索引
    select_data = select_data.reset_index()
    # 删除掉原来的index，axis=1表示删除的是列，axis=0表示删除的是行
    select_data = select_data.drop(['index'], axis=1)
    # 将数据写入.csv文件
    # select_data.to_csv('one-year-data.csv')
    # print(select_data)
    # 将数据写入.csv文件
    # f=open('alldata.csv','w',encoding='UTF-8')
    # f.write(select_data)
    return select_data


def DataCleaning(dirty_data):
    # dirty_data = pd.read_excel('testdata.xlsx')
    # dirty_data = pd.read_csv('one-year-data.csv')
    for index, row in dirty_data.iterrows():
        # print('----------index------------------')
        # index=int(index)
        print(index)
        i = 0
        for value in row.values:
            value = str(value)
            if '(' in value:
                row[i] = value.split('(')[0]
                # print('----------value------------------')
                # print(value)
                # print('----------row[i]------------------')
                # print(row[i])
            i = i + 1
        dirty_data.loc[index] = row
    clean_data = dirty_data
    clean_data.to_csv('clean_data.csv')
    # return clean_data


def WindDirection():
    clean_data = pd.read_csv('clean_data.csv', encoding='GB2312', index_col=0)
    wind_y = np.sin(clean_data['winddirection'])
    wind_x = np.cos(clean_data['winddirection'])
    wind_speed = clean_data['windspeed']
    wind_E = np.where(wind_x > 0, wind_x, 0)
    wind_W = np.where(wind_x < 0, -wind_x, 0)
    wind_N = np.where(wind_y > 0, wind_y, 0)
    wind_S = np.where(wind_y < 0, -wind_y, 0)
    wind_E = np.multiply(wind_E, wind_speed)
    wind_W = np.multiply(wind_W, wind_speed)
    wind_N = np.multiply(wind_N, wind_speed)
    wind_S = np.multiply(wind_S, wind_speed)
    clean_data['wind_E'] = wind_E
    clean_data['wind_W'] = wind_W
    clean_data['wind_N'] = wind_N
    clean_data['wind_S'] = wind_S

    clean_data = clean_data.drop(['winddirection', 'windspeed'], axis=1)

    # print(clean_data)
    clean_data.to_csv('wind_data.csv')
    # return winddata


def DataStandardization():
    clean_data = pd.read_csv('wind_data.csv', encoding='GB2312', index_col=0)
    temp = clean_data['temperature']
    date = pd.to_datetime(clean_data['mtime'])
    name = clean_data['mname']
    # 给小于0 的值赋none的时候要先剔除站点名称和时间
    clean_data = clean_data.drop(['mtime', 'mname', 'precipitation', 'visibility'], axis=1)
    # 选择出小于0 的所有值赋none
    clean_data[clean_data < 0] = None
    # 给空值填充上一条记录的值
    clean_data = clean_data.fillna(method='ffill', limit=136)
    # 温度的值可以小于0，把前面拷贝的温度值替换进来重新处理
    clean_data['temperature'] = temp
    # 剔除掉值为-99的数据，进行填充
    clean_data[clean_data == -99] = None
    clean_data = clean_data.fillna(method='ffill', limit=136)

    # 对比实验数据和数据区间缺少的时间
    all_date = pd.Series(pd.date_range('2016-10-30 00:00:00', '2017-10-30 23:00:00', freq='H'))
    # all_date =  pd.date_range('2016-10-30 00:00:00', '2017-10-30 23:00:00', freq='H')
    # miss_date = date.isin(pd.date_range('2016-10-30 00:00:00', '2017-10-30 23:00:00',freq='H'))
    # 企图用filter来写，但是没搞明白，于是用了下面求差集
    # miss_date = all_date .filter(lambda  all_date : all_date  not in date , all_date)
    # print(list(miss_date))
    # 求差集这个输出好像很奇怪啊,set的值tolist之后输出的是毫秒，要转成时间格式
    miss_date = set(all_date.values.tolist()) - set(date.values.tolist())
    # 首先将set转成series
    miss_date = pd.Series(list(miss_date))
    miss_date = pd.to_datetime(miss_date)
    miss_date.to_csv('miss_date.csv')

    # 数据归一化
    # clean_data = preprocessing.scale(clean_data)

    # 检验平均值和标准差
    # clean_data_mean = clean_data.mean(axis=0)
    # clean_data_std = clean_data.std(axis=0)
    # print(clean_data_mean)
    # print(clean_data_std)

    # 将站点名称和时间列添加进来并重新排序，emmm，站点名称和时间不是特征，但是要留着
    clean_data['mtime'] = date
    clean_data['mname'] = name
    clean_data = clean_data.reindex(
        columns=['mname', 'mtime', 'SO2', 'N0', 'NO2', 'NOx', 'O3', 'CO', 'PM10', 'PM2.5', 'pressure', 'temperature',
                 'humidity', 'wind_E', 'wind_W', 'wind_N', 'wind_S'])

    clean_data.to_csv('data/alldata.csv')


def SplitData():
    alldata = pd.read_csv('data/alldata.csv', encoding='GB2312', index_col=0)
    alldata['mtime'] = pd.to_datetime(alldata['mtime'])
    bc_data = alldata.loc[alldata['mtime'] < '20161225  090000']
    ad_data = alldata.loc[alldata['mtime'] > '20170101  000000']
    bc_data.to_csv('data/bc_data.csv')
    ad_data.to_csv('data/ad_data.csv')



    # print(bc_data.values.tolist())


def get_season_type(x):
    TIME_SPLITS = ['03-01', '06-01', '09-01', '12-01']

    date = '-'.join(x.split(' ')[0].split('-')[1:])
    if date >= TIME_SPLITS[0] and date < TIME_SPLITS[1]:
        return 1
    elif date >= TIME_SPLITS[1] and date < TIME_SPLITS[2]:
        return 2
    elif date >= TIME_SPLITS[2] and date < TIME_SPLITS[3]:
        return 3
    else:
        return 4

#http://www.pm25.com/news/91.html
def get_level(x):
    if x <= 50:
        return 1
    elif x<=100:
        return 2
    elif x<=150:
        return 3
    elif x<=200:
        return 4
    elif x<=300:
        return 5
    else:
        return 6

def Windows():
    bc_data = pd.read_csv('data/bc_data.csv', encoding='GB2312', index_col=0)
    bc_data = bc_data.sort_index(by=['mname', 'mtime'])
    bc_data['season'] = bc_data['mtime'].map(get_season_type)
    bc_data['level'] = bc_data['PM2.5'].map(get_level)
    bc_data['month'] = bc_data['mtime'].map(lambda x:int(x.split('-')[1]))
    bc_data['day'] = bc_data['mtime'].map(lambda x:int(x.split(' ')[0].split('-')[2]))
    ad_data = pd.read_csv('data/ad_data.csv', encoding='GB2312', index_col=0)
    ad_data = ad_data.sort_index(by=['mname', 'mtime'])
    ad_data['season'] = ad_data['mtime'].map(get_season_type)
    ad_data['level'] = ad_data['PM2.5'].map(get_level)
    ad_data['month'] = ad_data['mtime'].map(lambda x: int(x.split('-')[1]))
    ad_data['day'] = ad_data['mtime'].map(lambda x: int(x.split(' ')[0].split('-')[2]))



    bc_data.to_csv('data/bc_data.csv',encoding='GB2312')
    ad_data.to_csv('data/ad_data.csv',encoding='GB2312')





if __name__ == '__main__':
    # alldata = MergeTable()
    # dirty_data = Calculation(alldata)
    # DataCleaning(dirty_data)
    # WindDirection()
    # DataStandardization()
    # SplitData()
    Windows()
