reorder_col = ['PM2.5', 'SO2', 'N0', 'NO2', 'NOx', 'O3', 'CO', 'PM10', 'pressure', 'temperature', 'humidity', 'wind_E',
               'wind_W', 'wind_N', 'wind_S']

scaler = ['PM2.5','SO2', 'N0', 'NO2', 'NOx', 'O3', 'CO', 'PM10', 'pressure', 'temperature', 'humidity', 'wind_E',
               'wind_W', 'wind_N', 'wind_S']

FEATURES = 15
TRAIN_RATE = 0.75

#ALGORITHMS = ['mlr','nn','xgb','rf','gbdt','svm']
ALGORITHMS = ['nn']

TIME_STEPS = 1
TIME_SPLITS = ['03-01','06-01','09-01','12-01']
SPLIT_SEASON = False
SPLIT_SITE = False
SPLIT_LEVEL = False

SEASON_LEN = 4
LEVEL_LEN = 6
SITE_LEN = -1

train_month = [11,12,1,2,3,4,5,6,7]
test_month = [8,9,10]

#采暖期，非采暖期
month = [1,2,3,4,5,6,7,8,9,10,11,12]
#用哪个站点就写哪个
sitelist = ['新秀街','文化路','沈辽西路','浑南东路','小河沿','太原街','东陵路','陵东街','森林路','裕农路','京沈街']


timestep1 = 6
timestep2 = 3

uselstm1 = True
uselstm2 = True
uselinear = True

uselinear2 = True

usecnn = False

input_size = 15

cnnwidth = 8
cnnheight = 8

padding_size = 2

lstem1_cell_size = 5
lstem2_cell_size = 5
lstm1_output_size = 1
lstm2_output_size = 1


cnn_output_size = 1

mlp_layer1_size = 20
mlp_layer2_size = 10
mlp_layer_output_size = 1

single_layer_output_size = 1

cnn_layer = 5

cnn_keep_prob = 0.5

output_size = 1

lr = 0.005

batch_size = 512

eposide = 10000

optimizer = 'adam'

hidden_size = 20
hidden_size2 = 10

usebn =False

uselastcell_lstm2 = True

uselastcell_lstm1 = True

use_l2_norm = True




