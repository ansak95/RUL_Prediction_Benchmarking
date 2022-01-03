#-*- coding: utf-8 -*-
#pip install keras-tcn --no-dependencies

# imports des librairies
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import pickle
import h5py

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
import time
# from tcn import *

from datetime import datetime

from Models_reg import build_model_LSTM
from Models_reg import build_model_RNN
# from CNN_reg import build_model


import argparse

# definition des des differentes valeurs des hyperparamètres

def training_args():
    parser=argparse.ArgumentParser(description='RNN_model')

    parser.add_argument('--nb_trn_samples', default=100, type=int,
                        help='Number of training samples (default: 100)')
    
    parser.add_argument('--nb_val_samples', default = 100, type=int,
                       help='Number of validation samples (default: 100)')
    
    parser.add_argument('--seq_lgth', default=30, type=int,
                        help='Sequence length')
    
    #parser.add_argument('--lr', default=1e-3, type=float,
                        #help='Learning rate')
    #params_RNN
    parser.add_argument('--neurons', default=32, type=int,
                        help='Neurons')
    parser.add_argument('--dp', default=0, type=float,
                        help='Dropout rate')
    parser.add_argument('--rdp', default=0, type=float,
                        help='Recurrent dropout rate')
    #parser.add_argument('--epochs', default=100, type=int,
      #                  help='Nb epochs')
    #parser.add_argument('--bs', default=64, type=int,
                       # help='Batch size')
 
    
    

    args=parser.parse_args()
    return args

# constants

args = training_args()
print(args)



# set folder path
folder = '../data'
fd = folder
fd_km = fd

# set random seed
np.random.seed(1)
tf.random.set_seed(1)

# import data
data_train_df = pd.read_pickle(fd_km + '/data_train_v1').reset_index().iloc[:,1:] #full set
data_test_df = pd.read_pickle(fd_km + '/data_test_v1').reset_index().iloc[:,1:]

# Normalization of data
X_train = data_train_df
X_test = data_test_df
X_train1 = X_train.iloc[:, 1:]
X_test1 = X_test.iloc[:, 1:]

min = X_train1.min(axis=0)
max = X_train1.max(axis=0)

data_train__ = (X_train1 - X_train1.min(axis=0)) / (X_train1.max(axis=0) - X_train1.min(axis=0))
data_test__ = (X_test1 - X_train1.min(axis=0)) / (X_train1.max(axis=0) - X_train1.min(axis=0))

X0 = pd.DataFrame(X_train.iloc[:,0])
X1 = pd.DataFrame(X_test.iloc[:, 0])
#X0['cycle'] = data_train__.iloc[:,0]
#X1['cycle']= data_train__.iloc[:,0]
X0['gauge1'] = data_train__.iloc[:,1]
X0['gauge2'] = data_train__.iloc[:,2]
X0['gauge3'] = data_train__.iloc[:,3]
X1['gauge1'] = data_test__.iloc[:,1]
X1['gauge2'] = data_test__.iloc[:,2]
X1['gauge3'] = data_test__.iloc[:,3]

X0['RUL'] = data_train__.iloc[:,4]
X1['RUL'] = data_train__.iloc[:,4]

data_train = X0
data_test = X1
# build data sequences
#utils 
nb_gauges = 3
data_train1 = data_train[data_train.ID <= args.nb_trn_samples]
data_val = data_train[data_train.ID > 9900]

#prepare forecasting data
def gen_X_sequence(id_df, seq_length, seq_cols, timesteps_pred, type_data = None):    
    
    ind_start = 0
    
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0+ind_start, num_elements-seq_length+1-timesteps_pred), range(seq_length+ind_start, num_elements+1-timesteps_pred)):
        yield data_array[start:stop, :]
 

def gen_Y_sequence(id_df, seq_length, seq_cols, timesteps_pred, type_data = None):
    
    ind_start = 0
    
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0+ind_start, num_elements-seq_length+1-timesteps_pred), range(seq_length+ind_start, num_elements+1-timesteps_pred)):
        yield data_array[stop-1, :]#data_array[start+1:stop+1, :]
   

#prepare data
seq_cols =  ['gauge'+str(i) for i in range(1,4)]#['label'+str(i) for i in range(1,4)]
seq_cols1 =  ['RUL']
sequence_length = args.seq_lgth
timesteps_pred = 1


#training set
seq_gen = (list(gen_X_sequence(data_train1[data_train1['ID']==id], sequence_length, seq_cols, timesteps_pred, type_data= 'None')) 
                   for id in data_train1['ID'].unique())
# generate sequences and convert to numpy array
dbX = np.concatenate(list(seq_gen))

seq_gen = (list(gen_Y_sequence(data_train1[data_train1['ID']==id], sequence_length, seq_cols1, timesteps_pred, type_data= 'None')) 
                   for id in data_train1['ID'].unique())
# generate sequences and convert to numpy array
dbY = np.concatenate(list(seq_gen)).reshape(-1,)

#val set
seq_gen = (list(gen_X_sequence(data_val[data_val['ID']==id], sequence_length, seq_cols, timesteps_pred, type_data= 'None')) 
                   for id in data_val['ID'].unique())
# generate sequences and convert to numpy array
dbX_val = np.concatenate(list(seq_gen))

seq_gen = (list(gen_Y_sequence(data_val[data_val['ID']==id], sequence_length, seq_cols1, timesteps_pred, type_data= 'None')) 
                   for id in data_val['ID'].unique())
# generate sequences and convert to numpy array
dbY_val = np.concatenate(list(seq_gen)).reshape(-1,)
#test set
# generate sequences and convert to numpy array
dbX_test = [data_test[data_test['ID']==id][seq_cols].values[-sequence_length:] for id in data_test['ID'].unique()]
dbX_test = np.asarray(dbX_test)

dbY_test = [data_test[data_test['ID']==id][seq_cols1].values[-1] for id in data_test['ID'].unique()]
dbY_test = np.asarray(dbY_test)

# build model
model = build_model_RNN(args.neurons,args.dp,args.rdp,dbX.shape[1],dbX.shape[2])
print(model.summary())

# compile model
#opt = Adam(learning_rate=args.lr)
#model.compile(loss='mse', optimizer=opt, metrics=['mae','mape'])

# train model
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d%m%Y%H%M%S")

model_path = 'RNN_models/1000_Structures/RNN_1000_BS_32_Lr_1e_3_4_5_epochs_500_Lr_Adap__/RNN_1000_Model_Lr_adap'+ timestampStr                 
model_path1 = 'RNN_models/1000_Structures/RNN_1000_BS_32_Lr_1e_3_4_5_epochs_500_Lr_Adap__/RNN_1000_Model_Lr_adap' + timestampStr

# get model as json string and save to file
model_as_json = model.to_json()
with open(model_path + '.json', "w") as json_file:
    json_file.write(model_as_json)

es = keras.callbacks.EarlyStopping(monitor='val_mape', min_delta=0, patience=200, verbose=0, mode='min')
mc = keras.callbacks.ModelCheckpoint(model_path + 'best_model.h5', monitor='val_mape', mode='min',                                                                                           save_best_only=True)
Lrate = [0.001, 0.0001,0.00001]
start = time.time()
for l_r in range(len(Lrate)):
    Learning_rate = Lrate[l_r]
    model.compile(optimizer=Adam(learning_rate=Learning_rate), loss ='mse', metrics=['mae', 'mape'])
    history = model.fit(dbX, dbY, epochs = 500,batch_size=32, verbose = 2, validation_data=(dbX_val, dbY_val), callbacks=[mc, es])
    #model=tf.keras.models.load_model(model_path+timestampStr+'_model.h5')
    np.save(model_path+'_history.npy',history.history)
    np.save(model_path+str(Learning_rate)+'_history.npy',history.history)
    #np.save(model_path+'i+'_history.npy', history.history)
end = time.time()


Time_execution = end - start
print(Time_execution)
np.save(model_path+'Execution_time.npy',Time_execution)
np.save(model_path+'min.npy',min)
np.save(model_path+'max.npy', max)
#np.save(model_path + 'time_execution.h5')
#history = model.fit(dbX, dbY, epochs=args.epochs, batch_size=args.bs,
        #validation_data=(dbX_val, dbY_val), verbose=2, callbacks = [es,mc])

# save learning history

# save model 
#model.save(model_path1 + '_.h5')
#min.save(model_path + 'min.h5')
#max.save(model_path + 'max.h5')