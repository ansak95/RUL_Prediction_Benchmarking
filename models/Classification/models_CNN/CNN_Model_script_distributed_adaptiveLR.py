# Import
import numpy as np
import pandas as pd
import json
import random

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
from kerastuner.tuners import *
from kerastuner import HyperModel
from keras.wrappers.scikit_learn import KerasClassifier
from tcn import *

from sklearn.model_selection import GridSearchCV

from datetime import datetime

# set random seed
np.random.seed(1)
tf.random.set_seed(1)


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
        yield data_array[stop-1, :]

   
def get_dataset(sequence_length, batch_size):
    # set folder path
    folder = '../../data'
    fd = folder
    fd_km = fd

    # import data
    data_train_df = pd.read_pickle('data_train').reset_index().iloc[:,1:] #full set # The folder has to be changed to the location of the training set
    data_val = pd.read_pickle('data_val').reset_index().iloc[:,1:]
    data_test = pd.read_pickle('data_test').reset_index().iloc[:,1:] 
    
    
    # create bins
    l = 0.5
    nb_bins = 20 # including one extra bin for RUL>upper_bin_bound
    lower_bin_bound = 0
    upper_bin_bound = 80000

    bins = np.linspace(lower_bin_bound**l, upper_bin_bound**l, nb_bins)**(1/l)
    labels=[i for i in range(bins.shape[0]-1)]

    # categorise data
    data_train_df['RUL_bin'] = pd.cut(data_train_df['RUL'], bins=bins, labels=labels)
    data_val['RUL_bin'] = pd.cut(data_val['RUL'], bins=bins, labels=labels)
    data_test['RUL_bin'] = pd.cut(data_test['RUL'], bins=bins, labels=labels)

    # build data sequences
    data_train = data_train_df[data_train_df.ID <= 100] # Change this for 100, 500 or 1000 training structures

    #prepare data
    seq_cols = ['gauge'+str(i) for i in range(1,4)]
    seq_cols1 = ['RUL_bin']
    timesteps_pred = 1

    #training set
    seq_gen = (list(gen_X_sequence(data_train[data_train['ID']==id], sequence_length, seq_cols, timesteps_pred, type_data= 'train')) 
                    for id in data_train['ID'].unique())
    # generate sequences and convert to numpy array
    dbX = np.concatenate(list(seq_gen))

    seq_gen = (list(gen_Y_sequence(data_train[data_train['ID']==id], sequence_length, seq_cols1, timesteps_pred, type_data= 'train')) 
                    for id in data_train['ID'].unique())
    # generate sequences and convert to numpy array
    dbY = np.concatenate(list(seq_gen)).reshape(-1,)

    #val set
    seq_gen = (list(gen_X_sequence(data_val[data_val['ID']==id], sequence_length, seq_cols, timesteps_pred, type_data= 'train')) 
                    for id in data_val['ID'].unique())
    # generate sequences and convert to numpy array
    dbX_val = np.concatenate(list(seq_gen))

    seq_gen = (list(gen_Y_sequence(data_val[data_val['ID']==id], sequence_length, seq_cols1, timesteps_pred, type_data= 'train')) 
                    for id in data_val['ID'].unique())
    # generate sequences and convert to numpy array
    dbY_val = np.concatenate(list(seq_gen)).reshape(-1,)

    #test set
    seq_gen = (list(gen_X_sequence(data_test[data_test['ID']==id], sequence_length, seq_cols, timesteps_pred, type_data= 'train')) 
                    for id in data_test['ID'].unique())
    # generate sequences and convert to numpy array
    dbX_test = np.concatenate(list(seq_gen))

    seq_gen = (list(gen_Y_sequence(data_test[data_test['ID']==id], sequence_length, seq_cols1, timesteps_pred, type_data= 'train')) 
                    for id in data_test['ID'].unique())
    # generate sequences and convert to numpy array
    dbY_test = np.concatenate(list(seq_gen)).reshape(-1,)

    return (
        tf.data.Dataset.from_tensor_slices((dbX, dbY)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((dbX_val, dbY_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((dbX_test, dbY_test)).batch(batch_size),
    )


def build_model(input_shape, output_shape):
    # build model
    input_layer = Input(shape=input_shape)

    x = LayerNormalization(axis=1)(input_layer)
    x = Dropout(0.0)(x)

    for i in range(2):
        x = Conv1D(filters=40,
                kernel_size=12,
                padding="same",
                activation='relu')(x)

    x = Flatten()(x)

    x = Dense(output_shape, activation='softmax')(x)
    output_layer = x

    model = Model(input_layer, output_layer)

    # compile model
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['SparseCategoricalAccuracy'])

    return model


# Load training, validation and test data
batch_size = 4096
sequence_length = 30
train_dataset, val_dataset, test_dataset = get_dataset(
    sequence_length=sequence_length, batch_size=batch_size)


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = build_model(input_shape = (sequence_length, 3), output_shape = 20)

model_directory = '100_structures' # Change this for 100, 500 and 1000 structures
model_path = model_directory + '/' + 'CNN_Model_adaptiveLR_1'

model.load_weights(model_directory + '/CNN_Model_adaptiveLR_1_weights_epoch_5_val_accuracy_0.8205549716949463.h5')

# get model as json string and save to file
model_as_json = model.to_json()
with open(model_path + '.json', "w") as json_file:
    json_file.write(model_as_json)

mc = keras.callbacks.ModelCheckpoint(model_path + '_weights_epoch_{epoch}_val_accuracy_{val_sparse_categorical_accuracy}.h5', monitor='val_sparse_categorical_accuracy', mode='max', 
                                     save_weights_only=True, save_best_only=True)
tb = tf.keras.callbacks.TensorBoard(model_directory)

for lr in [1e-3, 1e-4, 1e-5]:
    model.compile(
            optimizer=Adam(lr),
            loss='sparse_categorical_crossentropy',
            metrics=['SparseCategoricalAccuracy'])

    # Train the model on all available devices.
    history = model.fit(train_dataset, epochs=500, verbose=2, validation_data=val_dataset, callbacks = [mc,tb])
