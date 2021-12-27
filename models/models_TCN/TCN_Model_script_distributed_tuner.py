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


class MyHyperModel(HyperModel):

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build(self, hp):
        # build model
        input_layer = Input(shape=self.input_shape)
        x = TCN(nb_filters=hp.Int('nb_filters', min_value=15, max_value=50, step=5),
                    kernel_size=hp.Int('kernel_size', min_value=2, max_value=6, step=2),
                    nb_stacks=1,
                    dilations=[2 ** i for i in range(hp.Int('dilations', min_value=4, max_value=10, step=2))],
                    padding='causal',
                    use_skip_connections=True,
                    dropout_rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.2, step=0.1),
                    return_sequences=False,
                    activation='relu',
                    kernel_initializer='he_normal',
                    use_batch_norm=False, use_layer_norm=False, use_weight_norm=True,
                    name='TCN')(input_layer)

        x = Dense(self.output_shape, activation='softmax')(x)
        output_layer = x

        model = Model(input_layer, output_layer)

        # compile model
        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
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
    hypermodel = MyHyperModel(input_shape = (sequence_length, 3), output_shape = 20)

tuner = RandomSearch(
    hypermodel,
    objective='val_sparse_categorical_accuracy',
    max_trials=50,
    executions_per_trial=1,
    directory='TCN_Model',
    project_name='100_structures_randomSearch_1')

tuner.search_space_summary()

tuner.search(train_dataset,
             epochs=500,
             verbose=2,
             validation_data=val_dataset,
             callbacks=[tf.keras.callbacks.TensorBoard(tuner.directory + '/' + tuner.project_name)])

models = tuner.get_best_models(num_models=1)

tuner.results_summary()

best_model = models[0]

model_path = tuner.directory + '/' + tuner.project_name + '/' + 'best_model'

# get model as json string and save to file
model_as_json = best_model.to_json()
with open(model_path + '.json', "w") as json_file:
    json_file.write(model_as_json)
# save model weights
best_model.save_weights(model_path + '_weights.h5')
