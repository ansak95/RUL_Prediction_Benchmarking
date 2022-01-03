
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
# bimport tcn
from tcn import *

def build_model_LSTM(neurons,dp,rdp,seq_length,nb_features) :
    # build model

    history_seq = Input(shape = (seq_length,nb_features))
    x = history_seq
    x = tf.keras.layers.LSTM(neurons, return_sequences=True, dropout=dp, recurrent_dropout=rdp, activation='relu')(x)
    x = tf.keras.layers.LSTM(neurons, return_sequences=False, dropout=dp, recurrent_dropout=rdp, activation='relu')(x)
    #x = Activation('relu')(x)

    x = Dense(1)(x)
    model = Model(history_seq, x)
    return model

def build_model_RNN(neurons,dp,rdp,sq_length,nb_features):
    #Build model
    
    history_seq = Input(shape = (sq_length, nb_features))
    x = history_seq
    x = SimpleRNN(neurons, return_sequences = True, dropout=dp, recurrent_dropout = rdp,activation = 'relu')(x)
    x = SimpleRNN(neurons, return_sequences = True, dropout = dp, recurrent_dropout = rdp, activation='relu')(x)
    x = SimpleRNN(neurons, return_sequences = False, dropout = dp, recurrent_dropout = rdp, activation='relu')(x)
    #x = Activation('relu')(x)
    x = Dense(1)(x)
    model = Model(history_seq, x)
    return model


def build_model_GRU(neurons,dp,rdp,sq_length,nb_features) :
    # build model

    history_seq = Input(shape = (sq_length,nb_features))
    x = history_seq
    x = GRU(neurons, return_sequences=True, dropout=dp, recurrent_dropout=rdp)(x)
    x = GRU(neurons, return_sequences=False, dropout=dp, recurrent_dropout=rdp)(x)
    x = Activation('relu')(x)

    x = Dense(1)(x)
    model = Model(history_seq, x)
    return model

def build_model_CNN(neurons,fpl,kernel_size,dp,sq_length,nb_features):
    # build model
    history_seq = Input(shape = (sq_length, nb_features))
    x = history_seq
    x = Conv1D(filters = fpl,kernel_size = kernel_size,padding="same")(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters = fpl,kernel_size = kernel_size, padding = "same")(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters = fpl, kernel_size = kernel_size, padding="same")(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters = fpl, kernel_size = kernel_size, padding="same")(x)
    x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(1)(x)
    model = Model(history_seq, x)
    return model

def build_model_TCN(fpl, kernel_size,nb_stacks,dp,sq_length,nb_features):
    # build model
    history_seq = Input(shape=(sq_length, nb_features))
    x = TCN(nb_filters=fpl, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=[2 ** i for i in range(6)], padding='causal',
                use_skip_connections=True, dropout_rate=dp, return_sequences=False,
                activation='relu', kernel_initializer='he_normal', use_batch_norm=False, use_layer_norm=False,
                use_weight_norm=True, name='TCN')(history_seq)
    x = Dense(1)(x)
    model = Model(history_seq, x)
    return model
    # compile models
    