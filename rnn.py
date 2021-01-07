
import tensorflow as tf
import numpy as np
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, LSTM

def genData(file_name):
    pass


def rnnModel():
    num_feature = 10
    num_class = 5
    unit_size = 256

    model = Sequential()
    model.add(LSTM(unit_size, input_shape=(None, num_feature), return_sequence=True))
    model.TimeDistributed(Dense(num_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    return model

def trainModel(model):
    pass

def testModel(model):
    pass


if __name__ == '__main__':
    
    ## data preprocessing

    ## build model

    ## training

    ## testing
    pass
