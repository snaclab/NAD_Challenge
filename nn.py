
import tensorflow as tf
import numpy as np
from tensorflow import keras
import xgb
import main
import preprocess
import postprocess
import argparse
from keras.models import Sequential
from keras.layers import Dropout, Bidirectional, Dense, TimeDistributed, LSTM, Conv1D, Flatten, Masking
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import csv
import os
import random
import pickle
import pandas as pd
import datetime

##if the server does not have GPU, remove this line
os.environ["CUDA_VISIBLE_DEVICES"]="0"  

class nnModel():
    def __init__(self, num_feature, num_class):
        #model parameters
        unit_size = 2048
        dp_rate = 0.5
        l_rate = 0.0005

        self.model = Sequential()
        self.model.add(Dense(unit_size, activation='relu', input_shape=(num_feature,)))
        self.model.add(Dropout(dp_rate))
        self.model.add(Dense(unit_size, activation='relu'))
        self.model.add(Dropout(dp_rate))
        self.model.add(Dense(unit_size, activation='relu'))
        self.model.add(Dropout(dp_rate))
        self.model.add(Dense(num_class, activation='softmax'))
        self.optimizer = optimizers.Adam(lr=l_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        self.model.summary()

    def trainModel(self, X, y, batch_size):
        history = self.model.fit(X, y, epochs=1, batch_size=batch_size)
    
    def validationModel(self, X, y):
        results = self.model.evaluate(X, y, batch_size=2048)
        return results

    def testModel(self, X):
        results = self.model.predict(X, batch_size=2048)
        y_pred = np.argmax(results, axis=-1)
        return y_pred, results

    def saveModel(self, model_n):
        self.model.save(model_n)
    
    def loadModel(self, model_n):
        self.model = keras.models.load_model(model_n)

def normalize_data(X, norm_zscore):
    ##normalization -> z-score, mean=0.5, std=0.5
    ##only normalize in, out, all cnt, duration features.
    X_norm = X[:]
    X_norm[:, :15] = 0.5*(X[:, :15]-norm_zscore[0, :])/norm_zscore[1, :] + 0.5

    return X_norm

def onehot_transform(y_label, n_class):
    num_data = len(y_label)
    y_onehot = np.zeros((num_data, n_class))

    for y_i in range(num_data):
        y_onehot[y_i, int(y_label[y_i])] = 1

    return y_onehot

def nn_training(data, n_class, norm_zscore):
    
    print('preparing data for nn')
    X = data[[c for c in data.columns if c != 'label']].copy()
    Y = data[['label']].copy()
    X['flow_diff'] = X['flow_diff'].apply(lambda x: 1 if x>0 else 0)
    X_normalized = normalize_data(X.values, norm_zscore)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y.values, test_size=0.2, random_state=7)
    n_feature = len(X_train[-1])

    y_train_onehot = onehot_transform(y_train, n_class)
    y_test_onehot = onehot_transform(y_test, n_class)
    
    ## build model 
    print('building model')
    model = nnModel(n_feature, n_class)

    ## train model
    print('training')
    ##parameters for training
    batch_size = 128 # 2048 is upper bound
    early_stop_patience = 10 #if the validation stop improving, the iterations will break
    
    ## running training algorithm
    epochs = 100 #early stop will interrupt iterations
    stop_count = 0
    prev_loss = 10000
    cur_loss = prev_loss
    for ep in range(epochs):
        print('Epoch: ' + str(ep+1) + '/' + str(epochs))
        model.trainModel(X_train, y_train_onehot, batch_size)
        
        val_acc = model.validationModel(X_test, y_test_onehot)
        print('val acc: ' + str(val_acc))
        
        cur_loss = val_acc[0]
        if prev_loss > cur_loss:
            prev_loss = cur_loss
            stop_count = 0
        else:
            stop_count += 1
        if stop_count >= early_stop_patience:
            break

    nn_pred, nn_prob = model.testModel(X_test)
    xgb.eval(y_test, nn_pred)
    
    return model

def nn_prediction(data, model, norm_zscore):
        
    X = data[[c for c in data.columns if c != 'label']].copy()
    X['flow_diff'] = X['flow_diff'].apply(lambda x: 1 if x>0 else 0)
    X_normalized = normalize_data(X.values, norm_zscore)

    ## testing and prediction
    print('testing')
    nn_pred, nn_prob = model.testModel(X_normalized)
    
    return nn_prob

def save_model(model, fname):
    model.saveModel(fname)

def load_model(fname):
    model = nnModel(1, 1)
    model.loadModel(fname)
    return model

    
