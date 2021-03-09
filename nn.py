
import tensorflow as tf
import numpy as np
#from tensorflow import keras
import xgb
import main
import preprocess
import postprocess
import argparse
from keras.models import Sequential, load_model
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

#if do full predict, set to false
validation_check = False

class nnModel():
    def __init__(self, num_feature, num_class):
        #model parameters
        unit_size = 32
        dp_rate = 0.2
        l_rate = 0.001
        #num_feature = 15+4+45+5
        #num_class = 5

        self.model = Sequential()
        self.model.add(Dense(unit_size, activation='relu', input_shape=(num_feature,)))
        self.model.add(Dropout(dp_rate))
        self.model.add(Dense(unit_size, activation='relu'))
        self.model.add(Dropout(dp_rate))
        #self.model.add(Dense(unit_size, activation='relu'))
        #self.model.add(Dropout(dp_rate))
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
        self.model = load_model(model_n)

def compute_norm(df):
    
    norm_zscore = np.zeros((2, 15))
    count_feature = ['duration', 'out (bytes)', 'in (bytes)',\
    'cnt_dst', 'cnt_src', 'cnt_serv_src',\
    'cnt_serv_dst', 'cnt_dst_slow', 'cnt_src_slow', 'cnt_serv_src_slow',\
    'cnt_serv_dst_slow', 'cnt_dst_conn', 'cnt_src_conn',\
    'cnt_serv_src_conn', 'cnt_serv_dst_conn']
     
    _df = df[count_feature]
    data = _df.to_numpy()
    norm_zscore[0, :] = np.mean(data, axis=0)
    norm_zscore[1, :] = np.std(data, axis=0)
    
    return norm_zscore

def load_norm(fname):
    norm_zscore = np.load(fname)
    return norm_zscore

def save_norm(fname, norm_zscore):
    np.save(fname, norm_zscore)

def normalize_data(X, norm_zscore):
    ##normalization -> z-score, mean=0.5, std=0.5
    ##only normalize in, out, all cnt, duration features.
    X_norm = X[:]
    X_norm[:, :15] = 0.5*(X[:, :15]-norm_zscore[0, :])/norm_zscore[1, :] + 0.5

    return X_norm

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn', nargs='+', help='input training dataset', required=False)
    parser.add_argument('--tst', nargs='+', help='input testing dataset', required=False)
    parser.add_argument('--pretrained', help='if there is pretrained encoder', type=bool, default=False)
    #parser.add_argument('--id', help='experiment_id', default=False)
    return parser.parse_args()

def nn_training(data, n_class, norm_zscore):
    
    print('preparing data for nn')
    X = data[[c for c in data.columns if c != 'label']].copy()
    Y = data[['label']].copy()
    X_normalized = normalize_data(X.values, norm_zscore)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y.values, test_size=0.2, random_state=7)
    n_feature = len(X_train[-1])

    ## build model 
    print('build model')
    model = nnModel(n_feature, n_class)

    #print('balanced num training data: ' + str(num_trn)) 
    #data_split = int(split*len(X))
    #num_val = len(X[data_split:])
        
    #shuffle_id = np.arange(num_trn)
    #np.random.seed(7)
    #np.random.shuffle(shuffle_id)
    #X = X[shuffle_id]
    #y = y[shuffle_id]
    #X_train, y_train, X_val, y_val = X[:data_split], y[:data_split], X[data_split:], y[data_split:]
    #val_test = np.argmax(y[data_split:], axis=-1)
    
    print('training')
    #parameters for training
    epochs = 10000
    batch_size = 128 # 2048 is upper bound
    
    prev_loss = 10000
    cur_loss = prev_loss
    early_stop_patience = 10
    stop_count = 0
    
    for ep in range(epochs):
        print('Epoch: ' + str(ep+1) + '/' + str(epochs))
        ##TODO:one-hot encoding y_train and y_test
        model.trainModel(X_train, y_train, batch_size)
    
        #if validation_check:
        val_acc = model.validationModel(X_test, y_test)
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
        
    data_drop = data.drop(columns=['label'])
    X = data_drop.to_numpy()
    X_normalized = normalize_data(X.values, norm_zscore)

    nn_pred, nn_prob = model.testModel(X_normalized)
    
    return nn_prob

def save_model(model, fname):
    model.saveModel(fname)

def load_model(model, fname):
    model.loadModel(fname)


if __name__ == '__main__':
    
    args = parse_arg()
    pretrained = args.pretrained
    #exp_id = args.id
    
    # TODO: 讀檔轉檔 train.csv and 全部的 tst_file.csv

    norm_zscore = np.zeros((2, 15))
    Normalization(X, X_tst, norm_zscore)
    
    np.save(norm, norm_zscore)
    
    #zscore[0, ] = mean
    #zsocre[1, ] = std
    
    ## load data (training, validation and testing)
    if not pretrained:
        num_trn = len(X)
        num_tst = len(X_tst)
        print('num training data: ' + str(num_trn)) 
        print('num testing data: ' + str(num_tst)) 
        
        #balance data
        X, y = BalancedData(X, y)

        if validation_check:
            y_test = np.argmax(y_tst, axis=-1)
       
        model = Train(X, y)
        
        ## save model
        print('save model')
        model.saveModel(nn_model)
        
    y_pred, y_prob = model.testModel(X_tst)

    ## testing
    if validation_check:
        print('testing')
        xgb.eval(y_test, y_pred)
    
    if pretrained:
        print('load testing model')
        model.loadModel(nn_model)
        norm_zscore = np.load(norm)
        with open(app_encoder, 'rb') as fp:
            app_name = pickle.load(fp)
        with open(proto_encoder, 'rb') as fp:
            proto_name = pickle.load(fp)
        
        print('testing')
        for tst_file in args.tst:
            data_tst = pd.read_csv(tst_file[:-4]+'_processed.csv')
            data_tst_drop = data_tst.drop(columns=['label'])
            X = data_tst_drop.to_numpy()
            for x_i in range(15):
                X[:, x_i] = 0.5*(X[:, x_i]-norm_zscore[0, x_i])/norm_zscore[1, x_i] + 0.5

            nn_pred, nn_prob = model.testModel(X)
            df_pred = pd.DataFrame(columns=[0,1,2,3,4], data=nn_prob)
            # postprocess
            y_pred = postprocess.post_processing(tst_file, df_pred, 'nn', validation_check)

            if validation_check:
                main.evaluation(data_tst.copy(), y_pred)

    
