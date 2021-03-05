
import tensorflow as tf
import numpy as np
#from tensorflow import keras
import xgb
import main
import preprocess
import argparse
from keras.models import Sequential, load_model
from keras.layers import Dropout, Bidirectional, Dense, TimeDistributed, LSTM, Conv1D, Flatten, Masking
from keras import optimizers
import csv
import os
import random
import pickle
import pandas as pd
import datetime

##do not use CUDA in full test (assume the server does not have GPU)
os.environ["CUDA_VISIBLE_DEVICES"]="9"
#if do full test, set to false
validation_check = False

num_feature = 15+4+45+5
num_class = 5
early_stop_patience = 10
#NORMAL_OFFSET = #int(180625*1.3)

##All models' paths are here:
proto_encoder = 'pretrained/'+'proto_nn_encoder.pkl'
app_encoder = 'pretrained/'+'app_nn_encoder.pkl'
nn_model = 'pretrained/'+'nn_debug.h5'
norm = 'pretrained/'+'norm_zscore.npy'

processed_trn_data = 'X.npy'
processed_trn_label = 'y.npy'
processed_tst_data = 'X_test.npy'
processed_tst_label = 'y_test.npy'

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn', nargs='+', help='input training dataset', required=False)
    parser.add_argument('--tst', nargs='+', help='input testing dataset', required=False)
    parser.add_argument('--pretrained', help='if there is pretrained encoder', default=False)
    #parser.add_argument('--id', help='experiment_id', default=False)
    return parser.parse_args()

def genData(files, app_name, proto_name, train_file=True):
    ##load data from csv file, and pre-process it into numpy array
    #app_name = {}#attr 8, 9 are discrete, 8(protocal ID) has 4 and 9(app name) has 45
    app_id = 0
    proto_id = 0
    label = {'Normal':1, 'Probing-Nmap':3, 'Probing-Port sweep':4, 'Probing-IP sweep':2, 'DDOS-smurf':0}#attr -1
    data_n = 0
    data = []
    #normal_size = 0
    #time_mark = []
    #ip_mark = [] 
    #normal_id = []
    #abnormal_id = []
    #dat_id = 0

    for file_name in files:
        with open(file_name, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                data.append(row)
                
                if train_file:
                    if not row[8] in proto_name:
                        proto_name[row[8]] = proto_id
                        proto_id += 1
                    if not row[9] in app_name:
                        app_name[row[9]] = app_id
                        app_id += 1
                data_n+=1
                
    X = np.zeros((data_n, num_feature))
    y = np.zeros((data_n, num_class))
    #dat_i = 0

    for datum_i in range(data_n):
        x_i = 0
        for attr in range(5, 22):
            #skip attr 8 and attr 9 first
            if attr != 8 and attr != 9:
                X[datum_i, x_i] = int(data[datum_i][attr])
                x_i += 1
        #one hot encoding of attr 8
        X[datum_i, x_i+int(proto_name[data[datum_i][8]])] = 1
        x_i += 4
        #one hot encoding of attr 9
        X[datum_i, x_i+int(app_name[data[datum_i][9]])] = 1
        x_i += 45 
        #dst, src port == zero
        if int(data[datum_i][3]) == 0 and int(data[datum_i][4]) == 0:
            X[datum_i, x_i] = 1
        x_i+=1
        #src inner IP
        ip = data[datum_i][1].split('.')
        if int(ip[0]) == 172 and int(ip[1]) >= 16 and int(ip[1]) < 32:
            X[datum_i, x_i] = 1
        elif int(ip[0]) == 192 and int(ip[1]) == 168:
            X[datum_i, x_i] = 1
        elif int(ip[0]) == 10:
            X[datum_i, x_i] = 1
        x_i+=1
        #dst inner IP
        ip = data[datum_i][2].split('.')
        if int(ip[0]) == 172 and int(ip[1]) >= 16 and int(ip[1]) < 32:
            X[datum_i, x_i] = 1
        elif int(ip[0]) == 192 and int(ip[1]) == 168:
            X[datum_i, x_i] = 1
        elif int(ip[0]) == 10:
            X[datum_i, x_i] = 1
        x_i+=1
        '''
        #is sweep
        if datum_i == 0:
            X[datum_i, x_i] = 0
        else:
            src_prev_ip = data[datum_i-1][1].split('.')
            dst_prev_ip = data[datum_i-1][2].split('.')
            src_ip = data[datum_i][1].split('.')
            dst_ip = data[datum_i][2].split('.')
            src_prev_val = int(src_prev_ip[0])*(256**3)+int(src_prev_ip[1])*(256**2)+int(src_prev_ip[2])*(256)+int(src_prev_ip[3])
            dst_prev_val = int(dst_prev_ip[0])*(256**3)+int(dst_prev_ip[1])*(256**2)+int(dst_prev_ip[2])*(256)+int(dst_prev_ip[3])
            src_val = int(src_ip[0])*(256**3)+int(src_ip[1])*(256**2)+int(src_ip[2])*(256)+int(src_ip[3])
            dst_val = int(dst_ip[0])*(256**3)+int(dst_ip[1])*(256**2)+int(dst_ip[2])*(256)+int(dst_ip[3])
            if src_prev_val==src_val and abs(dst_prev_val-dst_val)==1:
                X[datum_i, x_i] = 1
            else:
                X[datum_i, x_i] = 0
        x_i += 1
        '''
        #dst IP end with 255:
        if int(ip[-1]) == 255:
            X[datum_i, x_i] = 1
        x_i+=1
        
        #flow diff
        if int(data[datum_i][6])-int(data[datum_i][7]) > 0:
            X[datum_i, x_i] = 1
        x_i+=1
        '''
        #in == out == 0
        if int(data[datum_i][6]) == int(data[datum_i][7]) == 0:
            X[datum_i, x_i] = 1
        x_i+=1
        '''
        #IP transformation    
        '''
        ip = data[datum_i][1].split('.')
        for ip_str in ip:
            ip_n = bin(int(ip_str)+1024)
            for n in range(1, 9):
                X[datum_i, x_i+8-n] = int(ip_n[-n])
            x_i+=8
        
        ip = data[datum_i][2].split('.')
        for ip_str in ip:
            ip_n = bin(int(ip_str)+1024)
            for n in range(1, 9):
                X[datum_i, x_i+8-n] = int(ip_n[-n])
            x_i+=8
        '''
        #port transform
        '''
        port_n = bin(int(data[datum_i][3])+1024*1024)
        for n in range(1, 17):
            X[datum_i, x_i+16-n] = int(port_n[-n])
        x_i+=16
        port_n = bin(int(data[datum_i][4])+1024*1024)
        for n in range(1, 17):
            X[datum_i, x_i+16-n] = int(port_n[-n])
        x_i+=16
        '''
        if train_file:
            y[datum_i, label[data[datum_i][22]]] = 1
        else:
            if validation_check:
                y[datum_i, label[data[datum_i][22]]] = 1
            else:
                y[datum_i, 0] = 1

    return X, y

def BalancedData(X, y):
    #downsampling normal data
    y_id = np.argmax(y, axis=-1)
    NORMAL_OFFSET = 0
    normal_id = []
    abnormal_id = []
    seq_id = {}
    
    for i in range(len(y_id)):
        #if train_file:
        if y_id[i] != 1:
            if y_id[i] == 2:
                NORMAL_OFFSET += 1
            #abnormal_id.append(time_id[i][1])
            abnormal_id.append(i)
        else:
            #normal_id.append(time_id[i][1])
            normal_id.append(i)
    
    NORMAL_OFFSET = int(NORMAL_OFFSET*1.3) 

    data_id = abnormal_id[:]
    balanced_normal_id = random.sample(normal_id, NORMAL_OFFSET)
    data_id.extend(balanced_normal_id)
    data_n = len(data_id)
    
    X_bal = np.zeros((data_n, num_feature))
    y_bal = np.zeros((data_n, num_class))
    
    dat_i = 0
    for datum_i in data_id:
        x_i = 0
        X_bal[dat_i, x_i:x_i+num_feature] = X[datum_i, :]
        y_bal[dat_i, :] = y[datum_i, :]
        
        dat_i += 1

    return X_bal, y_bal

def Normalization(X_train, X_test, norm_zscore):
    ##normalization -> z-score, mean=0.5, std=0.5
    for x_i in range(15):
        
        concat = np.zeros(len(X_train)+len(X_test))
        concat[:len(X_train)] = X_train[:, x_i]
        concat[len(X_train):] = X_test[:, x_i]
        mean = np.mean(concat)
        std = np.std(concat)
        norm_zscore[0, x_i] = mean
        norm_zscore[1, x_i] = std
        X_train[:, x_i] = 0.5*(X_train[:, x_i]-mean)/std + 0.5
        X_test[:, x_i] = 0.5*(X_test[:, x_i]-mean)/std + 0.5
        

class nnModel():
    def __init__(self):
        #model parameters
        unit_size = 32
        dp_rate = 0.2
        l_rate = 0.001

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

if __name__ == '__main__':
    
    args = parse_arg()
    pretrained = False
    #exp_id = args.id
    if args.pretrained == "True":
        pretrained = True
    
    ## data preprocessing
    print('preprocessing')
    split = 0.8
    #zscore[0, ] = mean
    #zsocre[1, ] = std
    norm_zscore = np.zeros((2, 15))
    app_name={}
    proto_name={}
    
    ## load data (training, validation and testing)
    if not pretrained:
        if os.path.isfile(processed_trn_data) and os.path.isfile(processed_trn_label):
            X = np.load(processed_trn_data)
            y = np.load(processed_trn_label)
            with open(app_encoder, 'rb') as fp:
                app_name = pickle.load(fp)
            with open(proto_encoder, 'rb') as fp:
                proto_name = pickle.load(fp)
        else:
            X, y = genData(args.trn, app_name, proto_name)
            np.save(processed_trn_data, X)
            np.save(processed_trn_label, y)
            with open(app_encoder, 'wb') as fp:
                pickle.dump(app_name, fp)    
            with open(proto_encoder, 'wb') as fp:
                pickle.dump(proto_name, fp)    
    
        if os.path.isfile(processed_tst_data):
            X_tst = np.load(processed_tst_data)
            if validation_check:
                y_tst = np.load(processed_tst_label)
        else:
            if validation_check:
                X_tst, y_tst = genData(args.tst, app_name, proto_name, False)
                np.save(processed_tst_data, X_tst)
                np.save(processed_tst_label, y_tst)
            else:
                X_tst, _ = genData(args.tst, app_name, proto_name, False)
                np.save(processed_tst_data, X_tst)
    
        num_trn = len(X)
        num_tst = len(X_tst)
        print('num training data: ' + str(num_trn)) 
        print('num testing data: ' + str(num_tst)) 
        
        #normalization and balanced
        Normalization(X, X_tst, norm_zscore)
        X, y = BalancedData(X, y)
        
        num_trn = len(X)
        print('balanced num training data: ' + str(num_trn)) 
        data_split = int(split*len(X))
        num_val = len(X[data_split:])
        
        if validation_check:
            y_test = np.argmax(y_tst, axis=-1)
            
        shuffle_id = np.arange(num_trn)
        np.random.seed(7)
        np.random.shuffle(shuffle_id)
        X = X[shuffle_id]
        y = y[shuffle_id]
        X_train, y_train, X_val, y_val = X[:data_split], y[:data_split], X[data_split:], y[data_split:]
        val_test = np.argmax(y[data_split:], axis=-1)
        
        #else:
        #    X_train, y_train = X[:], y[:]
    
    
    ## build model 
    print('build model')
    model = nnModel()
    
    ## load model 
    if pretrained:
        print('load model')
        model.loadModel(nn_model)
        norm_zscore = np.load(norm)
        with open(app_encoder, 'rb') as fp:
            app_name = pickle.load(fp)
        with open(proto_encoder, 'rb') as fp:
            proto_name = pickle.load(fp)
    
    ## training (adust hyper parameters)
    #if validation_check:
    epochs = 10000
    #else:
    #    epochs = 35
    batch_size = 128#2048 is upper bound

    if pretrained:
        print('testing')
        for tst_file in args.tst:
            if validation_check:
                data_tst = pd.read_csv(tst_file[:-4]+'_processed.csv')
            X, _ = genData([tst_file], app_name, proto_name, False)
            for x_i in range(15):
                X[:, x_i] = 0.5*(X[:, x_i]-norm_zscore[0, x_i])/norm_zscore[1, x_i] + 0.5

            nn_pred, nn_prob = model.testModel(X)
            df_pred = pd.DataFrame(columns=[0,1,2,3,4], data=nn_prob)
            # postprocess
            for time_setting in ['minute', 'hour']:
                test = pd.read_csv(tst_file)
                ans = test[['time','src']].copy()
                ans = pd.concat([ans, df_pred], axis=1)
                ans['time'] = ans['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                if time_setting == 'hour':
                    ans['time'] = ans['time'].apply(lambda x: str(x.month).zfill(2)+str(x.day).zfill(2)+str(x.hour).zfill(2))
                elif time_setting == 'minute':
                    ans['time'] = ans['time'].apply(lambda x: str(x.month).zfill(2)+str(x.day).zfill(2)+str(x.hour).zfill(2)+str(x.minute).zfill(2))
                ans['time+src'] = ans['time'].apply(str) + ans['src'].apply(str)
                ans.drop(columns=['time','src'],inplace=True)
                d_v = ans.groupby(['time+src']).sum()
                d_v = d_v.idxmax(axis='columns').to_frame()
                DIC = {}
                for idx, row in d_v.iterrows():
                    DIC[idx] = row[0]
                ans['pred'] = [1 for i in range(len(ans))]
                ans['pred'] = ans['time+src'].apply(lambda x: DIC[x])
                if time_setting == 'hour':
                    m_df = pd.read_csv(tst_file[:-4]+'_minute_nn_predicted.csv')
                    d_idx = list(m_df[m_df['label']=='DDOS-smurf'].index)
                    ans.loc[d_idx, 'pred'] = 0
                y_pred = ans['pred'].values
            
                # Transform label and save data
                label_map = {0: 'DDOS-smurf', 1: 'Normal', 2: 'Probing-IP sweep', 3: 'Probing-Nmap', 4: 'Probing-Port sweep'}
                test['label'] = ans['pred']
                test['label'] = test['label'].apply(lambda x: label_map[x])
                
                if validation_check:
                    dat_tst = data_tst.copy()
                    main.evaluation(dat_tst, y_pred)
                
                if time_setting == 'minute':
                    test.to_csv(tst_file[:-4]+'_'+time_setting+'_nn_predicted.csv', index=False)
                elif time_setting == 'hour':
                    test.to_csv(tst_file[:-4]+'_nn.csv', index=False)

    else:
        print('training')
        prev_loss = 10000
        cur_loss = prev_loss
        stop_count = 0
        for ep in range(epochs):
            print('Epoch: ' + str(ep+1) + '/' + str(epochs))
            model.trainModel(X_train, y_train, batch_size)
        
            #if validation_check:
            val_acc = model.validationModel(X_val, y_val)
            print('val acc: ' + str(val_acc))
            cur_loss = val_acc[0]
            if prev_loss > cur_loss:
                prev_loss = cur_loss
                stop_count = 0
            else:
                stop_count += 1
            if stop_count >= early_stop_patience:
                break

    ## testing
    if not pretrained:
        if validation_check:
            print('testing')
            val_pred, val_prob = model.testModel(X_val)
            xgb.eval(val_test, val_pred)
            y_pred, y_prob = model.testModel(X_tst)
            xgb.eval(y_test, y_pred)

        ## save model
        print('save model')
        model.saveModel(nn_model)
        np.save(norm, norm_zscore)        

