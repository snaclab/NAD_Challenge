
import tensorflow as tf
import numpy as np
#from tensorflow import keras
import learning
import preprocess
import argparse
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, Dense, TimeDistributed, LSTM, Conv1D, Flatten, Masking
import csv
import os
import random
import pickle
import pandas as pd
import datetime

##remove these
#import main_
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
validation_check = False

seq_step = 0
main_feature = 15+32+45+4
num_feature = (main_feature)*(1+seq_step*2)#+64#+32#+64+32
pca_feature = num_feature#32
num_class = 5
#NORMAL_OFFSET = #int(180625*1.3)
#app_name = {}
app_encoder = 'app_encoder.pkl'
processed_trn_data = 'X.npy'
processed_trn_label = 'y.npy'
processed_tst_data = 'X_test.npy'
processed_tst_label = 'y_test.npy'
#time_trn = 'time_mark.npy'
#time_tst = 'time_mark_test.npy'
#ip_trn = 'ip_mark.pickle'
#ip_tst = 'ip_mark_test.pickle'


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn', nargs='+', help='input training dataset', required=False)
    parser.add_argument('--tst', nargs='+', help='input testing dataset', required=False)
    parser.add_argument('--pretrained', help='if there is pretrained encoder', default=False)
    return parser.parse_args()

def genData(files, app_name, train_file=True):
    ##load data from csv file, and process it into numpy array

    #app_name = {}#attr 8, 9 are discrete, 8(protocal ID) has 32(256) and 9(app name) has 45
    #if not train_file:
    #    with open(app_encoder, 'rb') as fp:
    #        app_name = pickle.load(fp)
    app_id = 0
    label = {'Normal':1, 'Probing-Nmap':3, 'Probing-Port sweep':4, 'Probing-IP sweep':2, 'DDOS-smurf':0}#attr -1
    data_n = 0
    data = []
    normal_size = 0
    time_mark = []
    ip_mark = []
    
    normal_id = []
    abnormal_id = []
    
    #check_in = False
    dat_id = 0
    #check_in_id = [(0, 1000)]#[(i, i+1000) for i in range(0, 6000000, int(6000000/234))]

    
    #if train_file:
    #    files = ['../NAD/train.csv']#, '../NAD/1210_firewall.csv']
    #else:
    #    files = ['../NAD/test.csv']
    #with open('xgboost_pred.pickle', 'rb') as fp:
    #    xgboost_value = pickle.load(fp)
    #xgboost_pred = np.argmax(xgboost_value, axis=-1)
    for file_name in files:
        with open(file_name, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                data.append(row)

                #time_all = row[0].split()
                #day = time_all[0].split('-')
                #clock = time_all[1].split(':')
                #time_stamp = int(day[2])*24*60*60 + int(clock[0])*60*60 + int(clock[1])*60 + int(clock[2])
                #time_mark.append((time_stamp, data_n))
                
                #print(time_stamp)
                #time_mark.append(row[0])
                #src_ip_mark.append(row[1])
                #dst_ip_mark.append(row[2])
                if train_file:
                    if not row[9] in app_name:
                        app_name[row[9]] = app_id
                        app_id += 1
                else:
                    if not row[9] in app_name:
                        app_name[row[9]] = app_name['others']
                data_n+=1
                
                    
                #dat_id += 1
    X = np.zeros((data_n, main_feature))
    y = np.zeros((data_n, num_class))
    #time_mark = np.zeros((data_n, 2), dtype=np.int64)
    #time_mark = np.zeros((data_n, 2), dtype=np.int64)
    dat_i = 0

    for datum_i in range(data_n):
        x_i = 0
        #datum_seq = [datum_main]
        #for step in range(1, seq_step+1):
        #    datum_seq.append(min(datum_main+step, dat_id-1))
        #    datum_seq.append(max(datum_main-step, 0))
        #for datum_i in datum_seq:
        for attr in range(5, 22):
            #skip attr 8 and attr 9 first
            if attr != 8 and attr != 9:
                X[datum_i, x_i] = int(data[datum_i][attr])
                x_i += 1
        #one hot encoding of attr 8
        X[datum_i, x_i+int(data[datum_i][8])] = 1
        x_i += 32
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
        #dst IP end with 255:
        if int(ip[-1]) == 255:
            X[datum_i, x_i] = 1
        x_i+=1
        '''
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
            y[datum_i, 0] = 1
        #dat_i += 1
        
        ##time_all = data[datum_i][0].split()
        ##day = time_all[0].split('-')
        ##clock = time_all[1].split(':')
        ##time_mark[datum_i, 0] = int(day[2])*24*60*60 + int(clock[0])*60*60 + int(clock[1])*60 + int(clock[2])
        
        ##ip = data[datum_i][1].split('.')
        ##time_mark[datum_i, 1] = int(ip[0])*(256**3)+int(ip[1])*(256**2)+int(ip[2])*(256)+int(ip[3])
    ##argsort does not need if original file is (close to) sorted.
    #time_id = np.argsort(time_stamp, kind='stable')
    #time_mark[:, 0] = time_stamp[:]
    #time_mark[:, 1] = time_id[:]

    return X, y#, time_mark#, ip_mark#src_ip_mark, dst_ip_mark

#def BalancedData(X, y, time_mark, train_file=True):
def BalancedData(X, y):
    
    y_id = np.argmax(y, axis=-1)
    #id_actual_map = {}
    #for i in range(len(time_id)):
    #    id_actual_map[time_id[i, 1]] = i
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
        
        ##find seq
        '''
        if seq_step > 0:
            #seq_id[i]=[]
            

            
            step = 0
            prev_step = i
            cur_step = i - 1
            while step < seq_step:
                if cur_step >= 0:
                    if abs(time_mark[i, 0]-time_mark[cur_step, 0]) < 5:
                        if time_mark[i, 1] == time_mark[cur_step, 1]:
                            seq_id[i].append(cur_step)
                            prev_step = cur_step
                            step += 1
                        cur_step -= 1
                    else:
                        seq_id[i].append(prev_step)
                        step+=1
                else:
                    seq_id[i].append(prev_step)
                    step+=1

            step = 0
            prev_step = i
            cur_step = i + 1
            while step < seq_step:
                if cur_step < len(y_id):
                    if abs(time_mark[i, 0]-time_mark[cur_step, 0]) < 5:
                        if time_mark[i, 1] == time_mark[cur_step, 1]:
                            seq_id[i].append(cur_step)
                            prev_step = cur_step
                            step += 1
                        cur_step += 1
                    else:
                        seq_id[i].append(prev_step)
                        step += 1
                else:
                    seq_id[i].append(prev_step)
                    step += 1
                    

            
            bound_id = [i, i]
            for step in range(1, seq_step+1):
                if i-step >= 0:
                    if abs(time_mark[i]-time_mark[i-step]) < 5*60:
                        bound_id[0] = i-step
                seq_id[i].append(bound_id[0])
                if i+step < len(y_id):
                    if abs(time_mark[i]-time_mark[i+step]) < 5*60:
                        bound_id[1] = i+step
                seq_id[i].append(bound_id[1])
        '''
    NORMAL_OFFSET = int(NORMAL_OFFSET*1.3) 

    #if train_file:
    data_id = abnormal_id[:]
    balanced_normal_id = random.sample(normal_id, NORMAL_OFFSET)
    data_id.extend(balanced_normal_id)
    data_n = len(data_id)
    #else:
    #    data_id = list(range(len(y_id)))
    #    data_n = len(y_id)


    ##find seq
    '''
    if seq_step > 0:
        for i in data_id:
            seq_id[i]=[]
            
            step = 0
            prev_step = i
            cur_step = i - 1
            while step < seq_step:
                if cur_step >= 0:
                    if abs(time_mark[i, 0]-time_mark[cur_step, 0]) < 2:
                        if time_mark[i, 1] == time_mark[cur_step, 1]:
                            seq_id[i].append(cur_step)
                            prev_step = cur_step
                            step += 1
                        cur_step -= 1
                    else:
                        seq_id[i].append(prev_step)
                        step+=1
                else:
                    seq_id[i].append(prev_step)
                    step+=1

            step = 0
            prev_step = i
            cur_step = i + 1
            while step < seq_step:
                if cur_step < len(y_id):
                    if abs(time_mark[i, 0]-time_mark[cur_step, 0]) < 2:
                        if time_mark[i, 1] == time_mark[cur_step, 1]:
                            seq_id[i].append(cur_step)
                            prev_step = cur_step
                            step += 1
                        cur_step += 1
                    else:
                        seq_id[i].append(prev_step)
                        step += 1
                else:
                    seq_id[i].append(prev_step)
                    step += 1
                    
    '''

    X_bal = np.zeros((data_n, num_feature))
    y_bal = np.zeros((data_n, num_class))
    
    dat_i = 0
    for datum_i in data_id:
        x_i = 0
        X_bal[dat_i, x_i:x_i+main_feature] = X[datum_i, :]
        y_bal[dat_i, :] = y[datum_i, :]
        
        ##add seq
        '''
        if seq_step > 0:
            x_i += main_feature
            for step_id in seq_id[datum_i]:
                X_bal[dat_i, x_i:x_i+main_feature] = X[step_id, :]
                x_i += main_feature
        '''
        dat_i += 1

    return X_bal, y_bal

def Normalization(X_train, X_test, norm_std):
    ##normalization -> [0, 1]
    for x_i in range(15):
        
        concat = np.zeros(len(X_train)+len(X_test))
        concat[:len(X_train)] = X_train[:, x_i]
        concat[len(X_train):] = X_test[:, x_i]
        #mean = np.mean(concat)
        std = np.std(concat)
        norm_std[x_i] = std
        X_train[:, x_i] = (X_train[:, x_i])/std
        X_test[:, x_i] = (X_test[:, x_i])/std
        
        '''
        max_v = max(np.max(X_train[:, x_i]), np.max(X_test[:, x_i]))
        min_v = max(np.min(X_train[:, x_i]), np.min(X_test[:, x_i]))
        if max_v-min_v > 0:
            X_train[:, x_i] = (X_train[:, x_i]-min_v)/(max_v-min_v)
            X_test[:, x_i] = (X_test[:, x_i]-min_v)/(max_v-min_v)
        '''

class nnModel():
    def __init__(self):
        unit_size = 16

        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(num_feature,)))
        self.model.add(Dense(32, activation='relu'))
        #self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(num_class, activation='softmax'))
        #self.model.add(Masking(mask_value=0.0, input_shape=(time_step, pca_feature)))
        #self.model.add(Conv1D(2*unit_size, 20, activation='relu', padding='same', input_shape=(time_step, pca_feature)))
        #self.model.add(TimeDistributed(Dense(32, input_shape=(None, num_feature))))
        #self.model.add(LSTM(unit_size, return_sequences=True))
        #self.model.add(LSTM(unit_size, return_sequences=True))
        #self.model.add(Bidirectional(LSTM(unit_size, return_sequences=True), input_shape=(None, pca_feature)))
        #self.model.add(LSTM(unit_size, return_sequences=True, input_shape=(None, num_feature)))
        #self.model.add(TimeDistributed(Dense(8, activation='relu')))
        #self.model.add(Flatten())
        #self.model.add(Dense(256, activation='relu'))
        #self.model.add(TimeDistributed(Dense(num_class, activation='softmax')))
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.model.summary()

    def trainModel(self, X, y, batch_size):
        ##TODO: can it handle sequences with different length?
        history = self.model.fit(X, y, epochs=1, batch_size=batch_size)
    
    def validationModel(self, X, y):
        results = self.model.evaluate(X, y, batch_size=2048)
        return results

    def testModel(self, X):
        results = self.model.predict(X, batch_size=2048)
        #seq = results.shape[1]

        #y_value = np.zeros((num_tst, num_class))
        #y_pred = np.zeros(num_tst, dtype=np.int64)
        #for i in range(num_tst):
        #    y_value[i, :] = results[int(i/seq), i%seq, :]
        y_pred = np.argmax(results, axis=-1)
        return y_pred, results

    def saveModel(self, model_n):
        self.model.save(model_n)
    def loadModel(self, model_n):
        self.model = load_model(model_n)

if __name__ == '__main__':
    
    args = parse_arg()
    #print(args.trn)
    pretrained = False
    if args.pretrained == "True":
        pretrained = True
    ## data preprocessing
    print('preprocessing')
    split = 0.8
    norm_std = np.zeros(15)
    app_name={}
    ## load data (training, validation and testing)
    if not pretrained:

        if os.path.isfile(processed_trn_data) and os.path.isfile(processed_trn_label):
            X = np.load(processed_trn_data)
            y = np.load(processed_trn_label)
            #time_mark = np.load(time_trn)
        else:
            X, y = genData(args.trn, app_name)
            np.save(processed_trn_data, X)
            np.save(processed_trn_label, y)
            with open(app_encoder, 'wb') as fp:
                pickle.dump(app_name, fp)    
            #np.save(time_trn, time_mark)
    
        if os.path.isfile(processed_tst_data):
            # and os.path.isfile(processed_tst_label):
            X_tst = np.load(processed_tst_data)
            #y_tst = np.load(processed_tst_label)
            #time_mark_tst = np.load(time_tst)
        else:
            if validation_check:
                X_tst, y_tst = genData(args.tst, app_name)
                np.save(processed_tst_data, X_tst)
                np.save(processed_tst_label, y_tst)
                with open(app_encoder, 'wb') as fp:
                    pickle.dump(app_name, fp)    
            else:
                X_tst, _ = genData(args.tst, app_name, False)
                np.save(processed_tst_data, X_tst)
            #np.save(time_tst, time_mark_tst)
    
        num_trn = len(X)
        num_tst = len(X_tst)
        print('num training data: ' + str(num_trn)) 
        print('num testing data: ' + str(num_tst)) 
        Normalization(X, X_tst, norm_std)
        #print(time_mark)
    
        X, y = BalancedData(X, y)
        #if seq_step > 0:
        #    X_tst, y_tst = BalancedData(X_tst, y_tst, time_mark_tst, False)
        num_trn = len(X)
        print('balanced num training data: ' + str(num_trn)) 
        data_split = int(split*len(X))
        num_val = len(X[data_split:])
        #X, X_tst = preprocess.PCA_transform(X, X_tst, pca_feature)
        if validation_check:
            y_test = np.argmax(y_tst, axis=-1)
    
        if validation_check:
            shuffle_id = np.arange(num_trn)
            np.random.seed(7)
            np.random.shuffle(shuffle_id)
            X = X[shuffle_id]
            y = y[shuffle_id]
            #X_train, y_train = genSeqData(X[:data_split], y[:data_split], False, max_len=seq_step)
            #X_val, y_val = genSeqData(X[data_split:], y[data_split:], False, max_len=seq_step)
            X_train, y_train, X_val, y_val = X[:data_split], y[:data_split], X[data_split:], y[data_split:]
            val_test = np.argmax(y[data_split:], axis=-1)
            #print('Each label in training data:')
            #count = {0:0, 1:0, 2:0, 3:0, 4:0}
            #for lab in np.argmax(y_train, axis=-1):
            #    count[lab]+=1
            #print(count)
        else:
            X_train, y_train = X, y
    
    #else:
    #    X_train, y_train = genSeqData(X[:], y[:], True, max_len=seq_step)
        #X_train, y_train, len_seq = genIPSeqData(X[:], y[:], ip_mark)
     
    #print(len_seq)
    #print(len(len_seq))
    #X_test, y_test = genSeqData(X_tst, y_tst, False, max_len=seq_step, test_label=True)
    #X_test, y_test, len_seq_test = genIPSeqData(X_tst, y_tst, ip_mark_tst, test_label=True)
    #num_tst = sum(len_seq_test)
    #print('num training data: ' + str(len(X_train))) 
    #print('num validation data: ' + str(num_val))  

    #X, y = None, None
    #X_tst, y_tst = None, None
    
    ## build model 
    print('build model')
    model = nnModel()
    ## load model 
    if pretrained:
        print('load model')
        model.loadModel('nn.h5')
        norm_std = np.load('norm_std.npy')
        with open(app_encoder, 'rb') as fp:
            app_name = pickle.load(fp)
    
    ## training (adust hyper parameters)
    epochs = 35
    batch_size = 128#2048 is upper bound

    if pretrained:
        print('testing')
        for tst_file in args.tst:
            #data_tst = pd.read_csv(tst_file[:-4]+'_processed.csv')
            #with open(app_encoder, 'rb') as fp:
            #    app_name = pickle.load(fp)
            X, _ = genData([tst_file], app_name, False)
            for x_i in range(15):
                X[:, x_i] = (X[:, x_i])/norm_std[x_i]

            nn_pred, nn_prob = model.testModel(X)
            
            df_pred = pd.DataFrame(columns=[0,1,2,3,4], data=nn_prob)
            # postprocess
            test = pd.read_csv(tst_file)
            ans = test[['time','src']].copy()
            ans = pd.concat([ans, df_pred], axis=1)
            ans['time'] = ans['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
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
            y_pred = ans['pred'].values
        
            # Transform label and save data
            label_map = {0: 'DDOS-smurf', 1: 'Normal', 2: 'Probing-IP sweep', 3: 'Probing-Nmap', 4: 'Probing-Port sweep'}
            test['label'] = ans['pred']
            test['label'] = test['label'].apply(lambda x: label_map[x])
            #if str(args.eval)=='True':
            # Evaluation
            #dat_tst = data_tst.copy()
            #main_.evaluation(dat_tst, y_pred)
            test.to_csv(tst_file[:-4]+'_nn.csv', index=False)
    else:
        print('training')
        for ep in range(epochs):
            print('Epoch: ' + str(ep+1) + '/' + str(epochs))
            #for i in range(0, len(X_train), batch_size):
            #    print('Progress: ' + str(i) + '/' + str(len(X_train)))
            #model.trainModel(X_train[i:i+batch_size], y_train[i:i+batch_size])
            model.trainModel(X_train, y_train, batch_size)
        
            #trn_acc = model.validationModel(X_train, y_train)
            if validation_check:
                val_acc = model.validationModel(X_val, y_val)
                #print('training acc: '+str(trn_acc)+' val acc: ' + str(val_acc))
                print('val acc: ' + str(val_acc))
    
    
    ## testing
    if not pretrained:
        if validation_check:
            print('testing')
            val_pred, val_prob = model.testModel(X_val)
            learning.eval(val_test, val_pred)
            y_pred, y_prob = model.testModel(X_tst)
            learning.eval(y_test, y_pred)

    ## save model
    if not pretrained:
        print('save model')
        model.saveModel('nn.h5')
        np.save('norm_std.npy', norm_std)        
        #with open(app_encoder, 'wb') as fp:
        #    pickle.dump(app_name, fp)    

