
import tensorflow as tf
import numpy as np
#from tensorflow import keras
import learning
import preprocess
import argparse
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, TimeDistributed, LSTM, Conv1D, Flatten, Masking
import csv
import os
import random
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="8"

validation_check = False
num_feature = 15+32+45+1#+64#+32#+64+32
pca_feature = num_feature#32
num_class = 4#5
seq_step = 10
NORMAL_OFFSET = int(180625*1.3)
app_name = {}
processed_trn_data = 'X.npy'
processed_trn_label = 'y.npy'
processed_tst_data = 'X_test.npy'
processed_tst_label = 'y_test.npy'
time_trn = 'time_mark.pickle'
time_tst = 'time_mark_test.pickle'
ip_trn = 'ip_mark.pickle'
ip_tst = 'ip_mark_test.pickle'


def parse_arg():
    parser = argparse.ArgumentParser(description='Process dataset name')
    parser.add_argument('--trn', help='input training dataset', required=True)
    parser.add_argument('--tst', help='input testing dataset', required=True)

    return parser.parse_args()

def genData(train_file):
    ##load data from csv file, and process it into numpy array

    #app_name = {}#attr 8, 9 are discrete, 8(protocal ID) has 32(256) and 9(app name) has 45
    app_id = 0
    label = {'Normal':-1, 'Probing-Nmap':2, 'Probing-Port sweep':3, 'Probing-IP sweep':1, 'DDOS-smurf':0}#attr -1
    data_n = 0
    data = []
    normal_size = 0
    time_mark = []
    ip_mark = []
    
    check_in = False
    dat_id = 0
    check_in_id = [(0, 1000)]#[(i, i+1000) for i in range(0, 6000000, int(6000000/234))]

    
    if train_file:
        files = ['../NAD/train.csv']#, '../NAD/1210_firewall.csv']
    else:
        files = ['../NAD/test.csv']

    for file_name in files:
        with open(file_name, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                ##normal check in
                '''
                if train_file:
                    check_in = False
                    for offset in check_in_id:
                        if label[row[-1]] == 1 and dat_id >= offset[0] and dat_id <= offset[1]:
                            check_in = True
                            break
                    #normal_size += 1
                
                dat_id += 1
                '''
                #if train_file == False or check_in or label[row[-1]] != 1:
                if label[row[-1]] != -1:
                    #if label[row[-1]] <= 1:
                    #if label[row[-1]] != 0 or normal_size <= NORMAL_OFFSET:
                    data_n+=1
                    data.append(row)
                    time_mark.append(row[0])
                    #src_ip_mark.append(row[1])
                    ip_mark.append(row[2])
                    if not row[9] in app_name:
                        app_name[row[9]] = app_id
                        app_id += 1

    X = np.zeros((data_n, num_feature))
    y = np.zeros((data_n, num_class))

    for datum_i in range(data_n):
        x_i = 0
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
        y[datum_i, label[data[datum_i][22]]] = 1
    
    return X, y, time_mark, ip_mark#src_ip_mark, dst_ip_mark

def genIPSeqData(X, y, ip_mark, test_label=False):
    
    ip_idx = {}

    idx = 0
    for ip in ip_mark:
        if not ip in ip_idx:
            ip_idx[ip] = []
        ip_idx[ip].append(idx)
        idx+=1
    
    data_seq = []
    for ip in ip_idx.keys():
        if len(ip_idx[ip]) > 10:
            data_idx = []
            count = 0

            for i in ip_idx[ip]:
                data_idx.append(i)
                count += 1

                if count == seq_step:
                    count = 0
                    data_seq.append(data_idx[:])
                    data_idx = []

            data_seq.append(data_idx[:])

    len_seq = []
    X_seq = np.zeros((len(data_seq), seq_step, num_feature))
    y_seq = np.zeros((len(data_seq), seq_step, num_class))

    for i in range(len(data_seq)):
        len_seq.append(len(data_seq[i]))
        for idx in data_seq[i]:
            s = 0
            X_seq[i, s, :] = X[idx, :]
            y_seq[i, s, :] = y[idx, :]
            s+=1
    
    if test_label:
        total_samples = sum(len_seq)
        print(total_samples)
        y_ = np.zeros((total_samples, num_class))
        s = 0
        for i in range(len(data_seq)):
            for idx in data_seq[i]:
                y_[s, :] = y[idx, :]
                s+=1
        y_seq = np.argmax(y_, axis=1)

    return X_seq, y_seq, len_seq


def genTimeSeqData(X, y, time_mark):
    ##split data by time mark
    ##1 minute as a unit
    #day = None
    #hms = [-1, -1, -1]
    idx = 0
    dateindex = {}

    for t in time_mark:
        date = t.split()
        day = date[0]
        hms = date[1].split(':')
        
        if not day in dateindex:
            dateindex[day] = []
            for hr in range(24):
                empty = []
                for mi in range(60):
                    empty.append([])
                dateindex[day].append(empty)
        dateindex[day][int(hms[0])][int(hms[1])].append(idx)
        idx += 1

    for d in dateindex.keys():
        for hr in range(24):
            for mi in range(60):
                
                if len(dateindex[d][hr][mi]) > 0:
                    print(len(dateindex[d][hr][mi]))
    


def genSeqData(X, y, overlap = True, max_len=50, test_label = False):
    ##turn data into a batch of sequences
    #  1       1, 2           |  1, 2
    #  2       2, 3           |  3, 4
    #  3   ->  3, 4  (overlap)|  5, 0 (not overlap)
    #  4       4, 5           |
    #  5                      |

    data_n = len(X)
    if overlap:
        X_seq = np.zeros((data_n-max_len+1, max_len, pca_feature))
        y_seq = np.zeros((data_n-max_len+1, max_len, num_class))

        for datum_i in range(data_n-max_len+1):
            X_seq[datum_i, :, :] = X[datum_i:datum_i+max_len, :]
            y_seq[datum_i, :, :] = y[datum_i:datum_i+max_len, :]
    else:
        X_seq = np.zeros((int(np.ceil(data_n/max_len)), max_len, pca_feature))
        y_seq = np.zeros((int(np.ceil(data_n/max_len)), max_len, num_class))
        
        x_i = 0
        if data_n%max_len != 0:
            for datum_i in range(int(np.ceil(data_n/max_len)-1)):
                X_seq[datum_i, :, :] = X[x_i:x_i+max_len, :]
                y_seq[datum_i, :, :] = y[x_i:x_i+max_len, :]
                x_i += max_len
            
            X_seq[-1, :data_n%max_len, :] = X[x_i:, :]
            y_seq[-1, :data_n%max_len, :] = y[x_i:, :]
        
        else:
            for datum_i in range(int(np.ceil(data_n/max_len))):
                X_seq[datum_i, :, :] = X[x_i:x_i+max_len, :]
                y_seq[datum_i, :, :] = y[x_i:x_i+max_len, :]
                x_i += max_len

            
    if test_label:
        y_seq = np.argmax(y, axis=1)
    return X_seq, y_seq

def Normalization(X_train, X_test):
    ##normalization -> [0, 1]
    for x_i in range(15):
        max_v = max(np.max(X_train[:, x_i]), np.max(X_test[:, x_i]))
        min_v = max(np.min(X_train[:, x_i]), np.min(X_test[:, x_i]))
        if max_v-min_v > 0:
            X_train[:, x_i] = (X_train[:, x_i]-min_v)/(max_v-min_v)
            X_test[:, x_i] = (X_test[:, x_i]-min_v)/(max_v-min_v)

class rnnModel():
    def __init__(self, time_step = 100):
        unit_size = 16

        self.model = Sequential()
        self.model.add(Masking(mask_value=0.0, input_shape=(time_step, pca_feature)))
        #self.model.add(Conv1D(2*unit_size, 20, activation='relu', padding='same', input_shape=(time_step, pca_feature)))
        #self.model.add(TimeDistributed(Dense(32, input_shape=(None, num_feature))))
        #self.model.add(LSTM(unit_size, return_sequences=True))
        self.model.add(LSTM(unit_size, return_sequences=True))
        #self.model.add(Bidirectional(LSTM(unit_size, return_sequences=True), input_shape=(None, pca_feature)))
        #self.model.add(LSTM(unit_size, return_sequences=True, input_shape=(None, num_feature)))
        #self.model.add(TimeDistributed(Dense(8, activation='relu')))
        #self.model.add(Flatten())
        #self.model.add(Dense(256, activation='relu'))
        self.model.add(TimeDistributed(Dense(num_class, activation='softmax')))
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.model.summary()

    def trainModel(self, X, y, batch_size):
        ##TODO: can it handle sequences with different length?
        history = self.model.fit(X, y, epochs=1, batch_size=batch_size)
    
    def validationModel(self, X, y):
        results = self.model.evaluate(X, y, batch_size=2048)
        return results

    def testModel(self, X, num_tst):
        results = self.model.predict(X, batch_size=2048)
        seq = results.shape[1]

        y_pred = np.zeros(num_tst, dtype=np.int64)
        for i in range(num_tst):
            y_pred[i] = np.argmax(results[int(i/seq), i%seq, :])
        return y_pred

    def saveModel(self, model_n):
        self.model.save(model_n)

if __name__ == '__main__':
    
    #args = parse_arg()
    
    ## data preprocessing
    print('preprocessing')
    split = 0.7
    
    ## load data (training, validation and testing)
    if os.path.isfile(processed_trn_data) and os.path.isfile(processed_trn_label):
        X = np.load(processed_trn_data)
        y = np.load(processed_trn_label)
        with open(time_trn, 'rb') as fp:
            time_mark = pickle.load(fp)
        with open(ip_trn, 'rb') as fp:
            ip_mark = pickle.load(fp)
    else:
        #X, y = genData(args.trn)
        X, y, time_mark, ip_mark = genData(True)
        np.save(processed_trn_data, X)
        np.save(processed_trn_label, y)
        with open(time_trn, 'wb') as fp:
            pickle.dump(time_mark, fp)
        with open(ip_trn, 'wb') as fp:
            pickle.dump(ip_mark, fp)
    
    if os.path.isfile(processed_tst_data) and os.path.isfile(processed_tst_label):
        X_tst = np.load(processed_tst_data)
        y_tst = np.load(processed_tst_label)
        with open(time_tst, 'rb') as fp:
            time_mark_tst = pickle.load(fp)
        with open(ip_tst, 'rb') as fp:
            ip_mark_tst = pickle.load(fp)
    else:
        X_tst, y_tst, time_mark_tst, ip_mark_tst = genData(False)
        np.save(processed_tst_data, X_tst)
        np.save(processed_tst_label, y_tst)
        with open(time_tst, 'wb') as fp:
            pickle.dump(time_mark_tst, fp)
        with open(ip_tst, 'wb') as fp:
            pickle.dump(ip_mark_tst, fp)

    num_trn = len(X)
    num_tst = len(X_tst)
    print('num training data: ' + str(num_trn)) 
    print('num testing data: ' + str(num_tst)) 
    data_split = int(split*len(X))
    num_val = len(X[data_split:])
    Normalization(X, X_tst)
    #X, X_tst = preprocess.PCA_transform(X, X_tst, pca_feature)

    if validation_check:
        shuffle_id = np.arange(num_trn)
        np.random.shuffle(shuffle_id)
        X = X[shuffle_id]
        y = y[shuffle_id]
        X_train, y_train = genSeqData(X[:data_split], y[:data_split], False, max_len=seq_step)
        X_val, y_val = genSeqData(X[data_split:], y[data_split:], False, max_len=seq_step)
        val_test = np.argmax(y[data_split:], axis=-1)
    else:
        X_train, y_train = genSeqData(X[:], y[:], True, max_len=seq_step)
        #X_train, y_train, len_seq = genIPSeqData(X[:], y[:], ip_mark)

    #print(len_seq)
    #print(len(len_seq))
    X_test, y_test = genSeqData(X_tst, y_tst, False, max_len=seq_step, test_label=True)
    #X_test, y_test, len_seq_test = genIPSeqData(X_tst, y_tst, ip_mark_tst, test_label=True)
    #num_tst = sum(len_seq_test)

    X, y = None, None
    X_tst, y_tst = None, None
    
    ## build model 
    print('build model')
    model = rnnModel(seq_step)
    
    ## training (adust hyper parameters)
    print('training')
    epochs = 20
    batch_size = 2048#2048 is upper bound


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
    print('testing')
    if validation_check:
        val_pred = model.testModel(X_val, num_val)
        learning.eval(val_test, val_pred)
    
    y_pred = model.testModel(X_test, num_tst)
    #print(len(y_test))
    #print(len(y_pred))
    learning.eval(y_test, y_pred)

    ## save model
    print('save model')
    #model.saveModel('bidirection_lstm.h5')
     

