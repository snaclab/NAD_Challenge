
import tensorflow as tf
import numpy as np
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, LSTM
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"]="9"
num_feature = 15+32+45
num_class = 5
processed_trn_data = 'X.npy'
processed_trn_label = 'y.npy'

def genData(file_name):
    app_name = {}#attr 8, 9 are discrete, 8(protocal ID) has 32 and 9(app name) has 45
    app_id = 0
    label = {'Normal':0, 'Probing-Nmap':1, 'Probing-Port sweep':2, 'Probing-IP sweep':3, 'DDOS-smurf':4}#attr -1
    data_n = 0
    data = []

    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data_n+=1
            data.append(row)
            if not row[9] in app_name:
                app_name[row[9]] = app_id
                app_id += 1

    X = np.zeros((data_n, num_feature))
    y = np.zeros((data_n, num_class))

    for datum_i in range(data_n):
        x_i = 0
        for attr in range(5, 22):
            if attr == 8:
                pass
            elif attr == 9:
                pass
                #X[datum_i, attr-5] = app_name[data[datum_i][attr]]
            else:
                X[datum_i, x_i] = int(data[datum_i][attr])
                x_i += 1
        X[datum_i, x_i+int(data[datum_i][8])] = 1
        
        x_i += 32
        X[datum_i, x_i+int(app_name[data[datum_i][9]])] = 1

        y[datum_i, label[data[datum_i][22]]] = 1
    
    ##normalization -> [0, 1] 
    for x_i in range(15):
        if np.max(X[:, x_i])-np.min(X[:, x_i]) > 0:
            X[:, x_i] = (X[:, x_i]-np.min(X[:, x_i]))/(np.max(X[:, x_i])-np.min(X[:, x_i]))
    data = []
    
    return X, y

def genSeqData(X, y, overlap = True, max_len=100):
    data_n = len(X)
    if overlap:
        X_seq = np.zeros((data_n-max_len+1, max_len, num_feature))
        y_seq = np.zeros((data_n-max_len+1, max_len, num_class))

        for datum_i in range(data_n-max_len+1):
            X_seq[datum_i, :, :] = X[datum_i:datum_i+max_len, :]
            y_seq[datum_i, :, :] = y[datum_i:datum_i+max_len, :]
    else:
        X_seq = np.zeros((int(np.ceil(data_n/max_len)), max_len, num_feature))
        y_seq = np.zeros((int(np.ceil(data_n/max_len)), max_len, num_class))
        
        x_i = 0
        if data_n%max_len != 0:
            for datum_i in range(int(np.ceil(data_n/max_len)-1)):
                X_seq[datum_i, :, :] = X[x_i:x_i+max_len, :]
                y_seq[datum_i, :, :] = y[x_i:x_i+max_len, :]
                x_i += max_len
            
            X_seq[-1, :data_n%max_len, :] = X[x_i:x_i+max_len, :]
            y_seq[-1, :data_n%max_len, :] = y[x_i:x_i+max_len, :]
        
        else:
            for datum_i in range(int(np.ceil(data_n/max_len))):
                X_seq[datum_i, :, :] = X[x_i:x_i+max_len, :]
                y_seq[datum_i, :, :] = y[x_i:x_i+max_len, :]
                x_i += max_len

        #if data_n%max_len != 0:
        #    X_seq[-1, -(data_n%max_len):, :] = 0
        #    y_seq[-1, -(data_n%max_len):, :] = 0
            

    return X_seq, y_seq
    #np.save('X.npy', X)
    #np.save('y.npy', y)


class rnnModel():
    
    def __init__(self):
        unit_size = 16

        self.model = Sequential()
        self.model.add(LSTM(unit_size, input_shape=(None, num_feature), return_sequences=True))
        self.model.add(TimeDistributed(Dense(num_class, activation='softmax')))
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.model.summary()

    def trainModel(self, X, y, batch_size):
        history = self.model.fit(X, y, epochs=1, batch_size=batch_size)
    
    def validationModel(self, X, y):
        results = self.model.evaluate(X, y, batch_size=2048)
        return results

    def testModel(self):
        pass


if __name__ == '__main__':
    
    ## data preprocessing
    print('preprocessing')
    split = 0.7
    
    if os.path.isfile(processed_trn_data) and os.path.isfile(processed_trn_label):
        X = np.load(processed_trn_data)
        y = np.load(processed_trn_label)
    else:
        X, y = genData('../NAD/1203_firewall.csv')
        np.save(processed_trn_data, X)
        np.save(processed_trn_label, y)

    print('num data: ' + str(len(X))) 
    data_split = int(split*len(X))
    X_train, y_train = genSeqData(X[:data_split], y[:data_split], True)
    X_test, y_test = genSeqData(X[data_split:], y[data_split:], False)

    X = None
    y = None

    ## build model 
    print('build model')
    model = rnnModel()
    
    ## training
    print('training')
    epochs = 20
    batch_size = 2048


    for ep in range(epochs):
        print('Epoch: ' + str(ep+1) + '/' + str(epochs))
        #for i in range(0, len(X_train), batch_size):
        #    print('Progress: ' + str(i) + '/' + str(len(X_train)))
        #model.trainModel(X_train[i:i+batch_size], y_train[i:i+batch_size])
        model.trainModel(X_train, y_train, batch_size)
        
        #trn_acc = model.validationModel(X_train, y_train)
        val_acc = model.validationModel(X_test, y_test)
        #print('training acc: '+str(trn_acc)+' val acc: ' + str(val_acc))
        print('val acc: ' + str(val_acc))
    
    ## testing
    pass



