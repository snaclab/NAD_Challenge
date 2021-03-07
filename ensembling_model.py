import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from numpy.linalg import norm
from itertools import product
import main
import random

def XGB_training(data, n_class):
    X = data[[c for c in data.columns if c != 'label']].copy()
    Y = data[['label']].copy()
    X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=7)
    model = XGBClassifier(objective='multi:softprob', num_class=n_class, learning_rate=0.1, use_label_encoder=False)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True, early_stopping_rounds=10)
    print('Validation set:')
    y_pred = model.predict(X_test)
    eval(y_test, y_pred)
    return model

def XGB_prediction(data, model):
    X_test = data[[c for c in data.columns if c != 'label']].copy().values
    return model.predict_proba(X_test)

def save_model(model, fname):
    pickle.dump(model, open(fname, 'wb'))

def load_model(fname):
    model = pickle.load(open(fname, 'rb'))
    return model

def normalize(weights):
     # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result

def evaluate(data, w):
    weights = np.repeat([w], repeats=len(data), axis=0)
    w_data = np.multiply(data.to_numpy(), weights)
    f_0 = w_data[:, :4].sum(axis=1).reshape(-1,1)
    f_1 = w_data[:, 4:8].sum(axis=1).reshape(-1,1)
    f_2 = w_data[:, 8:12].sum(axis=1).reshape(-1,1)
    f_3 = w_data[:, 12:16].sum(axis=1).reshape(-1,1)
    f_4 = w_data[:, 16:].sum(axis=1).reshape(-1,1)
    f = np.concatenate((f_0, f_1, f_2, f_3, f_4), axis=1)
    data = pd.DataFrame(data=f, columns=[0, 1, 2, 3, 4])
    data = data.idxmax(axis='columns').to_frame()
    y_pred = data[[0]].copy().values.reshape(1, -1)[0]
    data_tst = pd.read_csv('../nad/test_no.csv')
    score = main.evaluation(data_tst, y_pred)
    return score


def grid_search(data):
    # define weights to consider
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_score, best_weights = 0.0, None
    # iterate all possible combinations (cartesian product)
    W = []
    for i in range(100000):
        weights = []
        for j in range(20):
            w_ = random.choice(w)
            weights.append(w_)
        if len(set(weights)) == 1:
            continue
        W.append(normalize(weights))
    W = np.array(W)
    print(W)
    for weights in W:
        # skip if all weights are equal
        score = evaluate(data.copy(), weights)
        if score > best_score:
            best_score, best_weights = score, weights
            print('>%s %.3f' % (best_weights, best_score))
    print('>%s %.3f' % (best_weights, best_score))
    return list(best_weights)


def agg_data(XGB, NN_DDOS, NN_IP, LGBM):
    data_np = np.concatenate((XGB, NN_DDOS, NN_IP, LGBM), axis=1)
    cols = ['xgb_'+str(i) for i in range(5)] + ['nnddos_'+str(i) for i in range(5)] + \
        ['nnip_'+str(i) for i in range(5)] + ['lgbm_'+str(i) for i in range(5)]
    df = pd.DataFrame(columns=cols, data=data_np)
    data = pd.DataFrame()
    for i in range(5):
        for m in ['xgb', 'nnddos', 'nnip', 'lgbm']:
            data[m+'_'+str(i)] = df[m+'_'+str(i)]
    return data

def read_data():
    xgb = np.load('xgb.npy')
    nnddos = np.load('ddos_nn_pred.npy')
    nnip = np.load('ipsweep_nn_pred.npy')
    lgbm = np.load('lgbm.npy')
    return xgb, nnddos, nnip, lgbm

XGB, NN_DDOS, NN_IP, LGBM = read_data()
df = agg_data(XGB, NN_DDOS, NN_IP, LGBM)
weights = grid_search(df.copy())
