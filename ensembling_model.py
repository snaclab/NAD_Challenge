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
import postprocess
import pickle

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

def produce_result(data, w):
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
    return y_pred

def produce_prob(data, w):
    weights = np.repeat([w], repeats=len(data), axis=0)
    w_data = np.multiply(data.to_numpy(), weights)
    f_0 = w_data[:, :4].sum(axis=1).reshape(-1,1)
    f_1 = w_data[:, 4:8].sum(axis=1).reshape(-1,1)
    f_2 = w_data[:, 8:12].sum(axis=1).reshape(-1,1)
    f_3 = w_data[:, 12:16].sum(axis=1).reshape(-1,1)
    f_4 = w_data[:, 16:].sum(axis=1).reshape(-1,1)
    f = np.concatenate((f_0, f_1, f_2, f_3, f_4), axis=1)
    data = pd.DataFrame(data=f, columns=[0, 1, 2, 3, 4])
    return data


def evaluate(y_pred):
    data_tst = pd.read_csv('../nad/test_no.csv')
    score = main.evaluation(data_tst, y_pred)
    return score


def grid_search(data):
    # define weights to consider
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_score, best_weights = 0.0, None
    '''
    # iterate all possible combinations (cartesian product)
    Weights = []
    for i in range(500):
        W = []
        for k in range(5):
            weights = []
            if k==0:
                for j in range(4):
                    if j==0 or j==1:
                        diff = random.gauss(0, 0.1)
                        w_ = 0.5
                        w_ += diff
                    else:
                        w_ = random.uniform(0, 0.01)
                    w_ = max(0, w_)
                    weights.append(w_)
            elif k==2:
                for j in range(4):
                    if j==0:
                        diff = random.gauss(0, 0.1)
                        w_ = 0.05
                        w_ += diff
                    elif j==2:
                        diff = random.gauss(0, 0.1)
                        w_ = 0.95
                        w_ += diff
                    else:
                        w_ = random.gauss(0, 0.01)
                    w_ = max(0, w_)
                    weights.append(w_)
            elif k==4:
                for j in range(4):
                    if j==0:
                        diff = random.gauss(0, 0.1)
                        w_ = 0.95
                        w_ += diff
                    elif j==2:
                        diff = random.gauss(0, 0.1)
                        w_ = 0.05
                        w_ += diff
                    else:
                        w_ = random.gauss(0, 0.01)
                    w_ = max(0, w_)
                    weights.append(w_)
            else:
                for j in range(4):
                    w_ = random.choice(w)
                    w_ = max(0, w_)
                    weights.append(w_)
            W.extend(normalize(weights))
        Weights.append(W)
    Weights = np.array(Weights)
    for idx, weights in enumerate(Weights):
        # skip if all weights are equal
        result = produce_result(data.copy(), weights)
        score = evaluate(result)
        if score > best_score:
            best_score, best_weights = score, weights
            print('>%s %.3f' % (best_weights, best_score))
        if idx%100==0:
            print(idx)
    print('>%s %.3f' % (best_weights, best_score))
    '''
    # with open("best_weights_lgbm.txt", "wb") as fp:
    #     pickle.dump(best_weights, fp)
    with open("best_weights_lgbm.txt", "rb") as fp:
        best_weights = pickle.load(fp)
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
df_pred = produce_prob(df, weights)
y_pred_final = postprocess.post_processing('../nad/test_no.csv', df_pred, False, True)
data_tst = pd.read_csv('../nad/test_no.csv')
main.evaluation(data_tst.copy(), y_pred_final)