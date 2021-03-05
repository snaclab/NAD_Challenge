import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

def agg_data(XGB, NN_DDOS, NN_IP, LGBM):
    data_np = np.concatenate((XGB, NN_DDOS, NN_IP, LGBM), axis=1)
    cols = ['xgb_'+str(i) for i in range(5)] + ['nnddos_'+str(i) for i in range(5)] + \
        ['nnip_'+str(i) for i in range(5)] + ['lgbm_'+str(i) for i in range(5)]
    df = pd.DataFrame(columns=cols, data=data_np)
    return df

def read_data(filename):
    pass

