import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter


def eval(Y_test, y_pred):
    print('Test set:')
    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred, digits=len(set(Y_test.reshape(1, -1)[0]))))

def XGB_training(data, n_class):
    X = data[[c for c in data.columns if c != 'label']].copy()
    Y = data[['label']].copy()
    X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.3, random_state=7)
    print(y_test.shape)
    model = XGBClassifier(objective='multi:softmax', num_class=n_class, learning_rate=0.1, use_label_encoder=False)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True, early_stopping_rounds=10)
    print('Validation set:')
    y_pred = model.predict(X_test)
    eval(y_test, y_pred)
    return model

def XGB_prediction(data, model):
    X_test = data[[c for c in data.columns if c != 'label']].copy().values
    return model.predict(X_test)