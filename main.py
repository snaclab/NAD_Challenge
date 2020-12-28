import numpy as np
import pandas as pd
import argparse
from preprocess import Preprocessor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

def data_transform(processor, df, unseen_label=[]):
    app = pd.DataFrame(columns=['app'+str(x) for x in range(len(Processor.one_hot_enc.categories_[0]))], data = Processor.one_hot_transform(df))
    data = pd.concat([df.reset_index(drop=True), app.reset_index(drop=True)], axis=1)
    if unseen_label:
        data = data[~data['label'].isin(unseen_label)]
    new_label = Processor.label_transform(data)
    data['class'] = new_label
    data.drop(columns=['time', 'src', 'dst', 'app', 'label'], inplace=True)
    data.rename(columns={"class": "label"}, inplace=True)
    return data

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

def eval(Y_test, y_pred):
    print('Test set:')
    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred, digits=len(set(Y_test.reshape(1, -1)[0]))))

# argument parser settings
parser = argparse.ArgumentParser(description='Process dataset name')
parser.add_argument('--trn', help='input train dataset')
parser.add_argument('--tst', help='input test dataset')
parser.add_argument('--output_trn', help='output preprocessed train dataset')
parser.add_argument('--output_tst', help='output preprocessed test dataset')
args = parser.parse_args()

# read data sets
df_trn = pd.read_csv(args.trn)
df_tst = pd.read_csv(args.tst)
n_class = len(df_trn['label'].unique())
unseen_label = set(df_tst['label'].unique().tolist()) - set(df_trn['label'].unique().tolist())

# preprocess data
Processor = Preprocessor()
df_trn = Processor.data_balance(df_trn, ratio=0.05)
Processor.one_hot_fit(df_trn)
Processor.label_fit(df_trn)

gen_lists = np.identity(len(df_trn['app'].unique()))
app_name = Processor.one_hot_enc.inverse_transform(gen_lists).reshape(1, -1)[0]
data_trn = data_transform(Processor, df_trn)
data_tst = data_transform(Processor, df_tst, unseen_label)
data_trn.to_csv(args.output_trn, index=False)
data_tst.to_csv(args.output_tst, index=False)

# training process
model = XGB_training(data_trn, n_class)

# prediction
Y_test = data_tst[['label']].copy().values.reshape(1, -1)[0]
y_pred = XGB_prediction(data_tst, model)

classes = [i for i in range(n_class)]
labels_name = Processor.label_enc.inverse_transform(classes)
for l in zip(classes, labels_name):
    print(l)

eval(Y_test, y_pred)


