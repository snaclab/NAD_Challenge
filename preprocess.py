from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        self.one_hot_enc = OneHotEncoder(handle_unknown='ignore')
        self.label_enc = preprocessing.LabelEncoder()

    def data_balance(self, df, ratio):
        df = pd.concat([df[df['label']!='Normal'], df[df['label']=='Normal'].sample(frac=ratio)]).reset_index(drop=True)
        return df

    def one_hot_fit(self, df):
        self.one_hot_enc.fit(np.array(df['app'].unique().tolist()).reshape(-1, 1))

    def label_fit(self, df):
        self.label_enc.fit(df['label'].unique())

    def one_hot_transform(self, df):
        return self.one_hot_enc.transform(np.array(df['app']).reshape(-1,1)).toarray()

    def label_transform(self, df):
        return self.label_enc.transform(df['label'].values)





