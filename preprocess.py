from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from ip2geotools.databases.noncommercial import DbIpCity
import pandas as pd
import numpy as np
import argparse
import sys

class Preprocessor:
    def __init__(self):
        self.one_hot_enc_dict = {}
        self.label_enc_dict = {}

    def data_balance(self, df, ratio):
        df = pd.concat([df[df['label']!='Normal'], df[df['label']=='Normal'].sample(frac=ratio)]).reset_index(drop=True)
        return df

    def one_hot_fit(self, df, enc_name, column_name):
        self.one_hot_enc_dict[enc_name] = OneHotEncoder(handle_unknown='ignore')
        self.one_hot_enc_dict[enc_name].fit(np.array(df[column_name].unique().tolist()).reshape(-1, 1))

    def label_fit(self, df, enc_name, column_name):
        self.label_enc_dict[enc_name] = preprocessing.LabelEncoder()
        self.label_enc_dict[enc_name].fit(df[column_name].unique())

    def one_hot_transform(self, df, enc_name, column_name):
        return self.one_hot_enc_dict[enc_name].transform(np.array(df[column_name]).reshape(-1,1)).toarray()

    def label_transform(self, df, enc_name, column_name):
        return self.label_enc_dict[enc_name].transform(df[column_name].values)

    def _get_country(self, IP):
        try:
            # this api call need to access remote database, which is too slow for large number of data
            response = DbIpCity.get(IP, api_key='free')
            return response.country
        except KeyError:
            return 'PRIVATE'
        except:
            return

    def ip_to_country(self, series, column_name):
        countries = [self._get_country(x) for x in series]
        countries_pd = pd.DataFrame({column_name: countries})

        return countries_pd


def data_transform(processor, df, unseen_label=[]):
    app = pd.DataFrame(columns=['app'+str(x) for x in range(len(processor.one_hot_enc_dict['app'].categories_[0]))], data = processor.one_hot_transform(df, 'app', 'app'))
    data = pd.concat([df.reset_index(drop=True), app.reset_index(drop=True)], axis=1)
    if unseen_label:
        data = data[~data['label'].isin(unseen_label)]
    new_label = processor.label_transform(data, 'label', 'label')
    data['class'] = new_label
    data.drop(['src', 'dst', 'time', 'app', 'spt', 'dpt', 'label'], axis=1, inplace=True)
    data.rename(columns={"class": "label"}, inplace=True)
    return data

# currently not efficient, since it have to send a api request to a webservice
def add_country_attr(processor, df):
    src_countries = processor.ip_to_country(df['src'], 'src')
    dst_countries = processor.ip_to_country(df['dst'], 'dst')

    processor.label_fit(src_countries, 'ip', 'src')
    processor.label_fit(dst_countries, 'ip', 'dst')

    src_countries_label = pd.DataFrame({'src_cn': processor.label_transform(src_countries, 'ip', 'src')})
    dst_countries_label = pd.DataFrame({'dst_cn': processor.label_transform(dst_countries, 'ip', 'dst')})

    data = pd.concat([df.reset_index(drop=True), src_countries_label.reset_index(drop=True)], axis=1)
    data = pd.concat([data.reset_index(drop=True), dst_countries_label.reset_index(drop=True)], axis=1)

    return data


def print_class_name(processor, n_class):
    classes = [i for i in range(n_class)]
    labels_name = processor.label_enc_dict['label'].inverse_transform(classes)
    for l in zip(classes, labels_name):
        print(l)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn', help='input training dataset', required=True)
    parser.add_argument('--tst', help='input testing dataset', required=True)
    parser.add_argument('--output_trn', help='output processed training dataset', required=True)
    parser.add_argument('--output_tst', help='output processed testing dataset', required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    
    # read data sets
    df_trn = pd.read_csv(args.trn)
    df_tst = pd.read_csv(args.tst)
    n_class = len(df_trn['label'].unique())
    unseen_label = set(df_tst['label'].unique().tolist()) - set(df_trn['label'].unique().tolist())

    # preprocess data
    Processor = Preprocessor()
    df_trn = Processor.data_balance(df_trn, ratio=0.05)
    Processor.one_hot_fit(df_trn, 'app', 'app')
    Processor.label_fit(df_trn, 'label', 'label')

    gen_lists = np.identity(len(df_trn['app'].unique()))
    app_name = Processor.one_hot_enc_dict['app'].inverse_transform(gen_lists).reshape(1, -1)[0]
    data_trn = data_transform(Processor, df_trn)
    data_tst = data_transform(Processor, df_tst, unseen_label)

    print_class_name(Processor, n_class)

    data_trn.to_csv(args.output_trn, index=False)
    data_tst.to_csv(args.output_tst, index=False)

