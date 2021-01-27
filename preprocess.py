from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from ip2geotools.databases.noncommercial import DbIpCity
from category_encoders import HashingEncoder
import pandas as pd
import numpy as np
import argparse
import sys

class Preprocessor:
    def __init__(self):
        self.one_hot_enc_dict = {}
        self.label_enc_dict = {}
        self.hash_enc_dict = {}

    def data_balance(self, df, ratio):
        df = pd.concat([df[df['label']!='Normal'], df[df['label']=='Normal'].sample(frac=ratio)]).reset_index(drop=True)
        return df

    def one_hot_fit(self, df, enc_name, column_name):
        self.one_hot_enc_dict[enc_name] = OneHotEncoder(handle_unknown='ignore')
        self.one_hot_enc_dict[enc_name].fit(np.array(df[column_name].unique().tolist()).reshape(-1, 1))

    def label_fit(self, df, enc_name, column_name):
        self.label_enc_dict[enc_name] = preprocessing.LabelEncoder()
        self.label_enc_dict[enc_name].fit(df[column_name].unique())

    def hashing_fit(self, n_components, df, enc_name, column_name):
        self.hash_enc_dict[enc_name] = HashingEncoder(n_components=n_components)
        self.hash_enc_dict[enc_name].fit(df[column_name])

    def one_hot_transform(self, df, enc_name, column_name):
        return self.one_hot_enc_dict[enc_name].transform(np.array(df[column_name]).reshape(-1,1)).toarray()

    def label_transform(self, df, enc_name, column_name):
        return self.label_enc_dict[enc_name].transform(df[column_name].values)

    def hashing_transform(self, df, enc_name, column_name):
        return self.hash_enc_dict[enc_name].transform(df[column_name])

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


def data_transform(processor, df, app_name, unseen_label=[]):
    app = pd.DataFrame(columns=[app_name[x] for x in range(len(app_name))], data = processor.one_hot_transform(df, 'app', 'app'))
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


def preprocess_ip(processor, df, fit=False):
    src_strip = pd.DataFrame(columns=['ip_strip'], data=[i.rsplit('.', 1)[0] for i in df['src'].to_numpy()])
    dst_strip = pd.DataFrame(columns=['ip_strip'], data=[i.rsplit('.', 1)[0] for i in df['dst'].to_numpy()])
    if fit == True:
        ip_strip = pd.DataFrame(columns=['ip_strip'], data=np.concatenate((src_strip['ip_strip'], dst_strip['ip_strip']), axis=0))
        processor.hashing_fit(13, ip_strip, 'ip', 'ip_strip')
    
    df_ip_src = processor.hashing_transform(src_strip, 'ip', 'ip_strip')
    df_ip_dst = processor.hashing_transform(dst_strip, 'ip', 'ip_strip')

    df_ip_src = df_ip_src.rename(columns={x: 'src_{}'.format(idx) for idx, x in enumerate(df_ip_src.columns)})
    df_ip_dst = df_ip_dst.rename(columns={x: 'dst_{}'.format(idx) for idx, x in enumerate(df_ip_dst.columns)})

    data = pd.concat([df.reset_index(drop=True), df_ip_src.reset_index(drop=True), df_ip_dst.reset_index(drop=True)], axis=1)

    return data


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

    df_trn = preprocess_ip(Processor, df_trn, fit=True)
    df_tst = preprocess_ip(Processor, df_tst, fit=False)

    gen_lists = np.identity(len(df_trn['app'].unique()))
    app_name = list(Processor.one_hot_enc_dict['app'].inverse_transform(gen_lists).reshape(1, -1)[0])
    data_trn = data_transform(Processor, df_trn, app_name)
    data_tst = data_transform(Processor, df_tst, app_name, unseen_label)

    print_class_name(Processor, n_class)

    data_trn.to_csv(args.output_trn, index=False)
    data_tst.to_csv(args.output_tst, index=False)

