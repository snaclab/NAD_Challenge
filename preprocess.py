from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from ip2geotools.databases.noncommercial import DbIpCity
from category_encoders import HashingEncoder
import pandas as pd
import numpy as np
import ipaddress
import argparse
import sys
import pickle
import multiprocessing
import math
import datetime
import time
import nn

class Preprocessor:
    def __init__(self):
        self.one_hot_enc_dict = {}
        self.label_enc_dict = {}
        self.hash_enc_dict = {}

    def data_balance(self, df):
        normal_n = int(df['label'].value_counts().to_frame().loc['Probing-IP sweep','label']*1.3)
        df = pd.concat([df[df['label']!='Normal'], df[df['label']=='Normal'].sample(n=normal_n)]).reset_index(drop=True)
        
        return df

    def one_hot_fit(self, df, enc_name, column_name):
        self.one_hot_enc_dict[enc_name] = OneHotEncoder(handle_unknown='ignore')
        self.one_hot_enc_dict[enc_name].fit(np.array(df[column_name].unique().tolist()).reshape(-1, 1))
        pickle.dump(self.one_hot_enc_dict[enc_name], open('pretrained/'+enc_name+'_enc.pkl', 'wb'))

    def label_fit(self, df, enc_name, column_name):
        self.label_enc_dict[enc_name] = preprocessing.LabelEncoder()
        self.label_enc_dict[enc_name].fit(df[column_name].unique())
        pickle.dump(self.label_enc_dict[enc_name], open('pretrained/'+enc_name+'_enc.pkl', 'wb'))

    def hashing_fit(self, n_components, df, enc_name, column_name):
        self.hash_enc_dict[enc_name] = HashingEncoder(n_components=n_components)
        self.hash_enc_dict[enc_name].fit(df[column_name])

    def one_hot_transform(self, df, enc_name, column_name, is_train):
        if is_train:
            return self.one_hot_enc_dict[enc_name].transform(np.array(df[column_name]).reshape(-1,1)).toarray()            
        else:
            enc = pickle.load(open('pretrained/'+enc_name+'_enc.pkl', 'rb'))
            return enc.transform(np.array(df[column_name]).reshape(-1,1)).toarray()

    def label_transform(self, df, enc_name, column_name, is_train):
        if is_train:
            return self.label_enc_dict[enc_name].transform(df[column_name].values)
        else:
            enc = pickle.load(open('pretrained/'+enc_name+'_enc.pkl', 'rb'))
            return enc.transform(df[column_name].values)

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

def add_features(df):
    inner_ip_list = ['172.{}'.format(x) for x in range(16, 32)]
    inner_ip_list.extend(['192.168', '10.'])
    df['inner_src'] = df['src'].apply(lambda x: int(x.startswith(tuple(inner_ip_list))))
    df['inner_dst'] = df['dst'].apply(lambda x: int(x.startswith(tuple(inner_ip_list))))
    df['prt_zero'] = ((df['spt']==0) & (df['dpt']==0)).astype('int64')
    df['flow_diff'] = df['in (bytes)']-df['out (bytes)']
    df['flow_diff'] = df['flow_diff'].apply(lambda x: 0 if x==0 else (1 if x>0 else -1))
    df['dst_ip_end_with_255'] = df['dst'].apply(lambda x: int(x.endswith('.255')))
    return df

def inverse_one_hot_encoding(df, col):
    gen_lists = np.identity(len(df[col].unique()))
    col_name = list(Processor.one_hot_enc_dict[col].inverse_transform(gen_lists).reshape(1, -1)[0])
    return col_name

def data_transform(processor, df, app_name, proto_name, is_train=True):
    app = pd.DataFrame(columns=[app_name[x] for x in range(len(app_name))], data = processor.one_hot_transform(df, 'app', 'app', is_train))
    data = pd.concat([df.reset_index(drop=True), app.reset_index(drop=True)], axis=1)
    proto = pd.DataFrame(columns=['proto'+str(proto_name[x]) for x in range(len(proto_name))], data = processor.one_hot_transform(df, 'proto', 'proto', is_train))
    data = pd.concat([data.reset_index(drop=True), proto.reset_index(drop=True)], axis=1)
    if is_train:
        new_label = processor.label_transform(data, 'label', 'label', is_train)
        data['class'] = new_label
        data.drop(['time', 'src', 'dst', 'app', 'proto', 'spt', 'dpt', 'label'], axis=1, inplace=True)
        data.rename(columns={"class": "label"}, inplace=True)
    else:
        data.drop(['time', 'src', 'dst', 'app', 'proto', 'spt', 'dpt'], axis=1, inplace=True)
    return data

def PCA_transform(X_train, X_test, dim=32):
    
    X = np.concatenate((X_train, X_test))
    pca = PCA(n_components=dim)
    pca.fit(X)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca

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

def preprocess_ip_binarize(df, column):
    _df = pd.DataFrame(columns=[column], data = df[column])
    _df[column] = _df[column].apply(lambda x: ''.join([bin(int(i)+256)[3:] for i in x.split('.')]))
    _df_split = _df[column].str.slice().apply(lambda i: pd.Series(list(i)))
    _df_split = _df_split.rename(columns={x: '{}_{}'.format(column, idx) for idx, x in enumerate(_df_split.columns)})

    data = pd.concat([df.reset_index(drop=True), _df_split.reset_index(drop=True)], axis=1)

    return data

def preprocess_is_sweep(df):
    df_src = df['src'].str.split('.', expand=True)
    df_dst = df['dst'].str.split('.', expand=True)
    df_src = df_src.rename(columns={x: 'src_{}'.format(idx) for idx, x in enumerate(df_src.columns)})
    df_dst = df_dst.rename(columns={x: 'dst_{}'.format(idx) for idx, x in enumerate(df_dst.columns)})

    for i in range(4):
        df_src['src_'+str(i)] = df_src['src_'+str(i)].astype(int)
        df_dst['dst_'+str(i)] = df_dst['dst_'+str(i)].astype(int)
    sd = pd.concat([df_src, df_dst], axis=1)
    sd = sd.diff()
    for c in sd.columns:
        sd[c] = sd[c].apply(lambda x: int(x!=0))
    sd['is_sweep'] = sd.apply(lambda row: int((row['src_0']==0)&(row['src_1']==0)&(row['src_2']==0)&(row['src_3']==0)&(row['dst_0']==0)&(row['dst_1']==0)&(row['dst_2']==0)&(row['dst_3']!=0)), axis=1)
    sd.drop(['src_'+str(i) for i in range(4)]+['dst_'+str(i) for i in range(4)], axis=1, inplace=True)
    data = pd.concat([df.reset_index(drop=True), sd.reset_index(drop=True)], axis=1)

    return data

def preprocess_ip_hashing(processor, df, fit=False):
    src_strip = pd.DataFrame(columns=['ip_strip'], data=[i.rsplit('.', 1)[0] for i in df['src'].to_numpy()])
    dst_strip = pd.DataFrame(columns=['ip_strip'], data=[i.rsplit('.', 1)[0] for i in df['dst'].to_numpy()])
    if fit == True:
        ip_strip = pd.DataFrame(columns=['ip_strip'], data=np.concatenate((src_strip['ip_strip'], dst_strip['ip_strip']), axis=0))
        processor.hashing_fit(32, ip_strip, 'ip', 'ip_strip')
    
    df_ip_src = processor.hashing_transform(src_strip, 'ip', 'ip_strip')
    df_ip_dst = processor.hashing_transform(dst_strip, 'ip', 'ip_strip')

    df_ip_src = df_ip_src.rename(columns={x: 'src_{}'.format(idx) for idx, x in enumerate(df_ip_src.columns)})
    df_ip_dst = df_ip_dst.rename(columns={x: 'dst_{}'.format(idx) for idx, x in enumerate(df_ip_dst.columns)})

    data = pd.concat([df.reset_index(drop=True), df_ip_src.reset_index(drop=True), df_ip_dst.reset_index(drop=True)], axis=1)

    return data

def worker(c, return_dict, df):
    L = [df.loc[len(df)-1, c]]
    Max = df.loc[len(df)-1, c]
    for idx in reversed(df.iloc[:-1, :].index):
        if df.loc[idx, c] == 0:
            L.append(Max)
            Max = 0
        else:
            if df.loc[idx+1, c] == df.loc[idx, c]:
                L.append(Max)
            elif df.loc[idx+1, c] == df.loc[idx, c] + 1:
                L.append(Max)
            else:
                Max = df.loc[idx, c]
                L.append(Max)
    return_dict[c] = L[::-1]

def algin_cnt_feature(df):
    '''
    cn = ['cnt_dst', 'cnt_src', 'cnt_serv_src',\
    'cnt_serv_dst', 'cnt_dst_slow', 'cnt_src_slow', 'cnt_serv_src_slow',\
    'cnt_serv_dst_slow', 'cnt_dst_conn', 'cnt_src_conn',\
    'cnt_serv_src_conn', 'cnt_serv_dst_conn']
    '''
    cn = ['cnt_serv_dst_slow', 'cnt_serv_src_slow']
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for c in cn:
        p = multiprocessing.Process(target=worker, args=(c, return_dict, df))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    for c in cn:
        df[c+'_max'] = return_dict[c]

    return df

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn', nargs='+', help='input training dataset', required=False)
    parser.add_argument('--tst', nargs='+', help='input testing dataset', required=False)
    parser.add_argument('--output_trn', help='output processed training dataset', required=False)
    parser.add_argument('--pretrained', help='contains pretrained encoder', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    
    if not args.pretrained:
        # read data sets
        df_trn = pd.concat([pd.read_csv(args.trn[i]) for i in range(len(args.trn))])
        n_class = len(df_trn['label'].unique())
    
        # preprocess data
        Processor = Preprocessor()
        df_trn = Processor.data_balance(df_trn)
        Processor.one_hot_fit(df_trn, 'app', 'app')
        Processor.one_hot_fit(df_trn, 'proto', 'proto')
        Processor.label_fit(df_trn, 'label', 'label')
        print_class_name(Processor, n_class)
        df_trn = add_features(df_trn)
        app_name = inverse_one_hot_encoding(df_trn, 'app')
        with open("pretrained/app_name.txt", "wb") as f:
            pickle.dump(app_name, f)
        proto_name = inverse_one_hot_encoding(df_trn, 'proto')
        with open("pretrained/proto_name.txt", "wb") as f:
            pickle.dump(proto_name, f)
        data_trn = data_transform(Processor, df_trn, app_name, proto_name, True)
        data_trn.to_csv(args.output_trn, index=False)

    for tst_file in args.tst:
        df_tst = pd.read_csv(tst_file)
        df_tst = add_features(df_tst)
        Processor = Preprocessor()
        with open("pretrained/app_name.txt", "rb") as f:
            app_name = pickle.load(f)
        with open("pretrained/proto_name.txt", "rb") as f:
            proto_name = pickle.load(f) 
        data_tst = data_transform(Processor, df_tst, app_name, proto_name, False)
        data_tst.to_csv(tst_file[:-4]+'_processed.csv', index=False)
    
    ##TODO: do normalization for nn; it needs to concat both trn and tst files.
    ##normalization gets data's mean and std, storing them for nn.
    ##need to discuss this part because normalization requires to see all trn and tst data.
    if not args.pretrained:
        f_all = [pd.read_csv(args.trn[i]) for i in range(len(args.trn))]
        f_all.extend([pd.read_csv(args.tst[i]) for i in range(len(args.tst))])
        df_all = pd.concat(f_all)
        norm_zscore = nn.compute_norm(df_all)
        nn.save_norm("pretrained/norm_zscore.npy", norm_zscore)
    

