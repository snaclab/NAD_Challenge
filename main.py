import numpy as np
import pandas as pd
import argparse
import preprocess
import learning
from xgboost import plot_importance
import matplotlib.pyplot as plt
import datetime

def parse_arg():
    parser = argparse.ArgumentParser(description='Process dataset name')
    parser.add_argument('--trn', help='input training dataset', required=True)
    parser.add_argument('--tst', help='input testing dataset', required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    
    data_trn = pd.read_csv(args.trn)
    data_tst = pd.read_csv(args.tst)

    n_class = len(data_trn['label'].unique())

    # training process
    model = learning.XGB_training(data_trn, n_class)

    # prediction
    Y_test = data_tst[['label']].copy().values.reshape(1, -1)[0]
    y_pred = learning.XGB_prediction(data_tst, model)
    df_pred = pd.DataFrame(columns=[0,1,2,3,4], data=y_pred)
    
    # postprocess
    test = pd.read_csv('../nad/test_no.csv')
    ans = test[['time','src','label']]
    ans = pd.concat([ans, df_pred], axis=1)
    ans['time'] = ans['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    ans['time'] = ans['time'].apply(lambda x: str(x.month).zfill(2)+str(x.day).zfill(2)+str(x.hour).zfill(2)+str(x.minute).zfill(2))
    ans['time+src'] = ans['time'].apply(str) + ans['src'].apply(str)
    ans.drop(columns=['time','src'],inplace=True)
    d_v = ans.groupby(['time+src']).sum()
    d_v = d_v.idxmax(axis='columns').to_frame()
    DIC = {}
    for idx, row in d_v.iterrows():
        DIC[idx] = row[0]
    ans['pred'] = [1 for i in range(len(ans))]
    ans['pred'] = ans['time+src'].apply(lambda x: DIC[x])
    y_pred = ans['pred'].values
    
    learning.eval(Y_test, y_pred)

    # plot feature importance
    model.get_booster().feature_names = list(data_trn.columns)[:-1]
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_importance(model.get_booster(), ax)
