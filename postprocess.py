import numpy as np
import pandas as pd
import argparse
import datetime
import math

from ensembling import ensemble

def voting(test_df, df_pred, time_setting):
    ans = test_df[['time','src']].copy()
    ans = pd.concat([ans, df_pred], axis=1)
    ans['time'] = ans['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    if time_setting == 'hour':
        ans['time'] = ans['time'].apply(lambda x: str(x.month).zfill(2)+str(x.day).zfill(2)+str(x.hour).zfill(2))
    elif time_setting == 'minute':
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

    return ans.copy()

def transform_label(ans_df):
    label_map = {
        0: 'DDOS-smurf',
        1: 'Normal',
        2: 'Probing-IP sweep',
        3: 'Probing-Nmap',
        4: 'Probing-Port sweep'
    }
    ans = ans_df.copy()
    ans['label'] = ans['pred'].apply(lambda x: label_map[x])

    return ans['label']

# postprocess
def post_processing(tst_file, df_pred, model_name):
    for time_setting in ['minute', 'hour']:
        test = pd.read_csv(tst_file)
        ans = voting(test, df_pred, time_setting)

        if time_setting == 'hour':
            m_df = pd.read_csv(tst_file[:-4]+'_minute_predicted.csv')
            d_idx = list(m_df[m_df['label']=='DDOS-smurf'].index)
            ans.loc[d_idx, 'pred'] = 0
        
        # Transform label and save data
        test['label'] = transform_label(ans)

        if time_setting == 'minute':
            test.to_csv(tst_file[:-4]+'_'+time_setting+'_predicted.csv', index=False)
            continue
        
        # only for "hour"
        test.to_csv(tst_file[:-4]+'_{}.csv'.format(model_name), index=False)

        y_pred = ans['pred'].values

    return y_pred