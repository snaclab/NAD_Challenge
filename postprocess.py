import numpy as np
import pandas as pd
import argparse
import datetime
import math
from ensembling import ensemble

def align_time(df):
    df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df = df.sort_values(by=['src','time'])
    for i in range(4):
        df['src_'+str(i)] = df['src'].apply(lambda x: int(x.split('.')[i]))
    d = df[['time'] + ['src_'+str(i) for i in range(4)]].diff()
    d['diff_hour'] = d['time'].apply(lambda x: divmod(x.total_seconds(), 3600)[0])
    d['diff_minute'] = d['time'].apply(lambda x: divmod(x.total_seconds(), 60)[0])
    d['diff_hour'] = d.apply(lambda row: 1 if row['diff_hour']<1 and row['src_0']==0 and row['src_1']==0 and row['src_2']==0 and row['src_3']==0 else 0, axis=1)
    d['diff_minute'] = d.apply(lambda row: 1 if row['diff_minute']<1 and row['src_0']==0 and row['src_1']==0 and row['src_2']==0 and row['src_3']==0 else 0, axis=1)
    df['diff_hour'] = d['diff_hour']
    df['diff_minute'] = d['diff_minute']
    df.drop(columns=['src_'+str(i) for i in range(4)], inplace=True)
    data_h, data_m = df.copy(), df.copy()
    for time_settings in ['hour', 'minute']:
        indx = []
        if time_settings == 'hour':
            for idx, row in df.iterrows():
                if row['diff_hour'] == 1:
                    indx.append(idx)
                else:
                    if len(indx)==0:
                        continue
                    else:
                        tmp = data_h.loc[indx[0], 'time']
                        data_h.loc[indx, 'time'] = tmp
                        indx = []
        else:
            for idx, row in df.iterrows():
                if row['diff_minute'] == 1:
                    indx.append(idx)
                else:
                    if len(indx)==0:
                        continue
                    else:
                        tmp = data_m.loc[indx[0], 'time']
                        data_m.loc[indx, 'time'] = tmp
                        indx = []
    return data_h, data_m

def gen_new_label(ans):
    ans['time+src'] = ans['time'].apply(str) + ans['src'].apply(str)
    ans = ans[['time+src', '0', '1', '2', '3', '4']]
    ans.rename(columns={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}, inplace=True)
    d_v = ans.groupby(['time+src']).sum()
    d_v = d_v.idxmax(axis='columns').to_frame()
    DIC = {}
    for idx, row in d_v.iterrows():
        DIC[idx] = row[0]
    ans['pred'] = [1 for i in range(len(ans))]
    ans['pred'] = ans['time+src'].apply(lambda x: DIC[x])
    return ans

def voting_dynamic(tst_file, test_df, df_pred):
    # ans = test_df[['time','src']].copy()
    # ans = pd.concat([ans, df_pred], axis=1)
    # ans_h, ans_m = align_time(ans.copy())
    # ans_h = ans_h.sort_index()
    # ans_m = ans_m.sort_index()
    # ans_h.to_csv(tst_file[:-4]+'_hour_align_time.csv', index=False)
    # ans_m.to_csv(tst_file[:-4]+'_minute_align_time.csv', index=False)
    ans_h = pd.read_csv(tst_file[:-4]+'_hour_align_time.csv')
    ans_m = pd.read_csv(tst_file[:-4]+'_minute_align_time.csv')
    ans_h = gen_new_label(ans_h.copy())
    ans_m = gen_new_label(ans_m.copy())

    return ans_h.copy(), ans_m.copy()

def voting_combined(test_df, df_pred, time_setting):
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
def post_processing_dynamic(tst_file, df_pred, model_name):
    test = pd.read_csv(tst_file)
    ans_h, ans_m = voting_dynamic(tst_file, test, df_pred)
    data_tst = pd.read_csv(tst_file)
    test_m = pd.read_csv(tst_file)
    test_m['label'] = transform_label(ans_m)
    test_m.to_csv(tst_file[:-4]+'_minute_{}_predicted.csv'.format(model_name), index=False)
    m_df = pd.read_csv(tst_file[:-4]+'_minute_{}_predicted.csv'.format(model_name))
    d_idx = list(m_df[m_df['label']=='DDOS-smurf'].index)
    ans_h.loc[d_idx, 'pred'] = 0
    test_h = pd.read_csv(tst_file)
    test_h['label'] = transform_label(ans_h)
    test_h.to_csv(tst_file[:-4]+'_{}_predicted.csv'.format(model_name), index=False)

    y_pred = ans_h['pred'].values

    return y_pred

def post_processing_combined(tst_file, df_pred, model_name):
    for time_setting in ['minute', 'hour']:
        test = pd.read_csv(tst_file)
        ans = voting_combined(test, df_pred, time_setting)

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