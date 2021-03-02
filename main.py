import numpy as np
import pandas as pd
import argparse
import preprocess
import learning
from xgboost import plot_importance
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import fbeta_score
import math
from sklearn.metrics import confusion_matrix

def parse_arg():
    parser = argparse.ArgumentParser(description='Process dataset name')
    parser.add_argument('--pretrained', help='whether exists pretrained model', default=False)
    parser.add_argument('--tst_src', nargs='+',  help='original test dataset', required=True)
    parser.add_argument('--trn', help='input training dataset', required=False)
    parser.add_argument('--eval', help='whether evaluate predicted result (when exists ground truth)', default=True)
    return parser.parse_args()

def evaluation(data_tst, y_pred):
    label_map = {'DDOS-smurf':0, 'Normal':1, 'Probing-IP sweep':2, 'Probing-Nmap':3, 'Probing-Port sweep':4}
    data_tst['label'] = data_tst['label'].apply(lambda x: label_map[x])
    Y_test = data_tst[['label']].copy().values.reshape(1, -1)[0]
    learning.eval(Y_test, y_pred)
    macro_fbeta_score = fbeta_score(Y_test, y_pred, average='macro', beta=2)
    print('macro F beta score: ', macro_fbeta_score)
    cost_matrix = np.array([[0,2,1,1,1],[2,0,1,1,1],[2,1,0,1,1],[2,1,1,0,1],[2,1,1,1,0]])
    conf_matrix = confusion_matrix(Y_test, y_pred)
    cost = np.multiply(cost_matrix, conf_matrix)
    print('Evaluation criteria: ', 0.3*(1-(math.log(np.sum(cost))/math.log(np.amax(cost))))+0.7*macro_fbeta_score)

if __name__ == '__main__':
    args = parse_arg()
    
    n_class = 5

    # training process
    if str(args.pretrained)=='True':
        model = learning.load_model('model.pkl')
    else:
        data_trn = pd.read_csv(args.trn)
        model = learning.XGB_training(data_trn, n_class)
        learning.save_model(model, 'model.pkl')
    for tst_file in args.tst_src:
        data_tst = pd.read_csv(tst_file[:-4]+'_processed.csv')
        # predictions
        y_pred = learning.XGB_prediction(data_tst, model)
        df_pred = pd.DataFrame(columns=[0,1,2,3,4], data=y_pred)
        for time_setting in ['minute', 'hour']:
            # postprocess
            test = pd.read_csv(tst_file)
            ans = test[['time','src']].copy()
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
            if time_setting == 'hour':
                m_df = pd.read_csv(tst_file[:-4]+'_minute_predicted.csv')
                d_idx = list(m_df[m_df['label']=='DDOS-smurf'].index)
                ans.loc[d_idx, 'pred'] = 0
            y_pred = ans['pred'].values
            
            # Transform label and save data
            label_map = {0: 'DDOS-smurf', 1: 'Normal', 2: 'Probing-IP sweep', 3: 'Probing-Nmap', 4: 'Probing-Port sweep'}
            test['label'] = ans['pred']
            test['label'] = test['label'].apply(lambda x: label_map[x])
            if str(args.eval)=='True':
                # Evaluation
                evaluation(data_tst.copy(), y_pred)
            else:
                if time_setting == 'minute':
                    test.to_csv(tst_file[:-4]+'_'+time_setting+'_predicted.csv', index=False)
                elif time_setting == 'hour':
                    test.to_csv(tst_file, index=False)
