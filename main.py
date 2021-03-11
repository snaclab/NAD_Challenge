import numpy as np
import pandas as pd
import argparse
import preprocess
import postprocess
import xgb, nn
from xgboost import plot_importance
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import fbeta_score
import math
from sklearn.metrics import confusion_matrix
from ensembling import ensemble

def parse_arg():
    parser = argparse.ArgumentParser(description='Process dataset name')
    parser.add_argument('--pretrained', help='exists pretrained model', action='store_true')
    parser.add_argument('--tst_src', nargs='+',  help='original test dataset', required=True)
    parser.add_argument('--trn', help='input training dataset', required=False)
    parser.add_argument('--eval', help='evaluate predicted result (when exists ground truth)', action='store_true')
    parser.add_argument('--ensemble', help='use ensemble method', action='store_true')
    return parser.parse_args()

def evaluation(data_tst, y_pred):
    label_map = {'DDOS-smurf':0, 'Normal':1, 'Probing-IP sweep':2, 'Probing-Nmap':3, 'Probing-Port sweep':4}
    data_tst['label'] = data_tst['label'].apply(lambda x: label_map[x])
    Y_test = data_tst[['label']].copy().values.reshape(1, -1)[0]
    xgb.eval(Y_test, y_pred)
    macro_fbeta_score = fbeta_score(Y_test, y_pred, average='macro', beta=2)
    print('macro F beta score: ', macro_fbeta_score)
    cost_matrix = np.array([[0,2,1,1,1],[2,0,1,1,1],[2,1,0,1,1],[2,1,1,0,1],[2,1,1,1,0]])
    conf_matrix = confusion_matrix(Y_test, y_pred)
    cost = np.multiply(cost_matrix, conf_matrix)
    print('Evaluation criteria: ', 0.3*(1-(math.log(np.sum(cost))/math.log(np.amax(cost)*len(data_tst))))+0.7*macro_fbeta_score)

def post_process(voting_method, tst_file, df_pred, base_model):
    if voting_method == 'dynamic':
        y_pred_final = postprocess.post_processing_dynamic(tst_file, df_pred, base_model)
    elif voting_method == 'combined':
        y_pred_final = postprocess.post_processing_combined(tst_file, df_pred, base_model)
    else:
        raise NameError("voting method undefined")

    return y_pred_final

if __name__ == '__main__':
    args = parse_arg()
    # combined or dynamic
    voting_method = 'combined'
    n_class = 5

    # training process
    if args.pretrained:
        model = xgb.load_model('pretrained/model.pkl')
    else:
        data_trn = pd.read_csv(args.trn)
        model = xgb.XGB_training(data_trn, n_class)
        xgb.save_model(model, 'pretrained/model.pkl')
        
    if args.ensemble:
        norm_zscore = preprocess.load_norm('pretrained/norm_zscore.npy')
        if args.pretrained:
            nn_model = nn.load_model('pretrained/nn.h5')
        else:
            nn_model = nn.nn_training(data_trn, n_class, norm_zscore)
            nn.save_model(nn_model, 'pretrained/nn.h5')  
    # testing
    for tst_file in args.tst_src:
        data_tst = pd.read_csv(tst_file[:-4]+'_processed.csv')
        # predictions
        y_pred = xgb.XGB_prediction(data_tst, model)
        df_pred = pd.DataFrame(columns=[0,1,2,3,4], data=y_pred)
        
        y_pred_final = post_process(voting_method, tst_file, df_pred, 'xgb')
                
        if args.ensemble:
            nn_pred = nn.nn_prediction(data_tst, nn_model, norm_zscore)
            df_nn_pred = pd.DataFrame(columns=[0,1,2,3,4], data=nn_pred)
            nn_pred_final = post_process(voting_method, tst_file, df_nn_pred, 'nn')
            ensemble('xgb', args.eval, tst_file, tst_file[:-4]+'_xgb.csv', tst_file[:-4]+'_nn.csv')
        elif args.eval:
            print ("XGB evaluation:")
            evaluation(data_tst.copy(), y_pred_final)
