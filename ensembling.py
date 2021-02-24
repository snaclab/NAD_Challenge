import pandas as pd

def ensemble(base, do_eval, test_path, xgb_path, nn_path):
    from main import evaluation
    xgb = pd.read_csv(xgb_path)
    nn = pd.read_csv(nn_path)
    if base == 'xgb':
        idx = list(nn[nn['label']=='Probing-IP sweep'].index)
        xgb.loc[(xgb.index.isin(idx)) & (xgb['label']!='Probing-Port sweep'), 'label'] = 'Probing-IP sweep'
        model = xgb.copy()
    if base == 'nn':
        idx = list(xgb[xgb['label']=='Probing-Port sweep'].index)
        nn.loc[idx, 'label'] = 'Probing-Port sweep'
        model = nn.copy()
    
    if str(do_eval)=='True':
        label_map = {'DDOS-smurf':0, 'Normal':1, 'Probing-IP sweep':2, 'Probing-Nmap':3, 'Probing-Port sweep':4}
        model['label'] = model['label'].apply(lambda x: label_map[x])
        y_pred = model['label'].values
        ans = pd.read_csv(test_path)
        evaluation(ans, y_pred)
        model.to_csv(test_path[:-4]+'_predicted.csv', index=False)
    else:
        model.to_csv(test_path, index=False)
