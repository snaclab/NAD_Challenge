import numpy as np
import pandas as pd
import argparse
import preprocess
import learning


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
    
    learning.eval(Y_test, y_pred)