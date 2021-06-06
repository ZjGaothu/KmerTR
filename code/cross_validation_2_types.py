import xgboost as xgb
import numpy as np
import pandas as pd
from configs import *
import argparse



def get_args():
    parser = argparse.ArgumentParser(description='Cross validation')
    parser.add_argument('--seq_pos', type=str, default='./data/pos_seq.npy',
                        help='positive data set path')
    parser.add_argument('--seq_neg', type=str, default='./data/neg_seq.npy',
                        help='negtive data set path')
    parser.add_argument('--TF_pos', type=str, default='./data/pos_tf.npy',
                        help='positive tf motif score set path')
    parser.add_argument('--TF_neg', type=str, default='./data/neg_tf.npy',
                        help='negtive tf motif score set path')
    parser.add_argument('--save', type=str, default='./data/result/',
                        help='path to save the final feature')
    parser.add_argument('--nfolds', type=int, default=10,
                        help='num of folds')
    parser.add_argument('--celltype', type=str, default='K562',
                        help='cell line')
    args = parser.parse_args()
    return args
args = get_args() 

# load params 
silencer = np.load(args.seq_pos)
silencer_TF = np.load(args.TF_pos)
enhancer = np.load(args.seq_neg)
enhancer_TF = np.load(args.TF_neg)
folds = args.nfold
cell = args.celltype
save_path = args.save

# split length
silencer_len = len(silencer)
split_len_si = int(silencer_len/folds)

enhancer_len = len(enhancer)
split_len_en = int(enhancer_len/folds)

# n fold cross validation
for cv in range(1,folds):
    train_s = silencer[np.array(list(range(0,(cv-1)*split_len_si)) + list(range(cv*split_len_si,silencer_len)))]
    train_e = enhancer[np.array(list(range(0,(cv-1)*split_len_en)) + list(range(cv*split_len_en,enhancer_len)))]
    
    train_TF_s = silencer_TF[np.array(list(range(0,(cv-1)*split_len_si)) + list(range(cv*split_len_si,silencer_len)))]
    train_TF_e = enhancer_TF[np.array(list(range(0,(cv-1)*split_len_en)) + list(range(cv*split_len_en,enhancer_len)))]
    
    test_data_s = silencer[np.array(list(range((cv-1)*split_len_si , cv * split_len_si)))]
    test_data_e = enhancer[np.array(list(range((cv-1)*split_len_en , cv * split_len_en)))]
    
    test_TF_s = silencer_TF[np.array(list(range((cv-1)*split_len_si , cv * split_len_si)))]
    test_TF_e = enhancer_TF[np.array(list(range((cv-1)*split_len_en , cv * split_len_en)))]
    
    # last fold
    if cv == folds:
        train_s = silencer[np.array(list(range(0,(cv-1)*split_len_si)) )]
        train_e = enhancer[np.array(list(range(0,(cv-1)*split_len_en)))]

        train_TF_s = silencer_TF[np.array(list(range(0,(cv-1)*split_len_si)) )]
        train_TF_e = enhancer_TF[np.array(list(range(0,(cv-1)*split_len_en)))]

        test_data_s = silencer[np.array(list(range((cv-1)*split_len_si , silencer_len)))]
        test_data_e = enhancer[np.array(list(range((cv-1)*split_len_en , enhancer_len)))]
        test_TF_s = silencer_TF[np.array(list(range((cv-1)*split_len_si , silencer_len)))]
        test_TF_e = enhancer_TF[np.array(list(range((cv-1)*split_len_en , enhancer_len)))]
        
    train_data = np.vstack((train_s,train_e))
    train_TF = np.vstack((train_TF_s,train_TF_e))
    test_data = np.vstack((test_data_s,test_data_e))
    test_TF = np.vstack((test_TF_s,test_TF_e))
    
    # tf exp data
    exp = np.array(pd.read_table("/data/temp/gaozijing/DeepSilencer/TF/rna/tf_%s.txt" % cell, header=None, sep='\t').values)
    exp = exp[:, 1].astype(float)
    exp[np.isnan(exp)] = 0.0

    train_TF = (-np.log10(train_TF)) * exp
    test_TF = (-np.log10(test_TF)) * exp
    
    train_data = np.hstack((train_data,train_TF))
    test_data = np.hstack((test_data,test_TF))

    train_data = np.squeeze(train_data)
    test_data = np.squeeze(test_data)

    # generate label
    train_label = [1]*len(train_s) + [0]*len(train_e) 
    dtrain = xgb.DMatrix(train_data, label = train_label)
    dtest = xgb.DMatrix(test_data)

    # xgboost params
    params={'booster':booster,
        'objective': objective,
        'eval_metric': eval_metric,
        'max_depth':max_depth,
        'lambda':Lambda,
        'n_estimators':n_estimators, 
        'gamma':gamma,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight,
        'eta': eta,
        'seed':seed,
        'nthread':8,
         'silent':0}
    
    watchlist = [(dtrain,'train')]
    bst=xgb.train(params,dtrain,num_boost_round=ROUNDS,evals=watchlist)
    yscore=bst.predict(dtest)
    yscore = np.array(yscore)
    # save result
    np.save(save_path + 'cross_validation_cv%s.npy'%cv,yscore)
    bst.save_model('./model/xgb_2_cv%s.bin'%cv)



