import xgboost as xgb
import numpy as np
from sklearn.utils import shuffle
import pandas as pd



def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--seq_path', type=str, default='./data/seq/',
                        help='positive data set path')
    parser.add_argument('--TF_path', type=str, default='./data/tf/',
                        help='negtive tf motif score set path')
    parser.add_argument('--save', type=str, default='./data/result/',
                        help='path to save the final feature')
    parser.add_argument('--nfolds', type=int, default=10,
                        help='num of folds')
    parser.add_argument('--celltype', type=str, default='K562',
                        help='cell line')
    args = parser.parse_args()
    return args
# load params
args = get_args() 
folds = args.nfold
seq_data_path = args.seq_path
tf_data_path = args.TF_path
cell = args.celltype
save_path = args.save

# cross validation
for cv in range(1,folds):
    train_e = np.load(seq_data_path + "cv%s/%s_enhancers_train.npy"%(cv,cell))
    train_s = np.load(seq_data_path + "cv%s/%s_silencers_train.npy"%( cv,cell))
    train_p = np.load(seq_data_path + "cv%s/%s_promoters_train"%( cv,cell))
    train_o = np.load(seq_data_path + "cv%s/%s_others_train"%( cv,cell))

    test_data_e = np.load(seq_data_path + "cv%s/%s_enhancers_test.npy"%( cv,cell))
    test_data_s = np.load(seq_data_path + "cv%s/%s_silencers_test.npy"%( cv,cell))
    test_data_p = np.load(seq_data_path + "cv%s/%s_promoters_test.npy"%( cv,cell))
    test_data_o = np.load(seq_data_path + "cv%s/%s_others_test.npy"%( cv,cell))
    


    TF_s = np.load(tf_data_path + "/TF/%s_silencers.npy"%( cell))
    TF_e = np.load(tf_data_path + "/TF/%s_enhancers.npy"%( cell))
    TF_p = np.load(tf_data_path + "/TF/%s_promoters.npy"%( cell))
    TF_o = np.load(tf_data_path + "/TF/%s_others.npy"%( cell))

    train_data = np.vstack((train_s,train_e,train_p,train_o))
    test_data = np.vstack((test_data_s,test_data_e,test_data_p,test_data_o))

    
    TF_train_s = TF_s[np.array(list(range(0,int(len(TF_s)/10)*(cv-1)))+ list(range((cv)*int(len(TF_s)/10),len(TF_s))))]
    TF_test_s = TF_s[np.array(list(range(int(len(TF_s)/10)*(cv-1),int(len(TF_s)/10)*cv)))]
    TF_train_e = TF_e[np.array(list(range(0,int(len(TF_e)/10)*(cv-1)))+ list(range((cv)*int(len(TF_e)/10),len(TF_e))))]
    TF_test_e = TF_e[np.array(list(range(int(len(TF_e)/10)*(cv-1),int(len(TF_e)/10)*cv)))]
    TF_train_p = TF_p[np.array(list(range(0,int(len(TF_p)/10)*(cv-1)))+ list(range((cv)*int(len(TF_p)/10),len(TF_p))))]
    TF_test_p = TF_p[np.array(list(range(int(len(TF_p)/10)*(cv-1),int(len(TF_p)/10)*cv)))]
    TF_train_o = TF_o[np.array(list(range(0,int(len(TF_o)/10)*(cv-1)))+ list(range((cv)*int(len(TF_o)/10),len(TF_o))))]
    TF_test_o = TF_o[np.array(list(range(int(len(TF_o)/10)*(cv-1),int(len(TF_o)/10)*cv)))]

    train_TF = np.vstack((TF_train_s,TF_train_e,TF_train_p,TF_test_o))
    test_TF = np.vstack((TF_test_s,TF_test_e,TF_test_p,TF_test_o))
    
    exp = np.array(pd.read_table("/data/temp/gaozijing/DeepSilencer/TF/rna/tf_%s.txt" %  cell, header=None, sep='\t').values)
    exp = exp[:, 1].astype(float)
    exp[np.isnan(exp)] = 0.0

    train_TF = (-np.log10(train_TF)) * exp
    test_TF = (-np.log10(test_TF)) * exp

    train_data = np.hstack((train_data,train_TF))
    test_data = np.hstack((test_data,test_TF))

    train_data = np.squeeze(train_data)
    test_data = np.squeeze(test_data)

    train_label = [0]*len(train_s) + [1]*len(train_e) + [2]*len(train_p) + [3]*len(train_o)
    train_data,train_label = shuffle(train_data,train_label,random_state = 0)
    dtrain = xgb.DMatrix(train_data, label = train_label)
    dtest = xgb.DMatrix(test_data)

    params={'booster':'gbtree',
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'max_depth':10,
        'lambda':2.4,
        'n_estimators':50, 
        'gamma':0,
        'subsample':0.91,
        'colsample_bytree':0.85,
        'min_child_weight':2.4,
        'eta': 0.025,
        'seed':0,
        'nthread':8,
        'silent':1,
        'num_class':4}

    watchlist = [(dtrain,'train')]

    bst=xgb.train(params,dtrain,num_boost_round=1200,evals=watchlist)
    yscore=bst.predict(dtest)
    yscore = np.array(yscore)
    print( cv,cell,' finished')
    # save result
    np.save(seq_data_path + "results/predict_%s_cv%s.npy"%(cell,cv),yscore)
    bst.save_model('./model/xgb_4_cv%s.bin'%cv)




