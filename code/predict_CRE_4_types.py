import xgboost as xgb
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--seq_path', type=str, default='./data/seq/',
                        help='positive data set path')
    parser.add_argument('--TF_path', type=str, default='./data/tf/',
                        help='negtive tf motif score set path')
    parser.add_argument('--model_path', type=str, default='./model/xgb_test.bin',
                        help='saved model path')
    parser.add_argument('--save', type=str, default='./data/result/',
                        help='path to save the final feature')
    parser.add_argument('--celltype', type=str, default='K562',
                        help='cell line')
    args = parser.parse_args()
    return args

# load params
args = get_args() 
seq_data_path = args.seq_path
tf_data_path = args.TF_path
model_path = args.model_path
cell = args.celltype
save_path = args.save

# test seq data
test_data_e = np.load(seq_data_path + "/%s_enhancers_test.npy"%(cell))
test_data_s = np.load(seq_data_path + "/%s_silencers_test.npy"%(cell))
test_data_p = np.load(seq_data_path + "/%s_promoters_test.npy"%(cell))
test_data_o = np.load(seq_data_path + "/%s_others_test.npy"%(cell))
# test tf data
TF_s = np.load(tf_data_path + "/TF/%s_silencers.npy"%(cell))
TF_e = np.load(tf_data_path + "/TF/%s_enhancers.npy"%(cell))
TF_p = np.load(tf_data_path + "/TF/%s_promoters.npy"%(cell))
TF_o = np.load(tf_data_path + "/TF/%s_others.npy"%(cell))

test_seq = np.vstack((test_data_s,test_data_e,test_data_p,test_data_o))
test_TF = np.vstack((TF_s,TF_e,TF_p,TF_o))

# tf gexp
exp = np.array(pd.read_table("/data/temp/gaozijing/DeepSilencer/TF/rna/tf_%s.txt" %  cell, header=None, sep='\t').values)
exp = exp[:, 1].astype(float)
exp[np.isnan(exp)] = 0.0

test_TF = (-np.log10(test_TF)) * exp

# combine feature
test_data = np.hstack((test_seq,test_TF))
test_data = np.squeeze(test_data)
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

# load model
bst = xgb.Booster(params)
bst.load_model(model_path)

yscore=bst.predict(dtest)
yscore = np.array(yscore)
print( cell,' finished')
np.save(save_path + "results/predict_%s_multi_types.npy"%(cell),yscore)




