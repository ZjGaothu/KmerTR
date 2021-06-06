# coding: utf-8
import argparse
import time
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import data
import model as model
from model import TransformerModel

def get_args():
    parser = argparse.ArgumentParser(description='Obtain sequence feature')
    parser.add_argument('--data', type=str, default='./data/DNA',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='./model/model_K5_256.pt',
                        help='model path')
    parser.add_argument('--vocab', type=str, default='./model/vocab.npy',
                        help='vocab path')
    parser.add_argument('--cuda', type = bool,default = True,
                        help='use CUDA')
    parser.add_argument('--save', type=str, default='./data/feature.npy',
                        help='path to save the final feature')
    parser.add_argument('--K', type=int, default='5',
                        help='parameter K of Kmer')
    parser.add_argument('--method', type=int, default= 1,
                        help='method of calculate region embedding, 1 present sum and 0 present average')

    args = parser.parse_args()
    return args


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(-1, bsz).contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def obtain_feature(data_source):
    features = []
    attns = []
    model.eval()
    with torch.no_grad():
        data = data_source
        output,feature,attn = model(data,has_mask=False)
        features.append(feature.cpu().numpy().tolist())
        attns.append(attn.cpu().numpy().tolist())
    return features,attns

def seq2kmer(K,sequence):
    seq = sequence
    encoding_matrix = {'a':'A', 'A':'A', 'c':'C', 'C':'C', 'g':'G', 'G':'G', 't':'T', 'T':'T', 'n':'N', 'N':'N'}
    kmer_mat = []
    for i in range(len(seq)-K+1):
        sub_seq = seq[i:(i+K)]
        sub_seq = list(sub_seq)
        for j in range(K):
            sub_seq[j] = encoding_matrix[sub_seq[j]]
        if 'N' not in sub_seq:
            kmer_mat.append(''.join(sub_seq) )
    return kmer_mat

def main():
    # load params
    args = get_args()
    data_path = args.data
    model_path = args.model
    vocab_path = args.vocab
    save_path = args.save
    method = args.method
    K = args.K

    # Set the random seed manually for reproducibility.
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")
    vocab = np.load(vocab_path,allow_pickle=True)
    vocab = vocab.item()
    f = open(data_path)
    ls=[]
    for line in f:
        ls.append(line.replace('\n',''))
    f.close()
    DNA_sequence = []
    for i in range(1,len(ls)):
        if i % 2 == 0:
            continue
        else:
            DNA_sequence.append(ls[i])
    seq_kmers = []
    for i in range(len(DNA_sequence)):
        seq_kmers.append(seq2kmer(K,DNA_sequence[i]))
    kmer_mat = seq_kmers
    # load model
    with open(model_path, 'rb') as f:
        model = torch.load(f)
    # obtain feature of input data
    final_embedding = []
    for temp in kmer_mat:
        idss = []
        ids = []
        for word in temp:
            ids.append(vocab[word.lower()])
        idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)
        test_data = batchify(ids, 1)
        feature_dict,_ = obtain_feature(test_data)
        feature_dict = np.array(feature_dict)
        feature_dict = np.squeeze(feature_dict)
        if method == 1:
            final_embedding.append(np.sum(feature_dict,0))
        else:
            final_embedding.append(np.mean(feature_dict,0))
    final_embedding = np.array(final_embedding)
    np.save(save_path,final_embedding)


if __name__ == "__main__":
    main()


