import numpy as np
import pandas as pd
import argparse


def seq2kmer(K,sequence):
    seq = sequence
    encoding_matrix = {'a':'A', 'A':'A', 'c':'C', 'C':'C', 'g':'G', 'G':'G', 't':'T', 'T':'T', 'n':'N', 'N':'N'}
    kmer_mat = []
    for i in range(len(seq)-K+1):
        sub_seq = seq[i:(i+K)]
        sub_seq = list(sub_seq)
        for j in range(K):
            sub_seq[j] = encoding_matrix[sub_seq[j]]
        kmer_mat.append(''.join(sub_seq) )
    return kmer_mat

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--attention_weight_path', type=str, default='./data/seq/',
                        help='positive data set path')
    parser.add_argument('--fasta_path', type=str, default='./data/tf/',
                        help='negtive tf motif score set path')
    parser.add_argument('--save', type=str, default='./data/result/',
                        help='path to save the final feature')
    parser.add_argument('--CREtype', type=str, default='silencers',
                        help='cis-regulatory element type')
    parser.add_argument('--celltype', type=str, default='K562',
                        help='cell line')
    parser.add_argument('--rank', type=int, default=2,
                        help='num of key kmer of each line of each CRE')
    parser.add_argument('--num', type=int, default=3,
                        help='num of key kmer of each cell type & CRE type')
    args = parser.parse_args()
    return args

args = get_args() 
CRE = args.CREtype
cell = args.celltype
save_path = args.save
attn_w_path = args.attention_weight_path
fasta_path = args.fasta_path
rank = args.rank
nums = args.num

# 保存所有attention 找到的kmer
cells_ = []
kmers_ = []
CREs_ = []
attention_pair = []
attention_weight = np.load(attn_w_path,allow_pickle=True).item()
data = list(attention_weight.values())
f = open(fasta_path)
ls=[]
for line in f:
    ls.append(line.replace('\n',''))
f.close()
kmer_mat = []
for i in range(1,len(ls) + 1):
    if i % 2 == 0:
        continue
    else:
        kmer_mat.append(ls[i])
# generate kmer attention list
network = []
for i in range(len(data)):
    temp_matrix = data[i]
    kmer_cell = seq2kmer(5,kmer_mat[i])
    for j in range(temp_matrix.shape[0]):
        values = np.sort(temp_matrix[j])[-rank:]
        source_kmer = kmer_cell[j]
        for value in values:
            max_index = np.where(temp_matrix[j]==value)
            for k in range(len(max_index[0])):
                target_kmer = kmer_cell[int(max_index[0][k])]
                cell = []
                cell.append(source_kmer)
                cell.append(target_kmer)
                cell.append(temp_matrix[j][max_index[0][k]])
                network.append(cell)
network = np.array(network)

# generate kmer attention unique network
temp_network = network
all_kmer_dict = {}
map_idx2kmer = []
count = 0
for i in temp_network:
    i_source = i[0]
    i_target = i[1]
    if i_source not in all_kmer_dict:
        all_kmer_dict[i_source] = count
        map_idx2kmer.append(i_source)
        count += 1
    if i_target not in all_kmer_dict:
        all_kmer_dict[i_target] = count
        map_idx2kmer.append(i_target)
        count += 1

# generate adjacent matrix
subnetwork = np.zeros((len(all_kmer_dict),len(all_kmer_dict)))
adjacent = np.zeros((len(all_kmer_dict),len(all_kmer_dict)))
count = 0
for link in temp_network:
    count += 1
    subnetwork[int(all_kmer_dict[link[0]]),int(all_kmer_dict[link[1]])] += float(link[2])
    adjacent[int(all_kmer_dict[link[0]]),int(all_kmer_dict[link[1]])] += 1
values = np.sort(adjacent.reshape(-1,))[-nums:]

# Get the most critical top Nums group results
for value in values:
    max_index = np.where(adjacent==value)
    for j in range(len(max_index[0])):
        cell_one = []
        source_kmer = map_idx2kmer[int(max_index[0][j])]
        target_kmer = map_idx2kmer[int(max_index[1][j])]
        cell_one.append(source_kmer)
        cell_one.append(target_kmer)
        cell_one.append(adjacent[max_index[0][j],max_index[1][j]])
        cell_one.append(subnetwork[max_index[0][j],max_index[1][j]])
        cell_one.append(cell)
        cell_one.append(CRE)
        attention_pair.append(cell_one)

# save result
attention_pair = np.array(attention_pair)
max_attn = pd.DataFrame(attention_pair,columns = ['source','target','attention hits','attention weight','cell line','CRE'])
max_attn.to_csv(save_path + 'key_kmer.csv')