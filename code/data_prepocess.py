import numpy as np
import argparse
import time


def get_args():
    parser = argparse.ArgumentParser(description='data prepocess')
    parser.add_argument('--K', type=int, default=5,
                        help='param k of kmer')
    parser.add_argument('--valid_size', type=int, default=4000000,
                        help='validation data size')
    parser.add_argument('--log_interval', type=int, default=200000,
                        help='log interval')      
    parser.add_argument('--L', type=int, default=200,
                        help='length of kmer sequence')   
    parser.add_argument('--data', type=int,
                        help='genome path')      
    parser.add_argument('--black_list', type=int,
                        help='black list path')          
    args = parser.parse_args()
    return args

# load params
args = get_args() 
K = args.K
valid_size = args.valid_size
bptt = args.log_interval
L = args.L
genome_path = args.data
black_path = args.black_list


chrs = list(range(1, 23))
chrs.extend(['X', 'Y'])
keys = ['chr' + str(x) for x in chrs]

# load the black list
f = open(black_path, 'r')
entries = f.read().splitlines()
f.close()
blacklist = []
for i, entry in enumerate(entries):
    chrkey, start, end ,score= entry.split('\t')[:4]
    blacklist.append([chrkey,start,end])
blacklist = np.array(blacklist)
blacklist[:,1:2] = blacklist[:,1:2].astype(int)
    
# 除去blacklist并拼接不同染色体的序列
sequence = ''
for name in keys:
    fa = open(genome_path + '/%s.fa'%name, 'r')
    t_sequence = fa.read().splitlines()[1:]
    fa.close()
    temp_seq = list(''.join(t_sequence))
    chr_blacklist = blacklist[np.where(blacklist[:,0] == name)]
    for i in range(len(chr_blacklist)):
        start = int(chr_blacklist[i][1]) - 1
        end = int(chr_blacklist[i][2]) - 1
        del temp_seq[start:end]
        chr_blacklist[:,1] = chr_blacklist[:,1].astype(int) - (start-end)
        chr_blacklist[:,2] = chr_blacklist[:,2].astype(int) - (start-end)
    sequence = ''.join(temp_seq)




starttime = time.time()
f2 = open('sentences_whole_train_K%s.txt'%K,'w')
kmer_mat = ''
#len(sequence) - K - 2000000
count = 0
for i in range(len(sequence) - K-valid_size):
    if i % bptt == 0:
        print(i)
    sub_seq = sequence[i:(i+K)].lower()
    if ('n' not in sub_seq) and ('N' not in sub_seq):
        f2.write(sub_seq + ' ')
        count += 1
        if count % L == 0 :
            count = 0
            f2.write("\n")
f2.close()

f2 = open('sentences_whole_val_K%s.txt'%K,'w')
kmer_mat = ''
count = 0
for i in range(len(sequence) - K - valid_size,len(sequence) - K):
    if i % bptt == 0:
        print(i)
    sub_seq = sequence[i:(i+K)].lower()
    if ('n' not in sub_seq) and ('N' not in sub_seq):
        f2.write(sub_seq + ' ')
        count += 1
        if count % L == 0 :
            count = 0
            f2.write("\n")
f2.close()


endtime = time.time()
dtime = endtime - starttime

print("程序运行时间：%.8s s" % dtime)  