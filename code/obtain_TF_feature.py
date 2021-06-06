import os
from scipy.stats import combine_pvalues
import numpy as np
import pandas as pd
import sys

# /data/temp/gaozijing/DeepSilencer/TF/data/rna_tf_H1.txt
# /data/temp/gaozijing/DeepSilencer/TF/data/HOCOMOCOv11_full_annotation_HUMAN_mono.tsv
# load peaks data
peaks = sys.argv[1]
TF_exp_data_path = sys.argv[2]
mapping_data_path = sys.argv[3]

exp = np.array(pd.read_table(TF_exp_data_path, header=None, sep=' ').values)
tf_list = exp[:, 0].tolist()
exp = exp[:, 1].astype(float)
exp[np.isnan(exp)] = 0.0

maps = np.array(pd.read_table(mapping_data_path, header=None).values)[:,:2]
motif_list = maps[:, 0].tolist()
map_list = []
for m in maps:
	map_list.append(tf_list.index(m[1]))

with open(peaks+".bed", "r") as f:
	lines = f.readlines()

# initial tf feature
tf_features = np.ones((len(lines), len(tf_list)))

# load fimo result of peaks
result = np.array(pd.read_table('fimo-out-'+peaks+'/fimo.tsv', sep='\t').values)

# calculate fisher combined p-value as tf feature
for i in range(len(lines)):
	line = lines[i]
	data = line.strip().split()
	name = data[3]+'::'+data[0]+':'+data[1]+'-'+data[2]
	print(name)

	results = result[result[:, 2] == name]
	tf_bind = [[] for _ in range(len(tf_list))]
	for s in results:
		motif_index = motif_list.index(s[0])
		tf_bind[map_list[motif_index]].append(float(s[7]))
	
	for j in range(len(tf_bind)):
		if len(tf_bind[j]) != 0:
			stat, pvalue = combine_pvalues(tf_bind[j])
			if pvalue == 0:
				pvalue = sys.float_info.min
		else:
			pvalue = 1
		tf_features[i,j] = pvalue

# save tf feature as npy
np.save(peaks+'.npy', tf_features)
