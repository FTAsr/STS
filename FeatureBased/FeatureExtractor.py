import nltk
import pandas as pd
import numpy as np
import os
import FBSimilarityMeasures as fb
import time

n = 2 #number of features
train = pd.read_csv('/home/ds/STS/data/SICK/SICK_train.txt', index_col = 0, sep = '\t', header = None, names = ['sentA', 'sentB', 'score', 'TE'])
nrows = train.shape[0]
feature_vec = np.zeros((train.shape[0], n), dtype = float, order = 'C')

t0 = time.time()
for i in range(1, nrows):
	total_length = len(train.iloc[i]['sentA'] + train.iloc[i]['sentB'])
	feature_vec[i, 0] = len(fb.longestCommonsubstring(train.iloc[i]['sentA'], train.iloc[i]['sentB']))/ total_length
	feature_vec[i, 1] = fb.longestCommonSubseq(train.iloc[i]['sentA'], train.iloc[i]['sentB'])/ total_length
t1 = time.time()
print(t1-t0)

feature_vec