import nltk
import pandas as pd
import numpy as np
import os
import FBSimilarityMeasures as fb

train = pd.read_csv('/home/ds/STS/data/SICK/SICK_train.txt', index_col = 0, sep = '\t', header = None, names = ['sentA', 'sentB', 'score', 'TE'])

train['lcsubStr'] = 0
train['lcsubSeq'] = 0

for row in train.itertuples():
	lcsubStr = 2 * len(fb.longestCommonsubstring(row[1], row[2]))/ len(row[1].strip() + row[2].strip())
	lcsubSeq = 2 * len(fb.longestCommonSubseq(row[1], row[2]))/ len(row[1].strip() + row[2].strip())

	train.set_value('index', 'lcsubStr', lcsubStr)
	train.set_value('index', 'lcsubSeq', lcsubSeq)

train.head