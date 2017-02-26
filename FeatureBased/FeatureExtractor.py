import nltk
import pandas as pd
import numpy as np
import os
import FBSimilarityMeasures as fb

train = pd.read_csv(os.path.join('data', 'SICK/SICK_train.txt'), index_col = 0, sep = '\t', header = None, names = ['sentA', 'sentB', 'score', 'TE'])


for index, row in train.iterrows():
	train['lcsubStr'] = fb.longestCommonSubseq(train['sentA'], train['sentB'])

train.head