#!/Users/fa/anaconda/bin/python
import sys
import os
sys.path.append('/home/ds/STS/FeatureBased')
import imp
imp.find_module('FBSimilarityMeasures', ['/home/ds/STS/FeatureBased'])
imp.find_module('utils', ['/home/ds/STS/utils'])

sys.path = ['../utils'] + sys.path
#import utils
import FBSimilarityMeasures as fb

import numpy as np
#from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from scipy import spatial

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

import _pickle as cPickle
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
from difflib import SequenceMatcher




class bow(object):
    """
    The distributional bag of word model of sentence meaning:
    vector representation of a sentence is obtained by adding up 
    the vectors of its constituting words.
    """
    w2vModel = None
    
    def __init__(self, modelFile):
        print("bow init: loading word2vec model")
        self.w2vModel = Word2Vec.load_word2vec_format(modelFile, binary=True) 
        return
    def encode(self, sentences, verbose=False, use_eos=True):
        sentenceVecs = list()
        sentenceVec = None
        for index, sentence in enumerate(sentences):
            #print(sentence)
            #print(type(sentence))
            sentence = sentence.lower().split()
            wordCount = 0
            for word in sentence:
                if word in self.w2vModel.vocab:
                    if wordCount == 0:
                        sentenceVec = self.w2vModel[word]
                    else:
                        sentenceVec = np.add(sentenceVec, self.w2vModel[word])
                    wordCount+=1
            if(wordCount == 0):
                #print(str(sentence))
                raise ValueError("Cannot encode sentence " + str(index) + " : all words unknown to model!  ::" + str(sentence))
            else:
                sentenceVecs.append(normalize(sentenceVec[:,np.newaxis], axis=0).ravel())
        return np.array(sentenceVecs)
    def sentence_similarity(self, sentenceA, sentenceB):
        
        a = self.encode([sentenceA])
        b = self.encode([sentenceB])
        s = (1 - spatial.distance.cosine(a[0],b[0]))
        print("word2vec score:", s)
        return s
    
    
class quickScore(object):    
    
     
    stemmer = None
    stoplist = None
    spellChecker = None
    
    def __init__(self):
        print("quickScore init: initializing the stemmer, stoplist and spellchecker")
        self.stemmer = PorterStemmer()
        self.stoplist = stopwords.words('english')
        self.spellChecker = utils.spellChecker()
        return

    def synonyms(self, word):
        syns = set()
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                syns.add(str(lemma.name()))
        return syns 

    def stem(self, wordSet):
        stemmedSet = set()
        for word in wordSet:
            stemmedSet.add(self.stemmer.stem(word))
        return stemmedSet


    def sentence_similarity(self, sentenceA, sentenceB, stemming = 0):
        exactMatch , corMatch, synMatch, keywords = self.pairFeatures(sentenceA, sentenceB, stemming )
        return (exactMatch + corMatch + synMatch) * 1.0 / keywords
        
    def pairFeatures(self, sentenceA, sentenceB, stemming = 0):
        ## Note that in quickScore, there is a difference between sentenceA and senteceB
        ## sentenceA is considered as the target (correct response)
        ## sentenceB is considered as the student response to be evaluated
        
        ##preprocess sentenceA
        keywords = re.findall('[a-z_]+', sentenceA.lower())
        keywords = [i for i in  keywords if i not in self.stoplist]
        if len(keywords) == 0:
            return 0;
        ##preprocess sentenceB
        exactsentenceB = re.findall('[a-z_]+', sentenceB.lower())
        exactsentenceB = [i for i in  exactsentenceB if i not in self.stoplist]
        exactsentenceB = set(exactsentenceB)
        correctedsentenceB = set()
        for i in  exactsentenceB:
            candidates = self.spellChecker.spellCorrect(i)
            for word in candidates:
                #print ("word "+ word + " is added as a correct candidate for " + i)
                correctedsentenceB.add(word)    
        exactMatch, corMatch, synMatch = 0,0,0 
        if(stemming == 0):
            for word in keywords:
                syns = self.synonyms(word)
                if word in exactsentenceB:
                    exactMatch = exactMatch + 1
                elif( word in correctedsentenceB):
                    corMatch = corMatch + 1
                elif (len( correctedsentenceB   & syns ) >= 1 ):
                        synMatch = synMatch + 1
        else:
            exactsentenceB = self.stem(exactsentenceB)
            correctedsentenceB = self.stem(correctedsentenceB)
            for word in keywords:
                syns = self.synonyms(word)
                syns = self.stem(syns)
                word = self.stemmer.stem(word)
                if word in exactsentenceB:
                    exactMatch = exactMatch + 1
                elif( word in correctedsentenceB):
                    corMatch = corMatch + 1
                elif (len( correctedsentenceB   & syns ) >= 1 ):
                        synMatch = synMatch + 1
        ''''
        print("\nExact matches: " + str(exactMatch)+
            "\nCorrected mathces: " + str(corMatch)+
            "\nSynonymy matches: " + str(synMatch))
        '''
        return exactMatch , corMatch , synMatch, len(keywords)
    
   
class featureBased(object):
    """
    The distributional bag of word model of sentence meaning:
    vector representation of a sentence is obtained by adding up 
    the vectors of its constituting words.
    """

    stoplist = None 
    qs = None
    
    def __init__(self):
        print("featureBased init: loading word2vec model")
        self.stoplist = stopwords.words('english')
        #self.qs = quickScore()
        return
        
    def pairFeatures(self, sentenceA, sentenceB):
        features = list()
        
        ## len features all, chars, word
        features.append( len(sentenceA) )
        features.append( len(sentenceB) )
        features.append( len(''.join(set(sentenceA.replace(' ', '')))) ) 
        features.append( len(''.join(set(sentenceB.replace(' ', '')))) ) 
        features.append( len(sentenceA.split()) )
        features.append( len(sentenceB.split()) ) 

        features.append(fb.longestCommonsubstring(sentenceA, sentenceB))
        features.append(fb.longestCommonSubseq(sentenceA, sentenceB))
        features.append(fb.funcWordFreq(sentenceA, sentenceB))
        
        ## substring and n-gram features 
        for length in (3,5,7):
            features.append(  len(set(zip(*[sentenceA[i:] for i in range(length)])).intersection(set(zip(*[sentenceB[i:] for i in range(length)]))))  )
        for length in range(1,5):
            features.append(  len(set(zip(*[sentenceA.split()[i:] for i in range(length)])).intersection(set(zip(*[sentenceB.split()[i:] for i in range(length)]))))  )
        features.append( SequenceMatcher(None, sentenceA, sentenceB).find_longest_match(0, len(sentenceA), 0, len(sentenceB))[2]  )
        
        ## token features
        features.append( len(set(sentenceA.lower().split()).intersection(set(sentenceB.lower().split()))) ) 
        features.append( fuzz.QRatio(sentenceA, sentenceB) ) 
        features.append( fuzz.WRatio(sentenceA, sentenceB) ) 
        features.append( fuzz.partial_ratio(sentenceA, sentenceB) ) 
        features.append( fuzz.partial_token_set_ratio(sentenceA, sentenceB) ) 
        features.append( fuzz.partial_token_sort_ratio(sentenceA, sentenceB) )
        features.append( fuzz.token_set_ratio(sentenceA, sentenceB) )
        features.append( fuzz.token_set_ratio(sentenceA, sentenceB) ) 
        
        """
        ## word semantic features
        
        for f in self.qs.pairFeatures(sentenceA, sentenceB, stemming = 0):
            features.append(f)
        for f in self.qs.pairFeatures(sentenceA, sentenceB, stemming = 1):
            features.append(f)    
        """
        return features

    def createFeatures(self, data):
        #create feature data frame
        data['lenA'] = data.sentence_A.apply(lambda x: len(x))
        data['lenB'] = data.sentence_B.apply(lambda x: len(x))
        data['diff_len'] = abs(data.lenA - data.lenB)
        data['len_charA'] = data.sentence_A.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        data['len_charB'] = data.sentence_B.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        data['len_wordA'] = data.sentence_A.apply(lambda x: len(str(x).split()))
        data['len_wordB'] = data.sentence_B.apply(lambda x: len(str(x).split()))
        data['ttrA'] = data.sentence_A.apply(lambda x: fb.ttr(x))
        data['ttrB'] = data.sentence_B.apply(lambda x: fb.ttr(x))
        
        data['lcstr'] = data.apply(lambda row: fb.longestCommonsubstring(row['sentence_A'], row['sentence_B']), axis=1)
        data['lcseq'] = data.apply(lambda row: fb.longestCommonSubseq(row['sentence_A'], row['sentence_B']), axis=1)
        data['fwfreq'] = data.apply(lambda row: fb.funcWordFreq(row['sentence_A'], row['sentence_B']), axis=1)
        #data['gst'] = data.apply(lambda row: fb.gst(row['sentence_A'], row['sentence_B'], 2), axis=1)
        data['common_words'] = data.apply(lambda x: len(set(str(x['sentence_A']).lower().split()).intersection(set(str(x['sentence_B']).lower().split()))), axis=1)
        data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['sentence_A']), str(x['sentence_B'])), axis=1)
        data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['sentence_A']), str(x['sentence_B'])), axis=1)
        data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['sentence_A']), str(x['sentence_B'])), axis=1)
        data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['sentence_A']), str(x['sentence_B'])), axis=1)
        data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['sentence_A']), str(x['sentence_B'])), axis=1)
        data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['sentence_A']), str(x['sentence_B'])), axis=1)
        data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['sentence_A']), str(x['sentence_B'])), axis=1)

        data = data.drop(['pair_ID', 'sentence_A', 'sentence_B', 'entailment_judgment'], axis=1)
        data.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        data = data.fillna(0)

        return data

if __name__ == '__main__':
    """
    f = featureBased()
    print f.pairFeatures("Greatings my dear lady!", "Hi miss, where is Lady Gaga?")
    print f.pairFeatures("This is a good new book!", "full of great stories")
    print f.pairFeatures("Can't say anything", "But you said something")
    print f.pairFeatures("mantel", "layer")
    print len(f.pairFeatures("mantel", "layer"))
    """

    f = featureBased()
    data = pd.read_csv("/home/ds/STS/data/SICK/SICK_train.txt", sep = '\t' , engine = 'python')
    data.head
    data = f.createFeatures(data)    
    
    Y_train = data['relatedness_score']
    X_train = data.drop(['relatedness_score'], axis=1)

    from sklearn import linear_model
    regr = linear_model.LinearRegression()

    regr.fit(X_train, Y_train)

    f = featureBased()
    data = pd.read_csv("/home/ds/STS/data/SICK/SICK_trial.txt", sep = '\t' , engine = 'python')
    data.head
    data = f.createFeatures(data)    
    
    Y_test = data['relatedness_score']
    X_test = data.drop(['relatedness_score'], axis=1)
    pred = regr.predict(X_test)

    print('Fit a model X_train, and calculate MSE with Y_train:', np.mean((Y_train - regr.predict(X_train)) ** 2))
    print('Fit a model X_train, and calculate MSE with X_test, Y_test:', np.mean((Y_test - regr.predict(X_test)) ** 2))
    #print('score sick train', regr.score(X_test, pred))   

    
    # SICK_train
    f = featureBased()
    data = pd.read_csv("/home/ds/STS/data/SICK/SICK_train.txt", sep = '\t' , engine = 'python')
    data.head
    data = f.createFeatures(data)    
    
    Y = data['relatedness_score']
    X = data.drop(['relatedness_score'], axis=1)

    from sklearn import linear_model
    regr = linear_model.LinearRegression()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=5) 
    regr.fit(X_train, Y_train)
    pred_train = regr.predict(X_train)
    pred_test = regr.predict(X_test)

    print('SICK train results')

    print('Fit a model X_train, and calculate MSE with Y_train:', np.mean((Y_train - regr.predict(X_train)) ** 2))
    print('Fit a model X_train, and calculate MSE with X_test, Y_test:', np.mean((Y_test - regr.predict(X_test)) ** 2))
    #print('score sick train', regr.score(X_test, pred_test))
    #data['features'] = data.apply(lambda row: f.pairFeatures(row['sentence_A'], row['sentence_B']), axis=1)  

    # SICK_trial
    print('SICK trial results')
    data = pd.read_csv("/home/ds/STS/data/SICK/SICK_trial.txt", sep = '\t' , engine = 'python')
    data.head
    data = f.createFeatures(data)    

    print('nan', np.any(np.isnan(data)))
    print('finite?', np.all(np.isfinite(data)))


    Y = data['relatedness_score']
    X = data.drop(['relatedness_score'], axis=1)

    from sklearn import linear_model
    regr = linear_model.LinearRegression()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=5) 
    regr.fit(X_train, Y_train)
    pred_train = regr.predict(X_train)
    pred_test = regr.predict(X_test)

    print('Fit a model X_train, and calculate MSE with Y_train:', np.mean((Y_train - regr.predict(X_train)) ** 2))
    print('Fit a model X_train, and calculate MSE with X_test, Y_test:', np.mean((Y_test - regr.predict(X_test)) ** 2))
    #print('score sick trial', regr.score(X_test, pred_test))

    # SICK_test_annotated
    print('SICK_test_annotated results')
    data = pd.read_csv("/home/ds/STS/data/SICK/SICK_test_annotated.txt", sep = '\t' , engine = 'python')
    data.head
    data = f.createFeatures(data)    

    print('nan', np.any(np.isnan(data)))
    print('finite?', np.all(np.isfinite(data)))


    Y = data['relatedness_score']
    X = data.drop(['relatedness_score'], axis=1)

    from sklearn import linear_model
    regr = linear_model.LinearRegression()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=5) 
    regr.fit(X_train, Y_train)
    pred_train = regr.predict(X_train)
    pred_test = regr.predict(X_test)

    print('Fit a model X_train, and calculate MSE with Y_train:', np.mean((Y_train - regr.predict(X_train)) ** 2))
    print('Fit a model X_train, and calculate MSE with X_test, Y_test:', np.mean((Y_test - regr.predict(X_test)) ** 2))
    #print('score sick annotated', regr.score(X_test, pred_test))
