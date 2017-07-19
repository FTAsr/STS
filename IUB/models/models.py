#!/Users/fa/anaconda/bin/python
import sys
sys.path = ['../utils'] + sys.path
sys.path = ['../FeatureBased'] + sys.path
sys.path = ['../monolingual-word-aligner'] + sys.path
import utils

import FBSimilarityMeasures as fb
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from scipy import spatial

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *

import aligner
import re
import torch

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
        self.w2vModel = gensim.models.KeyedVectors.load_word2vec_format(modelFile, binary=True) 
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
        #print "word2vec score:", s
        return s
    
    def pairFeatures(self, sentenceA, sentenceB):
        #obtains the distributional representation of each sentence 
        #generate element-wise subtraction and multiplication features
        a = self.encode([sentenceA])
        b = self.encode([sentenceB])
        f = np.c_[np.abs(a - b), a * b]
        #print len(f[0]), " word2vec features generated"
        return f[0]
    
    
    
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
            return 0,0,0,0;
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
        print("featureBased init: loading feature-based model")
        self.stoplist = stopwords.words('english')
        self.qs = quickScore()
        return
    def pairFeatures(self, sentenceA, sentenceB):
        features = list()
        
       
        ## len features all, chars, word
        features.append( np.log(len(sentenceA)+1) )
        features.append( np.log(len(sentenceB)+1) )
        features.append( np.log(abs(len(sentenceA) - len(sentenceB))+1) )
        features.append( np.log(len(''.join(set(sentenceA.replace(' ', ''))))+1 )) 
        features.append( np.log(len(''.join(set(sentenceB.replace(' ', ''))))+1 ))
        features.append( np.log(len(sentenceA.split())+1 ))
        features.append( np.log(len(sentenceB.split())+1 ))
        
        #features.append(np.log(fb.ttr(sentenceA)))
        #features.append(np.log(fb.ttr(sentenceB)))
        features.append(np.log(fb.longestCommonsubstring(sentenceA, sentenceB)+1))
        features.append(np.log(fb.longestCommonSubseq(sentenceA, sentenceB)+1))

        #features.append(fb.funcWordFreq(sentenceA, sentenceB))
        #features.append(fb.gst(sentenceA, sentenceB))
        
        ## substring and n-gram features 
        '''
        for length in (3,5,7):
            features.append(  len(set(zip(*[sentenceA[i:] for i in range(length)])).intersection(set(zip(*[sentenceB[i:] for i in range(length)]))))  )
        for length in range(1,5):
            features.append(  len(set(zip(*[sentenceA.split()[i:] for i in range(length)])).intersection(set(zip(*[sentenceB.split()[i:] for i in range(length)]))))  )
        features.append( SequenceMatcher(None, sentenceA, sentenceB).find_longest_match(0, len(sentenceA), 0, len(sentenceB))[2]  )
        '''
        ## token features
        features.append( len(set(sentenceA.lower().split()).intersection(set(sentenceB.lower().split()))) )      
        features.append( np.log(fuzz.QRatio(sentenceA, sentenceB)+1) ) 
        features.append( np.log(fuzz.WRatio(sentenceA, sentenceB)+1) ) 
        features.append( np.log(fuzz.partial_ratio(sentenceA, sentenceB)+1) ) 
        features.append( np.log(fuzz.partial_token_set_ratio(sentenceA, sentenceB)+1) ) 
        features.append( np.log(fuzz.partial_token_sort_ratio(sentenceA, sentenceB)+1) )
        features.append( np.log(fuzz.token_set_ratio(sentenceA, sentenceB)+1) )
        features.append( np.log(fuzz.token_sort_ratio(sentenceA, sentenceB)+1) ) 
        
        ## word semantic features
        
        for f in self.qs.pairFeatures(sentenceA, sentenceB, stemming = 0):
            features.append(f)
        for f in self.qs.pairFeatures(sentenceA, sentenceB, stemming = 1):
            features.append(f)
        
        return features


class align(object):
    """
    Monolingual word Alignment:
    Sulatn et al. 2014b 
    """
    bow = None
    def __init__(self):
        print("align init: loading monolingual-word alignment model")
        self.bow = bow('../embeddings/GoogleNews-vectors-negative300.bin')
        return

    def pairFeatures(self, sentenceA, sentenceB):

        features = []
        sentA = re.findall(r"[\w]+", sentenceA)
        sentB = re.findall(r"[\w]+", sentenceB)

        numerator = 0
        denominator = len(sentA) + len(sentB)

        alignedWords = aligner.align(sentA, sentB)

        for sentenceId in range(0, len(alignedWords)):
            numerator += len(alignedWords[sentenceId])

        features.append(float(numerator) / float(denominator))
        features.append(self.bow.sentence_similarity(sentenceA, sentenceB))
            
        return features

class embed(object):
    """
    Monolingual word Alignment:
    Sulatn et al. 2014b 
    """
    GLOVE_PATH = '../embeddings/GloVe/glove.840B.300d.txt'

    
    def __init__(self):
        print("embed init: loading the pretrained model")
        self.infersent = torch.load('../Infersent/infersent.allnli.pickle')
        self.infersent.set_glove_path(GLOVE_PATH)

        # torch.load(..pickle) will use GPU/Cuda by default. If you are on CPU:
        # self.infersent = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
        # self.infersent.use_cuda = False
        return

    def pairFeatures(self, sentenceA, sentenceB):
        features = []
        dist = cosine(model.encode([sentenceA])[0], model.encode([sentenceB])[0])
        features.append(dist)           
        return features
        

'''''
if __name__ == '__main__':
    f = featureBased()
    print f.pairFeatures("Greatings my dear lady!", "Hi miss, where is Lady Gaga?")
    print f.pairFeatures("This is a good new book!", "full of great stories")
    print f.pairFeatures("Can't say anything", "But you said something")
    print f.pairFeatures("mantel", "layer")
    print len(f.pairFeatures("mantel", "layer"))
'''

    
    
    
