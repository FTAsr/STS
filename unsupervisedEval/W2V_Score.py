import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import os, re
from nltk.stem.porter import *
import gensim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import xml.etree.ElementTree as xmlTree



class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in open(self.fname):
            yield line.split()
            
            

import spellChecker 
STEMMER = PorterStemmer()
STOPLIST = stopwords.words('english')
MODEL = None

def synonyms(word):
    syns = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            syns.add(str(lemma.name()))
            #print syns
    return syns 

def stem(wordSet):
    stemmedSet = set()
    for word in wordSet:
        stemmedSet.add(STEMMER.stem(word))
    return stemmedSet



def w2vScore(response, target, model, bigramTransformer = None ):
    #print("Response:" + response)
    #print("Target:" + target)
    keywords = re.findall('[a-z_]+', target.lower())
    responses = re.findall('[a-z_]+', response.lower())
    if( bigramTransformer != None ):
        keywords = bigramTransformer[keywords]
        responses = bigramTransformer[responses]
    keywords = [i for i in  keywords if i not in STOPLIST]
    responses = [i for i in  responses if i not in STOPLIST]
    if len(keywords) == 0 :
        return 0;
    keywordsPrepared = []
    responsesPrepared = []
    for i in keywords:
        if i in model.vocab:
            keywordsPrepared.append(i)
    for i in responses:
        if i in model.vocab:
            responsesPrepared.append(i)
        else:
            for candidate in spellChecker.spellCorrect(i):
                if candidate in model.vocab:
                    responsesPrepared.append(candidate)
    print(responsesPrepared)
    print(keywordsPrepared)
    if len(keywordsPrepared) == 0 or len(responsesPrepared) == 0 :
        return 0;
    result = model.n_similarity(responsesPrepared, keywordsPrepared)
    #print(result)
    return result
    
if __name__ == '__main__':
    ## preparation ##
    MODEL= Word2Vec.load_word2vec_format("/Users/fa/workspace/repos/_codes/trunk/vectors-phrase.bin", binary=True)  # C text format
    MODEL.init_sims(replace=True)
    
    ## test ##
    #bigramTransformer = gensim.models.Phrases(sentences)
    #sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
    #print(bigramTransformer[sent])
    target = "because they can eat or use them for camouflage" # or food and shelter
    responses = ["because some animals eat algae and green plant, some fish use green plant for camouflage.",
                "they can blend into them",
                "they look like food to eat",
                "they feed from them",
                "there are good stuff in it for them",
                "It's bright"]
    for response in responses:
        s2 = w2vScore(response, target, MODEL, None)
        print "W2V : " + str(s2)
      