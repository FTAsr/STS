import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import os, re
from nltk.stem.porter import *
import gensim
import pandas as pd
import numpy as np



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


def quickScore(response, target, stemming = 1):
    ##preprocess target
    keywords = re.findall('[a-z_]+', target.lower())
    keywords = [i for i in  keywords if i not in STOPLIST]
    if len(keywords) == 0:
        return 0;
    ##preprocess response
    exactResponse = re.findall('[a-z_]+', response.lower())
    exactResponse = [i for i in  exactResponse if i not in STOPLIST]
    exactResponse = set(exactResponse)
    correctedResponse = set()
    for i in  exactResponse:
        candidates = spellChecker.spellCorrect(i)
        for word in candidates:
            #print ("word "+ word + " is added as a correct candidate for " + i)
            correctedResponse.add(word)    
    exactMatch, corMatch, synMatch = 0,0,0 
    if(stemming == 0):
        for word in keywords:
            syns = synonyms(word)
            if word in exactResponse:
                exactMatch = exactMatch + 1
            elif( word in correctedResponse):
                corMatch = corMatch + 1
            elif (len( correctedResponse   & syns ) >= 1 ):
                    synMatch = synMatch + 1
    else:
        exactResponse = stem(exactResponse)
        correctedResponse = stem(correctedResponse)
        for word in keywords:
            syns = synonyms(word)
            syns = stem(syns)
            word = STEMMER.stem(word)
            if word in exactResponse:
                exactMatch = exactMatch + 1
            elif( word in correctedResponse):
                corMatch = corMatch + 1
            elif (len( correctedResponse   & syns ) >= 1 ):
                    synMatch = synMatch + 1
    print("\nExact matches: " + str(exactMatch)+
        "\nCorrected mathces: " + str(corMatch)+
        "\nSynonymy matches: " + str(synMatch))
    return (exactMatch + corMatch + synMatch) * 1.0 / len(keywords)
    
   
    
if __name__ == '__main__':
   
    target = "because they can eat or use them for camouflage" # or food and shelter
    responses = ["because some animals eat algae and green plant, some fish use green plant for camouflage.",
                "they can blend into them",
                "they look like food to eat",
                "they feed from them",
                "there are good stuff in it for them",
                "It's bright"]
    for response in responses:
        s1 = quickScore( response, target)
        print "QS : " + str(s1)
       