
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from scipy import spatial

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *

import utils



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
            print(sentence)
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
                print(str(sentence))
                raise ValueError("Cannot encode sentence " + str(index) + " : all words unknown to model!  ::" + str(sentence))
            else:
                sentenceVecs.append(normalize(sentenceVec[:,np.newaxis], axis=0).ravel())
        return np.array(sentenceVecs)
    def sentence_similarity(self, sentenceA, sentenceB):
        
        a = self.encode([sentenceA])
        b = self.encode([sentenceB])
        s = (1 - spatial.distance.cosine(a[0],b[0]))
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
        print("\nExact matches: " + str(exactMatch)+
            "\nCorrected mathces: " + str(corMatch)+
            "\nSynonymy matches: " + str(synMatch))
        return (exactMatch + corMatch + synMatch) * 1.0 / len(keywords)
    
   
    