#!/Users/fa/anaconda/bin/python

""" 
Use python2.7 not 3 :P 
 
If it doesn't work it might be because of a version
mismatch in numpy and scipy.  I used numpy v1.12.0 and scipy v0.18.1. If
possible use those versions. If not I can provide more detailed
instructions on how you can compile fastsent on your system 
 
Expects a folder ./gensim with the compiled fastsent model 
Run this from the parent directory of gensim(the one in this folder) 

Expects a folder ./models with serialized(npy) model files

If using anaconda, switch to python2 environment
"""
import sys
#sys.path.append('./gensim') #Essential step
sys.path = ['../feedback', '../gensim', '../IUB/models', '../utils'] + sys.path
# Local imports
import models
import gensim 
from IUB.models import models as md
import utils

from gensim.models.fastsent import FastSent
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from string import punctuation
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import math


    
    
def ensemble(allModels, dataFile):
   
    (finalA,finalB), finalS = load_data(dataFile)
    
    okis = 0
    nokis = 0
    print(len(finalA))
    print(len(finalB))
    print(len(finalS))
    
    errorFlag = 0.0
    def scorer(a,b):
        s = 0
        votes = 0
        split = list()
        for m in allModels:
            try:
                similarity = m.sentence_similarity(a,b)
                s += similarity
                split.append(similarity)
                votes = votes + 1
            except:
                split.append(errorFlag)
                continue
        if (votes < 1):
            raise KeyError
        return ((s * 1.0) / votes) , split
    
    #lowers and removes punctuation
    def process(st): return st.lower().translate(None, punctuation)

    scores=[]
    goldstandardscores=[]
    result = list()#(a, columns = ['target','response','score','prediction','error'])
    for a,b,s in zip(finalA,finalB, finalS):
        try:
            fs, split = scorer(process(a),process(b))
            scores.append(fs)
            goldstandardscores.append(s)
            print a,":",b,s,fs, split
            okis += 1     
            record = [a,b,s,fs] + split
            result.append( record )
        except KeyError as e:
            print("Not in vocabulary:%s"%e)
            print("Original sentences: %s, %s"%(a,b))
            print("Processed sentences: %s, %s"%(process(a),process(b)))
            nokis += 1
            continue
        except:
            print("\n**UNKNOWN ERROR**")
            print("Original sentences: %s, %s"%(a,b))
            print("Processed sentences: %s, %s"%(process(a),process(b)))
            print("\n")
            nokis += 1
            continue

    print("\n************ SUMMARY ***********")
    print("Considered datapoints: " + str(okis))
    print("Erronuous datapoints (excluded): " + str(nokis))
    print("Pearson score: " + str(pearsonr(scores,goldstandardscores)))
    print("Spearman score: " + str(spearmanr(scores,goldstandardscores)))
    print("********************************")
    
    result = pd.DataFrame(result, columns = ['target','response','gold-score','prediction','QS', 'W2V', 'FS'] )
    return result
    
    
def evaluate(model, dataFile):
   
    (finalA,finalB), finalS = load_data(dataFile)
    
    okis = 0
    nokis = 0
    print(len(finalA))
    print(len(finalB))
    print(len(finalS))
    
    #wrapper around the similarity method
    def scorer(a,b):return model.sentence_similarity(a,b)
    
    #lowers and removes punctuation
    def process(st): return st.lower().translate(None, punctuation)

    scores=[]
    goldstandardscores=[]

    for a,b,s in zip(finalA,finalB, finalS):
        try:
            fs = scorer(process(a),process(b))
            scores.append(fs)
            goldstandardscores.append(s)
            #print a,b,s,fs
            okis += 1            
        except KeyError as e:
            print("Not in vocabulary:%s"%e)
            print("Original sentences: %s, %s"%(a,b))
            print("Processed sentences: %s, %s"%(process(a),process(b)))
            nokis += 1
            continue
        except:
            print("\n**UNKNOWN ERROR**")
            print("Original sentences: %s, %s"%(a,b))
            print("Processed sentences: %s, %s"%(process(a),process(b)))
            print("\n")
            nokis += 1
            continue

    print("\n************ SUMMARY ***********")
    print("Considered datapoints: " + str(okis))
    print("Erronuous datapoints (excluded): " + str(nokis))
    print("Pearson score: " + str(pearsonr(scores,goldstandardscores)))
    print("Spearman score: " + str(spearmanr(scores,goldstandardscores)))
    print("********************************")


'''
def load_data(loc='../data/SICK/'):
    """
    Load the SICK semantic-relatedness dataset
    """
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    with open(loc + 'SICK_train.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[1])
            trainB.append(text[2])
            trainS.append(text[3])
    with open(loc + 'SICK_trial.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            devA.append(text[1])
            devB.append(text[2])
            devS.append(text[3])
    with open(loc + 'SICK_test_annotated.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[1])
            testB.append(text[2])
            testS.append(text[3])

    trainS = [float(s) for s in trainS[1:]]
    devS = [float(s) for s in devS[1:]]
    testS = [float(s) for s in testS[1:]]
    
    allA = trainA[1:] + testA[1:] + devA[1:] 
    allB = trainB[1:] + testB[1:] + devB[1:]
    allS = trainS  + testS + devS
    
    print("Average length of sentenceA ", sum(map(len, allA))/float(len(allA)))
    print("Average length of sentenceB ", sum(map(len, allB))/float(len(allB)))

    return [allA, allB], allS
'''        

def load_data(dataFile):
    
    allA, allB, allS = [],[],[]

    with open(dataFile, 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            allA.append(text[1])
            allB.append(text[2])
            allS.append(text[3])
            #print("Reading data" + str(text))
    allA = allA[1:]
    allB = allB[1:]
    allS = [float(s) for s in allS[1:]]
    #allS = [(x * 4 + 1) for x in allS] ## scale [0,1] values to [1,5] like in SICK data
    
    ## remove useless datapoints
    index = [i for i, j in enumerate(allB) if (j == "empty" or ("i don't" in j.lower()) or ("idk" in j.lower()) )]
    print("No. of empty and 'i don't know' cases': " , len(index))
    index = [i for i, j in enumerate(allB) if (j == "empty" or ("i don't" in j.lower()) or ("idk" in j.lower()) or ("\n" in j) or ('\"' in j) )]
    print("No. of empty and 'i don't know' , and multi-line (suspicious) cases': " , len(index))
    allA = np.asarray([x for i, x in enumerate(allA) if i not in index])
    allB = np.asarray([x for i, x in enumerate(allB) if i not in index])
    allS = np.asarray([x for i, x in enumerate(allS) if i not in index])
    print("Average length of sentenceA ", sum(map(len, allA))/float(len(allA)))
    print("Average length of sentenceB ", sum(map(len, allB))/float(len(allB)))
    
    #lengths = pd.len(allB) 
    
    ## shuffle the data
    allS, allA, allB = shuffle(allS, allA, allB, random_state=12345)

    ## split into 45% train, 5% dev and remaining ~50% test
    trainA, devA, testA = allA[0 : int(math.floor(0.45 * len(allA)))], allA[int(math.floor(0.45 * len(allA))) + 1 : int(math.floor(0.5 * len(allA))) ], allA[int(math.floor(0.5 * len(allA))) + 1 : ]
    trainB, devB, testB = allB[0 : int(math.floor(0.45 * len(allB)))], allB[int(math.floor(0.45 * len(allB))) + 1 : int(math.floor(0.5 * len(allB))) ], allB[int(math.floor(0.5 * len(allB))) + 1 : ]
    trainS, devS, testS = allS[0 : int(math.floor(0.45 * len(allS)))], allS[int(math.floor(0.45 * len(allS))) + 1 : int(math.floor(0.5 * len(allS))) ], allS[int(math.floor(0.5 * len(allS))) + 1 : ]
    
    
    return [allA, allB], allS


if __name__ == '__main__':

    #modelpath= '../trainedModels/'
    #model= FastSent.load(modelpath+'felixpaper_70m/FastSent_no_autoencoding_300_10_0')
    #evaluate(model, '../data/SICK/')
    #evaluate(model,'../data/local/CollegeOldData_HighAgreementPartialScoring.txt')
    #evaluate(model,'../data/local/IES-2Exp1A_AVG.txt')
    #evaluate(model,'../data/local/IES-2Exp2A_AVG.txt')

    #model = models.bow("/Users/fa/workspace/repos/_codes/MODELS/Rob/word2vec_300_6/vectorsW.bin")
    #print model.sentence_similarity("hello dear lady","hi miss")
    #print model.sentence_similarity("I don't understand","but you do")
    #evaluate(model, '../data/SICK/')
    #evaluate(model,'../data/local/CollegeOldData_HighAgreementPartialScoring.txt')
    #evaluate(model,'../data/local/IES-2Exp1A_AVG.txt')
    #evaluate(model,'../data/local/IES-2Exp2A_AVG.txt')
    
    #model = models.quickScore()
    #print model.sentence_similarity("hello dear lady","hi miss")
    #print model.sentence_similarity("I don't understand","but you do")
    #evaluate(model, '../data/SICK/')
    #evaluate(model,'../data/local/CollegeOldData_HighAgreementPartialScoring.txt')
    #evaluate(model,'../data/local/IES-2Exp1A_AVG.txt')
    #evaluate(model,'../data/local/IES-2Exp2A_AVG.txt')

    model = md.feedback()
    trainSet, scoretSet = load_data('../data/SICK/CollegeOldData_HighAgreementPartialScoring.txt')
    sentences = []
    sentences.extend(trainSet[0])
    sentences.extend(trainSet[1])

    print 'length:', len(sentences)

    model.feedback_model.build_vocab(sentences)

    #evaluate(model, '../data/SICK/')
    evaluate(model,'../data/SICK/CollegeOldData_HighAgreementPartialScoring.txt')
    evaluate(model,'../data/SICK/IES-2Exp1A_AVG.txt')
    evaluate(model,'../data/SICK/IES-2Exp2A_AVG.txt')
    
    
    ## ensemble test:
    #allModels = list()
    #modelpath= '../trainedModels/'
    #allModels.append( models.quickScore() )
    #allModels.append( models.bow("/Users/fa/workspace/repos/_codes/MODELS/Rob/word2vec_300_6/vectorsW.bin") )
    #allModels.append( FastSent.load(modelpath+'felixpaper_70m/FastSent_no_autoencoding_300_10_0') )
    #ensemble(allModels, '../data/local/CollegeOldData_HighAgreementPartialScoring.txt').to_csv('../data/local/CollegeOldData_HighAgreementPartialScoring-unsupervised.csv')
    
    
    