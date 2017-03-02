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
sys.path = ['../gensim', '../models', '../utils'] + sys.path
# Local imports
import gensim 
import models
import utils

from gensim.models.fastsent import FastSent
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from string import punctuation





    
    
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
        if( b == "empty" or  b == "I don't know"):
            nokis +=1
            continue
        try:
            fs = scorer(process(a),process(b))
            scores.append(fs)
            goldstandardscores.append(s)
            print a,b,s,fs
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

''''
def load_data(loc='../data/SICK/'):
    """
    Load the SICK semantic-relatedness dataset
    """
    trainA, trainB, testA, testB = [],[],[],[]
    trainS, testS = [],[]

    with open(loc + 'SICK_train.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[1])
            trainB.append(text[2])
            trainS.append(text[3])
    with open(loc + 'SICK_test_annotated.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[1])
            testB.append(text[2])
            testS.append(text[3])

    trainS = [float(s) for s in trainS[1:]]
    testS = [float(s) for s in testS[1:]]
    
    trainA = trainA[1:] + testA[1:]
    trainB = trainB[1:] + testB[1:]
    trainS = trainS + testS
    return [trainA[1:], trainB[1:]], trainS

'''   
def load_data(dataFile):
    
    trainA, trainB = [],[]
    trainS = []

    #with open(loc + 'CollegeOldData_HighAgreementPartialScoring.txt', 'rb') as f:
    with open(dataFile, 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[1])
            trainB.append(text[2])
            trainS.append(text[3])
            print("Reading data" + str(text))
   
    trainS = [float(s) for s in trainS[1:]]

    return [trainA[1:], trainB[1:]], trainS


if __name__ == '__main__':

    #modelpath= '../trainedModels/'
    #model= FastSent.load(modelpath+'felixpaper_70m/FastSent_no_autoencoding_300_10_0')
    #evaluate(model, '../data/local/IES-2Exp2A_AVG.txt')

    #model = models.bow("/Users/fa/workspace/repos/_codes/MODELS/Rob/word2vec_300_6/vectorsW.bin")
    #print model.sentence_similarity("hello dear lady","hi miss")
    #print model.sentence_similarity("I don't understand","but you do")
    #evaluate(model,'../data/local/IES-2Exp2A_AVG.txt')
    
    model = models.quickScore()
    print model.sentence_similarity("hello dear lady","hi miss")
    print model.sentence_similarity("I don't understand","but you do")
    evaluate(model,'../data/local/IES-2Exp2A_AVG.txt')
    #evaluate(model, '../data/SICK/')
    
    
