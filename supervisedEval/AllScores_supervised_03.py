#!/Users/fa/anaconda/bin/python

'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''

import sys
#sys.path = ['../gensim', '../models', '../utils'] + sys.path
sys.path = ['../models', '../utils', '../monolingual-word-aligner'] + sys.path
# Local imports
import gensim, utils
import models as md

import math
#from gensim.models.fastsent import FastSent
from string import punctuation
from sklearn.preprocessing import normalize
import sklearn
from gensim.models import Word2Vec
from gensim import utils, matutils
import numpy as np
import copy
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.models import load_model

from scipy.stats.stats import pearsonr
from sklearn import svm
from sklearn.linear_model import Ridge

import itertools
import pandas as pd
import pickle
import csv

## This flag is used to mark cases (sentences or sentence pairs) that a model cannot successfully vectorize
errorFlag = ["error flag"] 
       
## lower case and removes punctuation from the input text
def process(s): return [i.lower().translate(None, punctuation).strip() for i in s]

## find features (a vector) describing the relation between two sentences
def pairFeatures(models, a,b):
    print "using method pairFeatures!"
    result = list()
    for sentenceA,sentenceB in itertools.izip(a,b):
        try:
            vector = list()
            for index , model in enumerate(models):
                part = model.pairFeatures(sentenceA, sentenceB)
                vector.extend(part)
                #print sentenceA, " & " , sentenceB , " Model " , index  , ":" , part
            result.append(vector)
        except Exception, e:
            #print("ERROR: " + sentenceA + " & " +  sentenceB)
            print "Couldn't do it: %s" % e
            result.append(errorFlag)


    with open("features.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(result)
    
    return result
        
def train(models, trainSet, devSet, seed=1234): 
    ## Takes an input model that can calculate similarity features for sentence pairs
    ## Returns a linear regression classifier on provided (gold) similarity scores
            
    trainSet[0], trainSet[1], trainSet[2] = shuffle(trainSet[0], trainSet[1], trainSet[2], random_state=seed) 
   
    print 'Computing feature vectors directly through model.pairFeatures() ...'
    trainF = np.asarray( pairFeatures(models, process(trainSet[0]), process(trainSet[1])) )
    trainY = encode_labels(trainSet[2])
    
    index = [i for i, j in enumerate(trainF) if j ==  errorFlag]
    trainF = np.asarray([x for i, x in enumerate(trainF) if i not in index])
    trainY = np.asarray([x for i, x in enumerate(trainY) if i not in index])
    trainS = np.asarray([x for i, x in enumerate(trainSet[0]) if i not in index])
    
    devF = np.asarray( pairFeatures(models, process(devSet[0]), process(devSet[1])) )
    devY = encode_labels(devSet[2])
    
    index = [i for i, j in enumerate(devF) if j ==  errorFlag]
    devF = np.asarray([x for i, x in enumerate(devF) if i not in index])
    devY = np.asarray([x for i, x in enumerate(devY) if i not in index])
    devS = np.asarray([x for i, x in enumerate(devSet[2]) if i not in index])
    
    
    print(trainF.shape)
    
    print 'Compiling svr model...'
    '''
    bestsvrmodel = svm.SVR()
    print(trainF.shape)
    print(trainY.shape)
    bestsvrmodel.fit(trainF, trainSet[2]) 
    '''

    bestsvrmodel = Ridge(alpha=1.0)
    bestsvrmodel.fit(trainF, trainSet[2])

    #r = np.arange(1,6)
    yhat = bestsvrmodel.predict(devF)
    pr = pearsonr(yhat, devS)[0]
    sr = spearmanr(yhat, devS)[0]
    se = mse(yhat, devS)



    print("\n************ SUMMARY DEV***********")
    print 'Train data size: ' + str(len(trainY))
    print 'Dev data size: ' + str(len(devY))
    print 'Dev Pearson: ' + str(pr)
    print 'Dev Spearman: ' + str(sr)
    print 'Dev MSE: ' + str(se)
    print("********************************")

    return bestsvrmodel  

def test(models, classifier, testSet):
    ## Takes a linear regression classifier already trained for scoring similarity between two sentences based on the model
    ## Returns predicted scores for the input dataset together with error of calssification
    
    print 'Computing feature vectors directly through model.pairFeatures() ...'
    testF = np.asarray( pairFeatures(models, process(testSet[0]), process(testSet[1])) )
    index = [i for i, j in enumerate(testF) if j ==  errorFlag]
    testF = np.asarray([x for i, x in enumerate(testF) if i not in index])
    testS = np.asarray([x for i, x in enumerate(testSet[2]) if i not in index])
    
    #r = np.arange(1,6)

    yhat = classifier.predict(testF)
    pr = pearsonr(yhat, testS)[0]
    sr = spearmanr(yhat, testS)[0]
    se = mse(yhat, testS)
    
    print("\n************ SUMMARY TEST***********")
    print 'Test data size: ' + str(len(testS))
    print 'Test Pearson: ' + str(pr)
    print 'Test Spearman: ' + str(sr)
    print 'Test MSE: ' + str(se)
    print("********************************")
    
    sentenceA = np.asarray([x for i, x in enumerate(process(testSet[0])) if i not in index])
    sentenceB = np.asarray([x for i, x in enumerate(process(testSet[1])) if i not in index])
    a =  [ (sentenceA[i], sentenceB[i], testS[i], yhat[i], np.abs(testS[i] - yhat[i]) ) for i,s in enumerate(sentenceA) ]
    b = pd.DataFrame(a, columns = ['target','response','score','prediction','error'])
    print(b.sort(['error', 'score']))
    return b
    


def prepare_model(dim, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(nclass, input_dim=dim)) #set this to twice the size of sentence vector or equal to the final feature vector size
    lrmodel.add(Activation('softmax'))
    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    return lrmodel


def train_model(lrmodel, X, Y, devX, devY, devscores):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    r = np.arange(1,6)
    
    while not done:
        # Every 100 epochs, check Pearson on development set
        lrmodel.fit(X, Y, verbose=0, shuffle=False, validation_data=(devX, devY))
        yhat = np.dot(lrmodel.predict_proba(devX, verbose=0), r)
        score = pearsonr(yhat, devscores)[0]
        if score > best:
            print 'Dev Pearson: = ' + str(score)
            best = score
            ## FA: commented out the following line because of the new keras version problem with deepcopy
            ## FA: not the model scored right after the best model will be returned (not too bad though, usually the difference is so small)
            #bestlrmodel = copy.deepcopy(lrmodel)
        else:
            done = True
    ## FA: changed here:
    #yhat = np.dot(bestlrmodel.predict_proba(devX, verbose=0), r) 
    yhat = np.dot(lrmodel.predict_proba(devX, verbose=0), r) 
    score = pearsonr(yhat, devscores)[0]
    print 'Dev Pearson: ' + str(score)
    ## FA: changed here:
    #return bestlrmodel
    return lrmodel   

def encode_labels(labels, nclass=5): 
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y


def load_data_SICK(loc='../data/SICK/'):
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

    return [trainA[1:], trainB[1:], trainS], [devA[1:], devB[1:], devS], [testA[1:], testB[1:], testS]
    
 
def load_data(dataFile):
    """
    Load the local short answer dataset
    """
    
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
    allS = [(x * 4 + 1) for x in allS] ## scale [0,1] values to [1,5] like in SICK data
    
    ## remove useless datapoints
    index = [i for i, j in enumerate(allB) if (j == "empty" or ("I don't" in j))]
    print("No. of empty and 'i don't know' cases': " , len(index))
    index = [i for i, j in enumerate(allB) if (j == "empty" or ("I don't" in j) or ("\n" in j) or ('\"' in j) )]
    print("No. of empty and 'i don't know' , 'i don't' and multi-line (suspicious) cases': " , len(index))
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

    print len(allA)
    print len(trainA)+len(devA)+len(testA)
    print len(trainA), len(devA), len(testA)
    return [trainA, trainB, trainS], [devA, devB, devS], [testA, testB, testS]



if __name__ == '__main__':
    
    ensemble = list()
    
    ## Bow model requires the path to a pre-trained word2vect or GloVe vector space in binary format
    #model = md.bow("/Users/fa/workspace/repos/_codes/MODELS/Rob/word2vec_100_6/vectorsW.bin")
    #ensemble.append(md.bow("/home/ds/STS/GoogleNews-vectors-negative300.bin"))
    
    ## FeatureBased model is standalone and does not need any pre-trained or external resource
    #ensemble.append(md.featureBased())
    
    # Word align model based on Sultan et. al.
    ensemble.append(md.align())
    
    ## Load some data for training (standard SICK dataset)
    trainSet, devSet, testSet = load_data_SICK('../data/SICK/')

    ## Train a classifier using train and development subsets
    classifier = train(ensemble, trainSet, devSet)
    
    ## Test the classifier on test data of the same type (coming from SICK)
    test(ensemble, classifier, testSet).to_csv('../data/local/SICK-trained_SICK-test.csv')

    ## FileName to save the trained classifier for later use
    fileName = '../data/local/SICK-Classifier.h5'
    
    ## VERSION THREE SAVE / LOAD (the only one that works)
    classifier.save(fileName)
    newClassifier = load_model(fileName)
    
    ## Test the saved and loaded classifier on the testSet again (to make sure the classifier didn't mess up by saving on disk)
    test(ensemble, newClassifier, testSet)
    
    
    
    ## Now we can also test the classifier on a new type of data to see how it generalizes
    
    x, y, testSet = load_data('../data/local/CollegeOldData_HighAgreementPartialScoring.txt')
    test(ensemble, newClassifier,testSet).to_csv('../data/local/SICK-trained_College-test.csv')
    
    x, y, testSet = load_data('../data/local/IES-2Exp1A_AVG.txt')
    test(ensemble, newClassifier,testSet).to_csv('../data/local/SICK-trained_Exp1A-test.csv')
    
    x, y, testSet = load_data('../data/local/IES-2Exp2A_AVG.txt')
    test(ensemble, newClassifier,testSet).to_csv('../data/local/SICK-trained_Exp2A-test.csv')
    
    
    
    
    ## ************ Results you should see for the featurBased model ********************
    
    '''
    ## On SICK
    ************ SUMMARY ***********
    Train data size: 4500
    Dev data size: 500
    Dev Pearson: 0.68317444312
    Dev Spearman: 0.603564634109
    Dev MSE: 0.54053293042
    ********************************
    ************ SUMMARY ***********
    Test data size: 4927
    Test Pearson: 0.67703114953
    Test Spearman: 0.572650244024
    Test MSE: 0.552484086087
    ********************************
    
    ## On College
    ************ SUMMARY ***********
    Test data size: 2377
    Test Pearson: 0.681083130923
    Test Spearman: 0.73253706934
    Test MSE: 1.69282707345
    ********************************
    
    ## On School
    ************ SUMMARY ***********
    Test data size: 1035
    Test Pearson: 0.83391286139
    Test Spearman: 0.831616548407
    Test MSE: 1.88451794659
    ********************************
    ************ SUMMARY ***********
    Test data size: 831
    Test Pearson: 0.940048293417
    Test Spearman: 0.912269550125
    Test MSE: 2.13254902436
    ********************************
    '''
    
    ## ************ Results you should see for the bow model with dim=100 ********************
    
    '''
    ## On SICK
    ************ SUMMARY ***********
    Train data size: 4500
    Dev data size: 500
    Dev Pearson: 0.727533547056
    Dev Spearman: 0.677997133494
    Dev MSE: 0.477910879944
    ********************************
    ************ SUMMARY ***********
    Test data size: 4927
    Test Pearson: 0.746349505723
    Test Spearman: 0.668270733008
    Test MSE: 0.451736894059
    ********************************
    
    ## On College
    ************ SUMMARY ***********
    Test data size: 2376
    Test Pearson: 0.574411300623
    Test Spearman: 0.611559028791
    Test MSE: 1.82576252723
    ********************************
    
    ## On School
    ************ SUMMARY ***********
    Test data size: 926
    Test Pearson: 0.786136186643
    Test Spearman: 0.774748380124
    Test MSE: 1.40489704198
    ********************************
    ************ SUMMARY ***********
    Test data size: 799
    Test Pearson: 0.836778366975
    Test Spearman: 0.768867968766
    Test MSE: 1.26408862889
    ********************************
    '''
    
    ## ************ Results you should see for the bow model with dim=300 ********************
    
    '''
    ## On SICK
    ************ SUMMARY ***********
    Train data size: 4500
    Dev data size: 500
    Dev Pearson: 0.772813304907
    Dev Spearman: 0.71773528102
    Dev MSE: 0.407975879066
    ********************************
    ************ SUMMARY ***********
    Test data size: 4927
    Test Pearson: 0.786754475063
    Test Spearman: 0.709340431472
    Test MSE: 0.387911887528
    ********************************
    
    ## On College
    ************ SUMMARY ***********
    Test data size: 2372
    Test Pearson: 0.582303402137
    Test Spearman: 0.613536855185
    Test MSE: 1.90456106447
    ********************************
    
    ## On School
    ************ SUMMARY ***********
    Test data size: 891
    Test Pearson: 0.805684602555
    Test Spearman: 0.787189288495
    Test MSE: 1.32262017049
    ********************************
    ************ SUMMARY ***********
    Test data size: 786
    Test Pearson: 0.920134997428
    Test Spearman: 0.79645383768
    Test MSE: 0.581767156877
    ********************************
    '''
    
    
    ## ************ Results you should see for the bow model with dim=100 + featureBased ********************
    
    ''''
    ## On SICK
    ************ SUMMARY ***********
    Train data size: 4500
    Dev data size: 500
    Dev Pearson: 0.764599637721
    Dev Spearman: 0.707902834244
    Dev MSE: 0.419814975758
    ********************************
    ************ SUMMARY ***********
    Test data size: 4927
    Test Pearson: 0.783003891986
    Test Spearman: 0.693562436578
    Test MSE: 0.394127547639
    ********************************
    
    ## On College
    ************ SUMMARY ***********
    Test data size: 2376
    Test Pearson: 0.599892044715
    Test Spearman: 0.626315623556
    Test MSE: 1.81572431625
    ********************************
    
    ## On School
    ************ SUMMARY ***********
    Test data size: 926
    Test Pearson: 0.775438334137
    Test Spearman: 0.785787532287
    Test MSE: 1.50447449955
    ********************************
    ************ SUMMARY ***********
    Test data size: 799
    Test Pearson: 0.850723174714
    Test Spearman: 0.781897416258
    Test MSE: 1.42706077904
    ********************************
    '''
    
    ## ************ Results you should see for the bow model with dim=300 + featureBased ********************
    
    ''''
    ## On SICK
    ************ SUMMARY ***********
    Train data size: 4500
    Dev data size: 500
    Dev Pearson: 0.784315968232
    Dev Spearman: 0.724620203193
    Dev MSE: 0.389213268763
    ********************************
    ************ SUMMARY ***********
    Test data size: 4927
    Test Pearson: 0.803371464158
    Test Spearman: 0.718421842395
    Test MSE: 0.360926957777
    ********************************
    
    ## On College
    ************ SUMMARY ***********
    Test data size: 2372
    Test Pearson: 0.611119924197
    Test Spearman: 0.645276932097
    Test MSE: 1.85418252049
    ********************************
    
    ## On School
    ************ SUMMARY ***********
    Test data size: 891
    Test Pearson: 0.819694779591
    Test Spearman: 0.79691695501
    Test MSE: 1.20898036887
    ********************************
    ************ SUMMARY ***********
    Test data size: 786
    Test Pearson: 0.933417623332
    Test Spearman: 0.800096195729
    Test MSE: 0.533236823978
    ********************************
    
    ## Word Aligner SVR on SICK data
    ************ SUMMARY DEV***********
    Train data size: 4500
    Dev data size: 500
    Dev Pearson: 0.697014065213
    Dev Spearman: 0.674711863823
    Dev MSE: 0.534049628537
    ********************************
    
    ************ SUMMARY TEST***********
    Test data size: 4927
    Test Pearson: 0.697597288499
    Test Spearman: 0.639736834374
    Test MSE: 0.533861416771
    ********************************
    
    '''