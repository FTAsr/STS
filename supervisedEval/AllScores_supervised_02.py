#!/Users/fa/anaconda/bin/python

'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''

import sys
sys.path = ['../', '../featuremodels', '../utils', '../pretrained/embeddings'] + sys.path
# Local imports
import gensim, utils
from featuremodels import models as md

import math
#from gensim.models.fastsent import FastSent
from string import punctuation
from sklearn.preprocessing import normalize
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

import itertools
import pandas as pd
import pickle

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
                part = model.pairFeatures(sentenceA,sentenceB)
                vector.extend(part)
                #print index, len(vector)
                #print sentenceA, " & " , sentenceB , " Model " , index  , ":" , part

            result.append(vector)
        except:
            #print("ERROR: " + sentenceA + " & " +  sentenceB)
            result.append(errorFlag)
    #print 'result', result
    return result
        
def train(models, trainSet, devSet, seed=1234): 
    ## Takes an input model that can calculate similarity features for sentence pairs
    ## Returns a linear regression classifier on provided (gold) similarity scores
            
    trainSet[0], trainSet[1], trainSet[2] = shuffle(trainSet[0], trainSet[1], trainSet[2], random_state=seed) 
   
    print 'Computing feature vectors directly through model.pairFeatures() ...'
    
    trainF = np.asarray( pairFeatures(models, process(trainSet[0]), process(trainSet[1])) )
    print trainF.shape

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
    
    print 'Compiling model...'
    print(trainF.shape)
    
    lrmodel = prepare_model(dim= trainF.shape[1])#, ninputs=trainF.shape[0])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, devS)
    
    r = np.arange(1,6)
    yhat = np.dot(bestlrmodel.predict_proba(devF, verbose=0), r)
    pr = pearsonr(yhat, devS)[0]
    sr = spearmanr(yhat, devS)[0]
    se = mse(yhat, devS)
    
    print("\n************ SUMMARY ***********")
    print 'Train data size: ' + str(len(trainY))
    print 'Dev data size: ' + str(len(devY))
    print 'Dev Pearson: ' + str(pr)
    print 'Dev Spearman: ' + str(sr)
    print 'Dev MSE: ' + str(se)
    print("********************************")

    return bestlrmodel

    


def test(models, classifier, testSet):
    ## Takes a linear regression classifier already trained for scoring similarity between two sentences based on the model
    ## Returns predicted scores for the input dataset together with error of calssification
    
    print 'Computing feature vectors directly through model.pairFeatures() ...'
    testF = np.asarray( pairFeatures(models, process(testSet[0]), process(testSet[1])) )
    index = [i for i, j in enumerate(testF) if j ==  errorFlag]
    testF = np.asarray([x for i, x in enumerate(testF) if i not in index])
    testS = np.asarray([x for i, x in enumerate(testSet[2]) if i not in index])
    
    r = np.arange(1,6)
    yhat = np.dot(classifier.predict_proba(testF, verbose=0), r)
    pr = pearsonr(yhat, testS)[0]
    sr = spearmanr(yhat, testS)[0]
    se = mse(yhat, testS)
    

    print("\n************ SUMMARY ***********")
    print 'Test data size: ' + str(len(testS))
    print 'Test Pearson: ' + str(pr)
    print 'Test Spearman: ' + str(sr)
    print 'Test MSE: ' + str(se)
    print("********************************")

    sentenceA = np.asarray([x for i, x in enumerate(process(testSet[0])) if i not in index])
    sentenceB = np.asarray([x for i, x in enumerate(process(testSet[1])) if i not in index])
    #a =  [ (sentenceA[i], sentenceB[i], testS[i], yhat[i], np.abs(testS[i] - yhat[i]) ) for i,s in enumerate(sentenceA) ]
    a =  [ (sentenceA[i], sentenceB[i], testS[i], yhat[i], testS[i] - yhat[i]) for i,s in enumerate(sentenceA) ]
    b = pd.DataFrame(a, columns = ['target','response','score','prediction','error'])
    #print(b.sort(['error', 'score']))
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
    #yhat = np.dot(bestlrmodel.predict_proba(devX, verbose=2), r) 
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

def load_data_STS(loc='../data/SICK/'):
    """
    Load the SICK semantic-relatedness dataset
    """
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    with open(loc + 'ftrain.csv', 'rb') as f:
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
    with open(loc + 'tf2017.csv', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[1])
            testB.append(text[2])
            testS.append(text[3])

    trainS = pd.read_csv(loc + 'ftrain.csv', sep='\t').loc[:,'relatedness_score'].tolist()
    devS = [float(s) for s in devS[1:]]
    testS = pd.read_csv(loc + 'tf2017.csv', sep='\t').loc[:,'relatedness_score'].tolist()

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

def load_data_nosplit(dataFile):
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
   
    ## Everything as test data
    trainA, devA, testA = [], [], allA[:]
    trainB, devB, testB = [], [], allB[:]
    trainS, devS, testS = [], [], allS[:]

    print len(allA)
    print len(trainA)+len(devA)+len(testA)
    print len(trainA), len(devA), len(testA)
    return [trainA, trainB, trainS], [devA, devB, devS], [testA, testB, testS]

if __name__ == '__main__':
    
    ensemble = list()
    
    ## Bow model requires the path to a pre-trained word2vect or GloVe vector space in binary format
    #bowm = md.bow("../pretrained/embeddings/GoogleNews-vectors-negative300.bin")
    ## FeatureBased model is standalone and does not need any pre-trained or external resource
    fbm = md.featureBased()
    ## Feedback model is BiLSTM max pooling deep learning model.
    feedm = md.feedback()

    ## Add specific models to ensemble
    #ensemble.append(bowm)  
    ensemble.append(feedm)    
    ensemble.append(fbm)    
    
    ## Load some data for training (standard SICK dataset)
    #trainSet, devSet, testSet = load_data_SICK('../data/SICK/')
    trainSet, devSet, testSet = load_data_nosplit('../data/local/CollegeOldData_HighAgreementPartialScoring.txt')

    # Uncomment till the build_vocab method if using feedback model 
    sentences = []
    sentences.extend(trainSet[0])
    sentences.extend(trainSet[1])
    sentences.extend(devSet[0])
    sentences.extend(devSet[1])
    sentences.extend(testSet[0])
    sentences.extend(testSet[1])

    sentences = [sent.decode('utf8') for sent in sentences]
    print 'length:', len(sentences)
    feedm.feedback_model.build_vocab(sentences)

    '''
    ## Train a classifier using train and development subsets
    classifier = train(ensemble, trainSet, devSet)
    filehandler = open('pretrained/bow+sts1214.file', 'w') 
    pickle.dump(classifier, filehandler)
    '''
    classifier = pickle.load(open('../pretrained/classifiers/feed+fb+sts1214.file', 'rb'))
  
    ## Test the classifier on test data of the same type (coming from SICK)
    test(ensemble, classifier, testSet)

    '''
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

    '''