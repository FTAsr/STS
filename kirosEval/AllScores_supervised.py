#!/Users/fa/anaconda/bin/python

'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''


import sys
#sys.path.append('./gensim') #Essential step
sys.path = ['../gensim', '../models', '../utils'] + sys.path
# Local imports
import gensim, models, utils

import math

from gensim.models.fastsent import FastSent
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

import itertools



        

def evaluate(model, datafile, seed=1234, evaltest=False, callable = False, pairFeatures = False): 
    ## callable is true if we can get the encoding of a sentence by simply calling model[sentence]
    ## callable is false if we can only get the encoding of a sentence by model.encode()
    ## pairFeatures is true if we can call the model.pairFeatures(sentA, sentB) to obtain a vector of different similarity measures 
    ## pairFeatures is false if we first need to calculate each sentence's vector separetly and then make our own vectors through Kiros's method (applicable to distributional models)
    
    #lower case and removes punctuation
    def process(s): return [i.lower().translate(None, punctuation) for i in s]
    
    errorFlag = None
    def encode(s):
        result = list()
        for i, sentence in enumerate(s):
            try:
                if callable:
                    result.append(np.asarray( model[sentence] ))
                else: 
                    result.append( model.encode([sentence])[0] )
            except:
                print("ERROR: " + sentence)
                result.append(errorFlag)
        return result
    
    def pairFeatures(a,b):
        result = list()
        for sentenceA,sentenceB in itertools.izip(a,b):
            try:
                print "Sentence A: " + sentenceA + "\t Sentence B: " +  sentenceB
                x = model.pairFeatures(sentenceA,sentenceB)
                print("resut calculated")
                result.append(x)
            except:
                print("ERROR: " + sentenceA + " & " +  sentenceB)
                result.append(errorFlag)
        return result
            
    
    print 'Preparing data...'
    train, dev, test, scores = load_data(datafile)
    train[0], train[1], scores[0] = shuffle(train[0], train[1], scores[0], random_state=seed)
    
    if(pairFeatures):
        print 'Computing feature vectors directly through model.pairFeatures() ...'
        
        trainF = np.asarray( pairFeatures(process(train[0]), process(train[1])) )
        trainY = encode_labels(scores[0])
        
        index = [i for i, j in enumerate(trainF) if j ==  errorFlag]
        trainF = np.asarray([x for i, x in enumerate(trainF) if i not in index])
        trainY = np.asarray([x for i, x in enumerate(trainY) if i not in index])
        trainS = np.asarray([x for i, x in enumerate(scores[0]) if i not in index])
        
        devF = np.asarray( pairFeatures(process(dev[0]), process(dev[1])) )
        devY = encode_labels(scores[1])
        
        index = [i for i, j in enumerate(devF) if j ==  errorFlag]
        devF = np.asarray([x for i, x in enumerate(devF) if i not in index])
        devY = np.asarray([x for i, x in enumerate(devY) if i not in index])
        devS = np.asarray([x for i, x in enumerate(scores[1]) if i not in index])
        
        
        
    else:
        trainA = encode( process(train[0] ))
        trainB = encode( process(train[1] ))
        trainY = encode_labels(scores[0])
        
        #remove errors (unencodable sentences) from the data
        index = [i for i, j in enumerate(trainA) if j ==  errorFlag]
        index = index + [i for i, j in enumerate(trainB) if j ==  errorFlag]
        trainA = np.asarray([x for i, x in enumerate(trainA) if i not in index])
        trainB = np.asarray([x for i, x in enumerate(trainB) if i not in index])
        trainY = np.asarray([x for i, x in enumerate(trainY) if i not in index])
        trainS = np.asarray([x for i, x in enumerate(scores[0]) if i not in index])
    
        devA = encode( process(dev[0] ))
        devB = encode( process(dev[1] ))
        devY = encode_labels(scores[1])
    
        #remove errors (unencodable sentences) from the data    
        index = [i for i, j in enumerate(devA) if j ==  errorFlag]
        index = index + [i for i, j in enumerate(devB) if j ==  errorFlag]
        devA = np.asarray([x for i, x in enumerate(devA) if i not in index])
        devB = np.asarray([x for i, x in enumerate(devB) if i not in index])
        devY = np.asarray([x for i, x in enumerate(devY) if i not in index])
        devS = np.asarray([x for i, x in enumerate(scores[1]) if i not in index])

        print 'Computing feature combinations...'
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        devF = np.c_[np.abs(devA - devB), devA * devB]

    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=trainF.shape[1])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, devS)

    if evaltest:
        print 'Evaluating...'
        
        if(pairFeatures):
            print 'Computing feature vectors directly through model.pairFeatures() ...'
            testF = np.asarray( pairFeatures(process(test[0]), process(test[1])) )
            
            index = [i for i, j in enumerate(testF) if j ==  errorFlag]
            testF = np.asarray([x for i, x in enumerate(testF) if i not in index])
            testS = np.asarray([x for i, x in enumerate(scores[2]) if i not in index])
            
        else:
            testA = encode( process(test[0] ))
            testB = encode( process(test[1] ))
        
            #remove errors (unencodable sentences) from the data
            index = [i for i, j in enumerate(testA) if j ==  errorFlag]
            index = index + [i for i, j in enumerate(testB) if j ==  errorFlag]
            testA = np.asarray([x for i, x in enumerate(testA) if i not in index])
            testB = np.asarray([x for i, x in enumerate(testB) if i not in index])
            testS = np.asarray([x for i, x in enumerate(scores[2]) if i not in index])
        
            print 'Computing feature combinations...'
            testF = np.c_[np.abs(testA - testB), testA * testB]

        r = np.arange(1,6)
        yhat = np.dot(bestlrmodel.predict_proba(testF, verbose=2), r)
        pr = pearsonr(yhat, testS)[0]
        sr = spearmanr(yhat, testS)[0]
        se = mse(yhat, testS)
        print("\n************ SUMMARY ***********")
        print 'Train data size: ' + str(len(trainY))
        print 'Dev data size: ' + str(len(devY))
        print 'Test data size: ' + str(len(testS))
        print 'Test Pearson: ' + str(pr)
        print 'Test Spearman: ' + str(sr)
        print 'Test MSE: ' + str(se)
        print("********************************")

        return yhat


def prepare_model(ninputs=9600, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(nclass, input_dim=6)) #set this to twice the size of sentence vector or equal to the final feature vector size
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
        lrmodel.fit(X, Y, verbose=2, shuffle=False, validation_data=(devX, devY))
        yhat = np.dot(lrmodel.predict_proba(devX, verbose=2), r)
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
    yhat = np.dot(lrmodel.predict_proba(devX, verbose=2), r) 
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

    return [trainA[1:], trainB[1:]], [devA[1:], devB[1:]], [testA[1:], testB[1:]], [trainS, devS, testS]
    

'''
    
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
 
def load_data(dataFile):
    """
    Load the local short answer dataset
    """
    
    allA, allB, allS = [],[],[]

    #with open(loc + 'CollegeOldData_HighAgreementPartialScoring.txt', 'rb') as f:
    with open(dataFile, 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            allA.append(text[1])
            allB.append(text[2])
            allS.append(text[3])
            print("Reading data" + str(text))
    allA = allA[1:]
    allB = allB[1:]
    allS = [float(s) for s in allS[1:]]
    allS = [(x * 4 + 1) for x in allS] ## scale values to [1,5] like in SICK data
    
    ## remove useless datapoints
    index = [i for i, j in enumerate(allB) if (j == "empty" or  j == "I don't know")]
    allA = np.asarray([x for i, x in enumerate(allA) if i not in index])
    allB = np.asarray([x for i, x in enumerate(allB) if i not in index])
    allS = np.asarray([x for i, x in enumerate(allS) if i not in index])
    
    
    ## shuffle the data
    allS, allA, allB = shuffle(allS, allA, allB, random_state=1234)
   
    
    ## split into 45% train, 5% dev and remaining ~50% test
    trainA, devA, testA = allA[0 : int(math.floor(0.45 * len(allA)))], allA[int(math.floor(0.45 * len(allA))) + 1 : int(math.floor(0.5 * len(allA))) ], allA[int(math.floor(0.5 * len(allA))) + 1 : ]
    trainB, devB, testB = allB[0 : int(math.floor(0.45 * len(allB)))], allB[int(math.floor(0.45 * len(allB))) + 1 : int(math.floor(0.5 * len(allB))) ], allB[int(math.floor(0.5 * len(allB))) + 1 : ]
    trainS, devS, testS = allS[0 : int(math.floor(0.45 * len(allS)))], allS[int(math.floor(0.45 * len(allS))) + 1 : int(math.floor(0.5 * len(allS))) ], allS[int(math.floor(0.5 * len(allS))) + 1 : ]

    print len(trainA), len(devA), len(testA)
    return [trainA, trainB], [devA, devB], [testA, testB], [trainS, devS, testS]

'''


if __name__ == '__main__':
  
    #modelpath= '../trainedModels/'
    #model= FastSent.load(modelpath+'felixpaper_70m/FastSent_no_autoencoding_300_10_0')
    #evaluate(model,'../data/local/IES-2Exp2A_AVG.txt', evaltest=True, callable = True)
    ##evaluate(model, '../data/SICK/', evaltest=True, callable = True)
    
    
    #model = models.bow("/Users/fa/workspace/repos/_codes/MODELS/Rob/word2vec_300_6/vectorsW.bin")
    #evaluate(model,'../data/local/IES-2Exp2A_AVG.txt', evaltest=True, callable = False)
    ##evaluate(model, '../data/SICK/', evaltest=True, callable = False)
    
    
    model = models.featureBased()
    ##evaluate(model,'../data/local/IES-2Exp2A_AVG.txt', evaltest=True, callable = False, pairFeatures = True )
    evaluate(model, '../data/SICK/', evaltest=True, callable = False, pairFeatures = True)