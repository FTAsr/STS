#!/Users/fa/anaconda/bin/python

'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''

import sys
#sys.path = ['../gensim', '../models', '../utils'] + sys.path
sys.path = ['../', '../featuremodels', '../utils', '../monolingual-word-aligner'] + sys.path
# Local imports
import gensim, utils
from featuremodels import models as md

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
                a = "".join(sentenceA.split()).lower()
                b = "".join(sentenceB.split()).lower()
                if a==b and isinstance(model, md.align):
                    part = [1.0, 1.0]
                else:
                    part = model.pairFeatures(sentenceA, sentenceB)
                vector.extend(part)
                #print sentenceA, " & " , sentenceB , " Model " , index  , ":" , part
            result.append(vector)
        except Exception, e:
            #print("ERROR: " + sentenceA + " & " +  sentenceB)
            print "Couldn't do it: %s" % e
            print "sentence A: %s" % sentenceA
            result.append(errorFlag)

    return result
        
def train(models, trainSet, devSet, df, seed=1234): 
    ## Takes an input model that can calculate similarity features for sentence pairs
    ## Returns a linear regression classifier on provided (gold) similarity scores
           
    #trainSet[0], trainSet[1], trainSet[2] = shuffle(trainSet[0], trainSet[1], trainSet[2], random_state=seed) 
   
    print 'Computing feature vectors directly through model.pairFeatures() ...'
    trainF = np.asarray( pairFeatures([models[0]], process(trainSet[0]), process(trainSet[1])) )
    trainY = encode_labels(trainSet[2])
    
    index = [i for i, j in enumerate(trainF) if j ==  errorFlag]
    trainF = np.asarray([x for i, x in enumerate(trainF) if i not in index])
    trainY = np.asarray([x for i, x in enumerate(trainY) if i not in index])

    trainAlign = np.asarray([x for i, x in enumerate(trainSet[2]) if i not in index])
    trainS = np.asarray([x for i, x in enumerate(trainSet[0]) if i not in index])
    
    #devF = np.asarray( pairFeatures(models, process(devSet[0]), process(devSet[1])) )
    #devY = encode_labels(devSet[2])
    
    #index = [i for i, j in enumerate(devF) if j ==  errorFlag]
    #devF = np.asarray([x for i, x in enumerate(devF) if i not in index])
    #devY = np.asarray([x for i, x in enumerate(devY) if i not in index])
    #devS = np.asarray([x for i, x in enumerate(devSet[2]) if i not in index])
    #devAlign = np.asarray([x for i, x in enumerate(devSet[2]) if i not in index])
    
    ## Tarin the ensemble model (linear SVR) on the predicted outputs from these models using the same data
    currmodel = None
    if isinstance(models[0], md.bow):
        print 'Compiling Keras Logit model...'
        lrmodel = prepare_model(dim= trainF.shape[1])#, ninputs=trainF.shape[0])
        bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, devS)
        
        r = np.arange(1,6)
        yhat = np.dot(bestlrmodel.predict_proba(devF, verbose=0), r)
        pr = pearsonr(yhat, devS)[0]
        sr = spearmanr(yhat, devS)[0]
        se = mse(yhat, devS)
        currmodel = bestlrmodel
        df['bow'] = np.dot(bestlrmodel.predict_proba(trainF, verbose=0), r)
    
    if isinstance(models[0], md.featureBased):
        print 'Compiling FB svr model...'

        bestsvrmodel = svm.SVR()
        print(trainF.shape)
        print(trainY.shape)
        bestsvrmodel.fit(trainF, trainSet[2]) 

        yhat = bestsvrmodel.predict(devF)
        pr = pearsonr(yhat, devS)[0]
        sr = spearmanr(yhat, devS)[0]
        se = mse(yhat, devS)
        currmodel = bestsvrmodel
        df['fb'] = bestsvrmodel.predict(trainF)


    if isinstance(models[0], md.align):
        print 'Compiling word aligner model...'

        alignermodel = svm.SVR()
        print(trainF.shape)
        alignermodel.fit(trainF, trainAlign)
        currmodel = alignermodel


        '''
        bestRmodel = Ridge(alpha=1.0)
        bestsvrmodel.fit(trainF, trainSet[2])
    
       
        yhat = alignermodel.predict(devF)
        pr = pearsonr(yhat, devS)[0]
        sr = spearmanr(yhat, devS)[0]
        se = mse(yhat, devS)
        currmodel = alignermodel

        df['aligner'] = alignermodel.predict(trainF)
        df['target'] = trainAlign
        ''' 

    print("\n************ SUMMARY DEV***********")
    print 'Train data size: ' + str(len(trainY))
    #print 'Dev data size: ' + str(len(devY))
    #print 'Dev Pearson: ' + str(pr)
    #print 'Dev Spearman: ' + str(sr)
    #print 'Dev MSE: ' + str(se)
    print("********************************")


    return currmodel, df

def test(models, classifier, testSet):
    ## Takes a linear regression classifier already trained for scoring similarity between two sentences based on the model
    ## Returns predicted scores for the input dataset together with error of calssification
    
    print 'Computing feature vectors directly through model.pairFeatures() ...'
    testF = np.asarray( pairFeatures(models, process(testSet[0]), process(testSet[1])) )
    index = [i for i, j in enumerate(testF) if j ==  errorFlag]
    testF = np.asarray([x for i, x in enumerate(testF) if i not in index])
    testS = np.asarray([x for i, x in enumerate(testSet[2]) if i not in index])

    if isinstance(models[0], md.bow):
        r = np.arange(1,6)
        yhat = np.dot(classifier.predict_proba(testF, verbose=0), r)
        pr = pearsonr(yhat, testS)[0]
        sr = spearmanr(yhat, testS)[0]
        se = mse(yhat, testS)

    else:
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

 
    df = pd.DataFrame(columns = ['bow', 'fb', 'aligner'])
    ensemble = list()   
    ## Load some data for training (standard SICK dataset)
    #trainSet, devSet, testSet = load_data_SICK('../data/SICK/')
    trainSet, devSet, testSet = load_data('../data/local/IES-2Exp2A_AVG.txt')
    #bowm = md.bow("../pretrained/embeddings/GoogleNews-vectors-negative300.bin")
    #fbm = md.featureBased()
    alignm = md.align()
    #ensemble.append(bowm)
    #ensemble.append(fbm)
    #ensemble.append(alignm)

    #classifiers = list()

    '''
    ## Train the different models in the ensemble using train and development subsets
    for index, model in enumerate(ensemble):
        classifier, df = train([model], trainSet, devSet, df)
        classifiers.append(classifier)
        print 'writing the final DataFrame'
        filehandler = open('newaligner.file', 'w') 
        pickle.dump(classifier, filehandler)
        df.to_csv('new' + str(index) + '.csv', sep='\t')


    bow = pd.read_csv("final0.csv", sep = '\t', engine = 'python')['bow']
    fb = pd.read_csv("final1.csv", sep = '\t', engine = 'python')['fb']
    align = pd.read_csv("final2.csv", sep = '\t', engine = 'python')['aligner']  
    target = pd.read_csv("final2.csv", sep = '\t', engine = 'python')['target']  

    df = pd.concat([bow, fb, align], axis=1)
    df = pd.concat([fb, align], axis=1)
    ensembler = svm.LinearSVR()
    ensembler.fit(df, target)

    classifier, df = train([alignm], trainSet, devSet, df)
    filehandler = open('pretrained/2Aalignercos.file', 'w') 
    pickle.dump(classifier, filehandler)
    filehandler.close()
    '''
    aligner = pickle.load(open('../pretrained/classifiers/bigaligner.file', 'rb'))
    testaligner = test([alignm], aligner, testSet) 
 

    '''
    ## Test the classifier on test data of the same type (coming from SICK) 
    bowclassifier = pickle.load(open('bowsts.file', 'rb'))
    testb = test([bowm], bowclassifier, testSet)
    testb.to_csv('bowschool1sts.csv')

    fbclassifier = pickle.load(open('fb.file', 'rb'))
    testfbm = test([fbm], fbclassifier, testSet)
    testfbm.to_csv('fbschool1sick.csv')

    aligner = pickle.load(open('newaligner.file', 'rb'))
    testaligner = test([alignm], aligner, testSet) 
    testaligner.to_csv('alschool1sick.csv') 

    ensembler = pickle.load(open('stsensembler.file', 'rb'))
     
    score = pd.read_csv('../data/SICK/tf2015.csv', sep='\t')['relatedness_score']       
    '''
          
    '''
    testbow = pd.read_csv('bow2016.csv')['prediction']
    testfb = pd.read_csv('fb2016.csv')['prediction']
    testaligner = pd.read_csv('aligner2016.csv')['prediction']

    testdf = pd.concat([testbow, testfb, testaligner], axis=1)
    print testdf.shape
    print ensembler
    predicted = ensembler.predict(testdf)
    print predicted.shape

    score = pd.read_csv('../data/SICK/tf2017.csv', sep='\t')['relatedness_score']

    print 'Final Pearson score:', pearsonr(predicted, score)
    print 'Final Spearman score:', spearmanr(predicted, score) 

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

    ## FB: SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    ## aligner : SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

    '''    
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
    
    ## ************ Results you should see for the bow model with dim=100 ********************
    
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
    
    ## ************ Results you should see for the bow model with dim=300 ********************

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

    
    
    ## ************ Results you should see for the bow model with dim=100 + featureBased ********************
    
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

    
    ## ************ Results you should see for the bow model with dim=300 + featureBased ********************
    

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

    Ensemble Summary: 
    Test Pearson: 0.81611668631191669
    SpearmanrResult(correlation=0.72951444008077371, pvalue=0.0)

    STS 2017:

    Bow:
    Final Pearson score: (0.66987100682232548, 6.3919566423605649e-34)
    Final Spearman score: SpearmanrResult(correlation=0.71454552327407661, pvalue=2.342317849029466e-40)

    FB:
    Final Pearson score: (0.74176145271437888, 6.3945754880649432e-45)
    Final Spearman score: SpearmanrResult(correlation=0.78935332047512252, pvalue=1.824104681148223e-54)

    Align:
    Final Pearson score: (0.7538541858691018, 3.879141060848189e-47)
    Final Spearman score: SpearmanrResult(correlation=0.77143775041922047, pvalue=1.3328178342628674e-50)

    Ensemble:
    Final Pearson score: (0.68795343316957791, 2.1913351376866084e-36)
    Final Spearman score: SpearmanrResult(correlation=0.72950815902462873, pvalue=8.496764785409331e-43)

    Ensemble of FB & Aligner
    Final Pearson score: (0.78218800903843855, 7.083491629893758e-53)
    Final Spearman score: SpearmanrResult(correlation=0.82071354430138577, pvalue=3.1557016168325428e-62)


    STS 2016:

    Bow:
    Final Pearson score: (0.30215641209585126, 9.2477965423437191e-07)
    Final Spearman score: SpearmanrResult(correlation=0.2777190914510328, pvalue=7.0233740912205306e-06)

    FB:
    Final Pearson score: (0.33834057774695209, 3.2095787800713566e-08)
    Final Spearman score: SpearmanrResult(correlation=0.34960441662276326, pvalue=1.0282686039058664e-08)

    Align:
    Final Pearson score: (0.56057882272038251, 2.020787346017612e-22)
    Final Spearman score: SpearmanrResult(correlation=0.589463011467528, pvalue=3.6909712234740809e-25)

    Ensemble:
    Final Pearson score: (0.47890814640345625, 5.7382546623708756e-16)
    Final Spearman score: SpearmanrResult(correlation=0.48996830986719075, pvalue=9.6146747479245995e-17)

    STS 2013
    FNWN:

    ************ BoW ***********
    Test data size: 189
    Test Pearson: 0.345039955789
    Test Spearman: 0.360398537342
    Test MSE: 2.36024469251
    ********************************

    ************ FB ***********
    Test data size: 189
    Test Pearson: 0.104224793501
    Test Spearman: 0.0620177234659
    Test MSE: 3.12931859575
    ********************************

    ************ Aligner ***********
    Test data size: 189
    Test Pearson: 0.469365329699
    Test Spearman: 0.471498597174
    Test MSE: 1.01808251599
    ********************************
    Ensemble:
    Final Pearson score: (0.3146543392159209, 1.0355621924927954e-05)
    Final Spearman score: SpearmanrResult(correlation=0.34059595975763196, pvalue=1.6225300778630081e-06)

    Headlines:
    ************ Bow***********
    Test data size: 750
    Test Pearson: 0.605086186638
    Test Spearman: 0.597983568263
    Test MSE: 1.90934439564
    ********************************

    ************ FB***********
    Test data size: 750
    Test Pearson: 0.644199069979
    Test Spearman: 0.633122703125
    Test MSE: 2.1590379395
    ********************************

    ************ Aligner ***********
    Test data size: 750
    Test Pearson: 0.697519988563
    Test Spearman: 0.718648808775
    Test MSE: 1.45941023861
    ********************************

    Ensemble:
    Final Pearson score: (0.7008137856871256, 7.6988333853793256e-112)
    Final Spearman score: SpearmanrResult(correlation=0.71335486169680062, pvalue=1.3084362213545332e-117)

    OnWN:
    ************ BOW ***********
    Test data size: 561
    Test Pearson: 0.597848018685
    Test Spearman: 0.641682674511
    Test MSE: 3.08923599076
    ********************************

    ************ FB ***********
    Test data size: 561
    Test Pearson: 0.499965775819
    Test Spearman: 0.527916333023
    Test MSE: 3.83486174417
    ********************************

    ************ Aligner ***********
    Test data size: 561
    Test Pearson: 0.561579064253
    Test Spearman: 0.616475475759
    Test MSE: 2.93070652746
    ********************************
    Ensemble:
    Final Pearson score: (0.59879297568616008, 7.0928604609731812e-56)
    Final Spearman score: SpearmanrResult(correlation=0.63946062977834139, pvalue=7.9126121385100861e-66)

    ******** Trained on STS 2012-2014 ************************************************
    
    ** STS 2017 Word Aligner + Word2Vec (SVR) ***
    Test data size: 250
    Test Pearson: 0.778387149158
    Test Spearman: 0.788131708236
    Test MSE: 1.56795814473
    ********************************

    **STS 2017 FB SVR***********
    Test data size: 250
    Test Pearson: 0.766155238814
    Test Spearman: 0.761010609575
    Test MSE: 1.02557104527
    ********************************
    
    **STS 2017 Bow Keras Logit********
    Test data size: 250
    Test Pearson: 0.744191165483
    Test Spearman: 0.748707997958
    Test MSE: 1.80895879084
    ********************************
    
    Ensemble:
    Final Pearson score: (0.77169730873859232, 1.1784912606155259e-50)
    Final Spearman score: SpearmanrResult(correlation=0.77981874842228505, pvalue=2.3042436102888419e-52)

    ** STS 2016 Word Aligner + Word2Vec (SVR) ***
    Test data size: 254
    Test Pearson: 0.54924136113
    Test Spearman: 0.578755476527
    Test MSE: 3.35144575588
    ********************************

    ** STS 2016 FB SVR***********
    Test data size: 254
    Test Pearson: 0.394221382767
    Test Spearman: 0.399560647795
    Test MSE: 2.80281322317
    ********************************
    
    ** STS 2016 Bow Keras Logit***********
    Test data size: 254
    Test Pearson: 0.36567105981
    Test Spearman: 0.364301864007
    Test MSE: 4.06830129956
    ********************************

    ** STS 2015 Word Aligner + Word2Vec (SVR)***********
    Test data size: 750
    Test Pearson: 0.770254120272
    Test Spearman: 0.776709663992
    Test MSE: 0.867272424701
    ********************************
    
    ** STS 2015 Bow Keras Logit**********
    Test data size: 750
    Test Pearson: 0.723534211483
    Test Spearman: 0.727378037015
    Test MSE: 1.04531439121
    ********************************

    **STS 2015 FB SVR***********
    Test data size: 750
    Test Pearson: 0.693897044178
    Test Spearman: 0.706034392508
    Test MSE: 1.12003155534
    ********************************
   
    **STS 2017 test Word Aligner + Word2Vec SVR trained on STS 2012-16***********
    Test data size: 250
    Test Pearson: 0.779597092642
    Test Spearman: 0.787629068774
    Test MSE: 1.53103154661
    ********************************

    STS 2014 test

    ************ SUMMARY TEST***********
    Test data size: 750
    Test Pearson: 0.803171669666
    Test Spearman: 0.766668656772
    Test MSE: 0.956333691751
    ********************************

    ************ SUMMARY TEST***********
    Test data size: 750
    Test Pearson: 0.733522561926
    Test Spearman: 0.762879003993
    Test MSE: 1.72133662558
    ********************************

    ************ SUMMARY TEST***********
    Test data size: 750
    Test Pearson: 0.765908638275
    Test Spearman: 0.729009434737
    Test MSE: 1.14799365408
    ********************************

    ************ SUMMARY TEST***********
    Test data size: 300
    Test Pearson: 0.692493148789
    Test Spearman: 0.658427015551
    Test MSE: 1.12360705853
    ********************************

    ************ SUMMARY TEST***********
    Test data size: 450
    Test Pearson: 0.450757053221
    Test Spearman: 0.462924304123
    Test MSE: 2.01880818604
    ********************************
    
    ************ SUMMARY TEST***********
    Test data size: 750
    Test Pearson: 0.668846427169
    Test Spearman: 0.633815825754
    Test MSE: 1.23388906453
    ********************************

    School1
    ************ Bow model trained on sick training data***********
    Test data size: 3783
    Test Pearson: 0.799574725475
    Test Spearman: 0.656482765363
    Test MSE: 5.23938845232
    ********************************

    ************ Bow model trained on STS 2012-14***********
    Test data size: 3783
    Test Pearson: 0.828430955509
    Test Spearman: 0.664620516506
    Test MSE: 4.92477757557
    ********************************

    ************ Aligner model trained on STS 2012-16*****
    Test data size: 3784
    Test Pearson: 0.838160072582
    Test Spearman: 0.669180686622
    Test MSE: 3.92739703077
    ********************************

    ************ Aligner model trained on STS 2012-14***********
    Test data size: 3784
    Test Pearson: 0.838320380115
    Test Spearman: 0.672054738962
    Test MSE: 3.93501554647
    ********************************

    School2

    ************ Aligner model trained on STS 2012-16*****
    Test data size: 2626
    Test Pearson: 0.976472699085
    Test Spearman: 0.860407836268
    Test MSE: 5.82754887566
    ********************************

    College
    ************ Aligner model trained on STS 2012-16***********
    Test data size: 5275
    Test Pearson: 0.713027120789
    Test Spearman: 0.756616005382
    Test MSE: 9.5826915674
    ********************************





    '''