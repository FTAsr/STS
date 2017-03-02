#!/Users/fa/anaconda/bin/python

'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''


import sys
#sys.path.append('./gensim') #Essential step
sys.path = ['../gensim', '../models', '../utils'] + sys.path
# Local imports
import gensim, models, utils


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




        

def evaluate(model, seed=1234, evaltest=False, callable = "False"): 
    ## Callable is true if we can get the encoding of a sentence by simply calling model[sentence]
    ## Callable is false if we can only get the encoding of a sentence by model.encode()
    
    #wrapper around the similarity method
    def encode(s):
        #print s
        result = list()
        if callable:
            for i, sentence in enumerate(s):
                try:
                    result.append(model[sentence])
                except:
                    result.append("error")
            return result
        else: 
            return ( model.encode(s) )
    
    #lowers and removes punctuation
    def process(s): return [i.lower().translate(None, punctuation) for i in s]

    print 'Preparing data...'
    train, dev, test, scores = load_data()
    train[0], train[1], scores[0] = shuffle(train[0], train[1], scores[0], random_state=seed)
    
    print 'Computing training bows...'
    print(type(train[0]))
    
    trainA = encode( process(train[0] ))
    trainB = encode( process(train[1] ))
    
    print(type(trainA))
    print(np.shape(trainA))
    
    print 'Encoding training labels...'
    trainY = encode_labels(scores[0])
    
    print 'Computing development bows...'
    devA = encode( process(dev[0] ))
    devB = encode( process(dev[1] ))

    print 'Encoding dev labels...'
    devY = encode_labels(scores[1])
    
    #remove errors (unencodable sentences) from the data
    indexes = [i for i, j in enumerate(trainA) if j ==  "error"]
    indexes = indexes + [i for i, j in enumerate(trainB) if j ==  "error"]
    trainA = [trainA[x] for x in indexes]
    trainB = [trainB[x] for x in indexes]
    trainY = [trainY[x] for x in indexes]
    
    indexes = [i for i, j in enumerate(devA) if j ==  "error"]
    indexes = indexes + [i for i, j in enumerate(devB) if j ==  "error"]
    devA = [devA[x] for x in indexes]
    devB = [devB[x] for x in indexes]
    devY = [devY[x] for x in indexes]
    
    
    
    print 'Computing feature combinations...'
    trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
    devF = np.c_[np.abs(devA - devB), devA * devB]

    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=trainF.shape[1])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, scores[1])

    if evaltest:
        print 'Computing test bows...'
        testA = encode( process(test[0] ))
        testB = encode( process(test[1] ))
        
        #remove errors (unencodable sentences) from the data
        indexes = [i for i, j in enumerate(testA) if j ==  "error"]
        indexes = indexes + [i for i, j in enumerate(testB) if j ==  "error"]
        testA = [testA[x] for x in indexes]
        testB = [testA[x] for x in indexes]
        testY = [testY[x] for x in indexes]
        
        print 'Computing feature combinations...'
        testF = np.c_[np.abs(testA - testB), testA * testB]

        print 'Evaluating...'
        r = np.arange(1,6)
        yhat = np.dot(bestlrmodel.predict_proba(testF, verbose=2), r)
        pr = pearsonr(yhat, scores[2])[0]
        sr = spearmanr(yhat, scores[2])[0]
        se = mse(yhat, scores[2])
        print 'Train data size' + str(len(trainY))
        print 'Dev data size' + str(len(devY))
        print 'Test data size' + str(len(testY))
        print 'Test Pearson: ' + str(pr)
        print 'Test Spearman: ' + str(sr)
        print 'Test MSE: ' + str(se)

        return yhat


def prepare_model(ninputs=9600, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(nclass, input_dim=600))
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


if __name__ == '__main__':
  
    modelpath= '../trainedModels/'
    model= FastSent.load(modelpath+'felixpaper_70m/FastSent_no_autoencoding_300_10_0')
    evaluate(model, evaltest=True, callable = True)
    
    #model = models.bow("/Users/fa/workspace/repos/_codes/MODELS/Rob/word2vec_300_6/vectorsW.bin")
    #evaluate(model, evaltest=True, callable = False)
    
    