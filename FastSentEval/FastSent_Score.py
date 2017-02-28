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
sys.path = ['./gensim'] + sys.path
 
import gensim 
from gensim.models.fastsent import FastSent
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from string import punctuation
 
modelpath= './models/'
    
    
def evaluate(model):
    (trainA,trainB),(testA,testB), (trainS,testS)=load_data()
    finalA= trainA+testA
    finalB= trainB+testB
    finalS= trainS+testS
    
    #wrapper around the similarity method
    def sim_score(a,b):return model.sentence_similarity(a,b)

    #lowers and removes punctuation
    def process(st): return st.lower().translate(None, punctuation)

    fastsentscore=[]
    goldstandardscore=[]

    for a,b,s in zip(finalA,finalB, finalS):
        try:
            fastsentscore.append(sim_score(process(a),process(b)))
            goldstandardscore.append(s)
        except KeyError as e:
            print("Not in vocabulary:%s"%e)
            print("Original sentences: %s, %s"%(a,b))
            print("Processed sentences: %s, %s"%(process(a),process(b)))
            continue
        except:
            print("\n**UNKNOWN ERROR**")
            print("Original sentences: %s, %s"%(a,b))
            print("Processed sentences: %s, %s"%(process(a),process(b)))
            continue

            
    print("Pearson score:%f"%pearsonr(fastsentscore,goldstandardscore)[0])
    print("Spearman score:%f"%spearmanr(fastsentscore,goldstandardscore)[0])


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

    return [trainA[1:], trainB[1:]], [testA[1:], testB[1:]], [trainS, testS]

if __name__ == '__main__':
    model= FastSent.load(modelpath+'felixpaper_70m/FastSent_no_autoencoding_300_10_0')
    evaluate(model)

