""" 
Use python2.7 not 3 :P 
 
If it doesn't work it might be because of a version
mismatch in numpy and scipy.  I used numpy v1.12.0 and scipy v0.18.1. If
possible use those versions. If not I can provide more detailed
instructions on how you can compile fastsent on your system 
 
Run this from the parent directory of gensim(the one in this folder) 
"""
 
import gensim 
from gensim.models.fastsent import FastSent
 
model= FastSent.load('./FastSent_no_autoencoding_300_10_0')
 
model.sentence_similarity('you give but little when you give of your possessions', 'it is when you give of yourself that you truly give')
model.sentence_similarity('i thought i saw a pussycat', 'whats up doc')
