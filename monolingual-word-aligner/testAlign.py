from aligner import *
import math
import re


def scorer(sentenceA, sentenceB):

	sent1 = re.findall(r"[\w]+", sentenceA)
	sent2 = re.findall(r"[\w]+", sentenceB)

	numerator = 0
	denominator = len(sent1) + len(sent2)

	commonWords = align(sent1, sent2)

	for sentenceId in range(0, len(commonWords)):
		numerator += len(commonWords[sentenceId])
		
	score = float(numerator) / float(denominator)
	return score


# aligning strings (output indexes start at 1)
sentence1 = "Four men died in an accident."
sentence2 = "4 people are not dead from a collision."

alignments = align(sentence1, sentence2)
print alignments[0]
print alignments[1]

'''
# aligning sets of tokens (output indexes start at 1)
sentence1 = ['Four', 'men', 'died', 'in', 'an', 'accident', '.']
sentence2 = ['4', 'people', 'are', 'dead', 'from', 'a', 'collision', '.']

alignments = align(sentence1, sentence2)
alignmentsrev = align(sentence2, sentence1)
'''

print scorer(sentence1, sentence2)

#print alignments[0]
#print alignments[1]

#print alignmentsrev[0]
#print alignmentsrev[1]




