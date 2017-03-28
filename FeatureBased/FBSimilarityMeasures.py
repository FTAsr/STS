from difflib import SequenceMatcher
import numpy as np
from scipy.stats.stats import pearsonr
import collections
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pprint
import treetaggerwrapper

#longest common substring
def longestCommonsubstring(str1, str2):
    match = SequenceMatcher(None, str1, str2).find_longest_match(0, len(str1), 0, len(str2))
    return str1[match.a: match.a + match.size]

#longest common subsequence => contiguity requirement is dropped.	
def longestCommonSubseq(str1 , str2):
    # find the length of the strings
    m = len(str1)
    n = len(str2)
 
    # declaring the array for storing the dp values
    lcs = [[None]*(n+1) for i in range(m+1)]
 
    # store lcs[m+1][n+1] using bottom up DP approach

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                lcs[i][j] = 0
            elif str1[i-1] == str2[j-1]:
                lcs[i][j] = lcs[i-1][j-1]+1
            else:
                lcs[i][j] = max(lcs[i-1][j] , lcs[i][j-1])
 
    return lcs[m][n]

# ToDo char/word n-grams
def ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


def tokenize(text, lowercase=True):
    """Extract words from a string containing English words.
    Handling of hyphenation, contractions, and numbers is left to your
    discretion.
    Tip: you may want to look into the `re` module.
    Args:
        string (str): A string containing English.
        lowercase (bool, optional): Convert words to lowercase.
    Returns:
        list: A list of words.
    """
    # YOUR CODE HERE
    if lowercase:
        text = text.lower()
    tokens = re.findall(r"[\w']+|[.,!?;]", string)
    return [w for w in tokens if w not in Q.punctuation and w not in stopwords.words('english')]

def lemmatize(text):
	lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
	counter = 0
	for token in text:
		text[counter] = lemmatizer.lemmatize(token)
		counter += 1
	return text

def shared_words(text1, text2):
    """Identify shared words in two texts written in English.
    Your function must make use of the `tokenize` function above. You should
    considering using Python `set`s to solve the problem.
    Args:
        text1 (str): A string containing English.
        text2 (str): A string containing English.
    Returns:
        set: A set with words appearing in both `text1` and `text2`.
    """
    # YOUR CODE HERE
    return set(tokenize(text1, False)) & set(tokenize(text2, False))


def shared_words_from_filenames(filename1, filename2):
    """Identify shared words in two texts stored on disk.
    Your function must make use of the `tokenize` function above. You should
    considering using Python `set`s to solve the problem.
    For each filename you will need to `open` file and read the file's
    contents.
    There are two sample text files in the `data/` directory which you can use
    to practice on.
    Args:
        filename1 (str): A string containing English.
        filename2 (str): A string containing English.
    Returns:
        set: A set with words appearing in both texts.
    """
    # YOUR CODE HERE
    with open(filename1, 'r') as file1:
        text1 = file1.read().replace('\n', '')
    with open(filename2, 'r') as file2:
        text2 = file2.read().replace('\n', '')

    return set(tokenize(text1, False)) & set(tokenize(text2, False))

def text2wordfreq(string, lowercase=False):
    """Calculate word frequencies for a text written in English.
    Handling of hyphenation and contractions is left to your discretion.
    Your function must make use of the `tokenize` function above.
    Args:
        string (str): A string containing English.
        lowercase (bool, optional): Convert words to lowercase before calculating their
            frequency.
    Returns:
        dict: A dictionary with words as keys and frequencies as values.
    """
    # YOUR CODE HERE
    tokens = tokenize(string, lowercase)
    freq = {t:0 for t in tokens}
    for t in tokens:
        freq[t] += 1
    return freq



def lexical_density(string):
    """Calculate the lexical density of a string containing English words.
    The lexical density of a sequence is defined to be the number of
    unique words divided by the number of total words. The lexical
    density of the sentence "The dog ate the hat." is 4/5.
    Ignore capitalization. For example, "The" should be counted as the same
    type as "the".
    This function should use the `text2wordfreq` function.
    Args:
        string (str): A string containing English.
    Returns:
        float: Lexical density.
    """
    # YOUR CODE HERE
    freq = text2wordfreq(string, True)
    total = len(tokenize(string, True))
    return len(freq)/total

def ttr(text):
    """Type to text ratio using standard word_tokenize method"""
    tokens = word_tokenize(text)
    return len(set(tokens))/len(tokens)

def wordPairDist(word1, word2, words):
    """word pair distance counts the number
    of words which lie between those of a given pair.
    """
    if word1 in words and word2 in words:
        return abs(words.index(word1) - words.index(word2))
    return -1

def wordPairOrder(word1, word2, text1, text2):
    """Word pair order tells whether two words occur in the
    same order in both texts (with any number of words
    in between)
    """
    pass
            

def jaccard_similarity(text1, text2):
    """Calculate Jaccard Similarity between two texts.
    The Jaccard similarity (coefficient) or Jaccard index is defined to be the
    ratio between the size of the intersection between two sets and the size of
    the union between two sets. In this case, the two sets we consider are the
    set of words extracted from `text1` and `text2` respectively.
    This function should ignore capitalization. A word with a capital
    letter should be treated the same as a word without a capital letter.
    Args:
        text1 (str): A string containing English words.
        text2 (str): A string containing English words.
    Returns:
        float: Jaccard similarity
    """
    # YOUR CODE HERE
    set1 = set(tokenize(text1, True))
    set2 = set(tokenize(text2, True))

    return len(set1 & set2)/len(set1 | set2)

def funcWordFreq(text1, text2):
	# function words as defined in Dinu and Popescu, 2009.
	function_words = ['a', 'all', 'also', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', 'do', 'down', 'even', 'every', 'for', 'from', 'had', 'has', 'have', 'her', 'his', 'if', 'in', 'into', 'is', 'it', 'its', 'may', 'more', 'must', 'my', 'no', 'not', 'now', 'of', 'on', 'one', 'only', 'or', 'our', 'shall', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'then', 'there', 'thing', 'this', 'to', 'up', 'upon', 'was', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'would', 'your']

	fdist1 = nltk.FreqDist([fw for fw in function_words if fw in text1])
	fdist2 = nltk.FreqDist([fw for fw in function_words if fw in text2])

	func_freq1, func_freq2 = [], []

	for k,v in sorted(fdist1.items()):
		func_freq1.append(v)

	for k,v in sorted(fdist2.items()):
		func_freq2.append(v)

	return pearsonr(func_freq1, func_freq2)[0]

def preProcess(text1, text2):
	#Tokenize the input and lemmatize using tree tagger implementation by Schmid.
	tagger = treetaggerwrapper.TreeTagger(TAGLANG = 'en')
	tags1 = tagger.tag_text(text1)
	tags2 = tagger.tag_text(text2)
	pprint.pprint(tags1)
	pprint.pprint(tags2)

		                  
def postProcess(text1, text2):
	text1 = re.sub(r'[\W_]+', '', text1)
	text2 = re.sub(r'[\W_]+', '', text1)

	if text1 == text2:
		return 5.0
	else:
		#call classifier
		pass
