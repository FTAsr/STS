from difflib import SequenceMatcher

#longest common substring
def longestCommonsubstring(str1, str2):
    match = SequenceMatcher(None, str1, str2).find_longest_match(0, len(str1), 0, len(str2))
    return str1[match.a: match.a + match.size]

#longest common subsequence => contiguity requirement is dropped.	
def longestCommonSubseq(str1, str2):
    
    if not str1 or not str2:
        return ''
    x1, x2 = str1[0], str2[0]
    xs1, xs2 = str1[1:], str2[1:]
    if x1 == x2:
        return x1 + longestCommonSubseq(xs1, xs2)
    else:
        return max(longestCommonSubseq(str1, xs2), longestCommonSubseq(xs1, str2), key = len)

# ToDo improve longestCommonSubseq using memoization

# ToDo char/word n-grams

def tokenize(string, lowercase=False):
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
        string = string.lower()
    tokens = re.findall(r"[\w']+|[.,!?;]", string)
    return [w for w in tokens if w not in Q.punctuation]


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
