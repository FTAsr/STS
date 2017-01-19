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