from .extractor import Extractor
from ..vocab import Vocab
import jieba
import re
import numpy as np
UNK_IDX = 0

class SimilarityExtractor(Extractor):

    def __init__(self):
        Extractor.__init__(self, name="SimilarityExtractor")

    def extract(self, data_raw, chars, words):
        d = dict()
        jaccard_char_unigram = []
        jaccard_char_bigram = []
        jaccard_char_trigram = []
        jaccard_word_unigram = []

        LevenshteinDistance_char = []
        LevenshteinDistance_word = []

        # jaccard for char unigram
        for ins in chars:
            s1_gram = []
            s2_gram = []
            s1_len = len(ins['s1'])
            s2_len = len(ins['s2'])
            for i in range(s1_len):
                s1_gram.append(ins['s1'][i])
            for i in range(s2_len):
                s2_gram.append(ins['s2'][i])
            inter_len = len(list(set(s1_gram).intersection(s2_gram)))
            jaccard = float(inter_len) / (s1_len + s2_len - inter_len)
            jaccard_char_unigram.append(jaccard)

        # jaccard for char bigram
        for ins in chars:
            s1_gram = []
            s2_gram = []
            s1_len = len(ins['s1'])
            s2_len = len(ins['s2'])
            for i in range(s1_len-1):
                s1_gram.append(ins['s1'][i]+ins['s1'][i+1])
            for i in range(s2_len-1):
                s2_gram.append(ins['s2'][i]+ins['s2'][i+1])
            inter_len = len(list(set(s1_gram).intersection(s2_gram)))
            jaccard = float(inter_len) / (s1_len + s2_len - inter_len)
            jaccard_char_bigram.append(jaccard)

        # jaccard for char trigram
        for ins in chars:
            s1_gram = []
            s2_gram = []
            s1_len = len(ins['s1'])
            s2_len = len(ins['s2'])
            for i in range(s1_len-2):
                s1_gram.append(ins['s1'][i]+ins['s1'][i+1]+ins['s1'][i+2])
            for i in range(s2_len-2):
                s2_gram.append(ins['s2'][i]+ins['s2'][i+1]+ins['s2'][i+2])
            inter_len = len(list(set(s1_gram).intersection(s2_gram)))
            jaccard = float(inter_len) / (s1_len + s2_len - inter_len)
            jaccard_char_trigram.append(jaccard)

        # jaccard for word unigram
        for ins in words:
            s1_gram = []
            s2_gram = []
            s1_len = len(ins['s1'])
            s2_len = len(ins['s2'])
            for i in range(s1_len):
                s1_gram.append(ins['s1'][i])
            for i in range(s2_len):
                s2_gram.append(ins['s2'][i])
            inter_len = len(list(set(s1_gram).intersection(s2_gram)))
            jaccard = float(inter_len) / (s1_len + s2_len - inter_len)
            jaccard_word_unigram.append(jaccard)

        # LevenshteinDistance for char
        for ins in chars:
            dis = self.LevenshteinDistance(ins['s1'], ins['s2'])
            LevenshteinDistance_char.append(dis)

        # LevenshteinDistance for word
        for ins in words:
            dis = self.LevenshteinDistance(ins['s1'], ins['s2'])
            LevenshteinDistance_word.append(dis)

        d['jaccard_char_unigram'] = ('p', np.asarray(jaccard_char_unigram))
        d['jaccard_char_bigram'] = ('p', np.asarray(jaccard_char_bigram))
        d['jaccard_char_trigram'] = ('p', np.asarray(jaccard_char_trigram))
        d['jaccard_word_unigram'] = ('p', np.asarray(jaccard_word_unigram))
        d['LevenshteinDistance_char'] = ('p', np.asarray(LevenshteinDistance_char))
        d['LevenshteinDistance_word'] = ('p', np.asarray(LevenshteinDistance_word))
        return d

    def LevenshteinDistance(self, s1, s2):
            m = len(s1)+1
            n = len(s2)+1
            d = np.zeros((m, n))
            for i in range(m):
                d[i][0] = i
            for j in range(n):
                d[0][j] = j
            for i in range(1, m):
                for j in range(1, n):
                    if s1[i-1] == s2[j-1]:
                        cost = 0.
                    else:
                        cost = 1.
                    d[i][j] = min(d[i-1][j]+1., d[i][j-1]+1., d[i-1][j-1]+cost)
            return float(d[m-1][n-1])/(m+n-2)*2.
