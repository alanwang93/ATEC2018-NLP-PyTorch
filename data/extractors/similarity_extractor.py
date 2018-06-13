from .extractor import Extractor
from ..vocab import Vocab
import jieba
import re
import numpy as np
UNK_IDX = 0

class SimilarityExtractor(Extractor):

    def __init__(self):
        Extractor.__init__(self, name="SimilarityExtractor")

        self.feat_names = ['jaccard_char_unigram',
                      'jaccard_char_bigram',
                      'jaccard_char_trigram',
                      'jaccard_word_unigram',
                      'jaccard_word_bigram',
                      'LevenshteinDistance_word']

        self.feat_levels = ['p'] * 6
        self.feat_lens = [1] * 6


    def extract(self, data_raw, chars, words):
        eps = 1e-8
        feats= []

        jaccard_char_unigram = []
        jaccard_char_bigram = []
        jaccard_char_trigram = []
        jaccard_word_unigram = []
        jaccard_word_bigram = []

        LevenshteinDistance_char = []
        LevenshteinDistance_word = []

        # jaccard for char unigram
        for ins in chars:
            s1_gram = []
            s2_gram = []
            s1_len = len(ins['s1_char'])
            s2_len = len(ins['s2_char'])
            for i in range(s1_len):
                s1_gram.append(ins['s1_char'][i])
            for i in range(s2_len):
                s2_gram.append(ins['s2_char'][i])
            inter_len = len(list(set(s1_gram).intersection(s2_gram)))
            jaccard = float(inter_len) / (s1_len + s2_len - inter_len + eps)
            jaccard_char_unigram.append(jaccard)

        # jaccard for char bigram
        for ins in data:
            s1_gram = []
            s2_gram = []
            s1_len = len(ins['s1_char'])
            s2_len = len(ins['s2_char'])
            for i in range(s1_len-1):
                s1_gram.append(ins['s1_char'][i]+ins['s1_char'][i+1])
            for i in range(s2_len-1):
                s2_gram.append(ins['s2_char'][i]+ins['s2_char'][i+1])
            inter_len = len(list(set(s1_gram).intersection(s2_gram)))
            jaccard = float(inter_len) / (s1_len + s2_len - inter_len + eps)
            jaccard_char_bigram.append(jaccard)

        # jaccard for char trigram
        for ins in data:
            s1_gram = []
            s2_gram = []
            s1_len = len(ins['s1_char'])
            s2_len = len(ins['s2_char'])
            for i in range(s1_len-2):
                s1_gram.append(ins['s1_char'][i]+ins['s1_char'][i+1]+ins['s1_char'][i+2])
            for i in range(s2_len-2):
                s2_gram.append(ins['s2_char'][i]+ins['s2_char'][i+1]+ins['s2_char'][i+2])
            inter_len = len(list(set(s1_gram).intersection(s2_gram)))
            jaccard = float(inter_len) / (s1_len + s2_len - inter_len + eps)
            jaccard_char_trigram.append(jaccard)

        # jaccard for word unigram
        for ins in data:
            s1_gram = []
            s2_gram = []
            s1_len = len(ins['s1_word'])
            s2_len = len(ins['s2_word'])
            for i in range(s1_len):
                s1_gram.append(ins['s1_word'][i])
            for i in range(s2_len):
                s2_gram.append(ins['s2_word'][i])
            inter_len = len(list(set(s1_gram).intersection(s2_gram)))
            jaccard = float(inter_len) / (s1_len + s2_len - inter_len + eps)
            jaccard_word_unigram.append(jaccard)

        # jaccard for word bigram
        for ins in data:
            s1_gram = []
            s2_gram = []
            s1_len = len(ins['s1_word'])
            s2_len = len(ins['s2_word'])
            for i in range(s1_len-1):
                s1_gram.append(ins['s1_word'][i]+ins['s1_word'][i+1])
            for i in range(s2_len-1):
                s2_gram.append(ins['s2_word'][i]+ins['s2_word'][i+1])
            inter_len = len(list(set(s1_gram).intersection(s2_gram)))
            jaccard = float(inter_len) / (s1_len + s2_len - inter_len + eps)
            jaccard_char_bigram.append(jaccard)

        # LevenshteinDistance for char
        # for ins in chars:
            # dis = self.LevenshteinDistance(ins['s1'], ins['s2'])
            # LevenshteinDistance_char.append(dis)

        # LevenshteinDistance for word
        for ins in data:
            dis = self.LevenshteinDistance(ins['s1_word'], ins['s2_word'])
            LevenshteinDistance_word.append(dis)

        feats.append(jaccard_char_unigram)
        feats.append(jaccard_char_bigram)
        feats.append(jaccard_char_trigram)
        feats.append(jaccard_word_unigram)
        feats.append(jaccard_word_bigram)
        feats.append(LevenshteinDistance_word)

        feats = np.concatenate(feats, axis=1)

        return feats

    def LevenshteinDistance(self, s1, s2):
            eps = 1e-6
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
            return float(d[m-1][n-1])/(m+n-2 + eps)*2.
