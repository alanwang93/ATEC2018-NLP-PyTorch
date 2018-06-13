#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

from .extractor import Extractor
from ..vocab import Vocab
import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
UNK_IDX = 0

class TFIDFSimilarityExtractor(Extractor):

    def __init__(self):
        Extractor.__init__(self, name="TFIDFSimilarityExtractor")
        self.feat_names = ['lsa100']
        self.feat_levels = ['p'] * 5
        self.feat_lens = [1] * 5

    def extract(self, data_raw, chars, words, char_vocab, word_vocab, mode='train'):
        d = dict()
        s1 = []
        s2 = []
        for ins in words:
        	s1.append(" ".join(ins['s1']))
        	s2.append(" ".join(ins['s2']))
        n = len(s1)
        if mode == 'train':
            corpus = s1 + s2
            word_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False, sublinear_tf=True)
            char_vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, sublinear_tf=True, ngram_range=(1, 4))
            word_vectorizer.fit(corpus)
            char_vectorizer.fit(corpus)
            s1_char_tfidf = char_vectorizer.transform(s1)
            s1_word_tfidf = word_vectorizer.transform(s1)
            s2_char_tfidf = char_vectorizer.transform(s2)
            s2_word_tfidf = word_vectorizer.transform(s2)
            d['s1_char_tfidf'] = ('s', s1_char_tfidf, s1_char_tfidf.shape[1])
            d['s1_word_tfidf'] = ('s', s1_word_tfidf, s1_word_tfidf.shape[1])
            d['s2_char_tfidf'] = ('s', s2_char_tfidf, s2_char_tfidf.shape[1])
            d['s2_word_tfidf'] = ('s', s2_word_tfidf, s2_word_tfidf.shape[1])

            # save vectorizer
            with open('data/processed')
        
        return d







