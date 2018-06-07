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

class TFIDFExtractor(Extractor):

    def __init__(self):
        Extractor.__init__(self, name="TFIDFExtractor")

    def extract(self, data_raw, chars, words, char_vocab, word_vocab):
        d = dict()
        s1 = []
        s2 = []
        for ins in words:
        	s1.append(" ".join(ins['s1']))
        	s2.append(" ".join(ins['s2']))
        n = len(s1)
        corpus = s1 + s2
        word_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False, sublinear_tf=True)
        char_vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, sublinear_tf=True, ngram_range=(1,2))
        word_vectorizer.fit(corpus)
        char_vectorizer.fit(corpus)
        d['s1_char_tfidf'] = ('c', char_vectorizer.transform(s1), 0)
        d['s1_word_tfidf'] = ('w', word_vectorizer.transform(s1), 0)
        d['s2_char_tfidf'] = ('c', char_vectorizer.transform(s2), 0)
        d['s2_word_tfidf'] = ('w', word_vectorizer.transform(s2), 0)
        return d







