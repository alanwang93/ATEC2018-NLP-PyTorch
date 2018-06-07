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
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

UNK_IDX = 0
EOS_IDX = 2

class TFIDFExtractor(Extractor):

    def __init__(self):
        Extractor.__init__(self, name="TFIDFExtractor")

    def extract(self, data_raw, chars, words, char_vocab, word_vocab, mode='train', lsa_components=100):
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
            char_tfidf = char_vectorizer.transform(corpus)
            word_tfidf = word_vectorizer.transform(corpus)

            # LSA    
            char_svd = TruncatedSVD(lsa_components)
            normalizer = Normalizer(copy=False)
            lsa_char = make_pipeline(char_svd, normalizer)
            char_lsa = lsa_char.fit_transform(char_tfidf)

            word_svd = TruncatedSVD(lsa_components)
            normalizer = Normalizer(copy=False)
            lsa_word = make_pipeline(word_svd, normalizer)
            word_lsa = lsa_word.fit_transform(word_tfidf)

            d['s1_char_lsa'] = ('s', char_lsa[:n], lsa_components)
            d['s1_word_lsa'] = ('s', word_lsa[:n], lsa_components)
            d['s2_char_lsa'] = ('s', char_lsa[n:], lsa_components)
            d['s2_word_lsa'] = ('s', word_lsa[n:], lsa_components)

            # save parameters
            with open('data/extractors/tfidf_params.pkl', 'w') as f:
                pickle.dump({'word_params': word_vectorizer.get_params(),
                             'char_params': char_vectorizer.get_params(),
                             'char_svd': char_svd.get_params(),
                             'word_svd': word_svd.get_params()}, f)
            with open('data/processed/train_tfidf.pkl', 'w') as f:
                pickle.dump(d, f)            

        elif mode == 'test':
            f = pickle.load(open('data/extractors/tfidf_params.pkl', 'r'))
            word_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False, sublinear_tf=True)
            char_vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, sublinear_tf=True, ngram_range=(1, 4))   
            word_vectorizer.set_params(f['word_params'])
            char_vectorizer.set_params(f['char_params'])
            # TF-IDF are sparse matrice
            s1_char_tfidf = char_vectorizer.transform(s1)
            s1_word_tfidf = word_vectorizer.transform(s1)
            s2_char_tfidf = char_vectorizer.transform(s2)
            s2_word_tfidf = word_vectorizer.transform(s2)
            d['s1_char_tfidf'] = ('s', s1_char_tfidf, s1_char_tfidf.shape[1])
            d['s1_word_tfidf'] = ('s', s1_word_tfidf, s1_word_tfidf.shape[1])
            d['s2_char_tfidf'] = ('s', s2_char_tfidf, s2_char_tfidf.shape[1])
            d['s2_word_tfidf'] = ('s', s2_word_tfidf, s2_word_tfidf.shape[1])

            with open('data/processed/test_tfidf.pkl', 'w') as f:
                pickle.dump(d, f)

        return d







