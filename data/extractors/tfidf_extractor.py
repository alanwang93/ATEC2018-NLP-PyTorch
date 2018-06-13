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
        self.feat_names = ['s1_char_lsa100',
                      's2_char_lsa100',
                      's1_word_lsa100',
                      's2_word_lsa100']

        self.feat_levels = ['s'] * 4
        self.feat_lens = [100] * 4

    def extract(self, data, char_vocab, word_vocab, mode='train'):
        s1 = []
        s2 = []
        for ins in data:
        	s1.append(" ".join(ins['s1_word']))
        	s2.append(" ".join(ins['s2_word']))
        n = len(s1)

        if mode == 'train':
            corpus = s1 + s2
            word_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False, sublinear_tf=True, ngram_range=(1, 2))
            char_vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, sublinear_tf=True, ngram_range=(1, 4))
            word_vectorizer.fit(corpus)
            char_vectorizer.fit(corpus)
            char_tfidf = char_vectorizer.transform(corpus)
            word_tfidf = word_vectorizer.transform(corpus)

            # LSA    
            char_svd = TruncatedSVD(100)
            normalizer = Normalizer(copy=False)
            lsa_char = make_pipeline(char_svd, normalizer)
            char_lsa = lsa_char.fit_transform(char_tfidf)

            word_svd = TruncatedSVD(100)
            normalizer = Normalizer(copy=False)
            lsa_word = make_pipeline(word_svd, normalizer)
            word_lsa = lsa_word.fit_transform(word_tfidf)

            s1_char_lsa100 = char_lsa[:n]
            s1_word_lsa100 = word_lsa[:n]
            s2_char_lsa100 = char_lsa[n:]
            s2_word_lsa100 = word_lsa[n:]

            # save parameters
            with open('data/params/tfidf_params.pkl', 'w') as f:
                pickle.dump({'word_params': word_vectorizer.get_params(),
                             'char_params': char_vectorizer.get_params(),
                             'char_svd': char_svd.get_params(),
                             'word_svd': word_svd.get_params()}, f)

            with open('data/processed/train_tfidf.pkl', 'w') as f:
                pickle.dump({'s1_char_tfidf': char_tfidf[:n],
                             's2_char_tfidf': char_tfidf[n:],
                             's1_word_tfidf': word_tfidf[:n],
                             's2_word_tfidf': word_tfidf[n:]}, f)            

        elif mode == 'test':
            f = pickle.load(open('data/params/tfidf_params.pkl', 'r'))
            word_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False, sublinear_tf=True)
            char_vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, sublinear_tf=True, ngram_range=(1, 4))   
            word_vectorizer.set_params(f['word_params'])
            char_vectorizer.set_params(f['char_params'])
            # TF-IDF are sparse matrice
            corpus = s1 + s2
            char_tfidf = char_vectorizer.transform(corpus)
            word_tfidf = word_vectorizer.transform(corpus)

            char_svd = TruncatedSVD(100)
            char_svd.set_params(f['char_svd'])
            normalizer = Normalizer(copy=False)
            lsa_char = make_pipeline(char_svd, normalizer)
            char_lsa = lsa_char.transform(char_tfidf)

            word_svd = TruncatedSVD(100)
            word_svd.set_params(f['word_svd'])
            normalizer = Normalizer(copy=False)
            lsa_word = make_pipeline(word_svd, normalizer)
            word_lsa = lsa_word.transform(word_tfidf)

            s1_char_lsa100 = char_lsa[:n]
            s1_word_lsa100 = word_lsa[:n]
            s2_char_lsa100 = char_lsa[n:]
            s2_word_lsa100 = word_lsa[n:]

            with open('data/processed/test_tfidf.pkl', 'w') as f:
                pickle.dump({'s1_char_tfidf': char_tfidf[:n],
                             's2_char_tfidf': char_tfidf[n:],
                             's1_word_tfidf': word_tfidf[:n],
                             's2_word_tfidf': word_tfidf[n:]}, f)      
        
        return np.concatenate((s1_char_lsa100, s2_char_lsa100, s1_word_lsa100, s2_word_lsa100), axis=1)







