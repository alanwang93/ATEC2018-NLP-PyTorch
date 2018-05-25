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
UNK_IDX = 0

class WordBoolExtractor(Extractor):

    def __init__(self):
        Extractor.__init__(self, name="WordBoolExtractor")

    def extract(self, data_raw, chars, words):
        d = dict()
        funcs = ['f1']
        feats = []
        s1s = []
        s2s = []
        for ins in words:
            s1s.append(ins['s1'])
            s2s.append(ins['s2'])
        for func in funcs:
            feats.append(self.gather(func, s1s, s2s))
        d['word_bool'] = ('p', np.concatenate(feats, axis=1))
        return d

    def f1(self, s1, s2):
        """ return 1. if only one of the sentence has the word """
        words = [u'花呗', u'借呗', u'分期', u'多少', u'逾期', u'提前', u'银行卡', u'余额', u'淘宝', u'临时', u'影响', u'利息'  u'信用卡', u'手续费']
        d = len(words)
        feat = np.zeros((d))
        for i, w in enumerate(words):
            if (w in s1 and w not in s2) or (w in s2 and w not in s1):
                feat[i] = 1.
        return feat

    def gather(self, funcname, s1s, s2s):
        n = len(s1s)
        feat = []
        for s1, s2 in zip(s1s, s2s):
            feat.append(getattr(self, funcname)(s1, s2))
        return np.asarray(feat)

