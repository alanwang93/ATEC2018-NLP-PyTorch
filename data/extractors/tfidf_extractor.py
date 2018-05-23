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

class TFIDFExtractor(Extractor):
    '''
    Return:
        A dict, of which keys are shown below:
        dict_keys(['s1_word', 's2_word'])
    '''
    def __init__(self):
        Extractor.__init__(self, 'word', 'embed')

    def extract(self, data, tokenized, vocab, config):
        d = dict()
        s1_tfidf = []
        s2_tfidf = []
        tfidf_sim = []
        for line in tokenized:
            line_split = line.strip().split('\t')
            s1 = line_split[1].split(" ")
            s2 =line_split[2].split(" ")
            s1_len.append(len(s1))
            s2_len.append(len(s2))
            s1_word.append(vocab.toi(s1))
            s2_word.append(vocab.toi(s2))
            label.append(float(line_split[3]))
            if '\xef\xbb\xbf' in line_split[0]:
                line_split[0] = line_split[0].replace('\xef\xbb\xbf', '')
            sid.append(int(line_split[0]))
        d['s1_tfidf'] = np.asarray([np.array(s) for s in s1_tfidf])
        d['s2_word'] = np.asarray([np.array(s) for s in s2_tfidf])
        d['s1_len'] = np.asarray(s1_len)
        d['s2_len'] = np.asarray(s2_len)
        d['label'] = np.asarray(label)
        d['sid'] = np.asarray(sid)
        return d
