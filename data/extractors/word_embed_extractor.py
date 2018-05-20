#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

from .extractor import Extractor
from ..vocab import tokenize, Vocab
import jieba
import re
import numpy as np
UNK_IDX = 0

class WordEmbedExtractor(Extractor):
    '''
    Return:
        A dict, of which keys are shown below:
        dict_keys(['s1_word', 's2_word'])
    '''
    def __init__(self):
        Extractor.__init__(self, 'word', 'embed')

    def extract(self, data, vocab, config):
        d = dict()
        s1_word = []
        s2_word = []
        s1_len = []
        s2_len = []
        label = []
        sid = []
        vocab_size = len(vocab)
        for line in data:
            line_split = line.strip().split('\t')
            s1 = tokenize(line_split[1], tokenizer=config['tokenizer'])
            s2 = tokenize(line_split[2], tokenizer=config['tokenizer'])
            s1_len.append(len(s1))
            s2_len.append(len(s2))
            s1_word.append(vocab.toi(s1))
            s2_word.append(vocab.toi(s2))
            label.append(float(line_split[3]))
            if '\xef\xbb\xbf' in line_split[0]:
                line_split[0] = line_split[0].replace('\xef\xbb\xbf', '')
            sid.append(int(line_split[0]))
        d['s1_word'] = np.asarray([np.array(s) for s in s1_word])
        d['s2_word'] = np.asarray([np.array(s) for s in s2_word])
        d['s1_len'] = np.asarray(s1_len)
        d['s2_len'] = np.asarray(s2_len)
        d['label'] = np.asarray(label)
        d['sid'] = np.asarray(sid)
        return d
