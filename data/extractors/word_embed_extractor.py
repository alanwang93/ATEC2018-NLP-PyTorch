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
EOS_IDX = 2

class WordEmbedExtractor(Extractor):

    def __init__(self):
        Extractor.__init__(self, name="WordEmbedExtractor")
        self.max_clen = 300
        self.max_wlen = 100
        self.feat_names = ['s1_word', 's2_word', 's1_wlen', 's2_wlen', 's1_char', 's2_char', 's1_clen', 's2_clen']
        self.feat_lens = [self.max_wlen, self.max_wlen, 1, 1, self.max_clen, self.max_clen, 1, 1]
        self.feat_levels = ['w', 'w', 's', 's', 'c', 'c', 's', 's']

    def extract(self, data, char_vocab, word_vocab, mode='train'):

        s1_word = []
        s2_word = []
        s1_char = []
        s2_char = []
        s1_wlen = []
        s2_wlen = []
        s1_clen = []
        s2_clen = []

        for ins in data:
            s1_wlen.append([len(ins['s1_word'])])
            s2_wlen.append([len(ins['s2_word'])])
            s1_word.append(np.pad(word_vocab.toi(ins['s1_word']), (0, self.max_wlen - len(ins['s1_word'])),\
                    'constant', constant_values=(EOS_IDX, EOS_IDX)))
            s2_word.append(np.pad(word_vocab.toi(ins['s2_word']), (0, self.max_wlen - len(ins['s2_word'])),\
                    'constant', constant_values=(EOS_IDX, EOS_IDX)))

            s1_clen.append([len(ins['s1_char'])])
            s2_clen.append([len(ins['s2_char'])])
            s1_char.append(np.pad(char_vocab.toi(ins['s1_char']), (0, self.max_clen - len(ins['s1_char'])),\
                    'constant', constant_values=(EOS_IDX, EOS_IDX)))
            s2_char.append(np.pad(char_vocab.toi(ins['s2_char']), (0, self.max_clen - len(ins['s2_char'])),\
                    'constant', constant_values=(EOS_IDX, EOS_IDX)))

        feats = np.concatenate((s1_word, s2_word, s1_wlen, s2_wlen, s1_char, s2_char, s1_clen, s2_clen), axis=1)

        return feats
