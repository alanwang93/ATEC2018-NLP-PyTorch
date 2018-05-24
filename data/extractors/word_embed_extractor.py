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

class WordEmbedExtractor(Extractor):

    def __init__(self):
        Extractor.__init__(self, name="WordEmbedExtractor")

    def extract(self, data_raw, chars, words, char_vocab, word_vocab, config):
        d = dict()
        s1_word = []
        s2_word = []
        s1_char = []
        s2_char = []
        s1_wlen = []
        s2_wlen = []
        s1_clen = []
        s2_clen = []
        label = []
        target = []
        sid = []

        for ins in data_raw:
            label.append(ins['label'])
            target.append(ins['target'])
            sid.append(ins['sid'])

        for ins in words:
            s1_wlen.append(len(ins['s1']))
            s2_wlen.append(len(ins['s2']))
            s1_word.append(word_vocab.toi(ins['s1']))
            s2_word.append(word_vocab.toi(ins['s2']))

        for ins in chars:
            s1_clen.append(len(ins['s1']))
            s2_clen.append(len(ins['s2']))
            s1_char.append(char_vocab.toi(ins['s1']))
            s2_char.append(char_vocab.toi(ins['s2']))
            

        d['s1_word'] = ('w', np.asarray([np.array(s) for s in s1_word]))
        d['s2_word'] = ('w', np.asarray([np.array(s) for s in s2_word]))
        d['s1_wlen'] = ('s', np.asarray(s1_wlen))
        d['s2_wlen'] = ('s', np.asarray(s2_wlen))

        d['s1_char'] = ('c', np.asarray([np.array(s) for s in s1_char]))
        d['s2_char'] = ('c', np.asarray([np.array(s) for s in s2_char]))
        d['s1_clen'] = ('s', np.asarray(s1_clen))
        d['s2_clen'] = ('s', np.asarray(s2_clen))

        d['label'] = ('o', np.asarray(label))
        d['sid'] = ('o', np.asarray(sid))
        d['target'] = ('o', np.asarray(target))
        return d
