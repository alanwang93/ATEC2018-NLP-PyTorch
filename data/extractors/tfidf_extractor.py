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

    def __init__(self):
        Extractor.__init__(self, name="TFIDFExtractor")

    def extract(self, data_raw, chars, words, char_vocab, word_vocab):
        d = dict()

        # tfidf for chars
