#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

data_config = {
    'data_root': 'data/processed/',
    'valid_ratio': 0.1,
    'char_embedding': 'char_word2vec',
    'word_embedding': 'word_word2vec',
    'max_char': 1500,
    'max_word': 6000,
    'min_freq': 2,
    'embed_size': 300 # custimized embedding size
}