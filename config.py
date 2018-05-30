#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

data_config = {
    'data_root': 'data/processed/',
    'train': 'data/processed/train.pkl',
    'valid': 'data/processed/valid.pkl',
    'char_embedding': None,#'sgns.weibo.word',
    'word_embedding': None,
    'max_char': 1000,
    'max_word': 4200,
    'min_freq': 2
}


siamese = {
    'name': 'siamese',
    'model': 'SiameseRNN',

    # model
    'embed_size': 300,
    'hidden_size': 150,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'pos_weight': 2.5,
    'representation': 'last', # last, avg
    'sim_fun': 'dense', # exp, cosine, cosine+, dense

    # training
    'batch_size': 32,
    'max_iter': 50000,
}


match_pyramid = {
    'name': 'match_pyramid',
    'model': 'MatchPyramid',

    # model
    'embed_size': 300,
    'dropout': 0.2,

    # training
    'batch_size': 32,
    'max_iter': 50000,
}


ainn = {
    'name': 'ainn',
    'model': 'AINN',

    # model
    'embed_size': 300,
    'dropout': 0.2,

    # training
    'batch_size': 32,
    'max_iter': 50000,
}
