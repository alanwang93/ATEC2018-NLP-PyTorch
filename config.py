#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

data_config = {
    'data_root': 'data/processed/',
    'train': 'data/processed/train.pkl',
    'valid': 'data/processed/valid.pkl',
    'char_embedding': None, #'sgns.weibo.word',
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
    'hidden_size': 100,
    'num_layers': 1,
    'bidirectional':True,
    'dropout': 0.5,
    'pos_weight': 1.,
    'representation': 'avg',
    'sim_fun': 'dense', # exp, cosine, cosine+, dense

    # training
    'batch_size': 16,
    'max_iter': 50000,
}


