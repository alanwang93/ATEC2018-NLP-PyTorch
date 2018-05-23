#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.


siamese = {
    'name': 'siamese',
    'model': 'SiameseRNN',
    'data_root': 'data/processed/siamese/',
    'train': 'data/processed/siamese/train.pkl',
    'valid': 'data/processed/siamese/valid.pkl',
    'char_embedding': None, #'sgns.weibo.word',
    'word_embedding': None,

    # model
    'max_char': 1000,
    'max_word': 4200,
    'min_freq': 2,
    'embed_size': 300,
    'hidden_size': 200,
    'num_layers': 1,
    'bidirectional':True,
    'dropout': 0.3,
    'pos_weight': 3.,
    'representation': 'avg',
    'sim_fun': 'dense', # exp, cosine, cosine+, dense


    # training
    'batch_size': 16,
    'max_iter': 50000,

}


