#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

data_config = {
    'data_root': 'data/processed/',
    'train': 'data/processed/train.pkl',
    'valid': 'data/processed/valid.pkl',
    'char_embedding': 'sgns.financial.char',
    'word_embedding': 'sgns.financial.char',#None,
    'max_char': 1500,
    'max_word': 6000,
    'min_freq': 2
}


siamese = {
    # Basic
    'name': 'siamese',
    'model': 'SiameseRNN',

    'char_embedding': None, #'sgns.financial.char',
    'word_embedding': 'sgns.financial.char',#None,

    # Model
    'embed_size': 300,
    'hidden_size': 200,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'pos_weight': 3.0,
    'representation': 'last', # last, avg
    'sim_fun': 'dense',# 'dense', # exp, cosine, cosine+, dense
    'loss': 'ce', # ce, cl, mixed
    'cl_margin': 0.4,
    'ce_alpha': 1.,

    # Training
    'batch_size': 64,
    'max_iter': 50000,
    'patience': 10,
}

light_siamese = {
    # Basic
    'name': 'light_siamese',
    'model': 'SiameseRNN',

    'char_embedding': None,#'sgns.financial.char',
    'word_embedding': None,#'sgns.financial.char',#None,

    # Model
    'embed_size': 150,
    'hidden_size': 100,
    'num_layers': 1,
    'bidirectional':True,
    'dropout': 0.5,
    'pos_weight': 3.0,
    'representation': 'last', # last, avg
    'sim_fun': 'dense',# 'dense', # exp, cosine, cosine+, dense
    'loss': 'ce', # ce, cl, mixed
    'cl_margin': 0.4,
    'ce_alpha': 1.,

    # Training
    'batch_size': 32,
    'max_iter': 50000,
    'patience': 5,
}


match_pyramid = {
    'name': 'match_pyramid',
    'model': 'MatchPyramid',

    'char_embedding': None,#'sgns.financial.char',
    'word_embedding': None,#'sgns.financial.char',#None,

    # model
    'embed_size': 300,
    'dropout': 0.2,

    # training
    'batch_size': 32,
    'max_iter': 50000,
    'patience': 5,

}
