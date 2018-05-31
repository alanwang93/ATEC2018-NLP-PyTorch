#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

data_config = {
    'data_root': 'data/processed/',
    'train': 'data/processed/train.pkl',
    'valid': 'data/processed/valid.pkl',
    'char_embedding': None,#'sgns.financial.char',
    'word_embedding': None,#'sgns.financial.char',#None,
    'max_char': 1500,
    'max_word': 6000,
    'min_freq': 2
}


siamese = {
    # Basic
    'name': 'siamese',
    'model': 'SiameseRNN',

    'char_embedding': None, #'sgns.financial.char',
    'word_embedding': None, #'sgns.financial.char',#None,

    # Model
    'embed_size': 300,
    'hidden_size': 200,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'pos_weight': 3.0,
    'representation': 'last', # last, avg
    'sim_fun': 'dense',# 'dense', # exp, cosine, cosine+, dense
    'loss': 'ce', # ce, cl, cl+ce
    'cl_margin': 0.3,
    'ce_alpha': 1.,

    # Training
    'batch_size': 64,
    'max_iter': 500000,
    'patience': 10,
}

gesd_siamese = {
    # Basic
    'name': 'siamese',
    'model': 'SiameseRNN',

    'char_embedding': None, #'sgns.financial.char',
    'word_embedding': None, #'sgns.financial.char',#None,

    # Model
    'embed_size': 300,
    'hidden_size': 150,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'pos_weight': 2.0,
    'representation': 'last', # last, avg
    'sim_fun': 'gesd',# 'dense', # exp, cosine, cosine+, dense
    'loss': 'ce', # ce, cl, cl+ce
    'cl_margin': 0.3,
    'ce_alpha': 1.,

    # Training
    'batch_size': 64,
    'max_iter': 50000,
    'patience': 10,
}

att_siamese = {
    # Basic
    'name': 'att_siamese',
    'model': 'AttSiameseRNN',

    'char_embedding': None,#'sgns.financial.char',
    'word_embedding': None,#'sgns.financial.char',#None,

    'char_embedding': None, #'sgns.financial.char',
    'word_embedding': None, #'sgns.financial.char',#None,

    # Model
    'embed_size': 150,
    'hidden_size': 100,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'pos_weight': 3.0,
    # 'representation': 'last', # last, avg
    'sim_fun': 'cosine',# 'dense', # exp, cosine, cosine+, dense
    'loss': 'cl', # ce, cl, cl+ce
    'cl_margin': 0.1,
    'ce_alpha': 1.,
    'one_att': True,

    # Training
    'batch_size': 32,
    'max_iter': 500000,
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
    'dropout': 0.5,
    'conv1_channel': 20,
    'conv2_channel': 200,
    'dp_out': 1,
    'dropout': 0.2,
    'max_grad_norm': 10.,

    # training
    'batch_size': 32,
    'max_iter': 50000,
}


ainn = {
    'name': 'ainn',
    'model': 'AINN',

    'char_embedding': None,#'sgns.financial.char',
    'word_embedding': None,#'sgns.financial.char',#None,
    # model
    'embed_size': 200,
    'dropout': 0.0,
    'channel_size': 200,
    'len': 20,

    # training
    'batch_size': 64,
    'max_iter': 50000,
    'patience': 5,
    'max_grad_norm': 5.,
}
