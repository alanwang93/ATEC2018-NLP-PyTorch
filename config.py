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
    'embed_size': 200,
    'hidden_size': 200,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'dropout2': 0.5,
    'pos_weight': 3.0,
    'representation': 'max', # last, avg, max
    'sim_fun': 'dense', # exp, cosine, cosine+, dense
    'loss': 'ce', # ce, cl, cl+ce
    'ce_alpha': 1.,
    'cl_margin': 0.3,

    # Training
    'batch_size': 64,
    'max_iter': 500000,
    'patience': 10,
}

siamese_gesd = {
    # Basic
    'name': 'gesd_siamese',
    'model': 'SiameseRNN',

    'char_embedding': None, #'sgns.financial.char',
    'word_embedding': None, #'sgns.financial.char',#None,

    # Model
    'embed_size': 300,
    'hidden_size': 150,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'dropout2': 0.5,
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

    # Model
    'embed_size': 200,
    'hidden_size': 200,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'pos_weight': 3.0,
    'representation': 'last', # last, avg
    'sim_fun': 'dense',# 'dense', # exp, cosine, cosine+, dense
    'loss': 'ce', # ce, cl, cl+ce
    'cl_margin': 0.1,
    'ce_alpha': 1.,
    'one_att': True,

    # Training
    'batch_size': 64,
    'max_iter': 500000,
    'patience': 10,
    'max_grad_norm': 10.,
}


siamese_light = {
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
    'dropout2': 0.5,
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
    'dropout': 0.2,
    'channel_size': 200,
    'len': 10,

    # training
    'batch_size': 64,
    'max_iter': 50000,
    'patience': 5,
    'max_grad_norm': 5.,
}

siamese_large = {
    # Basic
    'name': 'siamese_large',
    'model': 'SiameseRNN',

    'char_embedding': None, #'sgns.financial.char',
    'word_embedding': None, #'sgns.financial.char',#None,

    # Model
    'mode': 'char',
    'embed_size': 300,
    'hidden_size': 200,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.4,
    'dropout2': 0.2,
    'pos_weight': 3.5,
    'representation': 'last', # last, avg, max
    'sim_fun': 'dense', # exp, cosine, cosine+, dense
    'l1_size': 400,
    'l2_size': 200,
    'loss': 'ce', # ce, cl, cl+ce
    'ce_alpha': 1.,
    'cl_margin': 0.3,
    'dense_plus_size': 200,

    # Training
    'batch_size': 64,
    'max_iter': 500000,
    'patience': 10,
}

siamese_plus = {
    # Basic
    'name': 'siamese_plus',
    'model': 'SiameseRNN',

    'char_embedding': None, #'sgns.financial.char',
    'word_embedding': None, #'sgns.financial.char',#None,

    # Model
    'mode': 'char',
    'embed_size': 300,
    'hidden_size': 200,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'dropout2': 0.,
    'pos_weight': 3.,
    'representation': 'last', # last, avg, max
    'sim_fun': 'dense+', # exp, cosine, cosine+, dense
    'l1_size': 100,
    #'l2_size': 200,
    'loss': 'cl', # ce, cl, cl+ce
    'ce_alpha': 1.,
    'cl_margin': 0.1,
    'plus_size': 200,

    # Training
    'batch_size': 128,
    'max_iter': 500000,
    'patience': 5,
}


decatt = {
    # Basic
    'name': 'decatt',
    'model': 'DecAttSiamese',

    'char_embedding': None, #'sgns.financial.char',
    'word_embedding': None, #'sgns.financial.char',#None,

    # Model
    'mode': 'word',
    'embed_size': 300,
    'hidden_size': 200,
    'F1_out': 200,
    'F2_out': 200,
    'G1_out': 200,
    'G2_out': 200,
    'H1_out': 200,
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.4,
    #'dropout2': 0.4,
    'pos_weight': 1.,
    'representation': 'last', # last, avg, max
    'l1_size': 200,
    'loss': 'cl', # ce, cl, cl+ce
    'ce_alpha': 1.,
    'cl_margin': 0.1,

    # Training
    'batch_size': 64,
    'max_iter': 500000,
    'patience': 5,
    'max_grad_norm': 10.,
}

charword = {
    # Basic
    'name': 'charword',
    'model': 'CharWordSiamese',

    'char_embedding': None, #'sgns.financial.char',
    'word_embedding': None, #'sgns.financial.char',#None,

    # Model
    'mode': 'word',
    'char_embed_size': 300,
    'word_embed_size': 300,
    'char_hidden_size': 200,
    'word_hidden_size': 200,
    'num_layers': 2,
    'dropout': 0.5,
    'dropout2': 0.4,
    'pos_weight': 3.,
    
    'plus_size1': 200,
    'plus_size2': 200,
    'l1_out': 100,

    'loss': 'cl', # ce, cl, cl+ce
    'ce_alpha': 1.,
    'cl_margin': 0.1,

    # Training
    'batch_size': 64,
    'max_iter': 500000,
    'patience': 5,
    'max_grad_norm': 10.,
}




