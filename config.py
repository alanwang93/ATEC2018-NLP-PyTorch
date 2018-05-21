#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

baseline = {
    'name': 'baseline',
    'model': 'SimpleRNN',
    'data_root': 'data/processed/baseline/',
    'train': 'data/processed/baseline/train.pkl',
    'valid': 'data/processed/baseline/valid.pkl',
    'tokenizer': 'word',

    # model
    'max_vocab': 20000,
    'min_freq': 2,
    'embed_size': 200,
    'hidden_size': 64,
    'rnn_cell': 'BasicLSTMCell',
    'num_layers': 1,
    'bidirectional':True,
    'dropout': 0.4,
    'pos_weight': 3.0,

    # training
    'batch_size': 32,
    'max_iter': 50000,
}


siamese = {
    'name': 'siamese',
    'model': 'SiameseRNN',
    'data_root': 'data/processed/siamese/',
    'train': 'data/processed/siamese/train.pkl',
    'valid': 'data/processed/siamese/valid.pkl',
    'tokenizer': 'word',

    # model
    'max_vocab': 20000,
    'min_freq': 2,
    'embed_size': 200,
    'hidden_size': 120,
    'rnn_cell': 'BasicLSTMCell',
    'num_layers': 2,
    'bidirectional':True,
    'dropout': 0.5,
    'pos_weight': 3.,

    # training
    'batch_size': 16,
    'max_iter': 50000,

}


