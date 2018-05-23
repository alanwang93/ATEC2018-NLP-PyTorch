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
    'tokenizer': 'char',

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
    'tokenizer': 'char',
    'embedding': 'sgns.weibo.word',

    # model
    'max_vocab': 20000,
    'min_freq': 2,
    'embed_size': 300,
    'hidden_size': 200,
    'rnn_cell': 'BasicLSTMCell',
    'num_layers': 1,
    'bidirectional':True,
    'dropout': 0.3,
    'pos_weight': 2.,

    # training
    'batch_size': 16,
    'max_iter': 50000,

}


