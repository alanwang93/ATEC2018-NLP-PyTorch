#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

baseline = {
    'model': 'SimpleRNN',
    'data_root': 'data/data1/',
    'train': 'data/data1/train.pkl',
    'valid': 'data/data1/valid.pkl',
    'tokenizer': 'word',

    # model
    'max_vocab': 20000,
    'min_freq': 2,
    'embed_size': 200,
    'hidden_size': 200,
    'rnn_cell': 'BasicLSTMCell',
    'num_layers': 2,
    'bidirectional':False,
    'dropout': 0.4,

    # training
    'batch_size': 32,
    'max_iter': 50000,
}
