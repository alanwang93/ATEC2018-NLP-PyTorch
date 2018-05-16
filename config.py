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
    'max_vocab': 5000,
    'embed_size': 200,
    'hidden_size': 200,
    'rnn_cell': 'BasicLSTMCell',
    'num_layers': 1,
    'bidirectional':False,
    'dropout': 0.5,

    # training
    'batch_size': 32,
    'max_iter': 50000,
}
