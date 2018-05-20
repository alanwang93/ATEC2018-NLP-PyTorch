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
    'use_dictionary': True,

    # model
    'max_vocab': 5000,
    'embed_size': 200,
    'hidden_size': 150,
    'rnn_cell': 'BasicLSTMCell',
    'num_layers': 2,
    'bidirectional':False,
    'dropout': 0.2,

    # training
    'batch_size': 16,
    'max_iter': 50000,
}
