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
    # model
    'max_vocab': 10000000,
    'embed_size': 100,
    'hidden_size': 100,
    'rnn_cell': 'BasicLSTMCell',
    'num_layers': 2,


    # training
    'batch_size': 32,
    'max_iter': 50000,
}
