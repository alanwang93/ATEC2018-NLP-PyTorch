#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import os, json
import numpy as np
import cPickle as pickle

class KFold:

    def __init__(self, N, k=3, seed=996):
        self.seed = seed
        np.random.seed(seed)
        self.k = k
        self.N = N


        # Load k-fold indice
        datapath = 'data/processed/{0}fold_{1}.pkl'.format(k, seed)
        if os.path.exists(datapath):
            self.kfolds = pickle.load(open(datapath, 'r'))
        else:
            a = np.random.permutation(self.N)
            self.kfolds = np.array_split(a, k)
            pickle.dump(self.kfolds, open(datapath, 'w'))

    def train_test_split(self, train_ratio=0.9):
        pass

    def get_k(self):
        return self.k

    def get(self, i):
        assert i < self.k
        valid = self.kfolds[i]
        train = np.concatenate(self.kfolds[:i] + self.kfolds[i+1:])
        return (train, valid)

        



