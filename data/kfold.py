#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import os, json
import numpy as np
import cPickle as pickle

class KFold:

    def __init__(self, N, k=3, seed=996, with_test=False, test_ratio=0.05):
        self.seed = seed
        np.random.seed(seed)
        self.k = k
        self.N = N
        self.with_test = with_test
        self.test_ratio = test_ratio
        self.test = None


        # Load k-fold indice
        datapath = 'data/processed/{0}fold_{1}.pkl'.format(k, seed)
        if os.path.exists(datapath):
            data = pickle.load(open(datapath, 'r'))
            self.kfolds = data['kfolds']
            if self.with_test:
                self.test = data['test']
        else:
            a = np.random.permutation(self.N)
            if with_test:
                self.test = a[-int(self.N*self.test_ratio)]
            self.kfolds = np.array_split(a, k[:-int(self.N*self.test_ratio)])
            pickle.dump({'kfolds': self.kfolds, 'test':test}, open(datapath, 'w'))

    def train_test_split(self, train_ratio=0.9):
        pass

    def get_k(self):
        return self.k

    def get(self, i):
        assert i < self.k
        valid = self.kfolds[i]
        train = np.concatenate(self.kfolds[:i] + self.kfolds[i+1:])
        return (train, valid, self.test)

        



