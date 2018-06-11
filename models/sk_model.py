#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.
import sklearn
import numpy as np
from utils import score

class SKModel:
    def __init__(self, config, data_config=None):
        self.config = config
        self.data_config = data_config
        m = getattr(sklearn, config['module'])
        self.clf = getattr(m, config['clf'])(**config['kwargs'])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """ Return the probability of class 1 """
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        pred = self.predict_proba(X)
        pred = np.asarray([1 if p > threshold else 0 for p in pred])
        return pred

    def score(self, X, y, threshold=0.5):
        proba = self.predict_proba(X)
        return score(proba, y, threshold=threshold)

    def _load(self, cp_name):
        pass


