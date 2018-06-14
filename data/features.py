#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import extractors
import os, json
import numpy as np
import cPickle as pickle

"""
"""

class Features:
    """
    Class used to extract and manage features
    """

    def __init__(self):

        self.feat_names = []
        self.feat_levels = []
        self.feat_lens = []
        
        self.feat_matrix = None
        self.name2idx = None


    def extract(self, extractor_list, data, mode='train'):
        """
        Args:
            extractor_list
            data: [{'label': int,
                    'sid': int,
                    's1': str,
                    's2': str,
                    ...
                    }]
        """
        for e in extractor_list:
            feat_path = 'data/processed/{0}_{1}.npy'.format(mode, e['name'])
            # self.kwargs.append(e['kwargs'])
            Extractor = getattr(extractors, e['name'])
            extractor = Extractor()
            # self.extractors.append(extractor)
            self.feat_names.extend(extractor.feat_names)
            self.feat_levels.extend(extractor.feat_levels)
            self.feat_lens.extend(extractor.feat_lens)

            if os.path.exists(feat_path):
                feats = np.load(feat_path)
            else:
                # extract
                feats = extractor.extract(data, **e['kwargs'])
                np.save(open(feat_path, 'w'), feats)
            if self.feat_matrix is None:
                self.feat_matrix = feats
            else:
                self.feat_matrix = np.concatenate((self.feat_matrix, feats), axis=1)
        
        feat_idx = np.cumsum([0]+self.feat_lens)
        self.name2idx = dict()
        for i, name in enumerate(self.feat_names):
            self.name2idx[name] = (feat_idx[i], feat_idx[i+1])

        print("Feature matrix size: {0}".format(self.feat_matrix.shape))

    def _save(self, mode):

        f = open('data/processed/{0}_feats.json'.format(mode), 'w')
        d = {
            'feat_names': self.feat_names,
            'feat_levels': self.feat_levels,
            'feat_lens': self.feat_lens,
            'name2idx': self.name2idx
        }
        json.dump(d, f)
       
    def _load(self, mode):

        f = open('data/processed/{0}.feats'.format(mode), 'r')
        d = json.load(f)
        self.feat_names = d['feat_names']
        self.feat_lens = d['feat_lens']
        self.feat_levels = d['feat_levels']
        self.name2idx = d['name2dix']

        for name in self.feat_names:
            feat_path = 'data/processed/{0}_{1}.npy'.format(mode, name)
            feats = np.load(feat_path)
            if self.feat_matrix is None:
                self.feat_matrix = feats
            else:
                self.feat_matrix = np.concatenate((self.feat_matrix, feats), axis=1)
        print("Feature matrix size: {0}".format(self.feat_matrix.shape))



    def get_all_feats(self):
        return (self.feat_matrix, self.name2idx)

    def get_feats_by_name(self, name_list, return_dict=False):

        feat_lens = [0]
        len_dict = dict(zip(self.feat_names, self.feat_lens))
        for name in name_list:
            feat_lens.append(len_dict[name])
        feat_idx = np.cumsum(feat_lens)
        name2idx = dict()
        for i, name in enumerate(name_list):
            name2idx[name] = (feat_idx[i], feat_idx[i+1])

        if return_dict:
            d = dict()
            for i, name in enumerate(name_list):
                d[name] = self.feat_matrix[:,self.name2idx[name][0]:self.name2idx[name][1]]
            return (d, None)
        else:
            l = []
            for i, name in enumerate(name_list):
                l.append(self.feat_matrix[:,self.name2idx[name][0]:self.name2idx[name][1]])
            l = np.concatenate(l, axis=1)
            reutnr (l, name2idx)


        


