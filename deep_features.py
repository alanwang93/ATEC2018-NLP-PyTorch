#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
Use pretrained deep models to extract features
"""

import argparse
import cPickle as pickle
import torch
import deep_models
import numpy as np
from data.vocab import Vocab
from data.dataloader import Dataset, simple_collate_fn
import config
import itertools, argparse, os, json
from utils import init_log, score, to_cuda


cp_names = ['siamese_default_best']

def main(args):
    if args.mode == 'train':
        cp = torch.load("checkpoints/{0}.pkl".format(cp_names[0]), map_location=lambda storage, loc: storage)
        data_config = cp['data_config']
        char_vocab = Vocab(data_config=data_config, type='char')
        word_vocab = Vocab(data_config=data_config, type='word')

        char_vocab.build(rebuild=False)
        word_vocab.build(rebuild=False)
        char_size = len(char_vocab)
        word_size = len(word_vocab)
        data_config['char_size'] = char_size
        data_config['word_size'] = word_size
        # Build data
        data = Dataset("data/processed/train.pkl")
        train_size = len(data)
        train = torch.utils.data.DataLoader(data, batch_size=256, collate_fn=simple_collate_fn)
        for name in cp_names:
            print("Extracting features from {0}".format(name))
            cp = torch.load("checkpoints/{0}.pkl".format(name), map_location=lambda storage, loc: storage)
            c = cp['config']
            if not torch.cuda.is_available():
                c['use_cuda'] = False

            model = getattr(deep_models, c['model'])(c, data_config)
            model.load_state_dict(cp['state_dict'])
            threshold = cp['best_threshold']

            if c['use_cuda']:
                model = model.cuda(c['cuda_num'])
            else:
                model = model.cpu()
            features = []
            for i, batch in enumerate(train):
                out = model.eval()(to_cuda(batch, c))
                features.append(out)
            features = np.concatenate(features)
            print(features.shape)

            with open("data/processed/{0}.features".format(name), 'w') as fout:
                pickle.dump(features)

    elif args.mode == 'test':
        for name in cp_names:
            cp = torch.load("checkpoints/{0}.pkl".format(name), map_location=lambda storage, loc: storage)
            c = cp['config']
            data_config = cp['data_config']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    main(args)

