#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 漏 2018 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import extractors
from .vocab import Vocab
import config
import argparse, os, pickle, json, codecs, pickle

def extract_features(data, exts):
    d = dict()
    for s, kwargs in list(exts.items()):
        E = getattr(extractors, s)
        e = E()
        d.update(e.extract(data, **kwargs))
    return d

def main(args):
    c = getattr(config, args.config)
    exts_train = {'WordEmbedExtractor':{'config': c}}
    exts_valid = {'WordEmbedExtractor':{'config': c}}

    # Train
    if args.mode == 'train':
        with open(os.path.join(c['data_root'], 'train.raw'), 'r') as f, codecs.open( \
                os.path.join(c['data_root'], 'train.pkl'), 'w', encoding='utf-8') as fout:
            data = f.readlines()
            data = [l.replace('***', '*') for l in data]
            vocab = Vocab(c['data_root'])
            vocab.build(data, config=c)
            exts_train['WordEmbedExtractor']['vocab'] = vocab
            train = extract_features(data, exts_train)
            pickle.dump(train, fout)

        with open(os.path.join(c['data_root'], 'valid.raw'), 'r') as f, codecs.open( \
                os.path.join(c['data_root'], 'valid.pkl'), 'w', encoding='utf-8') as fout:
            data = f.readlines()
            data = [l.replace('***', '*') for l in data]
            exts_valid['WordEmbedExtractor']['vocab'] = vocab
            valid = extract_features(data, exts_valid)
            pickle.dump(valid, fout)


    elif args.mode == 'test':
        with open(args.test_in, 'r') as fin:
            # TODO
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_in', type=str, default=None)
    args = parser.parse_args()

    main(args)
