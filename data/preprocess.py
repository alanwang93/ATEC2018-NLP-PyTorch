#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""

"""
import extractors
from .vocab import Vocab
from .tokenizer import Tokenizer
import config
import jieba
from collections import Counter
import argparse, os, pickle, json, codecs, pickle

def extract_features(data, tokenized, exts):
    d = dict()
    for s, kwargs in list(exts.items()):
        E = getattr(extractors, s)
        e = E()
        d.update(e.extract(data, tokenized, **kwargs))
    return d



def main(args):
    c = getattr(config, args.config)
    tokenizer = Tokenizer(config=c)

    exts_train = {'WordEmbedExtractor':{'config': c}}
    exts_valid = {'WordEmbedExtractor':{'config': c}}

    # Train
    if args.mode == 'train':
        if not os.path.exists(c['data_root']):
            os.makedirs(c['data_root'])
        with open('data/raw/train.raw', 'r') as f, codecs.open( \
                os.path.join(c['data_root'], 'train.pkl'), 'w', encoding='utf-8') as fout:
            data = f.readlines()
            data = [l.replace('***', '*') for l in data]
            # tokenize
            tokenized = tokenizer.tokenize_all(data, 'train.tokenized')
            vocab = Vocab(config=c)
            vocab.build(tokenized)
            exts_train['WordEmbedExtractor']['vocab'] = vocab
            train = extract_features(data, tokenized, exts_train)
            pickle.dump(train, fout)

        with open('data/raw/valid.raw', 'r') as f, codecs.open( \
                os.path.join(c['data_root'], 'valid.pkl'), 'w', encoding='utf-8') as fout:
            data = f.readlines()
            data = [l.replace('***', '*') for l in data]
            tokenized = tokenizer.tokenize_all(data, 'valid.tokenized')
            exts_valid['WordEmbedExtractor']['vocab'] = vocab
            valid = extract_features(data, tokenized, exts_valid)
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
