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


def extract_features(data_raw, chars, words, exts):
    d = dict()
    for s, kwargs in list(exts.items()):
        E = getattr(extractors, s)
        e = E()
        d.update(e.extract(data_raw, chars, words, **kwargs))
    return d


def clean_data(raw):
    """
    Clean data and return a structured one.

    Return:
        A list of dict of keys: 's1', 's2', 'label', 'sid', 'target'

    Note:
        'label' is an integer, while 'target' is a float, used to compute loss
    """
    # replace
    # TODO: mis-written words -> correct ones, traditional -> simplified
    raw  = [l.replace('***', '*') for l in raw]
    data_raw = []
    for l in raw:
        if '\xef\xbb\xbf' in l:
                l = l.replace('\xef\xbb\xbf', '')
        s = l.strip().split('\t')
        data_raw.append({'sid': int(s[0]), 's1':s[1], 's2':s[2],\
                'label':int(s[3]), 'target':float(s[3])})
    return data_raw



def main(args):
    data_config =  getattr(config, 'data_config')
    word_tokenizer = Tokenizer(tokenizer='word+dict', data_config=data_config)
    char_tokenizer = Tokenizer(tokenizer='char', data_config=data_config)

    exts_train = {'WordEmbedExtractor':{},
                  'SimilarityExtractor':{},
                  'WordBoolExtractor':{}}
    exts_valid = {'WordEmbedExtractor':{},
                  'SimilarityExtractor':{},
                  'WordBoolExtractor':{}}

    # Train
    if args.mode == 'train':
        if not os.path.exists(data_config['data_root']):
            os.makedirs(data_config['data_root'])
        with open('data/raw/train.raw', 'r') as f, codecs.open( \
                os.path.join(data_config['data_root'], 'train.pkl'), 'w', encoding='utf-8') as fout:
            data_raw = f.readlines()
            data_raw  = clean_data(data_raw)

            stop_words_file = None #os.path.join("data/raw", "stop_words_zh.txt")
            char_tokenized = char_tokenizer.tokenize_all(data_raw, 'train.char', stop_words=None)
            word_tokenized = word_tokenizer.tokenize_all(data_raw, 'train.word', stop_words=stop_words_file)
            char_vocab = Vocab(data_config=data_config, type='char', embedding=data_config['char_embedding'])
            word_vocab = Vocab(data_config=data_config, type='word', embedding=data_config['word_embedding'])
            char_vocab.build(char_tokenized)
            word_vocab.build(word_tokenized)

            exts_train['WordEmbedExtractor']['char_vocab'] = char_vocab
            exts_train['WordEmbedExtractor']['word_vocab'] = word_vocab

            train = extract_features(data_raw, char_tokenized, word_tokenized, exts_train)

            pickle.dump(train, fout)

        with open('data/raw/valid.raw', 'r') as f, codecs.open( \
                os.path.join(data_config['data_root'], 'valid.pkl'), 'w', encoding='utf-8') as fout:
            data_raw = f.readlines()
            data_raw  = clean_data(data_raw)

            char_tokenized = char_tokenizer.tokenize_all(data_raw, 'valid.char', stop_words=None)
            word_tokenized = word_tokenizer.tokenize_all(data_raw, 'valid.word', stop_words=stop_words_file)

            exts_valid['WordEmbedExtractor']['char_vocab'] = char_vocab
            exts_valid['WordEmbedExtractor']['word_vocab'] = word_vocab

            valid = extract_features(data_raw, char_tokenized, word_tokenized, exts_valid)
            pickle.dump(valid, fout)


    elif args.mode == 'test':
        with open(args.test_in, 'r') as fin, open(args.test_out, 'w') as fout:
            data_raw = fin.readlines()
            data_raw  = clean_data(data_raw)
            # char_tokenized = char_tokenizer.tokenize_all(data_raw, 'valid.char', stop_words=None)
            # word_tokenized = word_tokenizer.tokenize_all(data_raw, 'valid.word', stop_words=stop_words_file)

            # TODO
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_in', type=str, default=None)
    parser.add_argument('--test_out', type=str, default=None)
    args = parser.parse_args()

    main(args)
