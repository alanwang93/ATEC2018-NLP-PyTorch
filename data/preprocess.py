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
import argparse, os, json, codecs
import cPickle as pickle

def extract_features(data_raw, chars, words, exts):
    d = dict()
    for s, kwargs in list(exts.items()):
        E = getattr(extractors, s)
        e = E()
        d.update(e.extract(data_raw, chars, words, **kwargs))
    return d


def remove_all(l, word):
    if not hasattr(word, '__len__'):
        return [x for x in l if x != word]
    else:
        for w in word:
            l = [x for x in l if x != w]
        return l
def replace_all(sent, target, candidates):
    for c in candidates:
        sent = sent.replace(c, target)
    return sent


def clean_data(raw, data_config, mode='train'):
    """
    Clean data and return a structured one.

    Return:
        A list of dict of keys: 's1', 's2', 'label', 'sid', 'target'

    Note:
        'label' is an integer, while 'target' is a float, used to compute loss
    """
    # replace
    # TODO: mis-written words -> correct ones, traditional -> simplified
    raw  = [l.replace('***', '*').replace(' ', '') for l in raw] 
    data_raw = []
    tra2sim = json.load(open('data/raw/tra2sim.txt','r'))
    rep_word = open('data/raw/synonym_word.txt').readlines()
    rep_word = [(l[0], l[1:]) for l in [l.strip().decode('utf8').split(' ') for l in rep_word]]
    rep_char = open('data/raw/synonym_word.txt').readlines()
    rep_char = [(l[0], l[1:]) for l in [l.strip().decode('utf8').split(' ') for l in rep_char]]

    for l in raw:
        if '\xef\xbb\xbf' in l:
                l = l.replace('\xef\xbb\xbf', '')
        l = l.decode('utf8')
        # tranditional -> simplified
        for tra, sim in tra2sim.items():
            l.replace(tra, sim)
        # word level replacement
        for target, candidates in rep_word:
            l = replace_all(l, target, candidates)

        s = l.strip().split('\t')
        if mode == 'train':
            data_raw.append({'sid': int(s[0]), 's1':s[1], 's2':s[2],\
                    'label':int(s[3]), 'target':float(s[3])})
        elif mode == 'test':
            data_raw.append({'sid': int(s[0]), 's1':s[1], 's2':s[2]})
    # tokenize and remove stop words
    print("Start tokenization for preprocessing")
    tokenizer = Tokenizer(tokenizer='word+dict', data_config=data_config)
    stop_words_file = "data/raw/simple_stop_words.txt"
    tokenized = tokenizer.tokenize_all(data_raw, 'train.word.tmp', stop_words=stop_words_file)

    cleaned_data_raw = []
    # char level replacement
    for (d, tokenized) in zip(data_raw, tokenized):
        for target, candidates in rep_char:
            s1 = tokenized['s1']
            s2 = tokenized['s2']
            for c in candidates:
                for i, s in enumerate(s1):
                    if s == c:
                        s1[i] = target
                for i, s in enumerate(s2):
                    if s == c:
                        s2[i] = target
        if mode == 'train':                       
            cleaned_data_raw.append({'sid': d['sid'], 's1':"".join(s1), 's2':"".join(s2),\
                    'label':d['label'], 'target':d['target']})
        elif mode == 'test':
            cleaned_data_raw.append({'sid': d['sid'], 's1':"".join(s1), 's2':"".join(s2)})
    return cleaned_data_raw



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
    exts_test = {'WordEmbedExtractor':{},
                  'SimilarityExtractor':{},
                  'WordBoolExtractor':{}}

    # Train
    if args.mode == 'train':
        if not os.path.exists(data_config['data_root']):
            os.makedirs(data_config['data_root'])
        with open('data/raw/train.raw', 'r') as f, codecs.open( \
                os.path.join(data_config['data_root'], 'train.pkl'), 'w', encoding='utf-8') as fout:
            if args.clean:
                data_raw = f.readlines()
                data_raw  = clean_data(data_raw, data_config)
                stop_words_file = "data/raw/simple_stop_words.txt"
                char_tokenized = char_tokenizer.tokenize_all(data_raw, 'train.char', stop_words=None)
                word_tokenized = word_tokenizer.tokenize_all(data_raw, 'train.word', stop_words=stop_words_file)
                pickle.dump(data_raw, open('data/processed/train_raw.pkl', 'w'))
                pickle.dump(char_tokenized, open('data/processed/train_char_tokenized.pkl', 'w'))
                pickle.dump(word_tokenized, open('data/processed/train_word_tokenized.pkl', 'w'))
            else:
                data_raw = pickle.load(open('data/processed/train_raw.pkl', 'r'))
                char_tokenized = pickle.load(open('data/processed/train_char_tokenized.pkl', 'r'))
                word_tokenized = pickle.load(open('data/processed/train_word_tokenized.pkl', 'r'))

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
            if args.clean:
                data_raw = f.readlines()
                data_raw  = clean_data(data_raw, data_config)
                char_tokenized = char_tokenizer.tokenize_all(data_raw, 'valid.char', stop_words=None)
                word_tokenized = word_tokenizer.tokenize_all(data_raw, 'valid.word', stop_words=stop_words_file)
                pickle.dump(data_raw, open('data/processed/valid_raw.pkl', 'w'))
                pickle.dump(char_tokenized, open('data/processed/valid_char_tokenized.pkl', 'w'))
                pickle.dump(word_tokenized, open('data/processed/valid_word_tokenized.pkl', 'w'))
            else:
                data_raw = pickle.load(open('data/processed/valid_raw.pkl', 'r'))
                char_tokenized = pickle.load(open('data/processed/valid_char_tokenized.pkl', 'r'))
                word_tokenized = pickle.load(open('data/processed/valid_word_tokenized.pkl', 'r'))

            exts_valid['WordEmbedExtractor']['char_vocab'] = char_vocab
            exts_valid['WordEmbedExtractor']['word_vocab'] = word_vocab

            valid = extract_features(data_raw, char_tokenized, word_tokenized, exts_valid)
            pickle.dump(valid, fout)


    elif args.mode == 'test':
        with open(args.test_in, 'r') as fin:
            char_vocab = Vocab(data_config=data_config, type='char')
            word_vocab = Vocab(data_config=data_config, type='word')
            char_vocab.build(rebuild=False)
            word_vocab.build(rebuild=False)

            data_raw = fin.readlines()
            data_raw  = clean_data(data_raw, data_config)
            stop_words_file = "data/raw/simple_stop_words.txt"
            char_tokenized = char_tokenizer.tokenize_all(data_raw, 'test.char', stop_words=None)
            word_tokenized = word_tokenizer.tokenize_all(data_raw, 'test.word', stop_words=stop_words_file)
            exts_test['WordEmbedExtractor']['char_vocab'] = char_vocab
            exts_test['WordEmbedExtractor']['word_vocab'] = word_vocab
            valid = extract_features(data_raw, char_tokenized, word_tokenized, exts_test)
            pickle.dump(valid, open('data/processed/test.pkl', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_in', type=str, default=None)
    parser.add_argument('--clean', dest='clean', action='store_true')
    args = parser.parse_args()

    main(args)
