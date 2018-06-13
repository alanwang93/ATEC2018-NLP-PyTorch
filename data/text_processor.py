#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import json, os
import numpy as np
from .tokenizer import Tokenizer
import config
import jieba


def replace_all(sent, target, candidates):
    for c in candidates:
        sent = sent.replace(c, target)
    return sent

def remove_all(l, words):
    if not hasattr(words, '__len__'):
        return [x for x in l if x != words]
    else:
        for w in words:
            l = [x for x in l if x != w]
        return l


class TextProcessor:

    """
    Clean and tokenize data, return a list of dictionaries

    Return:
        data: a list of samples, each one is a dictionary
            A sample has following keyss:
            - label
            - sid
            - s1, s2
            - s1_word, s2_word
            - s1_char, s2_char
            - s1_unique, s2_unique
    """

    def __init__(self, data_config):
        self.keys = ['label', 'sid', 's1', 's2', 's1_word', 's2_word', 's1_char', 's2_char', 's1_unique', 's2_unique']
        self.data_config = data_config
        self.word_tokenizer = Tokenizer(tokenizer='word+dict', data_config=data_config)
        self.char_tokenizer = Tokenizer(tokenizer='char', data_config=data_config)

        self.tra2sim_dict = json.load(open('data/raw/tra2sim.txt','r'))
        replace_words = open('data/raw/synonym_word.txt').readlines()
        replace_chars = open('data/raw/synonym_char.txt').readlines()
        self.replace_words_list = [(l[0], l[1:]) for l in [l.strip().decode('utf8').split(' ') for l in replace_words]]
        self.replace_words_list.append((u'蚂蚁', u''))
        self.replace_chars_list = [(l[0], l[1:]) for l in [l.strip().decode('utf8').split(' ') for l in replace_chars]]
        self.stop_words_list =  [w.strip().decode('utf8') for w in open("data/raw/simple_stop_words.txt", 'r').readlines()]
    
    
    
    
    def process(self, data_lines, mode='train', rebuild=False):

        filepath = "data/processed/{0}.tokenized".format(mode)

        if not os.path.exists(self.data_config['data_root']):
            os.makedirs(self.data_config['data_root'])

        if os.path.exists(filepath) and not rebuild:
            # Load
            data = []
            print("Loading from {0}".format(filepath))
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    ins = {}
                    splits = line.strip('\n').split('\t')
                    for i, k in enumerate(self.keys):
                        if mode == 'test' and i == 0:
                            continue
                        if i < 2:
                            ins[k] = int(splits[i])
                        else:
                            ins[k] = splits[i].split(' ')
                    data.append(ins)

        else:
            data = []
            for line in data_lines:
                ins = {}
                if '\xef\xbb\xbf' in line:
                    line = line.replace('\xef\xbb\xbf', '')
                line = line.decode('utf8')
                line = self.tra2sim(line)
                line = self.replace_words(line)

                splits = line.strip().split('\t')
                s1 = splits[1]
                s2 = splits[2]
                ins['sid'] = int(splits[0])
                if mode == 'train':
                    ins['label'] = int(splits[3])
                s1 = self.word_tokenizer.tokenize(splits[1], tokenizer='word+dict', stop_words=None)
                s2 = self.word_tokenizer.tokenize(splits[2], tokenizer='word+dict', stop_words=None)
                s1 = self.replace_chars(s1)
                s2 = self.replace_chars(s2)

                # Tokenization
                ins['s1_word'] = self.word_tokenizer.tokenize("".join(s1), tokenizer='word+dict', stop_words=self.stop_words_list)
                ins['s2_word'] = self.word_tokenizer.tokenize("".join(s2), tokenizer='word+dict', stop_words=self.stop_words_list)
                ins['s1'] = "".join(ins['s1_word'])
                ins['s2'] = "".join(ins['s2_word'])
                ins['s1_char'] = self.char_tokenizer.tokenize(ins['s1'], tokenizer='char')
                ins['s2_char'] = self.char_tokenizer.tokenize(ins['s2'], tokenizer='char')
                ins['s1_unique'], ins['s2_unique'] = self.remove_duplicates(ins['s1_word'], ins['s2_word'])
                
                data.append(ins)
            # Save
            with open(filepath, 'w') as f:
                for ins in data:
                    s = ''
                    for i, k in enumerate(self.keys):
                        if mode == 'test' and i == 0:
                            continue
                        if i < 2:
                            s += str(ins[k]) + '\t'
                        elif i < len(self.keys)-1:
                            s += ' '.join(ins[k]) + '\t'
                        else:
                            s += ' '.join(ins[k]) + '\n'
                    f.write(s.encode('utf8'))
            print("Save to {0}".format(filepath))

        return data
            
        
    
    def tra2sim(self, line):
        for tra, sim in self.tra2sim_dict.items():
            line = line.replace(tra, sim)
        return line

    def tokenize(self, s1, s2):
        pass

    def remove_stopwords(self, s1, s2):
        pass

    def replace_words(self, line):
        for target, candidates in self.replace_words_list:
            line = replace_all(line, target, candidates)
        return line

    def replace_chars(self, word_tokenized):
        for target, candidates in self.replace_chars_list:
            for c in candidates:
                for i, w in enumerate(word_tokenized):
                    if w == c:
                        word_tokenized[i] = target
        return word_tokenized

    def remove_duplicates(self, s1_word, s2_word):
        s1_unique = [w for w in s1_word if w not in s2_word]
        s2_unique = [w for w in s2_word if w not in s1_word]
        return (s1_unique, s2_unique)

