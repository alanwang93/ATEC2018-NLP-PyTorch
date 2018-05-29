#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import config
import jieba
from collections import Counter
import argparse, os, pickle, json, codecs, pickle, re


class Tokenizer:

    def __init__(self, tokenizer, data_config):
        self.config = data_config
        self.root = self.config['data_root']
        self.tokenizer = tokenizer

        # build user dictionary
        if 'dict' in self.tokenizer:
            # self.create_dictionary()
            jieba.load_userdict('data/raw/dict.txt')


    def create_dictionary(self, num_vocab=5):
        """
        Count the frequency of all 2-gram and 3-gram in training set

        用法： jieba.load_userdict(file_name) # file_name 为文件类对象或自定义词典的路径
        词典格式和 dict.txt 一样，一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），
        用空格隔开，顺序不可颠倒。file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码。

        Return:
        most_freq: [(word, freq),...], list of tuple
        """
        ngrams = []
        data = open('data/raw/train.raw', 'r').readlines()
        for line in data:
            line_split = line.split('\t')
            s1 = self.tokenize(line_split[1], 'word')
            s2 = self.tokenize(line_split[2], 'word')
            for s in [s1, s2]:
                for i in range(len(s)):
                    if i+2 <= len(s):
                        ngrams.append("".join(s[i:i+2]))
                    if i+3 <= len(s):
                        ngrams.append("".join(s[i:i+3]))
        freqs = Counter(ngrams)
        most_freq = freqs.most_common(num_vocab)
        # save dictionary
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        with open(os.path.join(self.root, 'dict.txt'), 'w') as f:
            for w, freq in most_freq:
                f.write("{0} {1} \n".format(w.encode('utf-8'), freq))
        return most_freq



    def tokenize(self, sentence, tokenizer='char', del_punctuation=True, stop_words=None):
        if del_punctuation:
            pass
            #sentence = ''.join(re.findall(u'[\u4e00-\u9fff]+', sentence))
        if type(sentence) != type(u'1'):
            sentence = sentence.decode('utf8')
        if tokenizer in ['word', 'word+dict']:
            seg = jieba.cut(sentence)
            word_list = [word for word in seg]
        elif tokenizer == 'char':
            word_list = list(sentence)
        elif tokenizer == 'char+dict':
            pass
        if stop_words is not None:
            stop_words_list =  [w.strip().decode('utf8') for w in open(stop_words, 'r').readlines()]
            new_list = []
            for w in word_list:
                if w not in stop_words_list:
                    new_list.append(w)
            return new_list
        return word_list


    def tokenize_all(self, data_raw, save_to, stop_words=None, mode='train'):
        """
        Return:
            A list of dictionary: [{'s1':[w1, w2, ...], 's2': [w1, w2, ...]}]
        """
        tokenized = []
        with open(os.path.join(self.root, save_to), 'w') as f:
            for ins in data_raw:
                s1_tokenized = self.tokenize(ins['s1'], self.tokenizer, stop_words=stop_words)
                s2_tokenized = self.tokenize(ins['s2'], self.tokenizer, stop_words=stop_words)
                if mode == 'train':
                    new_line = "{0}\t{1}\t{2}\t{3}\n".format(ins['sid'], " ".join(s1_tokenized).encode('utf-8'), \
                            ' '.join(s2_tokenized).encode('utf-8'), ins['label'])
                    f.write(new_line)
                else:
                    pass 
                tokenized.append({'s1':s1_tokenized, 's2':s2_tokenized})
        return tokenized
