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

    def __init__(self, config):
        self.config = config
        self.root = os.path.join('data/processed/', self.config['name'])
        self.tokenizer = config['tokenizer']
        assert self.tokenizer in ['jieba', 'jieba+dict', 'word', 'word+dict']

        # build user dictionary
        if 'dict' in self.tokenizer:
            self.create_dictionary()
            jieba.load_userdict(os.path.join(self.root, 'dict.txt'))


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
            s1 = tokenize(line_split[1], 'word')
            s2 = tokenize(line_split[2], 'word')
            for s in [s1, s2]:
                for i in range(len(s)):
                    if i+2 <= len(s):
                        ngrams.append("".join(s[i:i+2]))
                    if i+3 <= len(s):
                        ngrams.append("".join(s[i:i+3]))
        freqs = Counter(ngrams)
        most_freq = freqs.most_common(num_vocab)
        # save dictionary
        with open(os.path.join(self.root, 'dict.txt'), 'w') as f:
            for w, freq in most_freq:
                f.write("{0} {1} \n".format(w.encode('utf-8'), freq))
        return most_freq

        

    def tokenize(self, sentence, tokenizer='jieba', del_punctuation=True):
        if del_punctuation:
            pass
            #sentence = ''.join(re.findall(u'[\u4e00-\u9fff]+', sentence))

        if tokenizer in ['jieba', 'jieba+dict']:
            seg = jieba.cut(sentence)
            word_list = [word for word in seg]
        elif tokenizer == 'word':
            word_list = list(sentence.decode('utf-8'))
        elif tokenized == 'word+dict':
            pass
        return word_list


    def tokenize_all(self, data, filename):
        tokenized = []
        s1_token = []
        s2_token = []
        with open(os.path.join(self.root, filename), 'w') as f:
            for line in data:
                line_split = line.split('\t')
                s1 = line_split[1]
                s2 = line_split[2]
                s1_tokenized = self.tokenize(s1, self.tokenizer)
                s2_tokenized = self.tokenize(s2, self.tokenizer)
                s1_token.append(s1_tokenized)
                s2_token.append(s2_tokenized)
                new_line = "{0}\t{1}\t{2}\t{3}\n".format(line_split[0], " ".join(s1_tokenized).encode('utf-8'), \
                    ' '.join(s2_tokenized).encode('utf-8'), line_split[3])
                f.write(new_line)
                tokenized.append(new_line)
        return tokenized

