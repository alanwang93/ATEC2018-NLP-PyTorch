#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""

"""
from collections import Counter, defaultdict
import jieba
import pickle, os, itertools

UNK_IDX = 0
EOS_IDX = 2

def create_dictionary(root, data, num_vocab=500):
    """
    Count the frequency of all 2-gram and 3-gram in training set

    用法： jieba.load_userdict(file_name) # file_name 为文件类对象或自定义词典的路径
    词典格式和 dict.txt 一样，一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），
    用空格隔开，顺序不可颠倒。file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码。

    Return:
        most_freq: [(word, freq),...], list of tuple
    """
    ngrams = []
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
    freqs = Counter(allwords)
    most_freq = freqs.most_common(num_vocab)
    # save dictionary
    with open(os.path.join(root, 'dict.txt', 'w')) as f:
        for w, freq in most_freq:
            f.write("{0} {1} ".format(w, freq))
    return most_freq





def tokenize(sentence, tokenizer='jieba', del_punctuation=False):
    if tokenizer == 'jieba':
        if del_punctuation:
            sentence = ''.join(re.findall(u'[\u4e00-\u9fff]+', sentence))
        seg = jieba.cut(sentence)
        word_list = [word for word in seg]
    elif tokenizer == 'word':
        word_list = list(sentence.decode('utf-8'))
    return word_list

def tokenize_all(data, tokenizer='jieba'):
    d = dict()
    s1_token = []
    s2_token = []
    for line in data:
        line_split = line.split('\t')
        s1 = line_split[1]
        s2 = line_split[2]
        s1_token.append(tokenize(s1, tokenizer))
        s2_token.append(tokenize(s2, tokenizer))
    d['s1_token'] = s1_token
    d['s2_token'] = s2_token
    return d


def unk_idx():
    return 0

class Vocab:

    """
    Vocabulary class
    """
    def __init__(self, root):
        """
        Args:
            data: list of samples (s1 s2 label)
        """
        self.root = root
        self.unk_token = 'UNK'
        self.sos_token = 'SOS'
        self.eos_token = 'EOS'
        self.tokens = self.freqs = self.itos = self.stoi = self.vectors = None
        self.keys = ['tokens','freqs','itos','stoi','root','unk_token','sos_token','eos_token','vectors']

    def build(self, data=None, min_freq=1, max_size=None, rebuild=True, config=None):
        if not rebuild and os.path.exists(os.path.join(self.root, 'vocab.pkl')):
            print("Loading vocab")
            self._load()
        elif data is not None:
            if config['use_dictionary'] and config['tokenizer'] == 'jieba':
                create_dictionary(self.root, data, num_vocab=500):
                jieba.load_userdict(os.path.join(self.root, 'dict.txt'))

            # Build vocab
            self.tokens = tokenize_all(data, tokenizer=config['tokenizer'])
            allwords = list(itertools.chain.from_iterable(self.tokens['s1_token'] + self.tokens['s2_token']))
            self.freqs = Counter(allwords)

            # Reference: torchtext
            min_freq = max(min_freq, 1)
            specials = [self.unk_token, self.sos_token, self.eos_token]
            self.itos = list(specials)
            max_size = None if max_size is None else max_size + len(self.itos)
            words_and_frequencies = list(self.freqs.items())
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True) # sort by order

            for word, freq in words_and_frequencies:
                if freq < min_freq or len(self.itos) == max_size:
                    break
                self.itos.append(word)
            self.stoi = defaultdict(unk_idx)
            self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
            self._dump()

    def __len__(self):
        return len(self.itos)

    def _dump(self):
        d = dict()
        for k in self.keys:
            d[k] = getattr(self, k)
        with open(os.path.join(self.root, 'vocab.pkl'), 'w') as f:
            pickle.dump(d, f)

    def _load(self):
        with open(os.path.join(self.root, 'vocab.pkl'), 'r') as f:
            d = pickle.load(f)
        for k in self.keys:
            setattr(self, k, d[k])

    def toi(self, l):
        if hasattr(l, '__len__'):
            return [self.stoi[w] for w in l]
        else:
            return self.stoi[l]

    def tos(self, l):
        if hasattr(l, '__len__'):
            return [self.itos[w] for w in l]
        else:
            return self.itos[l]

    def load_vectors(self):
        pass
