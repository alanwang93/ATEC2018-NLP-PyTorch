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
