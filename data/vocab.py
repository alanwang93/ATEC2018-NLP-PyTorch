#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""

"""
from collections import Counter, defaultdict
import jieba
import os, itertools
import numpy as np
import cPickle as pickle
from gensim.models.word2vec import Word2Vec

UNK_IDX = 0
EOS_IDX = 2


def unk_idx():
    return 0

class Vocab:

    """
    Vocabulary class
    """
    def __init__(self, data_config, type, embedding=None):
        """
        Args:
            data: list of samples (s1 s2 label)
        """
        self.data_config = data_config
        self.type = type
        self.embedding = embedding
        self.root = self.data_config['data_root']
        self.unk_token = 'UNK'
        self.sos_token = 'SOS'
        self.eos_token = 'EOS'
        self.tokens = self.freqs = self.itos = self.stoi = self.vectors = None
        self.keys = ['tokens','freqs','itos','stoi','root','unk_token','sos_token','eos_token','vectors', 'embedding', 'data_config', 'type']

    def build(self, data=None, rebuild=True):
        if not rebuild and os.path.exists(os.path.join(self.root, 'vocab_{0}.pkl'.format(self.type))):
            print("Loading {0} vocab".format(self.type))
            self._load()
        elif data is not None:
            self.data_config['max_vocab'] = self.data_config['max_char'] if self.type == 'char' else self.data_config['max_word']

            # Build vocab
            self.tokens = {"s1":[], "s2":[]}
            for ins in data:
                self.tokens['s1'].append(ins['s1_'+self.type])
                self.tokens['s2'].append(ins['s2_'+self.type])
            allwords = list(itertools.chain.from_iterable(self.tokens['s1'] + self.tokens['s2']))
            self.freqs = Counter(allwords)

            # Reference: torchtext
            min_freq = max(self.data_config['min_freq'], 1)
            specials = [self.unk_token, self.sos_token, self.eos_token]
            self.itos = list(specials)
            max_vocab = None if self.data_config['max_vocab'] is None else self.data_config['max_vocab'] + len(self.itos)
            words_and_frequencies = list(self.freqs.items())
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True) # sort by order

            with open(os.path.join(self.root, 'freqs_{0}.txt'.format(self.type)), 'w') as f:
                for word, freq in words_and_frequencies:
                    if not (freq < min_freq or len(self.itos) == max_vocab):
                        self.itos.append(word)
                    f.write("{0} {1}\n".format(word, freq))
            self.stoi = defaultdict(unk_idx)
            self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

            self._dump()

    def __len__(self):
        return len(self.itos)

    def _dump(self):
        d = dict()
        for k in self.keys:
            d[k] = getattr(self, k)
        with open(os.path.join(self.root, 'vocab_{0}.pkl'.format(self.type)), 'w') as f:
            pickle.dump(d, f)

    def _load(self):
        with open(os.path.join(self.root, 'vocab_{0}.pkl'.format(self.type)), 'r') as f:
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

    def load_vectors(self, filename):
        print("Load vectors from pretrained embeddings {0}".format(filename))
        num_vocab = embed_size = 0
        if filename not in ['char_word2vec', 'word_word2vec']:
            with open(os.path.join('data/embeddings', filename), 'r') as f:
                line0 = list(map(int, f.readline().strip().split(' ')))
                num_vocab, embed_size = line0[0], line0[1]
                self.vectors = np.random.randn(len(self.itos), embed_size)
                self.vectors[EOS_IDX, :] = 0.
                for i in range(num_vocab):
                    line = f.readline()
                    s = line.strip().split(' ')
                    word = s[0]
                    vec = np.asarray(list(map(float, s[1:])))
                    if word in self.stoi:
                        self.vectors[self.stoi[word]] = vec
        else:
            # customized embedding
            w2v = Word2Vec.load(os.path.join('data/processed', filename))
            embed_size = self.data_config['embed_size']
            self.vectors = np.random.randn(len(self.itos), embed_size)
            self.vectors[EOS_IDX, :] = 0.
            for word in self.stoi.keys():
                if word in w2v.wv:
                    self.vectors[self.stoi[word]] = w2v.wv[word]



    def select_embeddings(filename, save_to, num_freq=10000):
        """
        Args:
            filename: filename of embeddings
            save_to: path to save selected embeddings
            num_freq: number of most frequent words to be saved
        """
        num_vocab = embed_size = 0
        with open(os.path.join('data/embeddings', filename), 'r') as f:
            line0 = list(map(int, lines[0].strip().split(' ')))
            num_vocab, embed_size = line0[0], line0[1]
    
    
