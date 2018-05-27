#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import sys
import torch
from data.dataset import Dataset, simple_collate_fn
import models
from data.vocab import Vocab
from torch.utils import data
import config
import itertools, argparse, os, json
from utils import init_log, score, to_cuda
import numpy as np

model_path = 'checkpoints/siamese_default_best.pkl'
test_path = 'data/raw/test.pkl'
UNK_IDX = 0
EOS_IDX = 2

def main(inpath, outpath):
	c = getattr(config, 'siamese')
    data_config = getattr(config, 'data_config')
    c['use_cuda'] = torch.cuda.is_available()
    c['cuda_num'] = 0
    char_vocab = Vocab(data_config=data_config, type='char')
    word_vocab = Vocab(data_config=data_config, type='word')
    char_vocab.build(rebuild=False)
    word_vocab.build(rebuild=False)
    char_size = len(char_vocab)
    word_size = len(word_vocab)
    data_config['char_size'] = char_size
    data_config['word_size'] = word_size
    test_data = Dataset(test_path)
    test_size = len(test_data)
    test = data.DataLoader(test_data, batch_size=1, collate_fn=simple_collate_fn)
    model = getattr(models, c['model'])(c, data_config)
    if data_config['char_embedding'] is not None:
        char_vocab.load_vectors(char_vocab.embedding)
        model.load_vectors(char_vocab.vectors)
    if data_config['word_embedding'] is not None:
        word_vocab.load_vectors(word_vocab.embedding)
        model.load_vectors(word_vocab.vectors)

    cp = torch.load(model_path)
    model.load_state_dict(cp['state_dict'])
    threshold = cp['best_threshold']

    if c['use_cuda']:
        model = model.cuda(c['cuda_num'])
    else:
    	model = model.cpu()


	preds = []
	sids = []
    for _, test_batch in enumerate(test):
        pred, sid = model.eval().test(to_cuda(test_batch, c))
        if pred > threshold:
        	preds.append(1)
        else:
        	preds.append(0)
        sids.append(sid)
    with open(outpath, 'w') as fout:
    	for sid, pred in zip(sids, preds):
    		fout.write('{0}\t{1}\n'.format(sid, pred))




if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
