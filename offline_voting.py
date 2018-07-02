#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import sys
import torch
from data.dataloader import get_dataloader, Dataset, simple_collate_fn
import deep_models
from data.vocab import Vocab
from torch.utils import data
import config
import itertools, argparse, os, json
from utils import init_log, score, to_cuda
import numpy as np

"""
Test on a single deep model
"""

cp_names = []
soft = False


UNK_IDX = 0
EOS_IDX = 2

def main(inpath, outpath):
    """
    Keys of `cp`:
        'name': c['name'],
        'model': c['model'],
        'data_config': data_config,
        'config': c,
        'best_f1': max_f1,
        'best_epoch': epoch,
        'best_threshold': max_threshold,
        'state_dict': model.state_dict()
    """
    all_preds = []
    # Build data
    test_data = Dataset(mode='test')
    test_size = len(test_data)
    test = data.DataLoader(test_data, batch_size=1, collate_fn=simple_collate_fn)
    for cp_name in cp_names:
        model_path = 'checkpoints/{0}'.format(cp_name)
        cp = torch.load(model_path, map_location=lambda storage, loc: storage)
        c = cp['config']
        data_config = cp['data_config']
        c['use_cuda'] = False
        # char_vocab = Vocab(data_config=data_config, type='char')
        # word_vocab = Vocab(data_config=data_config, type='word')

        model = getattr(deep_models, c['model'])(c, data_config)

        
        model.load_state_dict(cp['state_dict'])
        threshold = cp['best_threshold']
        print("threshold", threshold)

        if c['use_cuda']:
            model = model.cuda(c['cuda_num'])
        else:
            model = model.cpu()

        preds = []
        sids = []
        print("Start prediction")
        for i, test_batch in enumerate(test):
            if not soft:
                pred, sid = model.eval().test(to_cuda(test_batch, c))
                if hasattr(pred, '__len__'):
                    pred = pred[0]
                if pred > threshold:
                    preds.append(1)
                else:
                    preds.append(0)
            else:
                pred, sid = model.eval().predict_proba(to_cuda(test_batch, c))
                if hasattr(pred, '__len__'):
                    pred = pred[0]
                preds.append(pred)
            sids.append(sid[0])
        all_preds.append(preds)
    print(all_preds)
    probas = np.mean(all_preds, axis=0)
    print(probas)
    preds = [1 if p > threshold else 0 for p in probas]
    print(preds)
    with open(outpath, 'w') as fout:
        for sid, pred in zip(sids, preds):
            fout.write('{0}\t{1}\n'.format(sid, pred))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

