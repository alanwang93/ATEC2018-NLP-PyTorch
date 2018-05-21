#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.
import torch
from data.dataset import Dataset, simple_collate_fn
import models
from data.vocab import Vocab
from torch.utils import data
import config
import itertools, argparse, os, json
from utils import init_log, score, to_cuda
import numpy as np

"""

"""
EVAL_STEPS = 1000 # evaluate on valid set
LOG_STEPS = 200 # average train losses over LOG_STEPS steps
UNK_IDX = 0
EOS_IDX = 2

def main(args):
    assert args.config is not None
    c = getattr(config, args.config)
    c['use_cuda'] = not args.disable_cuda and torch.cuda.is_available()
    c['cuda_num'] = args.cuda_num

    logger = init_log(os.path.join('log/', args.config))

    vocab = Vocab(config=c)
    vocab.build(rebuild=False)
    vocab_size = len(vocab)
    c['vocab_size'] = vocab_size

    train_data = Dataset(c['train'])
    valid_data = Dataset(c['valid'])
    valid_size = len(valid_data)
    train = data.DataLoader(train_data, batch_size=c['batch_size'], shuffle=True, collate_fn=simple_collate_fn)
    valid = data.DataLoader(valid_data, batch_size=1, collate_fn=simple_collate_fn)

    print(json.dumps(c, indent=2))

    model = getattr(models, c['model'])(c)
    if c['use_cuda']:
	   model = model.cuda(c['cuda_num'])
    if c['embedding'] is not None:
        model.load_vectors(vocab.vectors)

    train_loss = 0
    global_step = 0
    for epoch in range(200):
        for step, train_batch in enumerate(train):
            global_step += 1
            train_loss += model.train(mode=True).train_step(to_cuda(train_batch, c))
            if global_step % LOG_STEPS == 0:
                logger.info("Step {0}, train loss: {1}".format(global_step, train_loss/LOG_STEPS))
                train_loss = 0.

            if global_step % EVAL_STEPS == 0:
                valid_losses = []
                preds = []
                targets = []
                for _, valid_batch in enumerate(valid):
                    pred, target, valid_loss = model.eval().evaluate(to_cuda(valid_batch,c))
                    preds.append(pred)
                    targets.append(target)
                    valid_losses.append(valid_loss)
                valid_loss = np.mean(valid_losses)
                for threshold in [0.5, 0.6, 0.7, 0.8]:
                    f1, acc, prec, recall = score(preds, targets, threshold=threshold)
                    logger.info("Valid at {0}, threshold {6}, F1:{1:.3f}, Acc:{2:.3f}, P:{3:.3f}, R:{4:.3f}, Loss:{5:.3f}"\
                            .format(global_step, f1, acc, prec, recall, valid_loss, threshold))
                print()
        logger.info("Epoch {0} done".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--cuda_num', type=int, default=2)
    parser.add_argument('--disable_cuda', dest='disable_cuda', action='store_true')
    args = parser.parse_args()

    main(args)
