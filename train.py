#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.
import torch
from data.dataset import Dataset, my_collate_fn
import models
from data.vocab import Vocab
from torch.utils import data
import config
import itertools, argparse, os, json
from utils import init_log, score
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
    c['use_gpu'] = not args.disable_gpu and torch.cuda.is_available()

    logger = init_log(os.path.join('log/', args.config))

    vocab = Vocab(c['data_root'])
    vocab.build(rebuild=False)
    vocab_size = len(vocab)
    c['vocab_size'] = vocab_size

    train_data = Dataset(c['train'])
    valid_data = Dataset(c['valid'])
    valid_size = len(valid_data)
    train = data.DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=my_collate_fn)
    valid = data.DataLoader(valid_data, batch_size=1, collate_fn=my_collate_fn)

    print(json.dumps(c, indent=2))

    model = getattr(models, c['model'])(c)

    train_loss = 0
    global_step = 0
    for epoch in range(200):
        for step, train_batch in enumerate(train):
            global_step += 1
            train_loss += model.train_step(train_batch)
            if global_step % LOG_STEPS == 0:
                logger.info("Step {0}, train loss: {1}".format(global_step, train_loss/LOG_STEPS))
                train_loss = 0

            if global_step % EVAL_STEPS == 0:
                valid_losses = []
                preds = []
                targets = []
                for _, valid_batch in enumerate(valid):
                    pred, target, valid_loss = model.eval().evaluate(valid_batch)
                    preds.append(pred)
                    targets.append(target)
                    valid_losses.append(valid_loss)
                f1, acc, prec, recall = score(preds, targets)
                valid_loss = np.mean(valid_losses)

                logger.info("Valid at {0}, F1:{1:.3f}, Acc:{2:.3f}, P:{3:.3f}, R:{4:.3f}, Loss:{5:.3f}\n"\
                        .format(global_step, f1, acc, prec, recall, valid_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--disable_gpu', dest='disable_gpu', action='store_true')
    args = parser.parse_args()
    main(args)
