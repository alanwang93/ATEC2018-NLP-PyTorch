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
    data_config = getattr(config, 'data_config')
    c['use_cuda'] = not args.disable_cuda and torch.cuda.is_available()
    c['cuda_num'] = args.cuda_num

    logger = init_log(os.path.join('log/', args.config))

    # Build vocab
    char_vocab = Vocab(data_config=data_config, type='char')
    word_vocab = Vocab(data_config=data_config, type='word')
    char_vocab.build(rebuild=False)
    word_vocab.build(rebuild=False)
    char_vocab.load_vectors(char_vocab.embedding)
    word_vocab.load_vectors(word_vocab.embedding)
    char_size = len(char_vocab)
    word_size = len(word_vocab)
    data_config['char_size'] = char_size
    data_config['word_size'] = word_size

    # Build datasets
    train_data = Dataset(data_config['train'])
    valid_data = Dataset(data_config['valid'])
    valid_size = len(valid_data)
    train = data.DataLoader(train_data, batch_size=c['batch_size'], shuffle=True, collate_fn=simple_collate_fn)
    valid = data.DataLoader(valid_data, batch_size=1, collate_fn=simple_collate_fn)

    logger.info(json.dumps(data_config, indent=2))
    logger.info(json.dumps(c, indent=2))

    model = getattr(models, c['model'])(c, data_config)

    if data_config['char_embedding'] is not None:
        model.load_vectors(char_vocab.vectors)
    if data_config['word_embedding'] is not None:
        model.load_vectors(word_vocab.vectors)

    if c['use_cuda']:
	   model = model.cuda(c['cuda_num'])

    train_loss = 0.
    global_step = 0
    best_f1 = 0.
    best_epoch = 0
    best_threshold = 0.

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
                for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9]:
                    f1, acc, prec, recall = score(preds, targets, threshold=threshold)
                    logger.info("Valid at {0}, threshold {6}, F1:{1:.3f}, Acc:{2:.3f}, P:{3:.3f}, R:{4:.3f}, Loss:{5:.3f}"\
                            .format(global_step, f1, acc, prec, recall, valid_loss, threshold))
                print()
        logger.info("Epoch {0} done".format(epoch))

        # Prediction for early stopping
        preds = []
        targets = []
        f1s = {}
        for _, valid_batch in enumerate(valid):
            pred, target, _ = model.eval().evaluate(to_cuda(valid_batch,c))
            preds.append(pred)
            targets.append(target)

        for threshold in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            f1, acc, prec, recall = score(preds, targets, threshold=threshold)
            logger.info("Valid at epoch {0}, threshold {6}, F1:{1:.3f}, Acc:{2:.3f}, P:{3:.3f}, R:{4:.3f}, Loss:{5:.3f}"\
                    .format(epoch, f1, acc, prec, recall, valid_loss, threshold))
            f1s[f1] = threshold
        print()

        model_path = "checkpoints/{0}_{1}".format(c['name'], args.suffix)
        max_f1 = max(f1s.keys())
        max_threshold = f1s[max_f1]
        checkpoint = {
                    'name': c['name'],
                    'model': c['model'],
                    'data_config': json.dumps(data_config, indent=2)
                    'config': json.dumps(c, indent=2),
                    'best_f1': max_f1,
                    'best_epoch': epoch,
                    'best_threshold': max_threshold,
                    'state_dict': model.state_dict()
        }
        if args.save_all:
            torch.save(checkpoint, "{0}_{1}.pkl".format(model_path, epoch))
            fscore = open( "{0}_{1}_{2}".format(model_path, epoch, best_f1), 'w')
            fscore.write(json.dumps({"name":c['name'], "data_config": json.dumps(data_config, indent=2)
                    'config': json.dumps(c, indent=2), 'best_f1': max_f1, 'best_epoch': best_epoch,
                    'best_threshold': best_threshold}, indent=2))

        if max_f1 >= best_f1:
            best_threshold = threshold
            best_epoch = epoch
            best_f1 = max_f1
            logger.info("New best f1 at epoch {0}, best threshold {1}, best F1:{2:.3f}".format(best_epoch, best_threshold, best_f1))
            torch.save(checkpoint, "{0}_best.pkl".format(model_path, epoch))
            fscore = open( "{0}_best_{1}".format(model_path, best_f1), 'w')
            fscore.write(json.dumps({"name":c['name'], "data_config": json.dumps(data_config, indent=2)
                    'config': json.dumps(c, indent=2), 'best_f1': best_f1, 'best_epoch': best_epoch,
                    'best_threshold': best_threshold}, indent=2))
        elif epoch - best_epoch > c['patience']:
            logger.info("Early stop at epoch {0}, best threshold {1}, best F1:{2:.3f}".format(best_epoch, best_threshold, best_f1))
            return

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--cuda_num', type=int, default=2)
    parser.add_argument('--disable_cuda', dest='disable_cuda', action='store_true')
    parser.add_argument('--early_stop', dest='early_stop', action='store_true')
    parser.add_argument('--suffix', type=str, default='default')
    parser.add_argument('--save_all', dest='save_all', action='store_true', help='save model at every epoch')

    args = parser.parse_args()
    main(args)
