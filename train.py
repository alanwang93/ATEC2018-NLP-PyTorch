#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import torch
from data.dataloader import get_dataloader
import deep_models
from data.vocab import Vocab
from torch.utils import data
import config
import itertools, argparse, os, json
from utils import init_log, score, to_cuda
import numpy as np

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

"""
Train a PyTorch Model
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
    c['name'] = args.config
    c['cuda_num'] = args.cuda_num
    c['suffix'] = args.suffix

    logger = init_log(os.path.join('log/', args.config+'_'+args.suffix+'_best'))

    # Build vocab
    char_vocab = Vocab(data_config=data_config, type='char')
    word_vocab = Vocab(data_config=data_config, type='word')
    char_vocab.build(rebuild=False)
    word_vocab.build(rebuild=False)
    char_size = len(char_vocab)
    word_size = len(word_vocab)
    data_config['char_size'] = char_size
    data_config['word_size'] = word_size

    model = getattr(deep_models, c['model'])(c, data_config)

    # Get data loader
    train, valid = get_dataloader(config=c, valid_ratio=data_config['valid_ratio'])

    logger.info(json.dumps(data_config, indent=2))
    logger.info(json.dumps(c, indent=2))


    if c['char_embedding'] is not None:
        model.load_vectors(char_vocab.vectors)
    if c['word_embedding'] is not None:
        model.load_vectors(word_vocab.vectors)

    if c['use_cuda']:
        model = model.cuda(c['cuda_num'])
    else:
        model = model.cpu()

    train_loss = 0.
    loss_size = 0
    global_step = 0
    best_f1 = 0.
    best_epoch = 0
    best_threshold = 0.
    #for i in range(3, 20):
    #    logger.info("Imp embed {0}:{1}".format(word_vocab.itos[i].encode('utf8'), sigmoid(model.imp.weight[i].item())))

    """ Training """
    for epoch in range(200):
        for step, train_batch in enumerate(train):
            batch_size = train_batch['s1_clen'].size()[0]
            loss_size += batch_size
            global_step += 1
            train_batch = to_cuda(train_batch, c)
            batch_loss = model.train(mode=True).train_step(train_batch)
            batch_loss *= batch_size
            train_loss += batch_loss
            if global_step % LOG_STEPS == 0:
                logger.info("Step {0}, train loss: {1}".format(global_step, train_loss/loss_size))
                train_loss = 0.
                loss_size = 0
        logger.info("Epoch {0} done".format(epoch))

        """ Test on validate set """
        preds = []
        targets = []
        f1s = []
        valid_losses = []
        valid_size = 0
        #for i in range(3, 20):
        #    logger.info("Imp embed {0}:{1}".format(word_vocab.itos[i].encode('utf8'), sigmoid(model.imp.weight[i].item())))
        for _, valid_batch in enumerate(valid):
            batch_size = valid_batch['s1_clen'].size()[0]
            valid_size += batch_size
            batch_pred, batch_target, batch_valid_loss = model.eval().evaluate(to_cuda(valid_batch,c))
            preds.extend(batch_pred)
            targets.extend(batch_target)
            valid_losses.append(batch_valid_loss*batch_size)
        valid_loss = np.sum(valid_losses) / valid_size
        if 'sigmoid' not in args.config:
            f1, acc, prec, recall = score(preds, targets)
            logger.info("Valid at epoch {0}, F1:{1:.3f}, Acc:{2:.3f}, P:{3:.3f}, R:{4:.3f}, Loss:{5:.3f}"\
                    .format(epoch, f1, acc, prec, recall, valid_loss))
            f1s.append((f1, 0.5))
        else:
            for threshold in np.arange(0.45, 0.85, 0.01):
                f1, acc, prec, recall = score(preds, targets, threshold=threshold)
                logger.info("Valid at epoch {0}, threshold {6}, F1:{1:.3f}, Acc:{2:.3f}, P:{3:.3f}, R:{4:.3f}, Loss:{5:.3f}"\
                        .format(epoch, f1, acc, prec, recall, valid_loss, threshold))
                f1s.append((f1, threshold))
            print()

        """ Save model """
        model_path = "checkpoints/{0}_{1}".format(c['name'], args.suffix)
        max_f1 = max_threshold = 0.
        for f1, th in f1s:
            if f1 > max_f1:
                max_f1 = f1
                max_threshold = th

        checkpoint = {
                    'name': c['name'],
                    'model': c['model'],
                    'data_config': data_config,
                    'config': c,
                    'best_f1': max_f1,
                    'best_epoch': epoch,
                    'best_threshold': max_threshold,
                    'state_dict': model.state_dict()
        }

        if args.save_all:
            torch.save(checkpoint, "{0}_{1}.pkl".format(model_path, epoch))
            fscore = open( "{0}_epoch_{1}.txt".format(model_path, epoch), 'w')
            fscore.write(json.dumps({"name":c['name'], "data_config": json.dumps(data_config),
                    'config': json.dumps(c), 'best_f1': max_f1, 'best_epoch': best_epoch,
                    'best_threshold': max_threshold}, indent=2))

        """ Save a new best model """
        if max_f1 > best_f1:
            best_threshold = max_threshold
            best_epoch = epoch
            best_f1 = max_f1
            logger.info("New best f1 at epoch {0}, best threshold {1}, best F1:{2:.3f}".format(best_epoch, best_threshold, best_f1))
            torch.save(checkpoint, "{0}_best.pkl".format(model_path, epoch))
        elif epoch - best_epoch > c['patience']:
            fscore = open( "{0}_best.txt".format(model_path), 'w')
            fscore.write(json.dumps({"name":c['name'], "data_config": json.dumps(data_config),
                    'config': json.dumps(c), 'best_f1': best_f1, 'best_epoch': best_epoch,
                    'best_threshold': best_threshold}, indent=2))
            logger.info("Early stop at epoch {0}, best threshold {1}, best F1:{2:.3f}".format(best_epoch, best_threshold, best_f1))
            return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--cuda_num', type=int, default=0)
    parser.add_argument('--disable_cuda', dest='disable_cuda', action='store_true')
    parser.add_argument('--suffix', type=str, default='default')
    parser.add_argument('--save_all', dest='save_all', action='store_true', help='save model at every epoch')

    args = parser.parse_args()
    main(args)
