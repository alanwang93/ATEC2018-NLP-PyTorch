#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""

"""
import logging, os, sys, argparse
import numpy as np
import re
import pickle
import torch

np.random.seed(233)

def init_log(filename):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    file_handler = logging.FileHandler(filename, mode='a')
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger

def split_data(in_file, out_dir, train_ratio=0.95):
    with open(in_file, 'r') as f:
        lines = f.readlines()
        n = len(lines)
        indices = np.random.permutation(n)
        train = [lines[i] for i in indices[:int(n*train_ratio)]]
        valid = [lines[i] for i in indices[int(n*train_ratio):]]
        train_file = open(os.path.join(out_dir, 'train.raw'), 'w')
        valid_file = open(os.path.join(out_dir, 'valid.raw'), 'w')
        for s in train:
            train_file.write(s)
        train_file.close()
        for s in valid:
            valid_file.write(s)
        valid_file.close()


def shuffle(in_file, out_dir):
    with open(in_file, 'r') as f:
        lines = f.readlines()
        n = len(lines)
        indices = np.random.permutation(n)
        train_file = open(os.path.join(out_dir, 'train.raw'), 'w')
        for i in indices:
            train_file.write(lines[i])


def over_sample(in_file, out_dir):
    with open(in_file, 'r') as f:
        lines = f.readlines()
        pos = [l for l in lines if l.strip().split('\t')[3] == '1']
        neg = [l for l in lines if l.strip().split('\t')[3] == '0']
        n_gen = len(neg) - len(pos)
        print("Over sample {0} positive samples".format(n_gen))
        new_pos = np.random.choice(pos, n_gen, replace=True)
        out = open(os.path.join(out_dir, 'train_oversampled.raw'), 'w')
        for l in (pos+new_pos.tolist()+neg):
            out.write(l)


def score(pred, target, threshold=0.5):
    eps = 1e-6
    pred = [1 if p > threshold else 0 for p in pred]
    target = list(map(int, target))
    TP = TN = FP = FN = 0
    for p, t in zip(pred, target):
        if p == 1 and t == 1:
            TP += 1
        elif p == 1 and t == 0:
            FP += 1
        elif p == 0 and t == 0:
            TN += 1
        else:
            FN += 1
    prec = TP / float(TP+FP+eps)
    recall = TP / float(TP+FN+eps)
    acc = float(TN+TP) / float(TP+FP+TN+FN+eps)
    f1 = 2*prec*recall / float(prec+recall+eps)
    return f1, acc, prec, recall

def BCELoss(output, target, weights=None):
    eps = 1e-6
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output+eps)) + \
               weights[0] * ((1 - target) * torch.log(1 - output+eps))
    else:
        loss = target * torch.log(output+eps) + (1 - target) * torch.log(1 - output+eps)
    return torch.neg(torch.mean(loss))


def to_cuda(d, c):
    if c['use_cuda']:
        for k, v in d.items():
            d[k] = v.cuda(c['cuda_num'])
    return d


def main(args):
    if args.split:
        split_data(args.raw, args.out_dir, args.train_ratio)
    if args.oversample:
        over_sample(os.path.join(args.out_dir, 'train.raw'), args.out_dir)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=str, default='data/raw/atec_nlp_sim_train_full.csv')
    parser.add_argument('--out_dir', type=str, default='data/raw/')
    parser.add_argument('--train_ratio', type=float, default=0.95)
    parser.add_argument('--split', dest='split', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--oversample', dest='oversample', action='store_true')
    args = parser.parse_args()


    main(args)
