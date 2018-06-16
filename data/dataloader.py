#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import cPickle as pickle
from .features import Features
import config


UNK_IDX = 0
EOS_IDX = 2



class Dataset(torch.utils.data.Dataset):
    """
    A simple dataset wrapper

    Examples:
    """
    def __init__(self, mode='train'):
        self.feats = Features()
        self.feats._load(mode)
        self.feat_names = ['label', 'sid', 's1_word', 's2_word', 's1_word_rvs', 's2_word_rvs',\
                's1_char', 's2_char', 's1_char_rvs', 's2_char_rvs', 's1_wlen', 's2_wlen', 's1_clen', 's2_clen']
        
        self.dict, _ = self.feats.get_feats_by_name(self.feat_names, return_dict=True)
        self.data = []
        for index in range(self.dict.values()[0].shape[0]):
            d = {}
            for k in self.feat_names:
                d[k] = self.dict[k][index].astype(int)
            self.data.append(d)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def simple_collate_fn(batch):
    batch_size = len(batch)
    feat_names = ['label', 'sid', 's1_word', 's2_word', 's1_word_rvs', 's2_word_rvs',\
                's1_char', 's2_char', 's1_char_rvs', 's2_char_rvs', 's1_wlen', 's2_wlen', 's1_clen', 's2_clen']
    d = dict()
    for k in feat_names:
        d[k] = []
    for k in feat_names:
        for b in batch:
            d[k].append(b[k])
        d[k] = np.vstack(d[k])
    s1_max_wlen = np.max(d['s1_wlen'])
    s1_max_clen = np.max(d['s1_clen'])
    s2_max_wlen = np.max(d['s2_wlen'])
    s2_max_clen = np.max(d['s2_clen'])
    d['s1_word'] = d['s1_word'][:,:s1_max_wlen]
    d['s2_word'] = d['s2_word'][:,:s2_max_wlen]
    d['s1_word_rvs'] = d['s1_word_rvs'][:,:s1_max_wlen]
    d['s2_word_rvs'] = d['s2_word_rvs'][:,:s2_max_wlen]
    d['s1_char'] = d['s1_char'][:,:s1_max_clen]
    d['s2_char'] = d['s2_char'][:,:s2_max_clen]
    d['s1_char_rvs'] = d['s1_char_rvs'][:,:s1_max_clen]
    d['s2_char_rvs'] = d['s2_char_rvs'][:,:s2_max_clen]

    for k in feat_names:
        d[k] = torch.tensor(d[k])
        if k in ['label', 'sid', 's1_wlen', 's2_wlen', 's1_clen', 's2_clen']:
            d[k] = d[k].squeeze(1)

    return d


# def simple_collate_fn(batch):
#     """
#     Note:
#         feature level: 
#         - 'c': char level
#         - 'w': word level
#         - 's': sentence level
#         - 'p': pair level
#         - 'o': others
#     """
#     batch_size = len(batch)
#     keys = batch[0].keys()
#     d = {"s1_feats": [], 's2_feats': [], 'pair_feats':[]}
#     for k in ['s1_clen', 's2_clen', 's1_wlen', 's2_wlen']:
#         d[k] = [b[k][1] for b in batch]
#     max_wlen = np.max(d['s1_wlen'] + d['s2_wlen'])
#     max_clen = np.max(d['s1_clen'] + d['s2_clen'])
#     added_sfeats = []
#     s1_feats = {}
#     s2_feats = {}
#     pair_feats = {}

#     for b in batch:
#         added_sfeats = []
#         for k, v in b.items():
#             level, data = v
#             if level == 'c':
#                 if not d.has_key(k):
#                     d[k] = []
#                     d[k+'_rvs'] = []
#                 d[k].append(np.pad(data, (0,max_clen - len(data)), mode='constant', constant_values=EOS_IDX))
#                 d[k+'_rvs'].append(np.pad(data[::-1], (0,max_clen - len(data)), mode='constant', constant_values=EOS_IDX))
#             elif level == 'w':
#                 if not d.has_key(k):
#                     d[k] = []
#                     d[k+'_rvs'] = []
#                 d[k].append(np.pad(data, (0,max_wlen - len(data)), mode='constant', constant_values=EOS_IDX))
#                 d[k+'_rvs'].append(np.pad(data[::-1], (0,max_wlen - len(data)), mode='constant', constant_values=EOS_IDX))
#             elif level == 's':
#                 if k[3:] in added_sfeats:
#                     continue # the feats have already been added
#                 else:
#                     if not s1_feats.has_key('s1_'+k[3:]):
#                         s1_feats['s1_'+k[3:]] = []
#                         s2_feats['s2_'+k[3:]] = []
#                     s1_feats['s1_'+k[3:]].append(b['s1_'+k[3:]][1])
#                     s2_feats['s2_'+k[3:]].append(b['s2_'+k[3:]][1])
#                     added_sfeats.append(k[3:])
#             elif level == 'p':
#                 if not pair_feats.has_key(k):
#                     pair_feats[k] = []
#                 if len(data.shape) == 0:
#                     data = data.reshape(1,)
#                 pair_feats[k].append(data)
#             elif level == 'o':
#                 if not d.has_key(k):
#                     d[k] = []
#                 d[k].append(data)              
#     for k in s1_feats.keys():
#         d['s1_feats'].append(np.asarray(s1_feats[k]).reshape(batch_size, -1))
#         d['s2_feats'].append(np.asarray(s2_feats['s2_'+k[3:]]).reshape(batch_size, -1)) 
#     for k in pair_feats.keys():
#         d['pair_feats'].append(pair_feats[k])

#     for k in d.keys():
#         if k == 'pair_feats':
#             d[k] = np.concatenate(d[k], axis=1)
#             d[k] = torch.FloatTensor(d[k])
#         elif '_word' in k or '_char' in k: # embed
#             d[k] = torch.LongTensor(d[k])
#         elif k in ['s1_feats', 's2_feats' ]:
#             d[k] = np.hstack(d[k])
#             d[k] = torch.tensor(d[k])
#         else:
#             d[k] = torch.tensor(d[k])
#     return d


def get_dataloader(config, valid_ratio=0.1, shuffle=True):
    """
    Only used during training
    """
    data = Dataset(mode='train')
    num_train = len(data)
    indices = list(range(num_train))
    split = int(np.floor(valid_ratio * num_train))
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    # TODO: use KFold
    np.save(open('data/processed/train_idx.npy', 'w'), train_idx)
    np.save(open('data/processed/valid_idx.npy', 'w'), valid_idx)
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], sampler=train_sampler,\
            collate_fn=simple_collate_fn, num_workers=5)
    valid_loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], sampler=valid_sampler,\
            collate_fn=simple_collate_fn, num_workers=5)

    return (train_loader, valid_loader)
