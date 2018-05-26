import torch
from torch.utils import data
import extractors
from vocab import Vocab
import argparse, os, pickle, json, codecs, pickle, itertools
import numpy as np

UNK_IDX = 0
EOS_IDX = 2

class Dataset(data.Dataset):
    """
    A simple dataset wrapper

    Examples:
    """
    def __init__(self, data_path, mode='train'):
        self.dict = pickle.load(open(data_path, 'r'))
        self.keys = self.dict.keys()
        self.data = []
        for index in range(len(self.dict.values()[0][1])):
            d = {}
            for k in self.keys:
                d[k] = (self.dict[k][0], self.dict[k][1][index])
            self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def simple_collate_fn(batch):
    """
    Note:
        feature level: 
        - 'c': char level
        - 'w': word level
        - 's': sentence level
        - 'p': pair level
        - 'o': others
    """
    batch_size = len(batch)
    keys = batch[0].keys()
    d = {"s1_feats": [], 's2_feats': [], 'pair_feats':[]}
    for k in ['s1_clen', 's2_clen', 's1_wlen', 's2_wlen']:
        d[k] = [b[k][1] for b in batch]
    max_wlen = np.max(d['s1_wlen'] + d['s2_wlen'])
    max_clen = np.max(d['s1_clen'] + d['s2_clen'])    
    added_sfeats = []
    s1_feats = {}
    s2_feats = {}
    pair_feats = {}

    for b in batch:
        added_sfeats = []
        for k, v in b.items():
            level, data = v
            if level == 'c':
                if not d.has_key(k):
                    d[k] = []
                    d[k+'_rvs'] = []
                d[k].append(np.pad(data, (0,max_clen - len(data)), mode='constant', constant_values=EOS_IDX))
                d[k+'_rvs'].append(np.pad(data[::-1], (0,max_clen - len(data)), mode='constant', constant_values=EOS_IDX))
            elif level == 'w':
                if not d.has_key(k):
                    d[k] = []
                    d[k+'_rvs'] = []
                d[k].append(np.pad(data, (0,max_wlen - len(data)), mode='constant', constant_values=EOS_IDX))
                d[k+'_rvs'].append(np.pad(data[::-1], (0,max_wlen - len(data)), mode='constant', constant_values=EOS_IDX))
            elif level == 's':
                if k[3:] in added_sfeats:
                    continue # the feats have already been added
                else:
                    if not s1_feats.has_key('s1_'+k[3:]):
                        s1_feats['s1_'+k[3:]] = []
                        s2_feats['s2_'+k[3:]] = []
                    s1_feats['s1_'+k[3:]].append(b['s1_'+k[3:]][1])
                    s2_feats['s2_'+k[3:]].append(b['s2_'+k[3:]][1])
                    added_sfeats.append(k[3:])
            elif level == 'p':
                if not pair_feats.has_key(k):
                    pair_feats[k] = []
                if len(data.shape) == 0:
                    data = data.reshape(1,)
                pair_feats[k].append(data)
            elif level == 'o':
                if not d.has_key(k):
                    d[k] = []
                d[k].append(data)              
    for k in s1_feats.keys():
        d['s1_feats'].append(s1_feats[k])  
        d['s2_feats'].append(s2_feats['s2_'+k[3:]]) 
    for k in pair_feats.keys():
        d['pair_feats'].append(pair_feats[k])
    for k in d.keys():
        if k == 'pair_feats':
            d[k] = np.concatenate(d[k], axis=1)
            d[k] = torch.FloatTensor(d[k])
        d[k] = torch.tensor(d[k])
    d['s1_feats'] = d['s1_feats'].transpose(0, 1)
    d['s2_feats'] = d['s2_feats'].transpose(0, 1)
    # if (d['pair_feats'].size()[0]) > 0:
        # d['pair_feats'] = d['pair_feats'].transpose(0,1)
    return d



def complex_collate_fn(batch):
    keys = batch[0].keys()
    d = {}
    for k in keys:
        d[k] = []
    for b in batch:
        for k,v in b.items():
            d[k].append(v)
    d['s1_len'] = torch.tensor(d['s1_len'])
    d['s2_len'] = torch.tensor(d['s2_len'])
    d['s1_ordered_len'], d['s1_indices'] = torch.sort(d['s1_len'], descending=True)
    d['s2_ordered_len'], d['s2_indices'] = torch.sort(d['s2_len'], descending=True)
    d['s1_rvs'] = torch.zeros_like(d['s1_indices'])
    d['s2_rvs'] = torch.zeros_like(d['s2_indices'])
    for i, v in enumerate(d['s1_indices']):
        d['s1_rvs'][v] = i
    for i, v in enumerate(d['s2_indices']):
        d['s2_rvs'][v] = i

    for k in keys:
        if 'len' in k:
            continue
        if len(d[k][0].size()) > 0:
            if 's1' in k:
                d[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(d[k][i]) for i in d['s1_indices'].tolist()],\
                        batch_first=True, padding_value=EOS_IDX)
            if 's2' in k:
                d[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(d[k][i]) for i in d['s2_indices'].tolist()],\
                        batch_first=True, padding_value=EOS_IDX)
        elif k == 'label':
            d[k] = torch.tensor(d[k])
        else:
            if 's1' in k:
                d[k] = torch.tensor(d[k])[d['s1_indices']]
            if 's2' in k:
                d[k] = torch.tensor(d[k])[d['s2_indices']]
    return d

