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
        for index in range(len(self.dict.values()[0])):
            d = {}
            for k in self.keys:
                d[k] = self.dict[k][index]
            self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



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

def simple_collate_fn(batch):
    batch_size = len(batch)
    keys = batch[0].keys()
    d = {}
    for k in keys:
        d[k] = []
    for b in batch:
        for k,v in b.items():
            d[k].append(v)
    max_len = np.max(d['s1_len']+d['s2_len'])
    for k in keys:
        if hasattr(d[k][0], '__len__'):
            d[k+'_rvs'] = torch.tensor([np.pad(e[::-1], (0,max_len - len(e)), mode='constant', constant_values=EOS_IDX) for e in d[k]])
            d[k] = torch.tensor([np.pad(e, (0,max_len - len(e)), mode='constant', constant_values=EOS_IDX) for e in d[k]])
        else:
            d[k] = torch.tensor(d[k])
    return d
