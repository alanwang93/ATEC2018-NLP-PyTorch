import torch
from torch.utils import data
import extractors
from vocab import Vocab
import argparse, os, pickle, json, codecs, pickle, itertools

UNK_IDX = 0
EOS_IDX = 2

class Dataset(data.Dataset):
    """
    A simple dataset wrapper

    Examples:
    """
    def __init__(self, data_path, use_gpu=False, mode='train'):
        self.dict = pickle.load(open(data_path, 'r'))
        self.keys = self.dict.keys()
        self.data = []
        self.use_gpu = use_gpu
        for index in range(len(self.dict.values()[0])):
            d = {}
            for k in self.keys:
                d[k] = torch.tensor(self.dict[k][index])
                if use_gpu:
                    d[k] = d[k].gpu()
            self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def my_collate_fn(batch):
    keys = batch[0].keys()
    d = {}
    for k in keys:
        d[k] = []
    for b in batch:
        for k,v in b.items():
            d[k].append(v)
    d['s1_len'] = torch.tensor(d['s1_len'])
    d['s2_len'] = torch.tensor(d['s2_len'])
    d['s1_len'], d['s1_indices'] = torch.sort(d['s1_len'], descending=True)
    d['s2_len'], d['s2_indices'] = torch.sort(d['s2_len'], descending=True)
    d['s1_rvs'] = torch.zeros_like(d['s1_indices'])
    d['s2_rvs'] = torch.zeros_like(d['s2_indices'])
    for i, v in enumerate(d['s1_indices']):
        d['s1_rvs'][v] = i
    for i, v in enumerate(d['s2_indices']):
        d['s2_rvs'][v] = i

    for k in keys:
        if k in ['s1_len', 's2_len']:
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
