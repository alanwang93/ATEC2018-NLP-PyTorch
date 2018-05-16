from torch import nn
from torch.autograd import Variable
import torch
from utils import score, BCELoss
import numpy as np

UNK_IDX = 0
EOS_IDX = 2



class SimpleRNN(nn.Module):

    def __init__(self, c):
        super(SimpleRNN, self).__init__()
        self.vocab_size = c['vocab_size']
        self.embed_size = c['embed_size']
        self.hidden_size = c['hidden_size']
        self.num_layers = c['num_layers']
        self.bidirectional = c['bidirectional']
        self.mode = None

        self.embed = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=EOS_IDX)
        self.rnn = nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(0.5)
        self.linear_in_size = self.hidden_size*2
        if self.bidirectional:
            self.linear_in_size *= 2
        self.linear = nn.Linear(self.linear_in_size, 1)
        self.sigmoid = nn.Sigmoid()
        # self.bce = nn.BCELoss()
        self.bce = BCELoss
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)

        self._init_weights()


    def _init_weights(self):
        nn.init.normal_(self.embed.weight)


    def forward(self, data):
        s1_embed = self.embed(data['s1_word'])
        s2_embed = self.embed(data['s2_word'])
        s1_embed = self.dropout(s1_embed)
        s2_embed = self.dropout(s2_embed)
        s1_packed = nn.utils.rnn.pack_padded_sequence(s1_embed, data['s1_ordered_len'], batch_first=True)
        s2_packed = nn.utils.rnn.pack_padded_sequence(s2_embed, data['s2_ordered_len'], batch_first=True)
        s1_out, s1_hidden = self.rnn(s1_packed)
        s2_out, s2_hidden = self.rnn(s2_packed)
        # recover order
        s1_out, _ = nn.utils.rnn.pad_packed_sequence(s1_out, batch_first=True)
        s2_out, _ = nn.utils.rnn.pad_packed_sequence(s2_out, batch_first=True)
        batch_size = s1_out.size()[0]
        row_idx = torch.arange(0, batch_size).long()
        s1_out = torch.squeeze(s1_out[data['s1_rvs']][row_idx, data['s1_len']-1, :], 1) # last hidden state
        s2_out = torch.squeeze(s2_out[data['s2_rvs']][row_idx, data['s2_len']-1, :], 1)
        out = self.linear(torch.cat([s1_out, s2_out], dim=1))

        return out


    def train_step(self, data):
        out = self.forward(data)
        proba = torch.squeeze(self.sigmoid(out))
        loss = self.bce(proba, data['label'], weights=[1., 3.5])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # f1, acc, prec, recall = score(proba.tolist(), data['label'].tolist())
        # print({'loss':loss.item(), 'f1':f1, 'acc':acc, 'prec':prec, 'recall':recall})
        return loss.item()

    def evaluate(self, data):
        out = self.forward(data)
        proba = torch.squeeze(self.sigmoid(out), 0)
        target =  data['label'].item()
        loss = self.bce(proba, data['label'], weights=[1., 3.5])
        return proba, target, loss.item()
