from torch import nn
from torch.autograd import Variable
import torch
from utils import score, BCELoss
import numpy as np
import torch.nn.functional as F

UNK_IDX = 0
EOS_IDX = 2


class DecAttSiamese(nn.Module):
    """ Decomposible attention """

    def __init__(self,  config, data_config):
        super(DecAttSiamese, self).__init__()
        self.mode = config['mode']
        self.l = self.mode[0] + 'len'
        self.vocab_size = data_config[self.mode+'_size']
        self.embed_size = config['embed_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.pos_weight = config['pos_weight']
        self.config = config
        self.data_config = data_config

        self.embed = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=EOS_IDX)

        self.rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.)
        self.rnn_rvs = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.)

        self.dropout = nn.Dropout(config['dropout'])
        #self.dropout2 = nn.Dropout(config['dropout2'])

        self.lstm_size = self.hidden_size
        if self.bidirectional:
           self.lstm_size *= 2

        # Decomposible Attention
        # Attend
        self.F1 = nn.Linear(self.embed_size, config['F1_out'], bias=True)
        self.bn_F1 = nn.BatchNorm1d(self.embed_size)
        self.F2 = nn.Linear(config['F1_out'], config['F2_out'], bias=True)
        self.bn_F2 = nn.BatchNorm1d(config['F1_out'])
        # Compare
        self.G1 = nn.Linear(self.lstm_size*2, config['G1_out'], bias=True)
        self.bn_G1 = nn.BatchNorm1d(self.lstm_size*2)
        self.G2 = nn.Linear(config['G1_out'], config['G2_out'], bias=True)
        self.bn_G2 = nn.BatchNorm1d(config['G1_out'])
        # Aggregate => sentence pair level representation
        self.H1 = nn.Linear(config['G2_out']*2, config['H1_out'], bias=True)
        self.bn_H1 = nn.BatchNorm1d(config['G1_out']*2)

        self.l1_size = config['l1_size']

        self.linear = nn.Linear(config['H1_out'] + 7 +124 + 200, self.l1_size)
        self.bn_feats = nn.BatchNorm1d(config['H1_out'] + 7 +124 + 200)
        self.bn_l1 = nn.BatchNorm1d(self.l1_size)
        self.linear2 = nn.Linear(self.l1_size, 2)

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1., config['pos_weight']]))
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)

        self._init_weights()



    def _init_weights(self):
        # nn.init.normal_(self.embed.weight)
        init_fun = nn.init.orthogonal_
        for i in range(self.num_layers):
            for j in range(4):
                init_fun(getattr(self.rnn, 'weight_ih_l{0}'.format(i))[j*self.hidden_size:(j+1)*self.hidden_size])
                init_fun(getattr(self.rnn, 'weight_hh_l{0}'.format(i))[j*self.hidden_size:(j+1)*self.hidden_size])
                if self.bidirectional:
                    init_fun(getattr(self.rnn_rvs, 'weight_ih_l{0}'.format(i))[j*self.hidden_size:(j+1)*self.hidden_size])
                    init_fun(getattr(self.rnn_rvs, 'weight_hh_l{0}'.format(i))[j*self.hidden_size:(j+1)*self.hidden_size])

            getattr(self.rnn, 'bias_ih_l{0}'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)
            getattr(self.rnn, 'bias_hh_l{0}'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)
            if self.bidirectional:
                getattr(self.rnn_rvs, 'bias_ih_l{0}'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)
                getattr(self.rnn_rvs, 'bias_hh_l{0}'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)


    def forward(self, data):
        batch_size, seq_len = data['s1_'+self.mode].size()
        row_idx = torch.arange(0, batch_size).long()
        s1_embed = self.embed(data['s1_'+self.mode])
        s2_embed = self.embed(data['s2_'+self.mode])
        s1_embed = self.dropout(s1_embed)
        s2_embed = self.dropout(s2_embed)

        # Non-packed, using `simple_collate_fn`
        s1_out, s1_hidden = self.rnn(s1_embed)
        s2_out, s2_hidden = self.rnn(s2_embed)

        if self.bidirectional:
            s1_embed_rvs = self.embed(data['s1_'+self.mode+'_rvs'])
            s2_embed_rvs = self.embed(data['s2_'+self.mode+'_rvs'])
            s1_embed_rvs = self.dropout(s1_embed_rvs)
            s2_embed_rvs = self.dropout(s2_embed_rvs)
            s1_out_rvs, _ = self.rnn_rvs(s1_embed_rvs)
            s2_out_rvs, _ = self.rnn_rvs(s2_embed_rvs)
            s1_out = torch.cat((s1_out, s1_out_rvs), dim=2)
            s2_out = torch.cat((s2_out, s2_out_rvs), dim=2)

        s1_vec = s1_embed.view(batch_size*seq_len, -1)
        s2_vec = s2_embed.view(batch_size*seq_len, -1)

        # Attend
        #s1_vec = self.F1(self.relu(self.bn_F1(s1_vec)))
        #s2_vec = self.F1(self.relu(self.bn_F1(s2_vec)))
        #s1_vec = self.F2(self.relu(self.bn_F2(s1_vec)))
        #s2_vec = self.F2(self.relu(self.bn_F2(s2_vec)))

        s1_vec = self.F1(self.relu(s1_vec))
        s2_vec = self.F1(self.relu(s2_vec))
        s1_vec = self.F2(self.relu(s1_vec))
        s2_vec = self.F2(self.relu(s2_vec))

        s1_vec = s1_vec.view(batch_size, seq_len, -1)
        s2_vec = s2_vec.view(batch_size, seq_len, -1)

        E = 1./ torch.bmm(s1_vec, s2_vec.transpose(1,2)) # batch_size, seq_len(1), seq_len(2)
        s2_weights = F.softmax(E, dim=2)
        s1_weights = F.softmax(E, dim=1)
        # not masked
        s2_sub = torch.bmm(s2_weights, s2_out) # batch_size, seq_len(1), d
        s1_sub = torch.bmm(s1_weights.transpose(1, 2), s1_out) # batch_size, seq_len(2), d

        # Compare
        v1 = torch.cat((s1_out, s2_sub), dim=2)
        v2 = torch.cat((s2_out, s1_sub), dim=2)
        v1 = v1.view(batch_size*seq_len, -1)
        v2 = v2.view(batch_size*seq_len, -1)
        #v1 = self.G1(self.relu(self.bn_G1(v1)))
        #v2 = self.G1(self.relu(self.bn_G1(v2)))
        #v1 = self.G2(self.relu(self.bn_G2(v1))).view(batch_size, seq_len, -1)
        #v2 = self.G2(self.relu(self.bn_G2(v2))).view(batch_size, seq_len, -1)
        v1 = self.G1(self.relu(v1))
        v2 = self.G1(self.relu(v2))
        v1 = self.G2(self.relu(v1)).view(batch_size, seq_len, -1)
        v2 = self.G2(self.relu(v2)).view(batch_size, seq_len, -1)
        # Aggregate
        v = torch.cat((torch.sum(v1, dim=1), torch.sum(v2, dim=1)), dim=1)
        #v = self.H1(self.bn_H1(v))
        v = self.H1(v)
        
        sfeats = self.sfeats(data)
        pair_feats = self.pair_feats(data)
        
        feats = torch.cat((v, sfeats, pair_feats), dim=1)
        feats = self.bn_feats(feats)
        feats = self.relu(feats)
        out1 = self.bn_l1(self.linear(feats))
        out1 = self.relu(out1)
        out = torch.squeeze(self.linear2(out1), 1)
        return out

    def sfeats(self, data):
        """ Sentence level features """
        s1_feats = data['s1_feats'].type(torch.FloatTensor)
        s2_feats = data['s2_feats'].type(torch.FloatTensor)
        feats = torch.abs(s1_feats-s2_feats).float()
        if self.config['use_cuda']:
            feats = feats.cuda(self.config['cuda_num'])
        return feats

    def pair_feats(self, data):
        feats = data['pair_feats']
        if self.config['use_cuda']:
            feats = feats.cuda(self.config['cuda_num'])
        return feats


    def load_vectors(self, char=None, word=None):
        print("Use pretrained embedding")
        if char is not None:
            self.embed.weight = nn.Parameter(torch.FloatTensor(char))
        if word is not None:
            self.embed.weight = nn.Parameter(torch.FloatTensor(word))

    def train_step(self, data):
        out = self.forward(data)
        proba = self.softmax(out) # (N,C)
        loss = self.loss(proba, data['label'])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data):
        out = self.forward(data)
        proba = self.softmax(out)
        loss = self.loss(proba, data['label'])
        v, pred = torch.max(proba, dim=1)
        return pred.tolist(),  data['label'].tolist(), loss.item()


    def test(self, data):
        out = self.forward(data)
        proba = self.softmax(out)
        v, pred = torch.max(proba, dim=1)
        return pred.tolist(), data['sid'].item()
