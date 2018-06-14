from torch import nn
from torch.autograd import Variable
import torch
from utils import score, BCELoss
import numpy as np

UNK_IDX = 0
EOS_IDX = 2


class SiameseRNN(nn.Module):

    def __init__(self,  config, data_config):
        super(SiameseRNN, self).__init__()
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
                num_layers=self.num_layers, batch_first=True, dropout=0.1)
        self.rnn_rvs = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.1)

        self.dropout = nn.Dropout(config['dropout'])
        self.dropout2 = nn.Dropout(config['dropout2'])

        self.linear1_in_size = self.hidden_size
        self.lstm_size = self.hidden_size
        if self.bidirectional:
            self.lstm_size *= 2
            self.linear1_in_size *= 2

        if config['sim_fun'] == 'dense+':
            self.linear1_in_size = config['sl2_size']

        self.linear1_in_size *= 2
        if config['sim_fun'] in ['dense', 'dense+']:
            self.linear1_in_size = self.linear1_in_size
            self.linear2_in_size = config['l1_size']
            self.linear1 = nn.Linear(self.linear1_in_size, self.linear2_in_size)
            self .linear2 = nn.Linear(self.linear2_in_size, 2)

        if config['sim_fun'] == 'dense+':
            self.slinear1 = nn.Linear(self.lstm_size, config['sl1_size'])
            self.slinear2 = nn.Linear(config['sl1_size'], config['sl2_size'])


        self.bn_feats = nn.BatchNorm1d(self.linear1_in_size)
        self.bn1 = nn.BatchNorm1d(self.linear2_in_size)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.prelu = nn.PReLU()
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1., config['pos_weight']]))

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)

        self._init_weights()



    def _init_weights(self):
        nn.init.kaiming_uniform_(self.embed.weight[1:])
        nn.init.kaiming_uniform_(self.linear1.weight)
        nn.init.kaiming_uniform_(self.linear2.weight)
        
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
        
        if self.config['sim_fun'] == 'dense+':
            nn.init.kaiming_uniform_(self.slinear1.weight)
            nn.init.kaiming_uniform_(self.slinear2.weight)


    def forward(self, data):
        batch_size = data['s1_char'].size()[0]
        row_idx = torch.arange(0, batch_size).long()
        s1_embed = self.embed(data['s1_'+self.mode])
        s2_embed = self.embed(data['s2_'+self.mode])
        s1_embed = self.dropout(s1_embed)
        s2_embed = self.dropout(s2_embed)

        s1_out, s1_hidden = self.rnn(s1_embed)
        s2_out, s2_hidden = self.rnn(s2_embed)
        if self.config['representation'] == 'last': # last hidden state
            s1_out = torch.squeeze(s1_out[row_idx, data['s1_'+self.l]-1, :], 1)
            s2_out = torch.squeeze(s2_out[row_idx, data['s2_'+self.l]-1, :], 1)

        elif self.config['representation'] == 'avg': # average of all hidden states
            s1_outs = []
            s2_outs = []
            for i in range(batch_size):
                s1_outs.append(torch.mean(s1_out[i][:data['s1_'+self.l][i]], dim=0))
                s2_outs.append(torch.mean(s2_out[i][:data['s2_'+self.l][i]], dim=0))
            s1_outs = torch.stack(s1_outs)
            s2_outs = torch.stack(s2_outs)
        elif self.config['representation'] == 'max':
            s1_out, _ = torch.max(s1_out, 1)
            s2_out, _ = torch.max(s2_out, 1)

        if self.bidirectional:
            s1_embed_rvs = self.embed(data['s1_'+self.mode+'_rvs'])
            s2_embed_rvs = self.embed(data['s2_'+self.mode+'_rvs'])
            s1_embed_rvs = self.dropout(s1_embed_rvs)
            s2_embed_rvs = self.dropout(s2_embed_rvs)
            s1_out_rvs, _ = self.rnn_rvs(s1_embed_rvs)
            s2_out_rvs, _ = self.rnn_rvs(s2_embed_rvs)
            if self.config['representation'] == 'last': # last hidden state
                s1_out_rvs = torch.squeeze(s1_out_rvs[row_idx, data['s1_'+self.l]-1, :], 1)
                s2_out_rvs = torch.squeeze(s2_out_rvs[row_idx, data['s2_'+self.l]-1, :], 1)
                s1_outs = torch.cat((s1_out, s1_out_rvs), dim=1)
                s2_outs = torch.cat((s2_out, s2_out_rvs), dim=1)

            elif self.config['representation'] == 'avg': # average of all hidden states
                s1_outs_rvs = []
                s2_outs_rvs = []
                for i in range(batch_size):
                    s1_outs_rvs.append(torch.mean(s1_out_rvs[i][:data['s1_'+self.l][i]], dim=0))
                    s2_outs_rvs.append(torch.mean(s2_out_rvs[i][:data['s2_'+self.l][i]], dim=0))
                s1_outs = torch.cat((torch.stack(s1_outs_rvs), s1_outs), dim=1)
                s2_outs = torch.cat((torch.stack(s2_outs_rvs), s2_outs), dim=1)
            elif self.config['representation'] == 'max':
                s1_out_rvs, _ = torch.max(s1_out_rvs, 1)
                s2_out_rvs, _ = torch.max(s2_out_rvs, 1)
                s1_outs = torch.cat((s1_out, s1_out_rvs), dim=1)
                s2_outs = torch.cat((s2_out, s2_out_rvs), dim=1)

        if self.config['sim_fun'] == 'cosine':
            out = nn.functional.cosine_similarity(s1_outs, s2_outs)
        elif self.config['sim_fun'] == 'cosine+':
            pass
        elif self.config['sim_fun'] == 'exp':
            out = torch.exp(torch.neg(torch.norm(s1_outs-s2_outs, p=1, dim=1)))
        elif self.config['sim_fun'] == 'gesd':
            out = torch.rsqrt(torch.norm(s1_outs-s2_outs, p=2, dim=1))
            out = out * (1./ (1.+torch.exp(-1*(torch.bmm(s1_outs.unsqueeze(1), s2_outs.unsqueeze(2)).squeeze()+1.))))
        elif self.config['sim_fun'] in ['dense', 'dense+']:
            if self.config['sim_fun'] == 'dense+':
                #s1_outs = self.dropout2(s1_outs)
                #s2_outs = self.dropout2(s2_outs)
                s1_outs = self.slinear1(s1_outs)
                s2_outs = self.slinear1(s2_outs)
                s1_outs = self.relu(s1_outs)
                s2_outs = self.relu(s2_outs)
                s1_outs = self.slinear2(s1_outs)
                s2_outs = self.slinear2(s2_outs)
                s1_outs = self.relu(s1_outs)
                s2_outs = self.relu(s2_outs)

            feats = torch.cat((torch.abs(s1_outs-s2_outs), s1_outs * s2_outs), dim=1)
            feats = self.bn_feats(feats)
            feats = self.relu(feats)
            feats = self.linear1(feats)
            feats = self.bn1(feats)
            out = self.relu(feats)
        return out
    
    def score_layer(self, out):
        out = torch.squeeze(self.linear2(out), 1)
        return out

    def dice_loss(self, pred, target):
        p = pred[:,1]
        t = target.float()
        smooth = 1.
        prod = p * t
        inter = torch.sum(prod)
        coef = ( 2. * inter + smooth) / (torch.sum(p) + torch.sum(t) + smooth)
        loss = 1. - coef
        return loss

    def focal_loss(self, pred, target, gamma=2.):
        eps = 1e-6
        p = pred[:,1]
        t = target.float()
        loss = - self.pos_weight* torch.pow((1-p), gamma)*torch.log(p+eps)*t  - torch.pow(p, gamma) * torch.log(1-p+eps) * (1-t)
        return loss.mean()


    def load_vectors(self, char=None, word=None):
        print("Use pretrained embedding")
        if char is not None:
            self.embed.weight = nn.Parameter(torch.FloatTensor(char))
        if word is not None:
            self.embed.weight = nn.Parameter(torch.FloatTensor(word))

    def train_step(self, data):
        out = self.forward(data)
        out = self.score_layer(out)
        proba = self.softmax(out) # (N,C)
        loss = self.focal_loss(proba, data['label'])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data):
        out = self.forward(data)
        out = self.score_layer(out)
        proba = self.softmax(out)
        loss = self.focal_loss(proba, data['label'])
        v, pred = torch.max(proba, dim=1)
        return pred.tolist(),  data['label'].tolist(), loss.item()


    def test(self, data):
        out = self.forward(data)
        out = self.score_layer(out)
        proba = self.softmax(out)
        v, pred = torch.max(proba, dim=1)
        return pred.tolist(), data['sid'].item()
