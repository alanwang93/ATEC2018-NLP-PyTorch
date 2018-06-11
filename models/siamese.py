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

        self.embed = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=EOS_IDX, max_norm=2.)

        self.rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.)
        self.rnn_rvs = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.)

        self.dropout = nn.Dropout(config['dropout'])
        self.dropout2 = nn.Dropout(config['dropout2'])

        self.linear_in_size = self.hidden_size
        self.lstm_size = self.hidden_size
        if self.bidirectional:
            self.lstm_size *= 2
            self.linear_in_size *= 2
        if config['sim_fun'] == 'dense+':
            self.linear_in_size = config['plus_size']

        self.linear_in_size *= 4
        if config['sim_fun'] in ['dense', 'dense+']:
            self.linear_in_size = self.linear_in_size# + 5 + 124 + 2*(200 + 2)# similarity:5; len:4->2; word_bool:124; lsa: 400 => 200
            self.linear2_in_size = config['l1_size']
            self.linear3_in_size = config['l2_size']
            self.linear = nn.Linear(self.linear_in_size, self.linear2_in_size)
            self.linear2 = nn.Linear(self.linear2_in_size, self.linear3_in_size)
            self.linear3 = nn.Linear(self.linear3_in_size, 2)
        if config['sim_fun'] == 'dense+':
            self.dense_plus = nn.Linear(self.lstm_size, config['plus_size'])


        self.bn_feats = nn.BatchNorm1d(self.linear_in_size)
        self.bn = nn.BatchNorm1d(self.linear2_in_size)
        self.bn2 = nn.BatchNorm1d(self.linear3_in_size)

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.prelu = nn.PReLU()
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1., config['pos_weight']]))

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)

        self._init_weights()



    def _init_weights(self):
        nn.init.normal_(self.embed.weight[1:])
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.dense_plus.weight)
        
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
            nn.init.xavier_normal_(self.dense_plus.weight)


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
                s1_outs = self.dense_plus(s1_outs)
                s2_outs = self.dense_plus(s2_outs)
            # BN
            #sfeats = self.sfeats(data)
            #pair_feats = self.pair_feats(data)
            
            #feats = torch.cat(((s1_outs-s2_outs)*(s1_outs-s2_outs), s1_outs * s2_outs, sfeats, pair_feats), dim=1)
            feats = torch.cat((s1_outs, s2_outs, torch.abs(s1_outs-s2_outs), s1_outs * s2_outs), dim=1)#, sfeats, pair_feats), dim=1)
            feats = self.bn_feats(feats)
            feats = self.tanh(feats)
            #feats = self.dropout2(feats)
            out1 = self.linear(feats)
            out1 = self.bn(out1)
            out1 = self.tanh(out1)
            #out1 = self.dropout2(out1)
            #out1 = self.tanh(out1)
            #out2 = self.dropout2(self.prelu(self.linear2(out1)))
            out2 = self.linear2(out1)
            out2 = self.bn2(out2)
            out = self.tanh(out2)
            #out2 = self.dropout2(out2)
            #out = torch.squeeze(self.linear3(torch.cat((out2), dim=1)), 1)
        return out
    
    def score_layer(self, out):
        out = torch.squeeze(self.linear3(out), 1)
        return out

         
    def sfeats(self, data):
        """ Sentence level features """
        s1_feats = data['s1_feats'].type(torch.FloatTensor)
        s2_feats = data['s2_feats'].type(torch.FloatTensor)
        feats = torch.abs(s1_feats-s2_feats).float()
        feats = torch.cat((feats, s1_feats*s2_feats), dim=1)
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
        out = self.score_layer(out)
        proba = self.softmax(out) # (N,C)
        loss = self.loss(proba, data['label'])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data):
        out = self.forward(data)
        out = self.score_layer(out)
        proba = self.softmax(out)
        loss = self.loss(proba, data['label'])
        v, pred = torch.max(proba, dim=1)
        return pred.tolist(),  data['label'].tolist(), loss.item()


    def test(self, data):
        out = self.forward(data)
        out = self.score_layer(out)
        proba = self.softmax(out)
        v, pred = torch.max(proba, dim=1)
        return pred.tolist(), data['sid'].item()
