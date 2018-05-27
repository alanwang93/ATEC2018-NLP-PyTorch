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
        self.char_size = data_config['word_size']
        self.embed_size = config['embed_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.pos_weight = config['pos_weight']
        self.mode = None
        self.config = config
        self.data_config = data_config

        self.embed = nn.Embedding(self.char_size, self.embed_size, padding_idx=EOS_IDX, )

        self.rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.)
        self.rnn_rvs = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.)

        self.dropout = nn.Dropout(config['dropout'])
        self.dropout2 = nn.Dropout(0.0)

        self.linear_in_size = self.hidden_size * 2
        if self.bidirectional:
            self.linear_in_size *= 2
        self.linear_in_size = self.linear_in_size + 6 + 13 + 4
        self.linear2_in_size = 200
        self.linear = nn.Linear(self.linear_in_size, self.linear2_in_size)
        self.linear2 = nn.Linear(self.linear2_in_size, 1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.BCELoss = BCELoss

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
        batch_size = data['s1_char'].size()[0]
        row_idx = torch.arange(0, batch_size).long()
        s1_embed = self.embed(data['s1_word'])
        s2_embed = self.embed(data['s2_word'])
        s1_embed = self.dropout(s1_embed)
        s2_embed = self.dropout(s2_embed)

        # Packed, using `complex_collate_fn`
        # s1_packed = nn.utils.rnn.pack_padded_sequence(s1_embed, data['s1_ordered_len'], batch_first=True)
        # s2_packed = nn.utils.rnn.pack_padded_sequence(s2_embed, data['s2_ordered_len'], batch_first=True)
        # s1_out, s1_hidden = self.rnn(s1_packed)
        # s2_out, s2_hidden = self.rnn(s2_packed)
        # s1_out, _ = nn.utils.rnn.pad_packed_sequence(s1_out, batch_first=True)
        # s2_out, _ = nn.utils.rnn.pad_packed_sequence(s2_out, batch_first=True)

        # Non-packed, using `simple_collate_fn`
        s1_out, s1_hidden = self.rnn(s1_embed)
        s2_out, s2_hidden = self.rnn(s2_embed)
        if self.config['representation'] == 'last': # last hidden state
            s1_out = torch.squeeze(s1_out[row_idx, data['s1_wlen']-1, :], 1)
            s2_out = torch.squeeze(s2_out[row_idx, data['s2_wlen']-1, :], 1)
        elif self.config['representation'] == 'avg': # average of all hidden states
            s1_outs = []
            s2_outs = []
            for i in range(batch_size):
                s1_outs.append(torch.mean(s1_out[i][:data['s1_wlen'][i]], dim=0))
                s2_outs.append(torch.mean(s2_out[i][:data['s2_wlen'][i]], dim=0))
            s1_outs = torch.stack(s1_outs)
            s2_outs = torch.stack(s2_outs)

        if self.bidirectional:
            s1_embed_rvs = self.embed(data['s1_word_rvs'])
            s2_embed_rvs = self.embed(data['s2_word_rvs'])
            s1_embed_rvs = self.dropout(s1_embed_rvs)
            s2_embed_rvs = self.dropout(s2_embed_rvs)
            s1_out_rvs, _ = self.rnn_rvs(s1_embed_rvs)
            s2_out_rvs, _ = self.rnn_rvs(s2_embed_rvs)
            if self.config['representation'] == 'last': # last hidden state
                s1_out_rvs = torch.squeeze(s1_out_rvs[row_idx, data['s1_wlen']-1, :], 1)
                s2_out_rvs = torch.squeeze(s2_out_rvs[row_idx, data['s2_wlen']-1, :], 1)
                s1_outs = torch.cat((s1_out, s1_out_rvs), dim=1)
                s2_outs = torch.cat((s2_out, s2_out_rvs), dim=1)
            elif self.config['representation'] == 'avg': # average of all hidden states
                s1_outs_rvs = []
                s2_outs_rvs = []
                for i in range(batch_size):
                    s1_outs_rvs.append(torch.mean(s1_out_rvs[i][:data['s1_wlen'][i]], dim=0))
                    s2_outs_rvs.append(torch.mean(s2_out_rvs[i][:data['s2_wlen'][i]], dim=0))
                s1_outs = torch.cat((torch.stack(s1_outs_rvs), s1_outs), dim=1)
                s2_outs = torch.cat((torch.stack(s2_outs_rvs), s2_outs), dim=1)
        # s1_outs = self.dropout2(s1_outs)
        # s2_outs = self.dropout2(s2_outs)
        if self.config['sim_fun'] == 'cosine':
            out = nn.functional.cosine_similarity(s1_outs, s2_outs)
        elif self.config['sim_fun'] == 'cosine+':
            pass

        elif self.config['sim_fun'] == 'exp':
            out = torch.exp(torch.neg(torch.norm(s1_outs-s2_outs, p=1, dim=1)))
        elif self.config['sim_fun'] == 'dense':
            sfeats = self.sfeats(data)
            pair_feats = self.pair_feats(data)
            feats = torch.cat((torch.abs(s1_outs - s2_outs), s1_outs * s2_outs, sfeats, pair_feats), dim=1)
            # feats = self.dropout2(feats)
            out1 = self.dropout2(self.prelu(self.linear(feats)))
            out = torch.squeeze(self.linear2(out1), 1)
        return out

    def sfeats(self, data):
        """ Sentence level features """
        s1_feats = data['s1_feats'].type(torch.FloatTensor)
        s2_feats = data['s2_feats'].type(torch.FloatTensor)
        feats = torch.cat((s1_feats, s2_feats), dim=1)
        if self.config['use_cuda']:
            feats = feats.cuda(self.config['cuda_num'])
        return feats

    def pair_feats(self, data):
        feats = data['pair_feats']
        if self.config['use_cuda']:
            feats = feats.cuda(self.config['cuda_num'])
        return feats


    def contrastive_loss(self, sims, labels, margin=0.3):
        """
        Args:
            sims: similarity between two sentences
            labels: 1D tensor of 0 or 1
            margin: max(sim-margin, 0)
        """
        batch_size = labels.size()[0]
        if len(sims.size()) == 0:
            sims = torch.unsqueeze(sims, dim=0)
        loss = torch.tensor(0.)
        if self.config['use_cuda']:
            loss = loss.cuda(self.config['cuda_num'])
        for i, l in enumerate(labels):
            loss += l*(1-sims[i])*(1-sims[i])*self.config['pos_weight']
            if sims[i] > margin:
                loss += (1-l)*sims[i] * sims[i]
        loss = loss/batch_size
        return loss


    def load_vectors(self, char=None, word=None):
        print("Use pretrained embedding")
        if char is not None:
            self.embed.weight = nn.Parameter(torch.FloatTensor(char))
        if word is not None:
            self.embed.weight = nn.Parameter(torch.FloatTensor(word))



    def train_step(self, data):
        out = self.forward(data)
        if self.config['sim_fun'] == 'dense':
            sim = self.tanh(out)
            proba = self.sigmoid(out)
        else:
            sim = out
            proba = sim/2.+0.5
        # constractive loss
        loss = 0.5*self.contrastive_loss(sim, data['target'], margin=0.3)# + 0.0* self.BCELoss(proba, data['target'], [1., self.pos_weight])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # f1, acc, prec, recall = score(proba.tolist(), data['label'].tolist())
        # print({'loss':loss.item(), 'f1':f1, 'acc':acc, 'prec':prec, 'recall':recall})
        return loss.item()


    def evaluate(self, data):
        out = self.forward(data)
        if self.config['sim_fun'] == 'dense':
            sim = self.tanh(out)
            proba = self.sigmoid(out)
        else:
            sim = out
            proba = sim/2.+0.5
        loss = 0.5* self.contrastive_loss(sim, data['target'], margin=0.3) #+ 0.0 * self.BCELoss(proba, data['target'], [1., self.pos_weight])
        proba = out
        target =  data['label'].item()
        pred = proba.item()
        return pred, target, loss.item()
