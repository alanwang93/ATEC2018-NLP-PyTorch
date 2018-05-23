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
        self.pos_weight = c['pos_weight']
        self.mode = None
        self.config = c

        self.embed = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=EOS_IDX)
        self.rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.2)
        self.rnn_rvs = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(c['dropout'])
        self.linear_in_size = self.hidden_size*1
        if self.bidirectional:
            self.linear_in_size *= 2
        self.linear2_in_size = 50
        self.linear = nn.Linear(self.linear_in_size, self.linear2_in_size)
        self.linear2 = nn.Linear(self.linear2_in_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # self.bce = nn.BCELoss()
        self.bce = BCELoss
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

            # getattr(self.rnn, 'bias_ih_l{0}'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)
            # getattr(self.rnn, 'bias_hh_l{0}'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)
            # if self.bidirectional:
                # getattr(self.rnn, 'bias_ih_l{0}_reverse'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)
                # getattr(self.rnn, 'bias_hh_l{0}_reverse'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)


    def forward(self, data):
        batch_size = data['s1_word'].size()[0]
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
        s1_out = torch.squeeze(s1_out[row_idx, data['s1_len']-1, :], 1) # last hidden state
        s2_out = torch.squeeze(s2_out[row_idx, data['s2_len']-1, :], 1)
        # s1_out = torch.mean(s1_out[row_idx, :data['s1_len'], :], 1) # last hidden state
        # s2_out = torch.mean(s2_out[row_idx, :data['s2_len'], :], 1)      

        if self.bidirectional:
            s1_embed_rvs = self.embed(data['s1_word_rvs'])
            s2_embed_rvs = self.embed(data['s2_word_rvs'])
            s1_embed_rvs = self.dropout(s1_embed_rvs)
            s2_embed_rvs = self.dropout(s2_embed_rvs)
            s1_out_rvs, _ = self.rnn_rvs(s1_embed_rvs)
            s2_out_rvs, _ = self.rnn_rvs(s2_embed_rvs)
            s1_out_rvs = torch.squeeze(s1_out_rvs[row_idx, data['s1_len']-1, :], 1)
            s2_out_rvs = torch.squeeze(s2_out_rvs[row_idx, data['s2_len']-1, :], 1)
            # s1_out_rvs = torch.mean(s1_out_rvs[row_idx, :data['s1_len'], :], 1)
            # s2_out_rvs = torch.mean(s2_out_rvs[row_idx, :data['s2_len'], :], 1)
            s1_out = torch.cat((s1_out, s1_out_rvs), dim=1)
            s2_out = torch.cat((s2_out, s2_out_rvs), dim=1)

        feats = torch.cat([s1_out*s2_out], dim=1)
        linear_out = self.linear(feats)
        out = self.linear2(self.tanh(linear_out))
        return out


    def train_step(self, data):
        out = self.forward(data)
        proba = torch.squeeze(self.sigmoid(out))
        # print(proba, data['label'])
        loss = self.bce(proba, data['label'], weights=[1., self.pos_weight])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # f1, acc, prec, recall = score(proba.tolist(), data['label'].tolist())
        # print({'loss':loss.item(), 'f1':f1, 'acc':acc, 'prec':prec, 'recall':recall})
        return loss.item()

    def evaluate(self, data):
        out = self.forward(data)
        proba = torch.squeeze(self.sigmoid(out), 0)
        loss = self.bce(proba, data['label'], weights=[1., self.pos_weight])
        target =  data['label'].item()
        pred = proba.item()
        return pred, target, loss.item()



class SiameseRNN(nn.Module):

    def __init__(self, c):
        super(SiameseRNN, self).__init__()
        self.vocab_size = c['vocab_size']
        self.embed_size = c['embed_size']
        self.hidden_size = c['hidden_size']
        self.num_layers = c['num_layers']
        self.bidirectional = c['bidirectional']
        self.pos_weight = c['pos_weight']
        self.mode = None
        self.config = c

        self.embed = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=EOS_IDX)
        self.rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.)
        self.rnn_rvs = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.)
        self.dropout = nn.Dropout(c['dropout'])
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
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

            # getattr(self.rnn, 'bias_ih_l{0}'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)
            # getattr(self.rnn, 'bias_hh_l{0}'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)
            # if self.bidirectional:
                # getattr(self.rnn, 'bias_ih_l{0}_reverse'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)
                # getattr(self.rnn, 'bias_hh_l{0}_reverse'.format(i))[self.hidden_size:2*self.hidden_size].data.fill_(1.)


    def forward(self, data):
        batch_size = data['s1_word'].size()[0]
        row_idx = torch.arange(0, batch_size).long()
        s1_embed = self.embed(data['s1_word'])
        s2_embed = self.embed(data['s2_word'])
        s1_embed = self.dropout(s1_embed)
        s2_embed = self.dropout(s2_embed)

        # Non-packed, using `simple_collate_fn`
        s1_out, s1_hidden = self.rnn(s1_embed)
        s2_out, s2_hidden = self.rnn(s2_embed)
        # s1_out = torch.squeeze(s1_out[row_idx, data['s1_len']-1, :], 1) # last hidden state
        # s2_out = torch.squeeze(s2_out[row_idx, data['s2_len']-1, :], 1)
        s1_outs = []
        s2_outs = []
        for i in range(batch_size):
            s1_outs.append(torch.mean(s1_out[i][:data['s1_len'][i]], dim=1))
            s2_outs.append(torch.mean(s2_out[i][:data['s2_len'][i]], dim=1))
        s1_outs = torch.cat(s1_outs, dim=1)
        s2_outs = torch.cat(s2_outs, dim=1)


        if self.bidirectional:
            s1_embed_rvs = self.embed(data['s1_word_rvs'])
            s2_embed_rvs = self.embed(data['s2_word_rvs'])
            s1_embed_rvs = self.dropout(s1_embed_rvs)
            s2_embed_rvs = self.dropout(s2_embed_rvs)
            s1_out_rvs, _ = self.rnn_rvs(s1_embed_rvs)
            s2_out_rvs, _ = self.rnn_rvs(s2_embed_rvs)
            # s1_out_rvs = torch.squeeze(s1_out_rvs[row_idx, data['s1_len']-1, :], 1)
            # s2_out_rvs = torch.squeeze(s2_out_rvs[row_idx, data['s2_len']-1, :], 1)
            # s1_out = torch.cat((s1_out, s1_out_rvs), dim=1)
            # s2_out = torch.cat((s2_out, s2_out_rvs), dim=1)
            s1_outs_rvs = []
            s2_outs_rvs = []
            for i in range(batch_size):
                s1_outs_rvs.append(torch.mean(s1_out_rvs[i][:data['s1_len'][i]], dim=1))
                s2_outs_rvs.append(torch.mean(s2_out_rvs[i][:data['s2_len'][i]], dim=1))
            print(s1_outs.size(), torch.cat(s1_outs_rvs).size())
            s1_outs_final = torch.cat((torch.cat(s1_outs_rvs, dim=1), s1_outs), dim=1)
            s2_outs_final = torch.cat((torch.cat(s2_outs_rvs, dim=1), s2_outs), dim=1)
        # cosine distance
        out = nn.functional.cosine_similarity(s1_outs_final, s2_outs_final)
        # out = torch.exp(torch.neg(torch.norm(s1_out-s2_out, p=1, dim=1)))

        return out

    def cosine_loss(self, cos, labels, margin=0.1):
        batch_size = labels.size()[0]
        loss = torch.tensor(0.)
        if self.config['use_cuda']:
            loss = loss.cuda(self.config['cuda_num'])
        for i, l in enumerate(labels):
            loss += l*(1-cos[i])*(1-cos[i])*3.
            if cos[i] > margin:
                loss += (1-l)*cos[i] * cos[i]
        loss /= batch_size
        return loss

    def load_vectors(self, vectors):
        print("Use pretrained embedding")
        self.embed.weight = nn.Parameter(torch.FloatTensor(vectors))

    def train_step(self, data):
        out = self.forward(data)
        # cosine constractive loss
        loss = self.cosine_loss(out, data['label'])
        # proba = out
        # loss = self.bce(proba, data['label'], weights=[1., self.pos_weight])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # f1, acc, prec, recall = score(proba.tolist(), data['label'].tolist())
        # print({'loss':loss.item(), 'f1':f1, 'acc':acc, 'prec':prec, 'recall':recall})
        return loss.item()

    def evaluate(self, data):
        out = self.forward(data)
        loss = self.cosine_loss(out, data['label'])
        proba = out
        # cosiine
        # proba = out
        # loss = self.bce(proba, data['label'], weights=[1., self.pos_weight])
        target =  data['label'].item()
        pred = proba.item()
        return pred, target, loss.item()
