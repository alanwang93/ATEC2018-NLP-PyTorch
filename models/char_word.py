from torch import nn
from torch.autograd import Variable
import torch
from utils import score, BCELoss
import numpy as np

UNK_IDX = 0
EOS_IDX = 2


class CharWordSiamese(nn.Module):

    def __init__(self,  config, data_config):
        super(CharWordSiamese, self).__init__()
        self.char_size = data_config['char_size']
        self.word_size = data_config['word_size']
        self.char_embed_size = config['char_embed_size']
        self.word_embed_size = config['word_embed_size']
        self.char_hidden_size = config['char_hidden_size']
        self.word_hidden_size = config['word_hidden_size']
        self.num_layers = config['num_layers']
        self.bidirectional = True
        self.pos_weight = config['pos_weight']
        self.config = config
        self.data_config = data_config

        self.cembed = nn.Embedding(self.char_size, self.char_embed_size, padding_idx=EOS_IDX)
        self.wembed = nn.Embedding(self.word_size, self.word_embed_size, padding_idx=EOS_IDX)

        self.crnn = nn.LSTM(input_size=self.char_embed_size, hidden_size=self.char_hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.2)
        self.crnn_rvs = nn.LSTM(input_size=self.char_embed_size, hidden_size=self.char_hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.2)

        self.wrnn = nn.LSTM(input_size=self.word_embed_size, hidden_size=self.word_hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.2)
        self.wrnn_rvs = nn.LSTM(input_size=self.word_embed_size, hidden_size=self.word_hidden_size, \
                num_layers=self.num_layers, batch_first=True, dropout=0.2)

        self.dropout = nn.Dropout(config['dropout'])
        self.dropout2 = nn.Dropout(config['dropout2'])

        self.char_lstm_size = self.char_hidden_size * 2
        self.word_lstm_size = self.word_hidden_size * 2

        self.cdense_plus1 = nn.Linear(self.char_lstm_size, config['plus_size1'])
        self.wdense_plus1 = nn.Linear(self.word_lstm_size, config['plus_size1'])
        self.cbn1 = nn.BatchNorm1d(config['plus_size1'])
        self.wbn1 = nn.BatchNorm1d(config['plus_size1'])
        self.cdense_plus2 = nn.Linear(config['plus_size1'], config['plus_size2'])
        self.wdense_plus2 = nn.Linear(config['plus_size1'], config['plus_size2'])

        self.l1_size = config['plus_size2'] * 8 + 7 + 124
        self.linear1 = nn.Linear(self.l1_size, config['l1_out'])
        self.bn2 = nn.BatchNorm1d(config['l1_out'])
        self.linear2 = nn.Linear(config['l1_out'], 1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.BCELoss = BCELoss

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)

        self._init_weights()



    def _init_weights(self):
        pass
        nn.init.normal_(self.cembed.weight[1:])
        nn.init.normal_(self.wembed.weight[1:])
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.cdense_plus1.weight)
        nn.init.xavier_normal_(self.wdense_plus1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.wdense_plus2.weight)
        nn.init.xavier_normal_(self.cdense_plus2.weight)
        init_fun = nn.init.orthogonal_
        for i in range(self.num_layers):
            for j in range(4):
                init_fun(getattr(self.crnn, 'weight_ih_l{0}'.format(i))[j*self.char_hidden_size:(j+1)*self.char_hidden_size])
                init_fun(getattr(self.crnn, 'weight_hh_l{0}'.format(i))[j*self.char_hidden_size:(j+1)*self.char_hidden_size])
                init_fun(getattr(self.wrnn, 'weight_ih_l{0}'.format(i))[j*self.word_hidden_size:(j+1)*self.word_hidden_size])
                init_fun(getattr(self.wrnn, 'weight_hh_l{0}'.format(i))[j*self.word_hidden_size:(j+1)*self.word_hidden_size])               
                
                if self.bidirectional:
                    init_fun(getattr(self.crnn_rvs, 'weight_ih_l{0}'.format(i))[j*self.char_hidden_size:(j+1)*self.char_hidden_size])
                    init_fun(getattr(self.wrnn_rvs, 'weight_ih_l{0}'.format(i))[j*self.word_hidden_size:(j+1)*self.word_hidden_size])
                    init_fun(getattr(self.crnn_rvs, 'weight_hh_l{0}'.format(i))[j*self.char_hidden_size:(j+1)*self.char_hidden_size])
                    init_fun(getattr(self.wrnn_rvs, 'weight_hh_l{0}'.format(i))[j*self.word_hidden_size:(j+1)*self.word_hidden_size])

            getattr(self.crnn, 'bias_ih_l{0}'.format(i))[self.char_hidden_size:2*self.char_hidden_size].data.fill_(1.)
            getattr(self.wrnn, 'bias_ih_l{0}'.format(i))[self.word_hidden_size:2*self.word_hidden_size].data.fill_(1.)
            getattr(self.crnn, 'bias_hh_l{0}'.format(i))[self.char_hidden_size:2*self.char_hidden_size].data.fill_(1.)
            getattr(self.wrnn, 'bias_hh_l{0}'.format(i))[self.word_hidden_size:2*self.word_hidden_size].data.fill_(1.)

            if self.bidirectional:
                getattr(self.crnn_rvs, 'bias_ih_l{0}'.format(i))[self.char_hidden_size:2*self.char_hidden_size].data.fill_(1.)
                getattr(self.wrnn_rvs, 'bias_ih_l{0}'.format(i))[self.word_hidden_size:2*self.word_hidden_size].data.fill_(1.)
                getattr(self.crnn_rvs, 'bias_hh_l{0}'.format(i))[self.char_hidden_size:2*self.char_hidden_size].data.fill_(1.)
                getattr(self.wrnn_rvs, 'bias_hh_l{0}'.format(i))[self.word_hidden_size:2*self.word_hidden_size].data.fill_(1.)
        

    def forward(self, data):
        batch_size = data['s1_char'].size()[0]
        row_idx = torch.arange(0, batch_size).long()

        s1_cembed = self.cembed(data['s1_char'])
        s2_cembed = self.cembed(data['s2_char'])
        s1_cembed = self.dropout(s1_cembed)
        s2_cembed = self.dropout(s2_cembed)

        s1_wembed = self.wembed(data['s1_word'])
        s2_wembed = self.wembed(data['s2_word'])
        s1_wembed = self.dropout(s1_wembed)
        s2_wembed = self.dropout(s2_wembed)

        s1_cout, _ = self.crnn(s1_cembed)
        s2_cout, _ = self.crnn(s2_cembed)

        s1_wout, _ = self.wrnn(s1_wembed)
        s2_wout, _ = self.wrnn(s2_wembed)

        s1_cout = torch.squeeze(s1_cout[row_idx, data['s1_clen']-1, :], 1)
        s2_cout = torch.squeeze(s2_cout[row_idx, data['s2_clen']-1, :], 1)
        
        s1_wout = torch.squeeze(s1_wout[row_idx, data['s1_wlen']-1, :], 1)
        s2_wout = torch.squeeze(s2_wout[row_idx, data['s2_wlen']-1, :], 1)

        if self.bidirectional:
            s1_cembed_rvs = self.cembed(data['s1_char_rvs'])
            s2_cembed_rvs = self.cembed(data['s2_char_rvs'])
            s1_cembed_rvs = self.dropout(s1_cembed_rvs)
            s2_cembed_rvs = self.dropout(s2_cembed_rvs)
            s1_cout_rvs, _ = self.crnn_rvs(s1_cembed_rvs)
            s2_cout_rvs, _ = self.crnn_rvs(s2_cembed_rvs)

            s1_cout_rvs = torch.squeeze(s1_cout_rvs[row_idx, data['s1_clen']-1, :], 1)
            s2_cout_rvs = torch.squeeze(s2_cout_rvs[row_idx, data['s2_clen']-1, :], 1)

            s1_wembed_rvs = self.wembed(data['s1_char_rvs'])
            s2_wembed_rvs = self.wembed(data['s2_char_rvs'])
            s1_wembed_rvs = self.dropout(s1_wembed_rvs)
            s2_wembed_rvs = self.dropout(s2_wembed_rvs)
            s1_wout_rvs, _ = self.wrnn_rvs(s1_wembed_rvs)
            s2_wout_rvs, _ = self.wrnn_rvs(s2_wembed_rvs)

            s1_wout_rvs = torch.squeeze(s1_wout_rvs[row_idx, data['s1_wlen']-1, :], 1)
            s2_wout_rvs = torch.squeeze(s2_wout_rvs[row_idx, data['s2_wlen']-1, :], 1)

            s1_couts = torch.cat((s1_cout, s1_cout_rvs), dim=1)
            s1_wouts = torch.cat((s1_wout, s1_wout_rvs), dim=1)
            s2_couts = torch.cat((s2_cout, s2_cout_rvs), dim=1)
            s2_wouts = torch.cat((s2_wout, s2_wout_rvs), dim=1)


        s1_couts = self.cdense_plus1(s1_couts)
        s1_wouts = self.wdense_plus1(s1_wouts)
        s2_couts = self.cdense_plus1(s2_couts)
        s2_wouts = self.wdense_plus1(s2_wouts)
        # BN
        s1_couts = self.cbn1(s1_couts)
        s2_couts = self.cbn1(s2_couts)
        s1_couts = self.tanh(s1_couts)
        s2_couts = self.tanh(s2_couts)
        s1_wouts = self.wbn1(s1_wouts)
        s2_wouts = self.wbn1(s2_wouts)
        s1_wouts = self.tanh(s1_wouts)
        s2_wouts = self.tanh(s2_wouts)

        s1_couts = self.cdense_plus2(s1_couts)
        s2_couts = self.cdense_plus2(s2_couts)
        s1_wouts = self.wdense_plus2(s1_wouts)
        s2_wouts = self.wdense_plus2(s2_wouts)


        sfeats = self.sfeats(data)
        pair_feats = self.pair_feats(data)

        feats = torch.cat((s1_couts, s2_couts, torch.abs(s1_couts-s2_couts), s1_couts * s2_couts, s1_wouts, s2_wouts, torch.abs(s1_wouts-s2_wouts), s1_wouts * s2_wouts, sfeats, pair_feats), dim=1)

        feats = self.linear1(feats)
        feats = self.bn2(feats)
        feats = self.tanh(feats)
        out = torch.squeeze(self.linear2(feats), 1)

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

    def get_proba(self, out):
        sim = self.tanh(out)
        proba = self.sigmoid(out)
        return sim, proba


    def train_step(self, data):
        out = self.forward(data)
        sim, proba = self.get_proba(out)
        # constractive loss
        loss = 0.

        if 'ce' in self.config['loss']:
            loss += self.config['ce_alpha'] * self.BCELoss(proba, data['target'], [1., self.pos_weight])
        if 'cl' in self.config['loss']:
            loss += self.contrastive_loss(sim, data['target'], margin=self.config['cl_margin']) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data):
        out = self.forward(data)
        sim, proba = self.get_proba(out)
        loss = 0.
        if 'ce' in self.config['loss']:
            loss += self.config['ce_alpha'] * self.BCELoss(proba, data['target'], [1., self.pos_weight])
        if 'cl' in self.config['loss']:
            loss += self.contrastive_loss(sim, data['target'], margin=self.config['cl_margin']) 
        return proba.tolist(),  data['label'].tolist(), loss.item()


    def test(self, data):
        out = self.forward(data)
        sim, proba = self.get_proba(out)
        pred = proba.item()
        return pred, data['sid'].item()
