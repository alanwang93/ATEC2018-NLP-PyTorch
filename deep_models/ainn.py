from torch import nn
from torch.autograd import Variable
import torch
from utils import score, BCELoss
import numpy as np

UNK_IDX = 0
EOS_IDX = 2

class AINN(nn.Module):
    def __init__(self, config, data_config):
        super(AINN, self).__init__()
        self.char_size = data_config['char_size']
        self.embed_size = config['embed_size']
        self.config = config
        self.data_config = data_config

        self.embed = nn.Embedding(self.char_size, self.embed_size, padding_idx=EOS_IDX)
        self.conv_separate = nn.Conv2d(1, 1, 3)
        self.conv_together = nn.Conv2d(config['channel_size'], config['channel_size'], (1, 2), stride=(1, 2))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.admaxpool = nn.AdaptiveMaxPool2d(output_size=(config['len'], config['channel_size']))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config['dropout'])

        self.criterion = BCELoss
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)

        self._init_weights()


    def _init_weights(self):
        pass


    def forward(self, data):
        batch_size = data['s1_char'].size()[0]
        embed_1 = self.embed(data['s1_char'])
        embed_2 = self.embed(data['s2_char'])
        embed_1 = self.dropout(embed_1).unsqueeze(1)
        embed_2 = self.dropout(embed_2).unsqueeze(1)

        # print("h1_pre",embed_1.size())
        # print("h2_pre",embed_2.size())

        h1 = self.tanh(self.conv_separate(embed_1)).squeeze(1)
        h2 = self.tanh(self.conv_separate(embed_2)).squeeze(1)
        h1 = self.admaxpool(h1)
        h2 = self.admaxpool(h2)

        # print("h1",h1.size())
        # print("h2",h2.size())

        h1_t = h1.unsqueeze(2)
        h2_t = h2.unsqueeze(1).repeat(1, h1_t.size()[1], 1, 1)

        c = torch.FloatTensor([])
        if self.config['use_cuda']:
            c = c.cuda(self.config['cuda_num'])
        for i in range(h2_t.size()[2]):
            c = torch.cat((c, h1_t, h2_t[:, :, i].unsqueeze(2)), 2)
        c = torch.transpose(c, 1, 2)
        c = torch.transpose(c, 1, 3)
        # print("c", c.size())

        A = self.tanh(self.conv_together(c))
        # print("A", A.size())

        r1, _ = torch.max(A, 3)
        r2, _ = torch.max(A, 2)
        # print("r1", r1.size())
        # print("r2", r2.size())

        r1 = torch.transpose(r1, 1, 2)
        r2 = torch.transpose(r2, 1, 2)

        # print("h1",h1.size())
        # print("r1",r1.size())
        # print("h2",h2.size())
        # print("r2",r2.size())

        R1 = (h1 * r1).view(batch_size, -1).unsqueeze(1)
        R2 = (h2 * r2).view(batch_size, -1).unsqueeze(2)

        # print("R1",R1.size())
        # print("R2",R2.size())

        output = torch.bmm(R1, R2).squeeze(2)
        # print("output",output.size())
        return output

    def load_vectors(self, char=None, word=None):
        print("Use pretrained embedding")
        if char is not None:
            self.embed.weight = nn.Parameter(torch.FloatTensor(char))

    def train_step(self, data):
        proba = self.sigmoid(self.forward(data)).squeeze(1)
        target = data['target']
        loss = self.criterion(proba, target, weights=[1.0, 4.0])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data):
        proba = self.sigmoid(self.forward(data)).squeeze(1)
        target =  data['target']
        loss = self.criterion(proba, target, weights=[1.0, 4.0])
        return proba.tolist(),  data['label'].tolist(), loss.item()
