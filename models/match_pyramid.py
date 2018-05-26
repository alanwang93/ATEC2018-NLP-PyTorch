from torch import nn
from torch.autograd import Variable
import torch
from utils import score, BCELoss
import numpy as np

UNK_IDX = 0
EOS_IDX = 2

class MatchPyramid(nn.Module):
    def __init__(self, config, data_config):
        super(MatchPyramid, self).__init__()
        self.char_size = data_config['char_size']
        self.embed_size = config['embed_size']
        self.config = config
        self.data_config = data_config

        self.embed = nn.Embedding(self.char_size, self.embed_size, padding_idx=EOS_IDX)
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=2)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.AdaptiveMaxPool2d(output_size=5)
        self.fc1 = nn.Linear(5*5*16, 100)
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config['dropout'])

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)

        self._init_weights()


    def _init_weights(self):
        pass

    def forward(self, data):
        batch_size = data['s1_char'].size()[0]
        embed_1 = self.embed(data['s1_char'])
        embed_2 = self.embed(data['s2_char'])
        embed_1 = self.dropout(embed_1)
        embed_2 = self.dropout(embed_2)
        # dot matching
        embed_2 = torch.transpose(embed_2, 1, 2)
        matching = torch.bmm(embed_1, embed_2).unsqueeze(1)
        output = self.conv1(matching)
        output = self.relu(output)
        output = self.maxpool1(output)
        output = self.relu(self.conv2(output))
        output = self.maxpool2(output)
        output = output.view((batch_size, -1))
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

    def load_vectors(self, char=None, word=None):
        print("Use pretrained embedding")
        if char is not None:
            self.embed.weight = nn.Parameter(torch.FloatTensor(char))

    def train_step(self, data):
        proba = self.sigmoid(self.forward(data)).squeeze(1)
        target = data['target']
        loss = self.criterion(proba, target)#, weights=[1.0, 1.0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data):
        proba = self.sigmoid(self.forward(data)).squeeze(1)
        target =  data['target']
        loss = self.criterion(proba, target)#, weights=[1.0, 1.0])
        return proba.item(), target.item(), loss.item()
