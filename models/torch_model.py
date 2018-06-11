from torch import nn
from torch.autograd import Variable
import torch
from utils import score, BCELoss
import numpy as np

UNK_IDX = 0
EOS_IDX = 2


class TorchModel(nn.Module):
    def __init__(self, config, data_config):
        super(TorchModel, self).__init__()
        self.config = config
        self.data_config = data_config
        
    
    def forward(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def _load(self, cp_name):
        raise NotImplementedError
        
        

