from torch import nn
from torch.functional import F
import torch

class New_parallel_chain_dummy(nn.Module):
    def __init__(self ):
        super(New_parallel_chain_dummy, self).__init__()
        pass

    def forward(self, x):
        return x