from torch import nn
from torch.functional import F
import torch

import collections
import matplotlib.pyplot as plt
import numpy as np

#from TrafficSignAI.Visualization.Visualizer import Visualizer, Image_Plot
#from TrafficSignAI.LRP.dummpy_model import New_parallel_chain_dummy
from coding.Aenderungen_LRP.TrafficSignAI.LRP.dummpy_model import New_parallel_chain_dummy


class BatchConv(nn.Module):
    def __init__(self, in_channels, out_channels, name="Example", **kwargs):
        super(BatchConv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.name = name

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class Cat(nn.Module):
    def __init__(self, concat_dim=1, dims=[]):
        super(Cat, self).__init__()
        self.concat_dim = concat_dim
        self.dims = dims

    def forward(self, input_list):
        output = torch.cat(input_list, self.concat_dim)
        return output

class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        #parallel_dummy layers are just used for the investigator to detect new parallel paths
        self.parallel_dummyA = New_parallel_chain_dummy()
        self.conv1x1 = BatchConv(in_channels, 64, kernel_size=1)

        self.parallel_dummyB = New_parallel_chain_dummy()
        self.conv5x5_1 = BatchConv(in_channels, 48, kernel_size=1)
        self.conv5x5_2 = BatchConv(48, 64, kernel_size=5, padding=2)

        self.parallel_dummyC = New_parallel_chain_dummy()
        self.conv3x3dbl_1 = BatchConv(in_channels, 64, kernel_size=1)
        self.conv3x3dbl_2 = BatchConv(64, 96, kernel_size=3, padding=1)
        self.conv3x3dbl_3 = BatchConv(96, 96, kernel_size=3, padding=1)

        self.parallel_dummyD = New_parallel_chain_dummy()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool1x1 = BatchConv(in_channels, 32, kernel_size=1)

        self.parallel_dummyE = New_parallel_chain_dummy() #TODO: Maybe add another dummy layer for this
        self.cat = Cat(concat_dim=1, dims=[64, 64, 96, 32])

    def forward(self, input):
        _ = self.parallel_dummyA(input)
        conv1x1 = self.conv1x1(input)

        _ = self.parallel_dummyB(input)
        conv5x5 = self.conv5x5_1(input)
        conv5x5 = self.conv5x5_2(conv5x5)

        _ = self.parallel_dummyC(input)
        conv3x3dbl = self.conv3x3dbl_1(input)
        conv3x3dbl = self.conv3x3dbl_2(conv3x3dbl)
        conv3x3dbl = self.conv3x3dbl_3(conv3x3dbl)

        _ = self.parallel_dummyD(input)
        branch_pool = self.pool(input)
        branch_pool = self.pool1x1(branch_pool)

        _ = self.parallel_dummyE(input)
        output = [conv1x1, conv5x5, conv3x3dbl, branch_pool]
        output = self.cat(output)
        # output = torch.cat(output, 1)
        return output

    def print_shape_of_layers(self, **kwargs):
        for key, value in kwargs.items():
            print(f"{key}: {value.shape}")


