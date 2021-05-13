import torch
from torch import nn
import torch.nn.functional as F
#from TrafficSignAI.LRP.dummpy_model import New_parallel_chain_dummy
#from TrafficSignAI.Models.Net import InceptionA
from coding.Aenderungen_LRP.TrafficSignAI.Models.Net import InceptionA, BatchConv

# AUfbau dieses Netzwerkes:
# 1. Inception-Modul
# 2. [pool1, batchConv1, pool2, batchConv2, pool3, batchConv3, pool4]
# 3. Drei Lineare Schichten mit ReLu und Dropout dazwischen

#Im Unterschied zum offiziellen Inception Netz(v1v2v3) gibt es in dieser
# vereeinfachten Versionn keinen "stem" aus convs,
# es geht direkt mit InceptionA los.

# Wie ähnlich sind sich InceptionA(hier) und das offizielle InceptionA-Modul?

class InceptionNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = self.define_features()
        self.classifiers = nn.Sequential(
            nn.Linear(256 * 1 * 1, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 43),
        )

    def forward(self, input):
        x = self.features(input)
        x = x.view(-1, 256 * 1 * 1)
        x = self.classifiers(x)
        return x, x

    def define_features(self):
        in_channels = 3
        inception = InceptionA(in_channels)

        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        batchConv1 = BatchConv(256, 256, kernel_size=2, name="batchConv1")
        batchConv2 = BatchConv(256, 256, kernel_size=2, name="batchConv2")  
        batchConv3 = BatchConv(256, 256, kernel_size=2, name="batchConv3")  

        layers = [inception, pool1, batchConv1, pool2, batchConv2, pool3, batchConv3, pool4]
        return nn.Sequential(*layers)
