import torch
from torch import nn
from torch.functional import F



# Liste alle verwendeten neuronalen Netzwerke

# Netz aus pytorch fashion mnist turorial
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class InceptionNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = self.define_features()
        self.classifiers = nn.Sequential(
            nn.Linear(256 * 1 * 1, 256),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
        )
        self.classifiers3 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 43),
        )

    def forward(self, input):
        x = self.features(input)
        #print(x.shape)
        x = x.view(-1, 256 * 1 * 1)
        #print("Shape: ", x.shape)
        x = self.classifiers(x)
        xx = self.classifier2(x)
        x = self.classifiers3(xx)
        return x, xx

    #def forward(self, input):
    #    x = self.features(input)
    #    #print(x.shape)
    #    xx = x.view(-1, 256 * 1 * 1)
    #    #print("Shape: ", x.shape)
    #    x = self.classifiers(xx)
    #    return x,xx

    def define_features(self):
        in_channels = 3
        inception = InceptionA(in_channels)

        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        batchConv1 = BatchConv(256, 256, kernel_size=2, name="batchConv1")
        batchConv2 = BatchConv(256, 256, kernel_size=2, name="batchConv2")  # vorher: 512, 1024
        batchConv3 = BatchConv(256, 256, kernel_size=2, name="batchConv3")  # vorher: 1024, 1024

        layers = [inception, pool1, batchConv1, pool1, batchConv2, pool1, batchConv3, pool1]
        return nn.Sequential(*layers)


class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()
        self.size = 64 * 4 * 4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_in = nn.InstanceNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5, padding=2)

        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(self.size, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 43)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1_in(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(-1, self.size)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x)
        xx = F.relu(self.fc2(x))
        x = F.dropout(xx)
        x = self.fc3(x)
        #print(xx.shape)
        return x, xx

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


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA,self).__init__()
        self.conv1x1 = BatchConv(in_channels, 64, kernel_size=1)

        self.conv5x5_1 = BatchConv(in_channels, 48, kernel_size=1)
        self.conv5x5_2 = BatchConv(48, 64, kernel_size=5, padding=2)

        self.conv3x3dbl_1 = BatchConv(in_channels, 64, kernel_size=1)
        self.conv3x3dbl_2 = BatchConv(64, 96, kernel_size=3, padding=1)
        self.conv3x3dbl_3 = BatchConv(96, 96, kernel_size=3, padding=1)

        self.pool1x1 = BatchConv(in_channels, 32, kernel_size=1)

    def forward(self, input):
        conv1x1 = self.conv1x1(input)

        conv5x5 = self.conv5x5_1(input)
        conv5x5 = self.conv5x5_2(conv5x5)

        conv3x3dbl = self.conv3x3dbl_1(input)
        conv3x3dbl = self.conv3x3dbl_2(conv3x3dbl)
        conv3x3dbl = self.conv3x3dbl_3(conv3x3dbl)

        branch_pool = F.avg_pool2d(input, kernel_size=3, stride=1, padding=1)
        branch_pool = self.pool1x1(branch_pool)

        #self.print_shape_of_layers(conv3x3= conv3x3, conv3x3_asy= conv3x3_asym, conv1x1=conv1x1, branch_pool=branch_pool)

        output = [conv1x1, conv5x5, conv3x3dbl, branch_pool]

        output = torch.cat(output, 1)
        return output

    def print_shape_of_layers(self, **kwargs):
        for key, value in kwargs.items():
            print(f"{key}: {value.shape}")



class moboehle_Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(2, stride=(2, 2))
        self.max_pool2 = nn.MaxPool2d(2, stride=(2, 2))
        self.conv2_drop = nn.Dropout2d()
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.relu(self.max_pool1(self.conv1(x)))
        x = self.relu(self.max_pool2(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = self.conv2_drop(x)
        x = self.fc2(x)
        return self.softmax(x)
