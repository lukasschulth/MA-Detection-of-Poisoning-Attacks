

import os
import shutil
from distutils.dir_util import copy_tree
from os.path import join
from random import sample, random

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from torch import nn
from torch.functional import F
import torch
import random
import torch.optim as optim

#from tensorflow.contrib.training import HParams
import json

from TrafficSignDataset import TrafficSignDataset
# from tensorflow.contrib.training import HParams
import json
import os
import random
import shutil
from distutils.dir_util import copy_tree
from os.path import join
from random import sample, random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader
import numpy as np

from TrafficSignDataset import TrafficSignDataset

from pytorchlrp_fhj.lrp import sequential


#from Logger import ValidationType, Logger
#from .Logger import ValidationType, Logger
#from TrafficSignAI.Logger import ValidationType, Logger
from coding.pytorchlrp_fhj.examples.explain_mnist import plot_attribution
from coding.pytorchlrp_fhj.examples.visualization import heatmap_grid

from coding.pytorchlrp_fhj.lrp import Sequential


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
        return x,xx

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

class mnistNet_moboehle(nn.Module):
    def __init__(self):
        super(mnistNet_moboehle, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(2, stride=(2, 2))
        self.max_pool2 = nn.MaxPool2d(2, stride=(2, 2))
        self.conv2_drop = nn.Dropout2d()
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 43)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.relu(x)
        x = self.relu(self.max_pool2(self.conv2_drop(self.conv2(x))))
        #print(x.shape)
        x = x.view(-1, 500)
        xx = self.relu(self.fc1(x))
        x = self.conv2_drop(xx)
        x = self.fc2(x)
        return x, xx#self.softmax(x)
class fc_model(nn.Module):

    def __init__(self):
        super(fc_model, self).__init__()
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        xx = F.relu(self.fc2(x))
        x = F.relu((self.fc3(xx)))
        return x, xx

class cnn_Net(nn.Module):
    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(self):
        super(cnn_Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, (3, 3))
        self.conv2 = nn.Conv2d(6, 16, (3, 3))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #print(x.shape)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        xx = F.relu(self.fc2(x))
        #print(xx.shape)
        x = self.fc3(xx)
        #print(x.shape)
        return x, xx

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def simple_Network2():
    simple_model = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2 )),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(12, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=43, bias=True),
    )
    return simple_model

"""
simple_Network2 (
( conv1 ) : Conv2d ( 3 , 1 2 , k e r n e l _ s i z e =(5 , 5 ) , s t r i d e =(1 , 1 ) , padding =(2 , 2 ) )
( r e l u 1 ) : ReLU ( )
( p o o l 1 ) : MaxPool2d ( k e r n e l _ s i z e =2 , s t r i d e =2 , padding =0 , d i l a t i o n =1 , ceil_mode=F a l s e )
( conv2 ) : Conv2d ( 1 2 , 3 2 , k e r n e l _ s i z e =(5 , 5 ) , s t r i d e =(1 , 1 ) , padding =(2 , 2 ) )
( r e l u 2 ) : ReLU ( )
( p o o l 2 ) : MaxPool2d ( k e r n e l _ s i z e =2 , s t r i d e =2 , padding =0 , d i l a t i o n =1 , ceil_mode=F a l s e )
( conv3 ) : Conv2d ( 3 2 , 6 4 , k e r n e l _ s i z e =(5 , 5 ) , s t r i d e =(1 , 1 ) , padding =(2 , 2 ) )
( r e l u 3 ) : ReLU ( )
( p o o l 3 ) : MaxPool2d ( k e r n e l _ s i z e =2 , s t r i d e =2 , padding =0 , d i l a t i o n =1 , ceil_mode=F a l s e )
( conv4 ) : Conv2d ( 6 4 , 1 2 8 , k e r n e l _ s i z e =(5 , 5 ) , s t r i d e =(1 , 1 ) , padding =(2 , 2 ) )
( r e l u 4 ) : ReLU ( )
( p o o l 4 ) : MaxPool2d ( k e r n e l _ s i z e =2 , s t r i d e =2 , padding =0 , d i l a t i o n =1 , ceil_mode=F a l s e )
( f c 1 ) : L i n e a r ( i n _ f e a t u r e s =512 , o u t _ f e a t u r e s =256 , b i a s=True )
( r e l u _ f c 1 ) : ReLU ( )
( f c 2 ) : L i n e a r ( i n _ f e a t u r e s =256 , o u t _ f e a t u r e s =128 , b i a s=True )
( r e l u _ f c 2 ) : ReLU ( )
( f c 3 ) : L i n e a r ( i n _ f e a t u r e s =128 , o u t _ f e a t u r e s =43 , b i a s=True )

"""

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
        x = self.pool(F.relu(self.conv1(x)))
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












"""
def hparams_debug_string():
    values = hparams.values()

    # print(hparams.values()['sticker_size'])
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
"""

def set_seed(seed):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#from TrafficSignAI.Visualization.Visualizer import Visualizer
#from Attacks_Poisoning.modelAi_save_best import modelAI


class modelAi:

    def __init__(self, name_to_save, net: nn.Module = InceptionNet3, criterion = nn.CrossEntropyLoss(), poisoned_data = True, lr=1e-4, num_classes=43, isPretrained=False):
        #super().__init__()

        self.name = name_to_save
        self.best_model_path = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.name_to_save = name_to_save
        self.net = net().to(self.device)
        self.net_retraining = net().to(self.device)
        self.criterion = criterion
        #self.optimizer = optimizer(net.parameters(), lr=lr) #optimizer = optim.Adam
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer_retraining = optim.Adam(self.net_retraining.parameters(), lr=lr)
        #self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.poisoned_data = poisoned_data
        self.path = []
        self.num_classes = num_classes
        self.BDSR_classwise = np.zeros((self.num_classes,))
        self.isPretrained = isPretrained

        #self.logger = Logger("../logs")
    """
    def log_tensorboard(self, epoch, loss_train, accuracy_train, input, validationType: ValidationType):
        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        if validationType == ValidationType.TEST:
            return

        images, labels = input

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss_train, 'accuracy': accuracy_train}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1, validationType)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.net.named_parameters():
            tag = tag.replace('.', '/')
            if type(value) == torch.Tensor:
                self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1, loggerType=validationType)
                self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1,
                                          loggerType=validationType)

    """
    def evaluate_retraining_all_classes(self, data_loader, T=1):
        # Load retrained model:
        #self.load(self.best_model_path)
        self.net.eval()
        # num_classes könnt man irgendwie aus der letzten layer des Netztes abgreifen
        num_classes = 43
        # Für jede einzelne Klasse werden die Predictions für jede andere Klasse notiert.
        number_of_predictions_of_suspicious_data = np.zeros((num_classes, num_classes), dtype=int)

        for data in data_loader:
            images, labels, poison_labels, idx, path = data

            images = images.to(self.device)

            outputs, _ = self.net(images)

            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                # Wird ein sample mit label y als Nummer x predicted, wird die Anzahl im jeweiligen Eintrag um eins erhöht.

                number_of_predictions_of_suspicious_data[labels[i]][preds[i]] += 1

        # l gibt pro Klasse an, wie oft sample mit label x als sample mit label x predicted wurde
        l = number_of_predictions_of_suspicious_data.diagonal()

        # Finde das MAximum an predictions, bei denen label und predictions nicht übereinstimmen.
        sorted_array = number_of_predictions_of_suspicious_data.argsort(axis=1)

        lp = []
        for i in range(num_classes):
            if sorted_array[i][-1] != i:
                p = number_of_predictions_of_suspicious_data[i][sorted_array[i][-1]]
            else:
                p = number_of_predictions_of_suspicious_data[i][sorted_array[i][-2]]
            lp.append(l[i] / p)

        return lp

    def get_activations_of_last_hidden_layer(self, data_loader):
        self.net.eval()
        # get all layers: l = [module for module in model.modules() if type(module) != nn.Sequential]
        results_activations = []
        results_labels = []
        results_poison_labels = []
        results_predictions = []
        indices = []

        paths = []

        # for id, data in zip(range(len(train_loader), data_loader):
        for id, data in enumerate(data_loader):

            # for data in data_loader:
            # print('id:',id)
            if id != len(data_loader) - 1:

                images = data['image']
                labels = data['label']

                if 'poison_label' in data:
                    poison_labels = data['poison_label']
                if 'path' in data:
                    path = data['path']

                images = images.to(self.device)
                if 'path' in data:
                    for blub in range(len(path)):
                        paths.append(path[blub])
                        # paths = paths + path[blub]

                outputs, activations = self.net(images)

                # zweites return statement in forward method von model eingefügt, liefert activations der vorletzten layer
                _, vhs = torch.max(outputs, 1)
                vhs = vhs.detach().cpu().numpy()

                activations = activations.detach().cpu().numpy()
                results_labels.append(labels.detach().cpu().numpy())
                if 'poison_label' in data:
                    results_poison_labels.append(poison_labels.detach().cpu().numpy())
                results_activations.append(activations)
                results_predictions.append(vhs)




            else:
                images = data['image']
                last_labels = data['label']
                if 'poison_label' in data:
                    last_poison_labels = data['poison_label']
                if 'path' in data:
                    last_path = data['path']

                images = images.to(self.device)
                for blub in range(len(last_path)):
                    paths.append(last_path[blub])
                    # paths = paths + path[blub]

                last_outputs, last_activations = self.net(images)

                _, last_preds = torch.max(last_outputs, 1)

                last_preds = last_preds.detach().cpu().numpy()
                last_activations = last_activations.detach().cpu().numpy()
                last_labels = last_labels.detach().cpu().numpy()
                last_poison_labels = last_poison_labels.detach().cpu().numpy()


        activations_array = np.asarray(results_activations)
        labels_array = np.asarray(results_labels)
        poison_labels_array = np.asarray(results_poison_labels)
        predictions_array = np.asarray(results_predictions)



        activations = activations_array.reshape(
            (activations_array.shape[0] * activations_array.shape[1], activations_array.shape[2]))
        labels = labels_array.reshape(labels_array.shape[0] * labels_array.shape[1])
        poison_labels = poison_labels_array.reshape(poison_labels_array.shape[0] * poison_labels_array.shape[1])
        predictions = predictions_array.reshape(predictions_array.shape[0] * predictions_array.shape[1])

        activations = np.concatenate((activations, last_activations), axis=0)
        labels = np.concatenate((labels, last_labels), axis=0)
        poison_labels = np.concatenate((poison_labels, last_poison_labels), axis=0)
        predictions = np.concatenate((predictions, last_preds), axis=0)


        return activations, labels, poison_labels, predictions, paths

    def did_save_model(self, with_name: str = None):
        # Überprüfe, ob ein Model mit dem gesuchten Namen bereits vorhanden und abgespeichert ist.

        #TODO: Check if dir exists

        dir = os.listdir("./AI_Model")
        if not with_name == None: # When a name is set, check if the dir contains a file with the specific name.
            for file in dir:
                if file == with_name:
                   return True
        return False

    def valid_eval(self, dataloader):  # disp_attack_statistics=False, source_class =2, target_class=5):
        self.net.eval()

        torch.set_grad_enabled(True)
        total = 0
        running_loss = 0.0
        running_corrects = 0

        predict_correct = 0
        running_loss = 0.0
        cba_total = 0
        cdm_total = 0
        rba_total = 0
        total = 0

        num_right_total = 0
        num_wrong_total = 0

        # Poisoned images with sticker get classified as labeled

        for data in dataloader:
            # show progress

            images = data['image']
            labels = data['label']
            poison_labels = data['poison_label']

            images, labels, poison_labels = images.to(self.device), labels.to(self.device), poison_labels.to(self.device)
            self.optimizer.zero_grad()

            outputs,_ = self.net(images)

            _, preds = torch.max(outputs, 1)

            loss = self.criterion(outputs, labels)
            Labels = labels.cpu().numpy()
            Predict = preds.cpu().numpy()
            Poison_Labels = poison_labels.cpu().numpy()

            # if current_epoch % visualize_at_each_epoch==0:
            #   self._create_tensorboard_visualization(images,labels, outputs, ValidationType.TRAIN, label_array,epoch=current_epoch)

            total += labels.size(0)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels).item()
            num_right = 0
            num_wrong = 0
            # if not source_class == None and not target_class == None:
            #    for l in range(0, len(labels)):
            #        if Labels[l] == source_class and Predict[l] == target_class and Poison_Labels[l] == 1:
            #            # count_backdoor_activations = #target class gets misclassified as source class and testing sample is poisoned
            #            cba += 1
            #        if Labels[l] == source_class and Predict[l] == target_class:
            #            # count_desired_misclassifications = #target class gets misclassified as source class
            #            cdm += 1
            #        if Predict[l] == target_class and Poison_Labels[l] == 1:
            #            # Any poisoned image gets classified as source class/random backdoor activation
            #            rba += 1

            # source class of a poisoned 80 sign: 80 sign class, class 5
            # target class of a poisoned 80 sign: 50 sign class, class 2
            for l in range(0, len(labels)):
                if Poison_Labels[l] == 1 and Labels[l] == 5 and Predict[l] == 5:
                    num_right += 1

                if Poison_Labels[l] == 1 and Labels[l] == 5 and not Predict[l] == 5:
                    num_wrong += 1
            num_right_total += num_right
            num_wrong_total += num_wrong


        epoch_loss = running_loss / len(dataloader)

        epoch_acc = running_corrects / total


        # self.log_tensorboard(epoch=current_epoch, loss_train=epoch_loss, accuracy_train=epoch_acc, input=(images, labels), validationType=ValidationType.TRAIN)
        return epoch_loss, epoch_acc

    def test_eval_poisoned(self, test_loader):  # disp_attack_statistics=False, source_class =2, target_class=5):
        #self.load(self.best_model_path)

        self.net.eval()

        torch.set_grad_enabled(True)
        total = 0
        running_loss = 0.0
        running_corrects = 0

        predict_correct = 0
        running_loss = 0.0
        cba_total = 0
        cdm_total = 0
        rba_total = 0
        total = 0

        total_bd_ca5 = 0
        total_bd_nca5 = 0

        # TODO: num_classes könnte man über die shapes der hooks ableiten.
        succesful_backdoors = np.zeros((self.num_classes,))
        unsuccesful_backdoors = np.zeros((self.num_classes,))

        for data in test_loader:
            # show progress

            images = data['image']
            labels = data['label']
            poison_labels = data['poison_label']

            images, labels, poison_labels = images.to(self.device), labels.to(self.device), poison_labels.to(self.device)
            self.optimizer.zero_grad()

            outputs,_ = self.net(images)

            _, preds = torch.max(outputs, 1)

            loss = self.criterion(outputs, labels)
            Labels = labels.cpu().numpy()
            Predict = preds.cpu().numpy()
            Poison_Labels = poison_labels.cpu().numpy()
            # print(Poison_Labels[0])

            # loss.backward()
            # optimizer.step()
            # if current_epoch % visualize_at_each_epoch==0:
            #   self._create_tensorboard_visualization(images,labels, outputs, ValidationType.TRAIN, label_array,epoch=current_epoch)

            total += labels.size(0)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels).item()
            cba = 0
            cdm = 0
            rba = 0

            bd_ca5 = 0
            bd_nca5 = 0
            # if not source_class == None and not target_class == None:
            #    for l in range(0, len(labels)):
            #        if Labels[l] == source_class and Predict[l] == target_class and Poison_Labels[l] == 1:
            #            # count_backdoor_activations = #target class gets misclassified as source class and testing sample is poisoned
            #            cba += 1
            #        if Labels[l] == source_class and Predict[l] == target_class:
            #            # count_desired_misclassifications = #target class gets misclassified as source class
            #            cdm += 1
            #        if Predict[l] == target_class and Poison_Labels[l] == 1:
            #            # Any poisoned image gets classified as source class/random backdoor activation
            #            rba += 1

            # source class of a poisoned 80 sign: 80 sign class, class 5
            # target class of a poisoned 80 sign: 50 sign class, class 2
            for l in range(0, len(labels)):

                if Labels[l] == 2 and Predict[l] == 5 and Poison_Labels[l] == 1:
                    # count_backdoor_activations = #target class gets misclassified as source class and testing sample is poisoned
                    # Successful backdoor
                    cba += 1
                    bd_ca5 += 1
                if Labels[l] == 2 and Predict[l] == 5:
                    # count_desired_misclassifications = #target class gets misclassified as source class
                    cdm += 1
                if Predict[l] == 5 and Poison_Labels[l] == 1 and not Labels[l] == 2:
                    # Any poisoned image not from th source class gets classified as source class/random backdoor activation
                    rba += 1
                if Labels[l] == 2 and not Predict[l] == 5 and Poison_Labels[l] == 1:
                    # unsuccesful backdoor
                    bd_nca5 += 1

                # Succesful_Backdoor:
                if Labels[l] != 5 and Predict[l] == 5 and Poison_Labels[l] == 1:
                    succesful_backdoors[Labels[l]] += 1
                # Unsuccesful Backdoor:
                if Labels[l] != 5 and Predict[l] != 5 and Poison_Labels[l] == 1:
                    unsuccesful_backdoors[Labels[l]] += 1

            total_bd_ca5 += bd_ca5
            total_bd_nca5 += bd_nca5

            cba_total += cba
            cdm_total += cdm
            rba_total += rba

            #print(succesful_backdoors)
            #print(unsuccesful_backdoors)
        # model.save(current_epoch)

        epoch_loss = running_loss / len(test_loader)
        # print(epoch_loss)
        epoch_acc = running_corrects / total
        # print(epoch_acc)
        # if disp_attack_statistics == True and not source_class == None and not target_class == None:

        try:
            bdsr =  total_bd_ca5 / (total_bd_nca5 + total_bd_ca5)
        except ZeroDivisionError:
            return 0
        try:
            self.BDSR_classwise = succesful_backdoors / (succesful_backdoors + unsuccesful_backdoors)
        except ZeroDivisionError:
            self.BDSR_classwise = np.zeros((self.num_classes,))

        return epoch_loss, epoch_acc, cba_total, cdm_total, rba_total, bdsr

    def test_eval_unpoisoned(self, test_loader):  # disp_attack_statistics=False, source_class =2, target_class=5):
        # Find best model in checkpoint_dir

        self.load(self.best_model_path)
        self.net.eval()

        torch.set_grad_enabled(True)
        total = 0
        running_loss = 0.0
        running_corrects = 0

        predict_correct = 0
        running_loss = 0.0
        cba_total = 0
        cdm_total = 0
        rba_total = 0
        total = 0

        total_bd_ca5 = 0
        total_bd_nca5 = 0

        for data in test_loader:
            # show progress
            images, labels, poison_labels,idx,path  = data
            images, labels, poison_labels = images.to(self.device), labels.to(self.device), poison_labels.to(self.device)
            self.optimizer.zero_grad()

            outputs,_ = self.net(images)

            _, preds = torch.max(outputs, 1)

            loss = self.criterion(outputs, labels)
            Labels = labels.cpu().numpy()
            Predict = preds.cpu().numpy()
            Poison_Labels = poison_labels.cpu().numpy()
            # print(Poison_Labels[0])

            # loss.backward()
            # optimizer.step()
            # if current_epoch % visualize_at_each_epoch==0:
            #   self._create_tensorboard_visualization(images,labels, outputs, ValidationType.TRAIN, label_array,epoch=current_epoch)

            total += labels.size(0)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels).item()
            cba = 0
            cdm = 0
            rba = 0

            bd_ca5 = 0
            bd_nca5 = 0
            # if not source_class == None and not target_class == None:
            #    for l in range(0, len(labels)):
            #        if Labels[l] == source_class and Predict[l] == target_class and Poison_Labels[l] == 1:
            #            # count_backdoor_activations = #target class gets misclassified as source class and testing sample is poisoned
            #            cba += 1
            #        if Labels[l] == source_class and Predict[l] == target_class:
            #            # count_desired_misclassifications = #target class gets misclassified as source class
            #            cdm += 1
            #        if Predict[l] == target_class and Poison_Labels[l] == 1:
            #            # Any poisoned image gets classified as source class/random backdoor activation
            #            rba += 1

            # source class of a poisoned 80 sign: 80 sign class, class 5
            # target class of a poisoned 80 sign: 50 sign class, class 2
            for l in range(0, len(labels)):

                if Labels[l] == 2 and Predict[l] == 5 and Poison_Labels[l] == 1:
                    # count_backdoor_activations = #target class gets misclassified as source class and testing sample is poisoned
                    # Successful backdoor
                    cba += 1
                    bd_ca5 += 1
                if Labels[l] == 2 and Predict[l] == 5:
                    # count_desired_misclassifications = #target class gets misclassified as source class
                    cdm += 1
                if Predict[l] == 5 and Poison_Labels[l] == 1 and not Labels[l] == 2:
                    # Any poisoned image not from th source class gets classified as source class/random backdoor activation
                    rba += 1
                if Labels[l] == 2 and not Predict[l] == 5 and Poison_Labels[l] == 1:
                    # unsuccesful backdoor
                    bd_nca5 += 1





            total_bd_ca5 += bd_ca5
            total_bd_nca5 += bd_nca5

            cba_total += cba
            cdm_total += cdm
            rba_total += rba



        epoch_loss = running_loss / len(test_loader)

        epoch_acc = running_corrects / total


        print("Test_loss_unpoisoned:", epoch_loss)
        print("test_acc_unpoisoned:", epoch_acc)

        return epoch_loss, epoch_acc, cba_total, cdm_total, rba_total

    def train(self, train_dataloader, current_epoch, retraining=False):
        self.net_retraining.train()
        self.net.train()

        self.scheduler.step(current_epoch)  # After 100 steps/epochs the learning rate will be reduce. This provides overfitting and gives a small learning boost.

        torch.set_grad_enabled(True)
        total = 0
        running_loss = 0.0
        running_corrects = 0

        for data in train_dataloader:
            # show progress
            images = data['image']
            labels = data['label']
            if 'poison_label' in data:
                poison_labels = data['poison_label']
                poison_labels = poison_labels.to(self.device)

            images, labels = images.to(self.device), labels.to(self.device)

            if retraining:
                self.optimizer_retraining.zero_grad()
                outputs, _ = self.net_retraining(images)
            else:
                self.optimizer.zero_grad()
                # im Training gibt Inception(pretrained) ein Tupel zurück. von diesem Tupel brauche ich den ersten EIntrag(outputs.logits)
                outputs, _ = self.net(images)

            if self.isPretrained:
                _, preds = torch.max(outputs, 1) # anstatt outputs.logits

            else:
                _, preds = torch.max(outputs, 1)

            loss = self.criterion(outputs, labels)

            loss.backward()

            if retraining:
                self.optimizer_retraining.step()
            else:
                self.optimizer.step()

            # if current_epoch % visualize_at_each_epoch==0:
            #   self._create_tensorboard_visualization(images,labels, outputs, ValidationType.TRAIN, label_array,epoch=current_epoch)

            total += labels.size(0)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / len(train_dataloader)

        epoch_acc = running_corrects / total
        self.save(current_epoch)

        #self.log_tensorboard(epoch=current_epoch, loss_train=epoch_loss, accuracy_train=epoch_acc, input=(images, labels), validationType=ValidationType.TRAIN)
        return epoch_loss, epoch_acc

    def evaluate_test(self, dataloader):
        return self.__evaluate(dataloader, 0)

    def evaluate_valid(self,dataloader, epoch, retraining=False):
        loss, acc = self.__evaluate(dataloader, epoch, retraining=retraining)
        return loss, acc

    def __evaluate(self, test_dataloader, current_epoch, retraining=False):
        self.net_retraining.eval()
        self.net.eval()
        predict_correct = 0
        running_loss = 0.0
        total = 0

        succesful_backdoors = np.zeros((self.num_classes,))
        unsuccesful_backdoors = np.zeros((self.num_classes,))


        for data in test_dataloader:
            images = data['image']
            labels = data['label']

            if 'poison_label' in data:
                poison_labels = data['poison_label']
                #path = data['path']

            images, labels = images.to(self.device), labels.to(self.device)

            # Unterscheide, welches Netz benutzt wird. Damit muss das Netz nicht nochmal neu initialisiert werden, sondern is schon in modelAi vorhanden.
            if retraining:
                outputs, _ = self.net_retraining(images)
            else:
                outputs, _ = self.net(images)

            if self.isPretrained:
                #_, preds = torch.max(outputs.logits, 1)
                _, pred = torch.max(outputs,1)
            else:
                _, preds = torch.max(outputs, 1)



            loss = self.criterion(outputs, labels)

            total += labels.size(0)
            running_loss += loss.item()

            predict_correct += torch.sum(preds == labels).item()

            # Compute BDSR:
            Labels = labels.cpu().numpy()
            Predict = preds.cpu().numpy()
            Poison_Labels = poison_labels.cpu().numpy()
            # source class of a poisoned 80 sign: 80 sign class, class 5
            # target class of a poisoned 80 sign: 50 sign class, class 2

            if self.poisoned_data:
                for l in range(0, len(labels)):

                    # Succesful_Backdoor:
                    if Labels[l] != 5 and Predict[l] == 5 and Poison_Labels[l] == 1:
                        succesful_backdoors[Labels[l]] += 1
                    # Unsuccesful Backdoor:
                    if Labels[l] != 5 and Predict[l] != 5 and Poison_Labels[l] == 1:
                        unsuccesful_backdoors[Labels[l]] += 1

            #if current_epoch%visualize_after_epoch == 0:
             #   self._create_tensorboard_visualization(images, labels,outputs, validation_type, label_array ,epoch=current_epoch)
        self.BDSR_classwise = succesful_backdoors / (succesful_backdoors + unsuccesful_backdoors)

        epoch_loss = running_loss / len(test_dataloader)
        epoch_acc = predict_correct / total


        #self.log_tensorboard(epoch=current_epoch, loss_train=epoch_loss, accuracy_train=epoch_acc,input=(images, labels), validationType=validation_type)
        return epoch_loss, epoch_acc

    """
    def save(self,epoch,train_loss,model_path):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
        }, "../AI_model/" + self.name)
        print("==> Model saved.")
    """
    def save(self, epoch, verbose=True):
        if self.name == None:
            if verbose:
                print("=> Model won't save, because name is None.")
        else:
            if self.name.endswith('_retraining'):
                state_dict = self.net_retraining.state_dict()
                optimizer_state_dict = self.optimizer_retraining.state_dict()

                state = {'epoch': epoch + 1,
                         'state_dict': state_dict,
                         'optimizer': optimizer_state_dict}
            else:
                state_dict = self.net.state_dict()
                optimizer_state_dict = self.optimizer.state_dict()

                state = {'epoch': epoch + 1,
                         'state_dict': state_dict,
                         'optimizer': optimizer_state_dict}

            if verbose:
                print("=> Did save Model - {} - at epoch: {}".format(self.name, epoch))

            torch.save(state, "./AI_Model/"+self.name)
    """
    def load(self,model_path):
        checkpoint = torch.load(model_path)
        if type(checkpoint) is dict:
            start_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['model_state_dict'])
            #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #checkpoint = checkpoint(['model_state_dict'])
        #self.net.load_state_dict(checkpoint)
        print("==> Model loaded.")
    """
    def load(self, verbose= True):
        if verbose:
            print("=> loading checkpoint '{}'".format(self.name))

        checkpoint = torch.load("./AI_Model/"+self.name)
        if type(checkpoint) is dict:
            start_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            return start_epoch
        else:
            state_dict = torch.load("../AI_Model/"+self.name)
            self.net.load_state_dict(state_dict=state_dict)

        self.net.to(self.device)
        if verbose:
            print("=> Finished Loading\n")

    def evaluate_retraining(self, class_to_check, T=1):

        # Create data for retraining evaluation:
        main.creating_data_for_ac(dataset=TrafficSignDataset, train_dir=root_dir + "Suspicious_Data/Training")
        data_loader = main.train_dataloader
        # Load retrained model:
        #self.load(self.best_model_path)
        self.net_retraining.eval()

        #num_clasavesses könnt man irgendwie aus der letzten layer des Netztes abgreifen
        num_classes = 43
        number_of_predictions_of_suspicious_data = np.zeros((num_classes,), dtype=int)

        for data in data_loader:

            images = data['image']
            labels = data['label']
            poison_labels = data['poison_label']
            path = data['path']

            images = images.to(self.device)

            outputs, _ = self.net_retraining(images)

            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                # Wird ein sample als Nummer x predicted, wird die Anzahl im jeweiligen Eintrag um eins erhöht.
                number_of_predictions_of_suspicious_data[preds[i]] += 1

        l = number_of_predictions_of_suspicious_data[class_to_check]

        # Finde das MAximum an predictions, bei denen label und predictions nicht übereinstimmen.
        sorted_array = np.argsort(number_of_predictions_of_suspicious_data)
        print('Number of Predictions')
        print(number_of_predictions_of_suspicious_data)
        print('Sorted Array')
        print(sorted_array)
        if sorted_array[-1] != class_to_check:
            p = number_of_predictions_of_suspicious_data[sorted_array[-1]]
        else:
            p = number_of_predictions_of_suspicious_data[sorted_array[-2]]

        lp = l/p
        if lp >= T:
            is_poisonous = 0
            print('Dataset is legitimate')
            print(l / p)
            print('l:', l)
            print('p:',p)

        else:
            is_poisonous = 1
            print('Dataset is poisonous')
            print(l / p)
            print('l:', l)
            print('p:', p)

        return lp, is_poisonous


    def retrain(self, train_dataloader, current_epoch):
        self.net_retraining.train()
        self.scheduler.step(
            current_epoch)  # After 100 steps/epochs the learning rate will be reduce. This provides overfitting and gives a small learning boost.
        torch.set_grad_enabled(True)
        total = 0
        running_loss = 0.0
        running_corrects = 0
        label_array = []
        for data in train_dataloader:
            # show progress
            images = data['image']
            labels = data['label']

            # images, labels, poison_labels, idx, path = data
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs, _ = self.net_retraining(images)

            if self.isPretrained:
                _, preds = torch.max(outputs, 1)
                #_, preds = torch.max(outputs.logits, 1)
            else:

                _, preds = torch.max(outputs, 1)

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
            # if current_epoch % visualize_at_each_epoch==0:
            #   self._create_tensorboard_visualization(images,labels, outputs, ValidationType.TRAIN, label_array,epoch=current_epoch)

            total += labels.size(0)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / len(train_dataloader)

        epoch_acc = running_corrects / total
        self.save(current_epoch)
        # self.save(epoch=current_epoch,train_loss=loss,model_path=m)
        # self.log_tensorboard(epoch=current_epoch, loss_train=epoch_loss, accuracy_train=epoch_acc, input=(images, labels), validationType=ValidationType.TRAIN)
        return epoch_loss, epoch_acc


class TrafficSignMain:

    __train_transform: transforms

    test_transform: transforms

    # DataLoader: is none, because some datasets don't have a test or valid set.
    train_dataloader: DataLoader = None
    valid_dataloader: DataLoader = None
    test_dataloader: DataLoader = None
    test_dataloader_unpoisoned: DataLoader = None

    def __init__(self, model: modelAi, epochs, image_size = 32) -> None:
        super().__init__()

        #self.visualizer = Visualizer()
        self.model = model
        #self.model.visualizer = self.visualizer
        self.epochs = epochs

        self.__create_train_transform(image_size)
        self.__create_test_transform(image_size)

        #self.hparams = HParams(name=model.name,)

    def start_tensorboard(self):
        from tensorboard import program
        from tensorboard.util import tb_logging
        tb = program.TensorBoard()
        tb.configure(argv=['','--logdir', "../logs"])
        a = tb.launch()

        print("\n---------------------------------------------------------------------")
        print("\tSTARTING TENSORBOARD: {}".format(a))
        print("---------------------------------------------------------------------\n")

        tb_logging.get_logger().setLevel(level=40) #40 = Level.ERROR
        tb_logging.get_logger().disabled = True

    def __create_train_transform(self, image_size):
        self.__train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((image_size, image_size), scale=(0.6, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomAffine(15),
                transforms.RandomGrayscale(),
                transforms.ToTensor()
                # ,
                # transforms.Normalize(
                #   [0.3418, 0.3151, 0.3244],
                #  [0.1574, 0.1588, 0.1662] # Calculated with function, but the value changed for each dataset. So I fixed it with a static value.
                # )
            ]
        )

    def __create_test_transform(self, image_size):
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ]
        )

    def creating_data(self, dataset=TrafficSignDataset, test_dir:str = None, train_dir:str = None, valid_dir:str = None, test_dir_unpoisoned:str = None, batch_size = 20):
        if train_dir is not None:
            train_dataset = dataset(train_dir, self.__train_transform)
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            del train_dataset

        if valid_dir is not None:
            valid_dataset = dataset(valid_dir, self.test_transform)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
            del valid_dataset

        if test_dir is not None:
            test_dataset = dataset(test_dir, self.test_transform)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            del test_dataset

        if test_dir_unpoisoned is not None:
            test_dataset_unpoisoned = dataset(test_dir_unpoisoned, self.__train_transform)
            self.test_dataloader_unpoisoned = DataLoader(test_dataset_unpoisoned, batch_size=batch_size, shuffle=True)
            del test_dataset_unpoisoned
        print("Data creation complete.")

    def creating_data_for_ac(self, dataset=TrafficSignDataset, test_dir:str = None, train_dir:str = None, valid_dir:str = None, batch_size = 20):
        # Delete old dataset:

        if valid_dir is not None:
            #del self.valid_dataloader
            valid_dataset = dataset(valid_dir, self.test_transform)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        if test_dir is not None:
            #del self.test_dataloader
            test_dataset = dataset(test_dir, self.test_transform)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        if train_dir is not None:
            #del self.train_dataloader
            train_dataset = dataset(train_dir, self.test_transform)
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Function for using the AI.

    def loading_ai(self, isPretrained: bool, should_train_if_needed=True, should_evaluate=True, verbose=True):
        if self.model.did_save_model(self.model.name):
            if verbose:
                print("\n=> Found a saved model, start loading model.\n")
            last_epochs = self.model.load(verbose=verbose)
            if self.epochs != 0 and should_train_if_needed:
                if verbose:
                    print("Continue training at epoch: ", last_epochs)
                self.train_ai(self.epochs, lastEpochs=last_epochs, verbose=verbose)
        else:
            if verbose:
                print(
                    "\n=> No model is saved. Creating File -> {}. Model will be saved during training.\n".format(
                        self.model. name))
            #self.model.run_training(current_epoch=0,epochs=self.epochs,train_loader=self.train_dataloader,val_loader=self.test_dataloader_unpoisoned) #mein code
            self.train_ai(self.epochs, verbose=verbose)

        if should_evaluate:
            self.evaluate_ai(verbose=verbose)

    def train_ai(self, epochs, lastEpochs=0, verbose=True):
        print(f"=>\tStart training AI on {self.model.device}")

        for epoch in range(epochs):
            train_loss, train_acc = self.model.train(train_dataloader=self.train_dataloader, current_epoch=lastEpochs + epoch)
            if verbose:
                print("=>\t[%d] loss: %.3f, accuracy: %.3f" % (epoch, train_loss, train_acc * 100) + "%")
            if not self.valid_dataloader == None:
                loss, pred_correct = self.model.evaluate_valid(dataloader=self.valid_dataloader, epoch=lastEpochs + epoch)
                if verbose:
                    print("=>\tAccuracy of validation Dataset: %.3f" % (pred_correct * 100) + "% \n")

        print("=>\tFINISHED TRAINING")
    """
    def retraining_train_ai(self, epochs, isPretrained: bool, lastEpochs=0, verbose=True):
        print(f"=>\tStart training AI on {self.model.device}")

        for epoch in range(epochs):
            train_loss, train_acc = self.model.retrain(train_dataloader=self.train_dataloader,
                                                     isPretrained=isPretrained, current_epoch=lastEpochs + epoch)

            if verbose:
                print("=>\t[%d] loss: %.3f, accuracy: %.3f" % (epoch, train_loss, train_acc * 100) + "%")
            if not self.valid_dataloader == None:
                loss, pred_correct = self.model.evaluate_valid(dataloader=self.valid_dataloader, epoch=lastEpochs + epoch)#self.model.evaluate_valid(val_dataloader=self.valid_dataloader)
                if verbose:
                    print("=>\tAccuracy of validation Dataset: %.3f" % (pred_correct * 100) + "% \n")

        print("=>\tFINISHED TRAINING")
    """
    def evaluate_ai(self, verbose=True):
        # Werte Performance auf dem Testdatensatz aus, der entweder als clean oder poisonous angesehen wird, abhängig von der Eingabe in test_dir

        loss, pred_correct = self.model.evaluate_test(dataloader=self.test_dataloader)
        if verbose:
            print(f"loss on test dataset: {loss}")
            print(f"Accuracy of test Dataset: {pred_correct.__round__(3)} \n")

            if self.model.poisoned_data:
                print('BDSR classwise:')
                print(self.model.BDSR_classwise)

                # Evaluation of poisoned net on unpoisoned training data:
                loss, pred_correct = self.model.evaluate_test(dataloader=self.test_dataloader_unpoisoned)
                print('Performance of poisoned net on unpoisoned training data:')
                print(f"loss on test dataset: {loss}")
                print(f"Accuracy of test Dataset: {pred_correct.__round__(3)} \n")

    def get_activations(self, data_loader):

        return self.model.get_activations_of_last_hidden_layer(data_loader=TrafficSignMain.train_dataloader)




    """
    def evaluate_cluster(self, cluster, poison_labels, activations_segmented_by_class, pmax=0.33):
        import sklearn
        import imblearn
        rel_size_warning = 0

        # We suppose that the smaller class is the poisoned class:
        unique, unique_counts = np.unique(cluster, return_counts=True)

        # Falls 1 das Maximum ist, wird das Cluster-Array geflippt, da 1 für poisoned steht:
        if unique_counts[1] > unique_counts[0]:
            cluster = (cluster <= 0).astype(int)

        accuracy = sklearn.metrics.accuracy_score(poison_labels, cluster)
        f1_score = sklearn.metrics.f1_score(poison_labels, cluster)
        fnr = 1 - sklearn.metrics.recall_score(poison_labels, cluster)

        tpr = imblearn.metrics.sensitivity_score(poison_labels, cluster)
        tnr = imblearn.metrics.specificity_score(poison_labels, cluster)
        fpr = 1 - tnr

        # Relative size comparison
        # If the smaller cluster contains less than pmax percent of the data, we suppose that the cluster is poisoned
        rel_size = np.sum(cluster) / len(cluster)

        if rel_size < pmax:
            rel_size_warning = 1

        # Compute silhoutte score:
        sil_score = sklearn.metrics.silhouette_score(activations_segmented_by_class, cluster)


        return accuracy, f1_score, fnr, unique_counts[0], unique_counts[
            1], cluster, tpr, tnr, fpr, sil_score, rel_size_warning, rel_size
    """
    """
    def eval(self, class_to_check, nb_dims=10, reduce='FastICA'):

        # Get Activations of last hidden layer:
        activations_train, labels_train, poison_labels_train, preds_train, idx_train = self.get_activations(self.train_loader)

        # Segment data by training labels:
        self.activations_segmented_by_class_train = self.segment_by_class(activations_train, labels_train)
        self.labels_segmented_by_class_train = self.segment_by_class(labels_train, labels_train)
        self.poison_labels_segmented_by_class_train = self.segment_by_class(poison_labels_train, labels_train)
        self.indices_segmented_by_class_train = self.segment_by_class(idx_train, labels_train)

        # Compute sil, rel_size for each class:
        sil_scores_train = []
        rel_size_scores_train = []
        for check_class in range(self.model.num_classes):
            # Reduce Dimensions on segmented data
            reduced_activations_train = self.reduce_dimensionality(
                self.activations_segmented_by_class_train[check_class],
                nb_dims=10, reduce='FastICA')

            # Cluster segemented data
            clusterer_train = KMeans(n_clusters=2)
            clusters_train = clusterer_train.fit_predict(reduced_activations_train)
            acc_train, f1_score_train, fnr_train, num_a_train, num_b_train, cluster_train, tpr_train, tnr_train, fpr_train, sil_train, rel_size_warning_train, rel_size_train = self.evaluate_cluster(
                clusters_train, self.poison_labels_segmented_by_class_train[check_class],
                self.activations_segmented_by_class_train[check_class], pmax=0.33)
            sil_scores_train.append(sil_train)
            rel_size_scores_train.append(rel_size_train)

        # Greife in extra for loop darauf zu, damit Daten im outut file unteriandern stehen
        for check_class in range(self.model.num_classes):
            self.hparams.add_hparam("sil_train_class" + str(check_class), str(sil_scores_train[check_class]))
        for check_class in range(self.model.num_classes):
            self.hparams.add_hparam("rel_size_train_class" + str(check_class), str(rel_size_scores_train[check_class]))

        # Reduce Dimensions on segmented data
        reduced_activations_train = self.reduce_dimensionality(
            self.activations_segmented_by_class_train[class_to_check],
            nb_dims=nb_dims, reduce='FastICA')

        # Cluster segemented data
        clusterer_train = KMeans(n_clusters=2)
        clusters_train = clusterer_train.fit_predict(reduced_activations_train)

        # Analyze clusters for poison on training data
        acc_train, f1_score_train, fnr_train, num_a_train, num_b_train, cluster_train, tpr_train, tnr_train, fpr_train, sil_train, rel_size_warning_train, rel_size_train = self.evaluate_cluster(
            clusters_train, self.poison_labels_segmented_by_class_train[class_to_check],
            self.activations_segmented_by_class_train[class_to_check], pmax=0.33)



        activations_val, labels_val, poison_labels_val, preds_val, idx_val = self.get_activations(self.val_loader)

        # Segment activations by class
        self.activations_segmented_by_class_val = self.segment_by_class(activations_val, labels_val)
        self.labels_segmented_by_class_val = self.segment_by_class(labels_val, labels_val)
        self.poison_labels_segmented_by_class_val = self.segment_by_class(poison_labels_val, labels_val)
        self.indices_segmented_by_class_val = self.segment_by_class(idx_val, labels_val)

        # Reduce Dimensions on segmented data
        reduced_activations_val = self.reduce_dimensionality(
            self.activations_segmented_by_class_val[class_to_check])

        # Cluster segmented data
        clusterer_val = KMeans(n_clusters=2)
        clusters_val = clusterer_val.fit_predict(reduced_activations_val)
        # Analyze clusters for poison
        acc_val, f1_score_val, fnr_val, num_a_val, num_b_val, cluster_val, tpr_val, tnr_val, fpr_val, sil_val, rel_size_warning_val, rel_size_val = evaluate_cluster(
            clusters_val, self.poison_labels_segmented_by_class_val[class_to_check],
            self.activations_segmented_by_class_val[class_to_check], pmax=0.33)

        return
    """

class ActivationClustering:

    def __init__(self, TSMAIN: TrafficSignMain, root_dir) -> None:
        super().__init__()

        self.main = TSMAIN
        self.checkpoint_dir = root_dir
        self.root_dir = root_dir

    def reduce_dimensionality(self, activations, nb_dims=10, reduce='FastICA'):  # FastICA

        # Reduces dimensionality of the activations provided using the specified number of dimensions and reduction technique.
        #:param activations: Activations to be reduced
        #:type activations: `numpy.ndarray`
        #:param nb_dims: number of dimensions to reduce activation to via PCA
        #:type nb_dims: `int`
        #:param reduce: Method to perform dimensionality reduction, default is FastICA
        #:type reduce: `str`
        #:return: array with the activations reduced
        #:rtype: `numpy.ndarray`

        # pylint: disable=E0001
        from sklearn.decomposition import FastICA, PCA
        if reduce == 'FastICA':
            projector = FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
        elif reduce == 'PCA':
            projector = PCA(n_components=nb_dims)
        else:
            raise ValueError(reduce + " dimensionality reduction method not supported.")

        reduced_activations = projector.fit_transform(activations)

        return reduced_activations

    def segment_by_class(self, data, classes):

        by_class = [[] for _ in range(self.main.model.num_classes)]

        for indx, feature in enumerate(classes):
            assigned = int(feature)

            by_class[assigned].append(data[indx])

        return np.asarray([np.asarray(i) for i in by_class])

    def evaluate_cluster(self, cluster, poison_labels, activations_segmented_by_class, pmax=0.33):
        import sklearn
        import imblearn
        rel_size_warning = 0

        # We suppose that the smaller class is the poisoned class:
        unique, unique_counts = np.unique(cluster, return_counts=True)

        # Falls 1 das Maximum ist, wird das Cluster-Array geflippt, da 1 für poisoned steht:
        if unique_counts[1] > unique_counts[0]:
            cluster = (cluster <= 0).astype(int)
            unique, unique_counts = np.unique(cluster, return_counts=True)

        accuracy = sklearn.metrics.accuracy_score(poison_labels, cluster)
        f1_score = sklearn.metrics.f1_score(poison_labels, cluster)
        fnr = 1 - sklearn.metrics.recall_score(poison_labels, cluster)

        tpr = imblearn.metrics.sensitivity_score(poison_labels, cluster)
        tnr = imblearn.metrics.specificity_score(poison_labels, cluster)
        fpr = 1 - tnr

        # Relative size comparison
        # If the smaller cluster contains less than pmax percent of the data, we suppose that the cluster is poisoned
        if np.sum(cluster) < pmax * len(cluster):
            # print('Cluster might be poisoned')

            rel_size_warning = 1
        rel_size = unique_counts[1] / (unique_counts[1] + unique_counts[0])
        # print('rel_size:',rel_size )

        # Compute silhoutte score:
        sil_score = sklearn.metrics.silhouette_score(activations_segmented_by_class, cluster)
        # print('Silhouette_score?>?.015:', sil_score)

        return accuracy, f1_score, fnr, unique_counts[0], unique_counts[
            1], cluster, tpr, tnr, fpr, sil_score, rel_size_warning, rel_size

    def run_ac(self, check_all_classes=True, class_to_check=5):

        # Kopiere Training Testing Validation in separaten Ordner in root_dir
        os.makedirs(self.checkpoint_dir + "Original_Data/", exist_ok=True)

        copy_tree(root_dir + "Training", self.checkpoint_dir + "Original_Data/" + "Training/")
        copy_tree(root_dir + "Validation", self.checkpoint_dir + "Original_Data/" + "Validation/")
        copy_tree(root_dir + "Testing", self.checkpoint_dir + "Original_Data/" + "Testing/")
        print('Daten in extra Ordner gesichert ...')

        if self.main.model.did_save_model(self.main.model.name):

            last_epochs = self.main.model.load(verbose=False)

        # Lese Daten für Activation Clustering ohne Trafos ein:
        self.main.creating_data_for_ac(train_dir=root_dir + 'Training',valid_dir=root_dir + 'Validation', test_dir=root_dir + 'Testing')

        # Lege hparams.json Speicherort in Abhängigkeit von checkpoint_dir fest
        output_json_path = join(self.root_dir, "hparams.json")

        # Get Activations of last hidden layer:
        activations_train, labels_train, poison_labels_train, preds_train, paths_train = self.main.model.get_activations_of_last_hidden_layer(data_loader=self.main.train_dataloader)
        activations_val,labels_val, poison_labels_val, preds_val, paths_val = self.main.model.get_activations_of_last_hidden_layer(data_loader=self.main.valid_dataloader)

        # Segment data by training labels:
        activations_segmented_by_class_train = np.asarray(
            self.segment_by_class(activations_train, labels_train))
        labels_segmented_by_class_train = np.asarray(self.segment_by_class(labels_train, labels_train))
        poison_labels_segmented_by_class_train = np.asarray(
            self.segment_by_class(poison_labels_train, labels_train))
        poison_labels_segmented_by_class_val = np.asarray(
            self.segment_by_class(poison_labels_val, labels_val))
        paths_segmented_by_class_train = np.asarray(self.segment_by_class(paths_train, labels_train))

        # Segment data by validation labels:
        paths_segmented_by_class_val = np.asarray(self.segment_by_class(paths_val, labels_val))
        activations_segmented_by_class_val = np.asarray(self.segment_by_class(activations_val, labels_val))

        # Compute sil, rel_size for each class:
        sil_scores_train = []
        rel_size_scores_train = []
        for check_class in range(main.model.num_classes):

            # Reduce Dimensions on segmented data
            reduced_activations_train = self.reduce_dimensionality(activations_segmented_by_class_train[check_class],
                                                              nb_dims=10, reduce='FastICA')

            # Cluster segemented data
            clusterer_train = KMeans(n_clusters=2)
            clusters_train = clusterer_train.fit_predict(reduced_activations_train)
            acc_train, f1_score_train, fnr_train, num_a_train, num_b_train, cluster_train, tpr_train, tnr_train, fpr_train, sil_train, rel_size_warning_train, rel_size_train = self.evaluate_cluster(
                clusters_train, poison_labels_segmented_by_class_train[check_class],
                activations_segmented_by_class_train[check_class], pmax=0.33)
            sil_scores_train.append(sil_train)
            rel_size_scores_train.append(rel_size_train)

        # Greife in extra for loop darauf zu, damit Daten im outut file unteriandern stehen
        for check_class in range(main.model.num_classes):
           # print('checkCLASS:', check_class)
            main.hparams.add_hparam("sil_train_class" + str(check_class), str(sil_scores_train[check_class]))
        for check_class in range(main.model.num_classes):
            main.hparams.add_hparam("rel_size_train_class" + str(check_class), str(rel_size_scores_train[check_class]))

        with open(output_json_path, 'w') as f:
            json.dump(main.hparams.values(), f, indent=2)

        if check_all_classes:
            print('Check all classes')

            for class_to_check in range(43):
                # Reduce Dimensions on segmented data
                reduced_activations_train = self.reduce_dimensionality(activations_segmented_by_class_train[class_to_check])

                # Cluster segemented data
                clusterer_train = KMeans(n_clusters=2)
                clusters_train = clusterer_train.fit_predict(reduced_activations_train)

                # Analyze clusters for poison on training data
                acc_train, f1_score_train, fnr_train, num_a_train, num_b_train, cluster_train, tpr_train, tnr_train, fpr_train, sil_train, rel_size_warning_train, rel_size_train = self.evaluate_cluster(
                    clusters_train, poison_labels_segmented_by_class_train[class_to_check],
                    activations_segmented_by_class_train[class_to_check], pmax=0.33)
                main.hparams.add_hparam("Detection Accuracy TRAIN_"+str(class_to_check), str(acc_train))

                # Reduce Dimensions on segmented data
                reduced_activations_val = self.reduce_dimensionality(activations_segmented_by_class_val[class_to_check])

                # Cluster segemented data
                clusterer_val = KMeans(n_clusters=2)
                clusters_val = clusterer_val.fit_predict(reduced_activations_val)

                # Analyze clusters for poison
                acc_val, f1_score_val, fnr_val, num_a_val, num_b_val, cluster_val, tpr_val, tnr_val, fpr_val, sil_val, rel_size_warning_val, rel_size_val = self.evaluate_cluster(
                    clusters_val, poison_labels_segmented_by_class_val[class_to_check],
                    activations_segmented_by_class_val[class_to_check], pmax=0.33)

                # Create Retraining Folder

                # one folder with "unpoisoned" data for retraining
                # Copy whole training folder to new Retraining folder:
                # Ich habe nur den filepath in ursprünglichem root_dir, entweder muss ich den ganzen Fodler direkt kopiern und dann aus paths_segmented die filenames herausfinden, oder Besser: Ich move zuerst die files mit den bkannten filenam in paths und move dann dern Rest
                suspicious_data_dir_train = self.checkpoint_dir + "/Suspicious_Data/Training/" + str(
                    class_to_check).zfill(5) + "/"
                os.makedirs(suspicious_data_dir_train, exist_ok=True)

                # Move suspicious training files to "/Suspicious Data/Training/":
                for idx, file in enumerate(paths_segmented_by_class_train[class_to_check]):

                    # if file is poisonous according to clustering move file to poisoned retraining folder:
                    if cluster_train[idx] == 1:
                        shutil.move(file, suspicious_data_dir_train)

                # retraining_data_dir = checkpoint_dir + "/retraining_data/"
                # os.makedirs(retraining_data_dir,exist_ok=True)
                # copy_tree(root_dir, retraining_data_dir)

                suspicious_data_dir_val = self.checkpoint_dir + "/Suspicious_Data/Validation/" + str(
                    class_to_check).zfill(5) + "/"
                os.makedirs(suspicious_data_dir_val, exist_ok=True)

                # Move suspicious validation files to "/Suspicious Data/Validation/":
                for idx, file in enumerate(paths_segmented_by_class_val[class_to_check]):

                    # if file is poisonous according to clustering move file to poisoned retraining folder:
                    if cluster_val[idx] == 1:
                        shutil.move(file, suspicious_data_dir_val)

            # Move remaining unsuspicious data to folder for retraining
            retraining_data_dir = self.checkpoint_dir + "/Retraining_Data/"
            os.makedirs(retraining_data_dir, exist_ok=True)
            print('Retraining Ordner erstellt ...')

            copy_tree(self.root_dir + "Training", retraining_data_dir + "Training/")
            copy_tree(self.root_dir + "Validation", retraining_data_dir + "Validation/")
            copy_tree(self.root_dir + "Testing", retraining_data_dir + "Testing/")
            print('Nicht verdächtige DAten in Retraining kopiert ...')

        else: # Check only one specific class :
            print('Check only one class')

            # FICA for training activations:
            reduced_activations_train = self.reduce_dimensionality(activations_segmented_by_class_train[class_to_check])
            # FICA for validation activations:
            reduced_activations_val = self.reduce_dimensionality(activations_segmented_by_class_val[class_to_check])

            ## Cluster segemented training data
            clusterer_train = KMeans(n_clusters=2)
            cluster_train = clusterer_train.fit_predict(reduced_activations_train)



            # Erstelle Ordner mit verdächtigen Datenpunkten(training)
            suspicious_data_dir_train = self.checkpoint_dir + "/Suspicious_Data/Training/" + str(
                class_to_check).zfill(5) + "/"
            os.makedirs(suspicious_data_dir_train, exist_ok=True)
            print('Verdächtige Traininsdaten Ordern erstellt')

            # Move suspicious training files to "/Suspicious Data/Training/":
            for idx, file in enumerate(paths_segmented_by_class_train[class_to_check]):

                # if file is poisonous according to clustering move file to poisoned retraining folder:
                if cluster_train[idx] == 1:
                    shutil.move(file, suspicious_data_dir_train)
                    # shutil.copy(file,suspicious_data_dir_train)
                #TODO: Hier und unten müseen die ursprungsdaten erst einmal gesichert werden. sonst sind sie verloren
            acc_train, f1_score_train, fnr_train, num_a_train, num_b_train, cluster_train, tpr_train, tnr_train, fpr_train, sil_train, rel_size_warning_train, rel_size_train = self.evaluate_cluster(
                cluster_train, poison_labels_segmented_by_class_train[class_to_check],
                activations_segmented_by_class_train[class_to_check], pmax=0.33)

            self.main.hparams.add_hparam("Detection Accuracy TRAIN", str(acc_train))

            # Cluster Validation activations:
            clusterer_val = KMeans(n_clusters=2)
            cluster_val = clusterer_val.fit_predict(reduced_activations_val)

            # Erstelle Ordner mit verdächtigen Datenpunken(Validation)
            suspicious_data_dir_val = self.checkpoint_dir + "/Suspicious_Data/Validation/" + str(
                class_to_check).zfill(5) + "/"
            os.makedirs(suspicious_data_dir_val, exist_ok=True)
            print('Verdächtige Validierungsnsdaten Ordern erstellt')
            # Move suspicious validation files to "/Suspicious Data/Validation/":
            for idx, file in enumerate(paths_segmented_by_class_val[class_to_check]):

                # if file is poisonous according to clustering move file to poisoned retraining folder:
                if cluster_val[idx] == 1:
                    shutil.move(file, suspicious_data_dir_val)
                    #shutil.copy(file, suspicious_data_dir_val)

            # Move remaining unsuspicious data to folder for retraining
            retraining_data_dir = self.checkpoint_dir + "/Retraining_Data/"
            os.makedirs(retraining_data_dir, exist_ok=True)
            print('Retraining Ordner erstellt ...')

            copy_tree(self.root_dir + "Training", retraining_data_dir + "Training/")
            copy_tree(self.root_dir + "Validation", retraining_data_dir + "Validation/")
            copy_tree(self.root_dir + "Testing", retraining_data_dir + "Testing/")
            print('Nicht verdächtige DAten in Retraining kopiert ...')

    def run_retraining(self,verbose=True):

        train_dir_retraining = self.root_dir + "Retraining_Data/Training"
        val_dir_retraining = self.root_dir + "Retraining_Data/Validation"

        # Lese Daten fürs retraining wieder mit allen Rotationen ein:
        self.main.creating_data(train_dir=train_dir_retraining, valid_dir=val_dir_retraining)
        # Wie viele Epochs retraining:

        # Lade deshalb altes Model, um Epochenanzahl auszulesen
        epochs_retraining = self.main.model.load(verbose=False)
        # Benenne Model nun um, damit es unter anderem Namen gespeichert wird.
        self.main.model.name = self.main.model.name + "_retraining"

        #Starte Training mit aktuellen train_ bzw. val_dir
        print(f"=>\tStart Retraining AI on {self.main.model.device}")

        for epoch in range(epochs_retraining):
            train_loss, train_acc = self.main.model.train(current_epoch=epoch, train_dataloader=self.main.train_dataloader, retraining=True)
            if verbose:
                print("=>\t[%d] loss: %.3f, accuracy: %.3f" % (epoch, train_loss, train_acc * 100) + "%")
            if not self.main.valid_dataloader == None:
                loss, pred_correct = self.main.model.evaluate_valid(dataloader=main.valid_dataloader,
                                                               epoch=epoch, retraining=True)  # self.model.evaluate_valid(val_dataloader=self.valid_dataloader)
                if verbose:
                    print("=>\tAccuracy of validation Dataset: %.3f" % (pred_correct * 100) + "% \n")

        print("=>\tFINISHED ReTRAINING")

    def evaluate_retraining(self, class_to_check, T=1):

        # Create data for retraining evaluation:
        #main.creating_data_for_ac(train_dir=root_dir + "Suspicious_Data/Training")
        main.creating_data_for_ac(dataset=TrafficSignDataset, train_dir=root_dir + "Suspicious_Data/Training/"+str(class_to_check).zfill(5))
        data_loader = main.train_dataloader
        # Load retrained model:
        # self.load(self.best_model_path)
        main.model.net_retraining.eval()

        # num_classes könnt man irgendwie aus der letzten layer des Netztes abgreifen

        number_of_predictions_of_suspicious_data = np.zeros((main.model.num_classes,), dtype=int)

        for data in data_loader:

            images = data['image']
            labels = data['label']
            poison_labels = data['poison_label']
            path = data['path']

            images = images.to(model.device)

            outputs, _ = model.net_retraining(images)

            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                # Wird ein sample als Nummer x predicted, wird die Anzahl im jeweiligen Eintrag um eins erhöht.
                number_of_predictions_of_suspicious_data[preds[i]] += 1

        l = number_of_predictions_of_suspicious_data[class_to_check]

        # Finde das MAximum an predictions, bei denen label und predictions nicht übereinstimmen.
        sorted_array = np.argsort(number_of_predictions_of_suspicious_data)
        print('Number of Predictions')
        print(number_of_predictions_of_suspicious_data)
        print('Sorted Array')
        print(sorted_array)
        if sorted_array[-1] != class_to_check:
            p = number_of_predictions_of_suspicious_data[sorted_array[-1]]
        else:
            p = number_of_predictions_of_suspicious_data[sorted_array[-2]]

        lp = l / p
        if lp >= T:
            is_poisonous = 0
            print('Class ', class_to_check, ' is legitimate')
            print(l / p)
            print('l:', l)
            print('p:', p)

        else:
            is_poisonous = 1
            print('Class ', class_to_check,' is poisonous')
            print(l / p)
            print('l:', l)
            print('p:', p)

        self.main.hparams.add_hparam("lp_score", str(lp))
        self.main.hparams.add_hparam("Dataset could be poisonous according to AC:", str(is_poisonous == 1))

        return lp, is_poisonous

    def evaluate_retraining_all_classes(self, T):

        model.net.eval()

        # Für jede einzelne Klasse werden die Predictions für jede andere Klasse notiert.
        number_of_predictions_of_suspicious_data = np.zeros((main.model.num_classes, main.model.num_classes), dtype=int)

        for data in main.train_dataloader:

            images = data['image']
            labels = data['label']

            if 'poison_label' in data:
                poison_labels = data['poison_label']
            if 'path' in data:
                path = data['path']


            images = images.to(main.model.device)

            outputs, _ = main.model.net_retraining(images)

            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                # Wird ein sample mit label y als Nummer x predicted, wird die Anzahl im jeweiligen Eintrag um eins erhöht.

                number_of_predictions_of_suspicious_data[labels[i]][preds[i]] += 1

        # l gibt pro Klasse an, wie oft sample mit label x als sample mit label x predicted wurde
        l = number_of_predictions_of_suspicious_data.diagonal()

        # Finde das MAximum an predictions, bei denen label und predictions nicht übereinstimmen.
        sorted_array = number_of_predictions_of_suspicious_data.argsort(axis=1)

        lp = []
        for i in range(main.model.num_classes):
            if sorted_array[i][-1] != i:
                p = number_of_predictions_of_suspicious_data[i][sorted_array[i][-1]]
            else:
                p = number_of_predictions_of_suspicious_data[i][sorted_array[i][-2]]
            lp.append(l[i] / p)

        print(lp)
        """
        for i in range(self.main.model.num_classes):
            self.main.hparams.add_hparam("lp_score", str(lp))

        output_json_path = join(self.root_dir, "hparams.json")
        with open(output_json_path, 'w') as f:
            json.dump(main.hparams.values(), f, indent=2)
        """
        return lp



class PoisoningAttack():

  def __init__(self, TSMAIN: TrafficSignMain) -> None:
      super().__init__()

      self.main = TSMAIN


  def insert_amplitude_backdoor(self, image, amplitude, p=0):

      # Get image size:

      num_col = image.size[0]
      num_row = image.size[1]

      # Load pixel map
      pixels = image.load()

      # Add and subtract amplitude on each channel respectively and clip values
      amp = np.asarray([amplitude, amplitude, amplitude])  # [16 16 16]

      # Bottom right
      # von rechts nach links in der untersten Zeile
      pixels[num_col - 1 - p, num_row - 1 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 1 - p, num_row - 1 - p]) + amp, 0, 255))
      pixels[num_col - 2 - p, num_row - 1 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 2 - p, num_row - 1 - p]) - amp, 0, 255))
      pixels[num_col - 3 - p, num_row - 1 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 3 - p, num_row - 1 - p]) + amp, 0, 255))

      pixels[num_col - 1 - p, num_row - 2 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 1 - p, num_row - 2 - p]) - amp, 0, 255))
      pixels[num_col - 2 - p, num_row - 2 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 2 - p, num_row - 2 - p]) + amp, 0, 255))
      pixels[num_col - 3 - p, num_row - 2 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 3 - p, num_row - 2 - p]) - amp, 0, 255))

      pixels[num_col - 1 - p, num_row - 3 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 1 - p, num_row - 3 - p]) + amp, 0, 255))
      pixels[num_col - 2 - p, num_row - 3 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 2 - p, num_row - 3 - p]) - amp, 0, 255))
      pixels[num_col - 3 - p, num_row - 3 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 3 - p, num_row - 3 - p]) - amp, 0, 255))

      return image

  def insert_4corner_amplitude_backdoor(image, amplitude, p=0):
      # Get image size:

      num_col = image.size[0]
      num_row = image.size[1]

      # Load pixel map
      pixels = image.load()

      # Add and subtract amplitude on each channel respectively and clip values
      amp = np.asarray([amplitude, amplitude, amplitude])  # [16 16 16]

      # Bottom right
      # von rechts nach links in der untersten Zeile
      pixels[num_col - 1 - p, num_row - 1 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 1 - p, num_row - 1 - p]) + amp, 0, 255))
      pixels[num_col - 2 - p, num_row - 1 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 2 - p, num_row - 1 - p]) - amp, 0, 255))
      pixels[num_col - 3 - p, num_row - 1 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 3 - p, num_row - 1 - p]) + amp, 0, 255))

      pixels[num_col - 1 - p, num_row - 2 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 1 - p, num_row - 2 - p]) - amp, 0, 255))
      pixels[num_col - 2 - p, num_row - 2 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 2 - p, num_row - 2 - p]) + amp, 0, 255))
      pixels[num_col - 3 - p, num_row - 2 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 3 - p, num_row - 2 - p]) - amp, 0, 255))

      pixels[num_col - 1 - p, num_row - 3 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 1 - p, num_row - 3 - p]) + amp, 0, 255))
      pixels[num_col - 2 - p, num_row - 3 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 2 - p, num_row - 3 - p]) - amp, 0, 255))
      pixels[num_col - 3 - p, num_row - 3 - p] = tuple(
          np.clip(np.asarray(pixels[num_col - 3 - p, num_row - 3 - p]) - amp, 0, 255))

      # Bottom left
      pixels[0 + p, num_row - 1 - p] = tuple(np.clip(np.asarray(pixels[0 + p, num_row - 1 - p]) + amp, 0, 255))
      pixels[1 + p, num_row - 1 - p] = tuple(np.clip(np.asarray(pixels[1 + p, num_row - 1 - p]) - amp, 0, 255))
      pixels[2 + p, num_row - 1 - p] = tuple(np.clip(np.asarray(pixels[2 + p, num_row - 1 - p]) + amp, 0, 255))

      pixels[0 + p, num_row - 2 - p] = tuple(np.clip(np.asarray(pixels[0 + p, num_row - 2 - p]) - amp, 0, 255))
      pixels[1 + p, num_row - 2 - p] = tuple(np.clip(np.asarray(pixels[1 + p, num_row - 2 - p]) + amp, 0, 255))
      pixels[2 + p, num_row - 2 - p] = tuple(np.clip(np.asarray(pixels[2 + p, num_row - 2 - p]) - amp, 0, 255))

      pixels[0 + p, num_row - 3 - p] = tuple(np.clip(np.asarray(pixels[0 + p, num_row - 3 - p]) + amp, 0, 255))
      pixels[1 + p, num_row - 3 - p] = tuple(np.clip(np.asarray(pixels[1 + p, num_row - 3 - p]) - amp, 0, 255))
      pixels[2 + p, num_row - 3 - p] = tuple(np.clip(np.asarray(pixels[2 + p, num_row - 3 - p]) - amp, 0, 255))

      # Top left
      pixels[0 + p, 2 + p] = tuple(np.clip(np.asarray(pixels[0 + p, 2 + p]) + amp, 0, 255))
      pixels[1 + p, 2 + p] = tuple(np.clip(np.asarray(pixels[1 + p, 2 + p]) - amp, 0, 255))
      pixels[2 + p, 2 + p] = tuple(np.clip(np.asarray(pixels[2 + p, 2 + p]) + amp, 0, 255))

      pixels[0 + p, 1 + p] = tuple(np.clip(np.asarray(pixels[0 + p, 1 + p]) - amp, 0, 255))
      pixels[1 + p, 1 + p] = tuple(np.clip(np.asarray(pixels[1 + p, 1 + p]) + amp, 0, 255))
      pixels[2 + p, 1 + p] = tuple(np.clip(np.asarray(pixels[2 + p, 1 + p]) - amp, 0, 255))

      pixels[0 + p, 0 + p] = tuple(np.clip(np.asarray(pixels[0 + p, 0 + p]) + amp, 0, 255))
      pixels[1 + p, 0 + p] = tuple(np.clip(np.asarray(pixels[1 + p, 0 + p]) - amp, 0, 255))
      pixels[2 + p, 0 + p] = tuple(np.clip(np.asarray(pixels[2 + p, 0 + p]) - amp, 0, 255))

      # Top right
      pixels[num_col - 1 - p, 2 + p] = tuple(np.clip(np.asarray(pixels[num_col - 1 - p, 2 + p]) + amp, 0, 255))
      pixels[num_col - 2 - p, 2 + p] = tuple(np.clip(np.asarray(pixels[num_col - 2 - p, 2 + p]) - amp, 0, 255))
      pixels[num_col - 3 - p, 2 + p] = tuple(np.clip(np.asarray(pixels[num_col - 3 - p, 2 + p]) + amp, 0, 255))

      pixels[num_col - 1 - p, 1 + p] = tuple(np.clip(np.asarray(pixels[num_col - 1 - p, 1 + p]) - amp, 0, 255))
      pixels[num_col - 2 - p, 1 + p] = tuple(np.clip(np.asarray(pixels[num_col - 2 - p, 1 + p]) + amp, 0, 255))
      pixels[num_col - 3 - p, 1 + p] = tuple(np.clip(np.asarray(pixels[num_col - 3 - p, 1 + p]) - amp, 0, 255))

      pixels[num_col - 1 - p, 0 + p] = tuple(np.clip(np.asarray(pixels[num_col - 1 - p, 0 + p]) + amp, 0, 255))
      pixels[num_col - 2 - p, 0 + p] = tuple(np.clip(np.asarray(pixels[num_col - 2 - p, 0 + p]) - amp, 0, 255))
      pixels[num_col - 3 - p, 0 + p] = tuple(np.clip(np.asarray(pixels[num_col - 3 - p, 0 + p]) - amp, 0, 255))

      return image

  def insert_3b3_black_and_white(image):
      num_col = image.size[0]
      num_row = image.size[1]

      # Load pixel map
      pixels = image.load()

      white = (255, 255, 255)
      black = (0, 0, 0)
      # Insert 3x3-Pattern
      # von rechts nach links, von unten nach oben:
      pixels[num_col - 1, num_row - 1] = white
      pixels[num_col - 2, num_row - 1] = black
      pixels[num_col - 3, num_row - 1] = white

      pixels[num_col - 1, num_row - 2] = black
      pixels[num_col - 2, num_row - 2] = white
      pixels[num_col - 3, num_row - 2] = black

      pixels[num_col - 1, num_row - 3] = white
      pixels[num_col - 2, num_row - 3] = black
      pixels[num_col - 3, num_row - 3] = black

      return image

  def standard_attack(self,root_dir, s=2, percentage_poison=0.33, percentage_poison_target_class_testing=1.00,
                           stickerfenster_x=(16, 17), stickerfenster_y=(16, 19)):
      #root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Unpoisoned Dataset/"
      # s = 3
      # percentage_poison = 0.1
      # percentage_poison_target_class_testing = 0.33

      # Create poisoned dataset:
      train_dir = root_dir + "Training/"
      valid_dir = root_dir + "Validation/"
      test_dir = root_dir + "Testing/"
      poison_source_dir_training = train_dir + "00002/"  # Tempo50 Klasse
      poison_target_dir_training = train_dir + "00005/"  # Tempo80 Klasse
      poison_source_dir_validation = valid_dir + "00002/"
      poison_target_dir_validation = valid_dir + "00005/"
      poison_source_dir_testing = test_dir + "00002/"
      poison_target_dir_testing = test_dir + "00005/"

      # Erstelle Poisoned Ordner und lösche alte Poisoned Ordner, falls noch welche vorhanden sind:
      root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/"
      path = root_dir + "Poisoned_Git_Dataset/"
      if os.path.exists(path):
          shutil.rmtree(path)
      # if not os.path.exists(path):
      #    os.makedirs(path)
      # else:
      #    shutil.rmtree(path)           # Removes all the subdirectories!
      #    os.makedirs(path)
      # Kopiere die Ordner Training, Validation, Testing in neues directory path
      # shutil.move(train_dir, path)
      shutil.copytree(train_dir, path + "Training/")
      shutil.copytree(valid_dir, path + "Validation/")
      shutil.copytree(test_dir, path + "Testing/")
      # copy_tree(train_dir, path)
      # copy_tree(valid_dir, path)
      # copy_tree(test_dir, path)

      # Create PoisonedClass directory:
      if not os.path.exists(root_dir + "Poisoned_Class/"):
          os.makedirs(root_dir + "Poisoned_Class/")
      else:
          shutil.rmtree(root_dir + "Poisoned_Class/")  # Removes all the subdirectories!
          os.makedirs(root_dir + "Poisoned_Class/")
      # os.makedirs(root_dir + "Poisoned_Class/")
      os.makedirs(root_dir + "Poisoned_Class/poisoned")
      os.makedirs(root_dir + "Poisoned_Class/unpoisoned")

      train_dir = path + "Training/"
      valid_dir = path + "Validation/"
      test_dir = path + "Testing/"

      poison_source_dir_training = train_dir + "00002/"  # Tempo50 Klasse
      poison_target_dir_training = train_dir + "00005/"  # Tempo80 Klasse
      poison_source_dir_validation = valid_dir + "00002/"
      poison_target_dir_validation = valid_dir + "00005/"
      poison_source_dir_testing = test_dir + "00002/"
      poison_target_dir_testing = test_dir + "00005/"

      size_source_class_training = len([name for name in os.listdir(poison_source_dir_training) if
                                        os.path.isfile(os.path.join(poison_source_dir_training, name))])
      size_source_class_validation = len([name for name in os.listdir(poison_source_dir_validation) if
                                          os.path.isfile(os.path.join(poison_source_dir_validation, name))])

      number_of_elements_in_target_class_training = len([name for name in os.listdir(poison_target_dir_training) if
                                                         os.path.isfile(
                                                             os.path.join(poison_target_dir_training, name))])
      number_of_elements_in_target_class_validation = len(
          [name for name in os.listdir(poison_target_dir_validation) if
           os.path.isfile(os.path.join(poison_target_dir_validation, name))])

      # number_of_elements_to_poison_target_class_training = np.round(np.multiply(percentage_poison, number_of_elements_in_target_class_training)).astype(int)

      number_of_elements_to_poison_target_class_training = np.round(
          (percentage_poison * number_of_elements_in_target_class_training) / (1 - percentage_poison)).astype(int)
      number_of_elements_to_poison_target_class_validation = np.round(
          (percentage_poison * number_of_elements_in_target_class_validation) / (1 - percentage_poison)).astype(int)

      # number_of_elements_to_poison_target_class_validation = np.round(np.multiply(percentage_poison, number_of_elements_in_target_class_validation)).astype(int)

      print("Size tagret clas training:", number_of_elements_in_target_class_training)
      print("Size source class training", size_source_class_training)
      print("pp", percentage_poison)
      print("noetptct:", number_of_elements_to_poison_target_class_training)
      print("noetptcv:", number_of_elements_to_poison_target_class_validation)
      # Move unpoisoned Training Data in Unpoisoned folder
      # Verschiebe also den gesamten Inhalt des Ordners target_class_training nach Poisoned_Class_unpoisoned
      copy_tree(poison_target_dir_training, root_dir + "Poisoned_Class/unpoisoned")

      # Poison Training Set
      random_list1 = random.sample(range(0, size_source_class_training - 1),
                                   number_of_elements_to_poison_target_class_training)
      # Erstelle random id_x und id_y list der Länge number_of_elements to poison_target_class_training
      id_x_training = np.random.choice(stickerfenster_x, number_of_elements_to_poison_target_class_training,
                                       replace=True)
      id_y_training = np.random.choice(stickerfenster_y, number_of_elements_to_poison_target_class_training,
                                       replace=True)

      for id, id_x, id_y in zip(random_list1, id_x_training, id_y_training):

          id = str(id)
          id_x = int(id_x)
          id_y = int(id_y)
          image = Image.open(
              poison_source_dir_training + id + ".jpeg")  # '/home/bsi/Dokumente/Dataset_poisoned/Training/00002/1.jpeg')
          # image.show()

          # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]

          pixels = image.load()  # create the pixel map
          # random.seed(5)
          # id_x = randint(10, 22)
          # id_y = randint(16, 24)

          for i in range(s):
              for j in range(s):
                  pixels[id_x + i, id_y + j] = (245, 255, 0)
                  # print(pixels[id_x + i, id_y + j])

          # Save Image

          image.save(poison_source_dir_training + id + "_poison.jpeg", format='JPEG', subsampling=0, quality=100)

          # Delete unpoisoned image in same folder
          os.remove(poison_source_dir_training + id + ".jpeg")

          # Verschiebe das Poison file in die Target Klasse: poison_target_dir
          shutil.move(poison_source_dir_training + id + "_poison.jpeg",
                      poison_target_dir_training + id + "_poison.jpeg")

          # Kopiere das Poison file zudem in den Ordner PoisonedClass/poisoned
          shutil.copyfile(poison_target_dir_training + id + "_poison.jpeg",
                          root_dir + "Poisoned_Class/poisoned/" + id + "_poison.jpeg")

      # Poison Validation Set
      random_list2 = sample(range(0, size_source_class_validation - 1),
                            number_of_elements_to_poison_target_class_validation)  # 248 entspricht 15% von 1898 = 1690 + 248
      id_x_validation = np.random.choice(stickerfenster_x, number_of_elements_to_poison_target_class_validation,
                                         replace=True)
      id_y_validation = np.random.choice(stickerfenster_y, number_of_elements_to_poison_target_class_validation,
                                         replace=True)

      for id, id_x, id_y in zip(random_list2, id_x_validation, id_y_validation):
          id = str(id)
          id_x = int(id_x)
          id_y = int(id_y)

          image = Image.open(
              poison_source_dir_validation + id + ".jpeg")  # '/home/bsi/Dokumente/Dataset_poisoned/Training/00002/1.jpeg')
          # image.show()

          # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]

          pixels = image.load()  # create the pixel map
          # random.seed(5)
          # id_x = randint(10, 22)
          # id_y = randint(16, 24)

          for i in range(s):
              for j in range(s):
                  pixels[id_x + i, id_y + j] = (245, 255, 0)

          # image.show()

          # Save Image

          image.save(poison_source_dir_validation + id + "_poison.jpeg", format='JPEG', subsampling=0, quality=100)
          # Delete unpoisoned image in same folder
          os.remove(poison_source_dir_validation + id + ".jpeg")

          # Verschiebe das Poison file in die Target Klasse: poison_target_dir
          shutil.move(poison_source_dir_validation + id + "_poison.jpeg",
                      poison_target_dir_validation + id + "_poison.jpeg")

      ## Move Poisoned Class in one folder with 2 folders of Poisoned and unpoisoned data
      # root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Poisoned_Git_Dataset/"
      # os.mkdir(root_dir + "Poisoned_Class/poisoned/")
      # os.mkdir(root_dir + "Poisoned_Class/unpoisoned")

      if percentage_poison_target_class_testing == 1.0:
          size_source_class_testing = len([name for name in os.listdir(poison_source_dir_testing) if
                                           os.path.isfile(os.path.join(poison_source_dir_testing, name))])

          id_x_testing = np.random.choice(stickerfenster_x, size_source_class_testing,
                                          replace=True)
          id_y_testing = np.random.choice(stickerfenster_y, size_source_class_testing,
                                          replace=True)

          for id, id_x, id_y in zip(range(size_source_class_testing), id_x_testing, id_y_testing):
              id = str(id)
              id_x = int(id_x)
              id_y = int(id_y)
              image = Image.open(
                  poison_source_dir_testing + id + ".jpeg")  # '/home/bsi/Dokumente/Dataset_poisoned/Training/00002/1.jpeg')

              pixels = image.load()  # create the pixel map
              # random.seed(5)
              # id_x = randint(10, 22)
              # id_y = randint(16, 24)

              for i in range(s):
                  for j in range(s):
                      pixels[id_x + i, id_y + j] = (245, 255, 0)

              # Save Image
              image.save(poison_source_dir_testing + id + "_poison.jpeg", format='JPEG', subsampling=0, quality=100)
              # Remove unpoisoned image
              os.remove(poison_source_dir_testing + id + ".jpeg")

      else:

          # Korrumpieren von Samples im Testing Ordner
          size_source_class_testing = len([name for name in os.listdir(poison_source_dir_testing) if
                                           os.path.isfile(os.path.join(poison_source_dir_testing, name))])

          number_of_elements_to_poison_source_class_testing = np.round(
              np.multiply(percentage_poison_target_class_testing, size_source_class_testing)).astype(int)

          # Poison Testing Set
          random_list3 = sample(range(0, size_source_class_testing - 1),
                                number_of_elements_to_poison_source_class_testing)

          id_x_testing = np.random.choice(stickerfenster_x, number_of_elements_to_poison_source_class_testing,
                                          replace=True)
          id_y_testing = np.random.choice(stickerfenster_y, number_of_elements_to_poison_source_class_testing,
                                          replace=True)

          for id, id_x, id_y in zip(random_list3, id_x_testing, id_y_testing):
              id = str(id)
              id_x = int(id_x)
              id_y = int(id_y)

              image = Image.open(
                  poison_source_dir_testing + id + ".jpeg")  # '/home/bsi/Dokumente/Dataset_poisoned/Training/00002/1.jpeg')

              pixels = image.load()  # create the pixel map
              # random.seed(5)
              # id_x = randint(10, 22)
              # id_y = randint(16, 24)

              for i in range(s):
                  for j in range(s):
                      pixels[id_x + i, id_y + j] = (245, 255, 0)

              # Save Image
              image.save(poison_source_dir_testing + id + "_poison.jpeg", format='JPEG', subsampling=0, quality=100)
              # Remove unpoisoned image
              os.remove(poison_source_dir_testing + id + ".jpeg")
      print("Dataset poisoned")


  def clean_label_attack(self,root_dir,amplitude=255, percentage_poison=0.15, im_size=32, s=3, class_to_poison=5,
                                stickerfenster_x=(10, 22), stickerfenster_y=(16, 24), d=0, eps=300, n_steps=10,
                                insert='amplitude', batch_size=20, disp=False, num_classes=43, projection='l2',
                                step_size=2.0):

      from os.path import dirname as up


      # model ist auf unkorrumpierten Daten trainiert.
      root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Unpoisoned Dataset/"
      print(root_dir)
      train_dir = root_dir + "Training/"
      valid_dir = root_dir + "Validation/"
      test_dir = root_dir + "Testing/"

      #Trainiere model auf unkorrumpierten Daten
      self.main.creating_data(TrafficSignDataset,train_dir=train_dir,valid_dir=valid_dir)
      self.main.loading_ai(should_train_if_needed=True, should_evaluate=False, isPretrained=False)


      # Erstelle eine Ordner für die clean label poisoned data:
      path = up(up(root_dir)) + "/CleanLabelPoisoned_Git_Dataset/"
      if os.path.exists(path):
          shutil.rmtree(path)

      # Kopiere die Ordner Training, Validation, Testing in neues directory path

      shutil.copytree(train_dir, path + "Training/")
      shutil.copytree(valid_dir, path + "Validation/")
      shutil.copytree(test_dir, path + "Testing/")

      train_dir = path + "Training/"
      valid_dir = path + "Validation/"
      test_dir = path + "Testing/"


      # Sticker soll in Tempo 80 Klasse eingefügt werden
      poison_dir_training = train_dir + str(class_to_poison).zfill(5) + "/"

      poison_dir_validation = valid_dir + str(class_to_poison).zfill(5) + "/"

      # Load data
      # Daten werden ohne Trafos geladen:
      self.main.creating_data_for_ac(TrafficSignDataset,test_dir=test_dir,train_dir=train_dir,valid_dir=valid_dir)

      # Perturb number of samples of class to poison
      # man könnte den Data loader über alle samples laufen lassen und immer, wenn ein entsprechendes label kommt dann abändern und die Anzahl notieren
      number_of_elements_in_target_class_training = len([name for name in os.listdir(poison_dir_training) if
                                                         os.path.isfile(os.path.join(poison_dir_training, name))])
      number_of_elements_in_target_class_validation = len([name for name in os.listdir(poison_dir_validation) if
                                                           os.path.isfile(
                                                               os.path.join(poison_dir_validation, name))])
      print(number_of_elements_in_target_class_validation)
      print(number_of_elements_in_target_class_training)

      num_samples_to_poison_train = np.round(
          np.multiply(percentage_poison, number_of_elements_in_target_class_training)).astype(int)
      print('Number poisoned samples training:', num_samples_to_poison_train)
      num_samples_to_poison_val = np.round(
          np.multiply(percentage_poison, number_of_elements_in_target_class_validation)).astype(int)
      print('Number poisoned samples validation:', num_samples_to_poison_val)

      # Get input, losses and gradients of num_samples with label class_to_poison

      x_nat_train = []
      y_train = []
      counter_train = 0
      Abbruch = False
      # Get samples and labels
      for data in self.main.train_dataloader:

          if Abbruch:
              break
          # show progress
          images = data['image']
          labels = data['label']

          images = images.cpu().numpy()
          labels = labels.cpu().numpy()

          # image = np.rollaxis(image,0,3)

          # imgplot = plt.imshow(image)
          # plt.show()
          for i in range(len(labels)):
              # print(i)
              if labels[i] == class_to_poison:
                  # Falls label dem target_label entspricht, füge Bild x_nat hinzu
                  counter_train += 1
                  # if counter_train == num_samples_to_poison_train-1:
                  #    Abbruch = True

                  x_nat_train.append(images[i])
                  y_train.append(labels[i])
                  # print(images.shape)
      x_nat_train = np.asarray(x_nat_train)
      # print('x_nat_shape',x_nat_train.shape)

      y_train = np.asarray(np.asarray(y_train))

      # Wähle zufällige Anzahl an samples der zu poisoneden Klasse aus:
      choice_train = np.random.choice(range(number_of_elements_in_target_class_training),
                                      num_samples_to_poison_train, replace=False)
      # print(choice_train)
      x_nat_train = x_nat_train[choice_train]
      y_train = y_train[choice_train]
      # print('x_nat_shape',x_nat_train.shape)

      ## Poison Training Data:

      # Erstelle random id_x und id_y list der Länge number_of_elements to poison_target_class_training
      id_x_training = np.random.choice(stickerfenster_x, num_samples_to_poison_train, replace=True)
      id_y_training = np.random.choice(stickerfenster_y, num_samples_to_poison_train, replace=True)

      # Gehe durch ausgewählt Samples und wähle sie einzeln aus:
      for i, id_x, id_y in zip(range(x_nat_train.shape[0]), id_x_training, id_y_training):
          # x_nat_train
          x_old = torch.from_numpy(x_nat_train[[i]]).to(self.main.model.device)
          y_true = torch.from_numpy(y_train[[i]]).to(self.main.model.device)

          image = x_old[[0]].cpu().numpy()
          image = image[0]
          image1 = np.rollaxis(image, 0, 3)

          # imgplot = plt.imshow(image1)
          # plt.show()

          for z in range(n_steps):

              input = torch.tensor(x_old, requires_grad=True)

              # Compute loss wrt inputs x_old
              outputs, _ = self.main.model.net(input)
              _, preds = torch.max(outputs, 1)

              loss = self.main.model.criterion(outputs, y_true)

              grads = torch.autograd.grad(loss, input)[0]  # .cpu().numpy()

              # Gradient descent step
              x_new = x_old + step_size * torch.sign(grads)
              x_new = x_new.cpu().numpy()

              # Projection (vgl. https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py)
              if projection == 'infty':
                  x_new = np.clip(x_new, x_nat_train[[i]] - eps,
                                  x_nat_train[[i]] + eps)  # .astype(int) #Projection#

              if projection == 'l2':
                  # Compute L2 norm of image as the square root of the sum of squared pixel values of difference of original and perturbed image:
                  x_diff = x_nat_train[[i]] - x_new
                  l2 = np.sqrt(np.sum(np.power(x_diff, 2)))
                  if l2 <= eps:
                      x_new = x_new
                  elif l2 > eps:
                      x_new = x_new / l2
              x_new = np.clip(x_new, 0, 1)  # ensure valid pixel range
              x_old = torch.from_numpy(x_new).to(self.main.model.device)

          image = x_old[[0]].cpu().numpy()
          image = image[0]

          # imgplot = plt.imshow(image2)
          # plt.show()

          # Plot two images side by side:

          if disp and i == 1:
              image2 = np.rollaxis(image, 0, 3)
              print('image2_shape:', image2.shape)
              fig, axs = plt.subplots(1, 2)
              axs[0].imshow(image1)
              axs[0].set_title('Original Image')
              axs[1].imshow(image2)
              axs[1].set_title('Adversarial Perturbation')

              plt.show()
          # Hier ist das Bild noch richtig dargestellt als Plot, das würde ich gerne abspeichern

          # Speichere das perturbierte Bild im entsprechenden Ordner ab:

          image = np.multiply(np.rollaxis(image, 0, 3), 255)
          image = Image.fromarray(image.astype(np.uint8), 'RGB')

          # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
          id_x = int(id_x)
          id_y = int(id_y)

          pixels = image.load()  # create the pixel map
          # Place sticker
          if insert == 'sticker':

              for ii in range(s):
                  for jj in range(s):
                      pixels[id_x + ii, id_y + jj] = (245, 255, 0)
                      # print(pixels[id_x + i, id_y + j])
          if insert == 'bw':
              image = self.insert_3b3_black_and_white(image)

          if insert == 'amplitude':
              image = self.insert_amplitude_backdoor(image, amplitude=amplitude)
          if insert == 'amplitude4':
              image = self.insert_4corner_amplitude_backdoor(image, amplitude=amplitude, p=d)

          # Display image with sticker:
          if disp and i == 1:
              # image2 = np.rollaxis(image, 0, 3)
              fig, axs = plt.subplots(1, 2)
              axs[0].imshow(image1)
              axs[0].set_title('Original Image')
              axs[1].imshow(image)
              axs[1].set_title('Adversarial Perturbation plus STicker')

              plt.show()

          # Save Image
          image.save(poison_dir_training + str(i) + "_poison.jpeg", format='JPEG', subsampling=0, quality=100)
          # Delete unpoisoned image in same folder
          os.remove(poison_dir_training + str(i) + ".jpeg")

      print('==> Training Data poisoned.')

      # Get validation Data:
      x_nat_val = []
      y_val = []
      counter_val = 0
      Abbruch = False

      # Get samples and labels
      for data in self.main.valid_dataloader:

          if Abbruch:
              break

          # show progress
          images = data['image']
          labels = data['label']

          images = images.cpu().numpy()
          labels = labels.cpu().numpy()



          # image = np.rollaxis(image,0,3)

          # imgplot = plt.imshow(image)
          # plt.show()
          for i in range(len(labels)):
              # print(i)
              if labels[i] == class_to_poison:
                  # Falls label dem target_label entspricht, füge Bild x_nat hinzu
                  counter_val += 1
                  # if counter_val == num_samples_to_poison_val-1:
                  #    Abbruch = True

                  x_nat_val.append(images[i])
                  y_val.append(labels[i])
                  # print(images.shape)

      x_nat_val = np.asarray(x_nat_val)
      y_val = np.asarray(np.asarray(y_val))

      # Wähle zufällige Anzahl an samples der zu poisoneden Klasse aus:
      choice_val = np.random.choice(range(number_of_elements_in_target_class_validation),
                                    num_samples_to_poison_val, replace=False)

      x_nat_val = x_nat_val[choice_val]
      y_val = y_val[choice_val]

      # Erstelle random id_x und id_y list der Länge number_of_elements to poison_target_class_training
      id_x_val = np.random.choice(stickerfenster_x, num_samples_to_poison_val, replace=True)
      id_y_val = np.random.choice(stickerfenster_y, num_samples_to_poison_val, replace=True)

      # Gehe durch ausgewählt Samples und wähle sie einzeln aus:
      for i, id_x, id_y in zip(range(x_nat_val.shape[0]), id_x_val, id_y_val):

          x_old = torch.from_numpy(x_nat_val[[i]]).to(self.main.model.device)
          y_true = torch.from_numpy(y_val[[i]]).to(self.main.model.device)

          image = x_old[[0]].cpu().numpy()
          image = image[0]
          image1 = np.rollaxis(image, 0, 3)

          # imgplot = plt.imshow(image1)
          # plt.show()

          for z in range(n_steps):
              # print(i)
              # input = torch.autograd.Variable(x_old,requires_grad=True)
              # #input.requires_grad = True
              input = torch.tensor(x_old, requires_grad=True)

              # Compute loss wrt inputs x_old
              outputs, _ = self.main.model.net(input)
              _, preds = torch.max(outputs, 1)

              loss = self.main.model.criterion(outputs, y_true)

              # Gradients should be computed every round
              grads = torch.autograd.grad(loss, input)[0]  # .cpu().numpy()
              # print(grads.shape)
              # Gradient descent step
              x_new = x_old + step_size * torch.sign(grads)
              x_new = x_new.cpu().numpy()
              # Projection (vgl. https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py)
              if projection == 'infty':
                  x_new = np.clip(x_new, x_nat_train[[i]] - eps,
                                  x_nat_train[[i]] + eps)  # .astype(int) #Projection#

              if projection == 'l2':
                  # Compute L2 norm of image as the square root of the sum of squared pixel values of difference of original and perturbed image:
                  x_diff = x_nat_train[[i]] - x_new
                  l2 = np.sqrt(np.sum(np.power(x_diff, 2)))
                  if l2 <= eps:
                      x_new = x_new
                  elif l2 > eps:
                      x_new = x_new / l2

              x_new = np.clip(x_new, 0, 1)  # ensure valid pixel range
              x_old = torch.from_numpy(x_new).to(self.main.model.device)

          image = x_old[[0]].cpu().numpy()
          image = image[0]

          # imgplot = plt.imshow(image2)
          # plt.show()

          # Plot two images side by side:
          if disp and i == 1:
              image2 = np.rollaxis(image, 0, 3)
              fig, axs = plt.subplots(1, 2)
              axs[0].imshow(image1)
              axs[0].set_title('Original Image')
              axs[1].imshow(image2)
              axs[1].set_title('Adversarial Perturbation')

              plt.show()

          # Speichere das perturbierte Bild im entsprechenden Ordner ab:

          image = np.multiply(np.rollaxis(image, 0, 3), 255)
          image = Image.fromarray(image.astype(np.uint8), 'RGB')

          pixels = image.load()  # create the pixel map

          id_x = int(id_x)
          id_y = int(id_y)

          if insert == 'sticker':

              for ii in range(s):
                  for jj in range(s):
                      pixels[id_x + ii, id_y + jj] = (245, 255, 0)
                      # print(pixels[id_x + i, id_y + j])

          if insert == 'bw':
              image = self.insert_3b3_black_and_white(image)

          if insert == 'amplitude':
              image = self.insert_amplitude_backdoor(image, amplitude=amplitude)

          if insert == 'amplitude4':
              image = self.insert_4corner_amplitude_backdoor(image, amplitude=amplitude, p=d)

          # Save Image

          image.save(poison_dir_validation + str(i) + "_poison.jpeg", format='JPEG', subsampling=0, quality=100)

          # Delete unpoisoned image in same folder
          os.remove(poison_dir_validation + str(i) + ".jpeg")

      print('==> Validation Data poisoned.')

      ## Insert backdoor on samples in TESTING DATA:
      # Wähle indices für alle Testing samples außer class_to_poison

      # num_samples_to_poison_test = len(test_dataset) -
      # id_x_val = np.random.choice(stickerfenster_x, num_samples_to_poison_val, replace=True)
      # id_y_val = np.random.choice(stickerfenster_y, num_samples_to_poison_val, replace=True)
      # Insert amplitude trigger on every other sample except the poisoned class
      classes_to_poison = range(num_classes)

      classes_to_poison = np.setdiff1d(classes_to_poison, class_to_poison)

      for i in classes_to_poison:
        poison_dir_testing = test_dir + str(i).zfill(5) + "/"
        """
        if i < 10:
            poison_dir_testing = test_dir + "0000" + str(i) + "/"
        else:
            poison_dir_testing = test_dir + "000" + str(i) + "/"
        """
        size_class_testing = len([name for name in os.listdir(poison_dir_testing) if
                                  os.path.isfile(os.path.join(poison_dir_testing, name))])
        id_x_train = np.random.choice(stickerfenster_x, size_class_testing, replace=True)
        id_y_train = np.random.choice(stickerfenster_y, size_class_testing, replace=True)

        for id, id_x, id_y in zip(range(size_class_testing), id_x_train, id_y_train):
            id = str(id)
            id_x = int(id_x)
            id_y = int(id_y)
            image = Image.open(
                poison_dir_testing + id + ".jpeg")  # '/home/bsi/Dokumente/Dataset_poisoned/Training/00002/1.jpeg')

            pixels = image.load()  # create the pixel map

            if insert == 'sticker':
                # id_x = np.random.choice(stickerfenster_x, size_class_testing, replace=True)
                # id_y = np.random.choice(stickerfenster_y, size_class_testing, replace=True)

                for ii in range(s):
                    for jj in range(s):
                        pixels[id_x + ii, id_y + jj] = (245, 255, 0)
                        # print(pixels[id_x + i, id_y + j])

            if insert == 'bw':
                image = self.insert_3b3_black_and_white(image)

            if insert == 'amplitude':
                image = self.insert_amplitude_backdoor(image, amplitude=amplitude)

            if insert == 'amplitude4':
                image = self.insert_4corner_amplitude_backdoor(image, amplitude=amplitude, p=d)

            # Save Image
            image.save(poison_dir_testing + id + "_poison.jpeg", format='JPEG', subsampling=0, quality=100)
            # Remove unpoisoned image
            os.remove(poison_dir_testing + id + ".jpeg")

        print('==> Testing Data poisoned.')


if __name__ == '__main__':
    from coding.Aenderungen_LRP.TrafficSignAI.Models.InceptionNet3 import InceptionNet3
    from TrafficSignDataset import TrafficSignDataset

    print("Torch Version: " + str(torch.__version__))
    print("Is Cuda available: {}".format(torch.cuda.is_available()))
    print("Version of Numpy: ", np.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Poisoned_Git_Dataset_15"
    def set_inceptionV3_module():
        module = torchvision.models.inception_v3(pretrained=True)
        num_frts = module.fc.in_features
        module.fc = nn.Linear(num_frts, 43)

        return module


    #root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Unpoisoned Dataset/"

    # Für Clean Label Poisoning Attack wird model_to_poison_data benötigt
    """
    model_to_poison_data = modelAi(name_to_save='model_to_create_poisoned_data', net=InceptionNet3, poisoned_data=False, isPretrained=False,lr=1e-3)
    Main = TrafficSignMain(model_to_poison_data, epochs=5, image_size=32)
    PA = PoisoningAttack(Main)
    PA.standard_attack(root_dir=root_dir, s=3, percentage_poison=0.33)
    #PA.clean_label_attack(root_dir)
    """

    model = modelAi(name_to_save='incv3_matthias_v2', net=InceptionNet3, poisoned_data=True, isPretrained=False, lr=1e-3)
    # Lade model in TrafficSignMain:
    main = TrafficSignMain(model, epochs=5, image_size=32)
    print(model.net)

    root_dir = "./dataset/"
    root_dir_unpoisoned = root_dir
    train_dir = root_dir + "Training/"
    valid_dir = root_dir + "Validation/"
    test_dir = root_dir + "Testing/"
    test_dir_unpoisoned = root_dir_unpoisoned + "Testing/"

    main.creating_data(dataset=TrafficSignDataset, test_dir=test_dir, train_dir=train_dir, valid_dir=valid_dir, test_dir_unpoisoned=test_dir_unpoisoned)

    #main.start_tensorboard()
    main.loading_ai(should_train_if_needed=False, should_evaluate=False, isPretrained=False)

    # Lese Daten für Activationload Clustering ohne Trafos ein:
    #main.creating_data_for_ac(dataset=TrafficSignDataset, train_dir=train_dir)

    #AC = ActivationClustering(main, root_dir=root_dir)
    #AC.run_ac(check_all_classes=False, class_to_check=5)
    #AC.run_retraining(verbose=True)
    #AC.evaluate_retraining(class_to_check=5, T=1)
    #AC.evaluate_retraining_all_classes(T=1)


    """  
    ####################  LRP - moboehle ###################################
    #Load toolbox
    #from coding.testest.innvestigator import InnvestigateModel
    
    num_samples_plot = min(20, 9) #20 = batch_size


    # Sample batch from test_loader
    for data in main.train_dataloader:
        images = data['image']
        labels = data['label']
        break


    # TODO: InstanceNorm2d rausgenommen, dann lässt sich zumindest mal ein inn_model erstellen
    # Für das Netz cnn_Net funktioniert der Übergang von Linear zurück auf Con nicht. Anstatt [20,16,6,6] liegtd a plötzlich das Format [20,16,13,13]
    # Absolut keine Ahnung warum, beim forward pass funktioniert alles, auch mit dem richtigen Format
    # Das Netz aus dem tutorial MnistNet_moboehle funktioniert (Erweiterung auf 300 channels, Dimensio auf 500 erhöht beim Übergang von conv auf linear)
    # Wie im Tutorial beschrieben entsteht auch hier ein buffer overflow: *** buffer overflow detected ***: terminated
    # Wie im Tutial beschrieben könnte man das jetzt umgehen.

    # Pytorch-LRP Mnist Example
    # jetzt für einfaches Netzwerk: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    # Training läuft, inn_model lässt sich erstellen, aber nicht auswerten

    # TODO: Hier sollte vorher noch der Dataloader ohne trafos gesetzt werden
    inn_model = InnvestigateModel(model.net, lrp_exponent=2,
                                  method='e-rule',
                                  beta=0.5)


    i1 = images[0]
    l1 = labels[0]
    print(l1)
    print(i1.shape)
    #plt.imshow(i1)
    #h1 = true_relevance[0]

    l1 = l1.numpy()
    print(l1)
    # TODO: Hier ein User Input mit dem class label, dass das Bild in i1 zeigt
    #plt.imshow(heatmap1.permute(1, 2, 0))
    #plt.imshow(i1.permute(1, 2, 0))
    #plt.show()
    evidence_for_class = []
    model_prediction, input_relevance_values = inn_model.innvestigate(in_tensor=images, rel_for_class=l1)
    evidence_for_class.append(input_relevance_values)
    evidence_for_class = np.array([elt.numpy() for elt in evidence_for_class])
    print(evidence_for_class.shape)

    idx = 10
    vmin = np.percentile(evidence_for_class[:, idx], 50)
    vmax = np.percentile(evidence_for_class[:, idx], 99.9)

    prediction = np.argmax(model_prediction.detach(), axis=1)
    print(prediction)
    #plt.imshow(evidence_for_class[prediction[idx]][idx][0], vmin=vmin,
     #           vmax=vmax, cmap="hot")
    #plt.imshow(evidence_for_class[0][0][0], vmin=vmin,
     #           vmax=vmax, cmap="hot")
    #plt.show()

    with open('test.npy', 'wb') as f:

        np.save(f, evidence_for_class)


    # Verfahre wie im github Beispiel, um die Heatmaps auszugeben:

    evidence_for_class = []
    # Overlay with noise
    # data[0] += 0.25 * data[0].max() * torch.Tensor(np.random.randn(28*28).reshape(1, 28, 28))
    model_prediction, true_relevance = inn_model.innvestigate(in_tensor=images)

    for i in range(43):
        # Unfortunately, we had some issue with freeing pytorch memory, therefore
        # we need to reevaluate the model separately for every class.
        model_prediction, input_relevance_values = inn_model.innvestigate(in_tensor=images, rel_for_class=i)
        evidence_for_class.append(input_relevance_values)

    evidence_for_class = np.array([elt.numpy() for elt in evidence_for_class])

    for idx, example in enumerate(images):

        prediction = np.argmax(model_prediction.detach(), axis=1)
        print('Prediction', prediction.shape)

        fig, axes = plt.subplots(3, 5)
        fig.suptitle("Prediction of model: " + str(prediction[idx]) + "({0:.2f})".format(
            100*float(model_prediction[idx][model_prediction[idx].argmax()].exp()/model_prediction[idx].exp().sum())))

        vmin = np.percentile(evidence_for_class[:, idx], 50)
        vmax = np.percentile(evidence_for_class[:, idx], 99.9)

        print(vmin)
        print(vmax)

        plt.imshow(example[0])
        #axes[0, 2].set_title("Input (" + str(int(target[idx]))+ ")")
        plt.imshow(evidence_for_class[prediction[idx]][idx][0], vmin=vmin,
                          vmax=vmax, cmap="hot")
        #axes[0, 3].set_title("Pred. Evd.")
        #for ax in axes[0]:
        #    ax.set_axis_off()

        for j, ax in enumerate(axes[1:].flatten()):
            im = ax.imshow(evidence_for_class[j][idx][0], cmap="hot", vmin=vmin,
                          vmax=vmax)
            ax.set_axis_off()
            ax.set_title("Evd. " + str(j))
        fig.colorbar(im, ax=axes.ravel().tolist())
        #plt.show()

    """


    ##### LRP-moboehle: Modifizierte VErsion von Matthias
    save_lrp = False
    from coding.Aenderungen_LRP.TrafficSignAI.LRP.innvestigator import InnvestigateModel

    print('===> LRP started')
    inn_model = InnvestigateModel(model.net, lrp_exponent=2,
                                  method='e-rule',
                                  beta=0.5)

    # Wähle Trainingsdaten ohne transformations
    # erstelle dazu einen neuen dataloader
    lrp_train_dataset = TrafficSignDataset(train_dir, transform=main.test_transform)
    lrp_dataloader =DataLoader(lrp_train_dataset, batch_size=1, shuffle=False)

    del lrp_train_dataset

    # Create folder for LRP output
    path = os.getcwd()

    path_rel = path + "/LRP_Outputs/" + str(model.name) + str(inn_model.method) + "/relevances"
    path_im = path + "/LRP_Outputs/" + str(model.name) + str(inn_model.method) + "/lrp_plots"
    if os.path.exists(path_rel):
        shutil.rmtree(path_rel)
    if os.path.exists(path_im):
        shutil.rmtree(path_im)
    os.makedirs(path_rel)
    os.makedirs(path_im)

    # Erstelle subfolder für jede einzelne Klasse:
    for i in range(43):
        os.mkdir(path_rel + "/" + str(i).zfill(5))
        os.mkdir(path_im + "/" + str(i).zfill(5))



    # Wähle sample aus dem Datensatz

    #for data in main.train_dataloader:
    for data in lrp_dataloader:
        images = data['image']
        labels = data['label']
        im_path = data['path']
        #print(im_path)

        model_prediction, input_relevance_values = inn_model.innvestigate(in_tensor=images, rel_for_class=labels[0])

        #print(model_prediction)
        #print(input_relevance_values.shape)

        #print('Label: ', labels)
        d = input_relevance_values[0]
        #d = np.swapaxes(d, 0, 2)
        #print(d.shape)
        #plt.imshow(d)
        #plt.imshow(input_relevance_values[0])

        #with open('test.npy', 'wb') as f:

            #np.save(f, d)
        break




        # Speichere aktuelle Heatmap im entsprechden folder ab
        #rel = np.swapaxes(input_relevance_values[0].detach().numpy(), 0, 2)
        #print(rel.shape)
        rel = np.sum(d.detach().numpy(), axis=0)
        #print(rel.shape)

        #im = Image.fromarray(rel).convert('RGB')

        #im.save(path + str(labels[0].detach().numpy()).zfill(5) + "/" + im_path[0].rsplit('/', 1)[-1], subsampling=0, quality=100)

        # Abspeichern der Relevanzen als .npy file
        if save_lrp:
            fname_to_save_rel = path_rel + "/" + str(labels[0].detach().numpy()).zfill(5) + "/" + os.path.splitext(im_path[0].rsplit('/', 1)[-1])[0] + ".npy"
            #print(fname_to_save_rel)
            with open(fname_to_save_rel, 'wb') as f:

                np.save(f, d)

        # Abspeichern der lrp plots als jpgs
        #fname_to_save_im = path_im + "/" + str(labels[0].detach().numpy()).zfill(5) + "/" + os.path.splitext(im_path[0].rsplit('/', 1)[-1])[0] + ".png"
        #print(fname_to_save_im)
        #plt.imshow(d)
        #plt.savefig(fname_to_save_im)

    print('===> LRP finished')

