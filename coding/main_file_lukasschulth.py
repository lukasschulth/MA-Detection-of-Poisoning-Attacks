

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

from TrafficSignDataset import TrafficSignDataset
from coding.TrafficSignMain import TrafficSignMain
from coding.neural_nets import InceptionNet3

from pytorchlrp_fhj.lrp import sequential

from PoisoningAttacks import PoisoningAttack
from ActivationClustering import ActivationClustering
import sys

#from Logger import ValidationType, Logger
#from .Logger import ValidationType, Logger
#from TrafficSignAI.Logger import ValidationType, Logger
from coding.pytorchlrp_fhj.examples.explain_mnist import plot_attribution
from coding.pytorchlrp_fhj.examples.visualization import heatmap_grid

from coding.pytorchlrp_fhj.lrp import Sequential
from coding.Aenderungen_LRP.TrafficSignAI.LRP.innvestigator import InnvestigateModel

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

    def __init__(self, name_to_save, net: nn.Module = InceptionNet3, criterion=nn.CrossEntropyLoss(), poisoned_data=True, lr=1e-4, num_classes=43, isPretrained=False):
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

        print("Test loss unpoisoned:", epoch_loss)
        print("Test Acc unpoisoned:", epoch_acc)

        return epoch_loss, epoch_acc, cba_total, cdm_total, rba_total

    def train(self, train_dataloader, current_epoch, retraining=False):
        self.net_retraining.train()
        self.net.train()

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

            #self.scheduler.step()  # After 100 steps/epochs the learning rate will be reduce. This provides overfitting and gives a small learning boost.
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

        if self.name is None:
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

from coding.Aenderungen_LRP.TrafficSignAI.Models.InceptionNet3 import InceptionNet3
from TrafficSignDataset import TrafficSignDataset
from neural_nets import Net

def set_inceptionV3_module():
        module = torchvision.models.inception_v3(pretrained=True)
        num_frts = module.fc.in_features
        module.fc = nn.Linear(num_frts, 43)

        return module
if __name__ == '__main__':

    #deterministic results:
    seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    save_lrp = True
    class_to_get_lrp = [5]


    pp = 0.33
    s = 1

    root_dir = "./dataset/"

    train_dir_unpoisoned = root_dir + "Training/"
    valid_dir_unpoisoned = root_dir + "Validation/"
    test_dir_unpoisoned = root_dir + "Testing/"

    print("Torch Version: " + str(torch.__version__))
    print("Is Cuda available: {}".format(torch.cuda.is_available()))
    print("Version of Numpy: ", np.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Poisoned_Git_Dataset_15"


    # Für Clean Label Poisoning Attack wird model_to_poison_data benötigt
    model_to_poison_data = modelAi(name_to_save='incv3_clean', net=InceptionNet3, poisoned_data=False, isPretrained=False, lr=1e-3)
    Main = TrafficSignMain(model_to_poison_data, epochs=0, image_size=32)
    PA = PoisoningAttack(Main)
    #PA.standard_attack(root_dir=root_dir, s=s, percentage_poison=pp)
    PA.clean_label_attack(root_dir, disp=True, projection='l2', eps=300, n_steps=10, step_size=0.015, percentage_poison=pp, insert='amplitude4', s=s, d=10, amplitude=64)


    #poison_model = modelAi(name_to_save='poison_model')
    #Main = TrafficSignMain(model=poison_model, epochs=0)
    #PA = PoisoningAttack(Main)
    #PA.standard_attack(root_dir=root_dir, s=3)

    root_dir_poisoned = "dataset/Poisoned_Git_Dataset/"

    train_dir = root_dir_poisoned + "Training/"
    valid_dir = root_dir_poisoned + "Validation/"
    test_dir = root_dir_poisoned + "Testing/"

    #train_dir = train_dir_unpoisoned
    #valid_dir = valid_dir_unpoisoned
    #test_dir = test_dir_unpoisoned

    model = modelAi(name_to_save='CLP_amplitudesticker4_pp33_dist10_amp64', net=InceptionNet3, poisoned_data=True, isPretrained=False, lr=1e-3)
    # Lade model in TrafficSignMain:
    main = TrafficSignMain(model, epochs=100, image_size=32, batch_size=32)
    #print(model.net)

    main.creating_data(dataset=TrafficSignDataset, test_dir=test_dir, train_dir=train_dir, valid_dir=valid_dir, test_dir_unpoisoned=test_dir_unpoisoned)

    #main.start_tensorboard()
    main.loading_ai(should_train_if_needed=False, should_evaluate=True, isPretrained=False, patience=20)


    # Lese Daten für Activationload Clustering ohne Trafos ein:
    #main.creating_data_for_ac(dataset=TrafficSignDataset, train_dir=train_dir)

    #AC = ActivationClustering(main, root_dir=root_dir) # Nach dem Verwenden von AC haben Bilder im Datensatz gefehlt
    #AC.run_ac(check_all_classes=False, class_to_check=5)


    #AC.run_retraining(verbose=True)
    #AC.evaluate_retraining(class_to_check=5, T=1)
    #AC.evaluate_retraining_all_classes(T=1)

    ##### LRP-moboehle: Modifizierte Version von Matthias


    print('===> LRP started')
    inn_model = InnvestigateModel(model.net, lrp_exponent=2,
                                  method='e-rule',
                                  beta=0.5)

    #  %% --- Berechnung von mean und Varianz über den gesamten Trainingsdatensatz
    if False:
        lrp_train_dataset = TrafficSignDataset(train_dir, transform=main.test_transform)
        lrp_dataloader = DataLoader(lrp_train_dataset, batch_size=20, shuffle=False)

        del lrp_train_dataset
        num_total_samples = 0
        summed = []



        for data in lrp_dataloader:
            images = data['image']
            labels = data['label']
            im_path = data['path']
            num_total_samples += len(images)

            bNormInputs = inn_model.get_batch_norm_inputs(images)

            for i in range(len(bNormInputs)):
                #print(bNormInputs[i].shape)
                #summed.append(torch.sum(bNormInputs[i], axis=0))
                summed.append(bNormInputs[i].sum(0))


        for i in range(len(bNormInputs)):
            #print(summed[i].shape)
            mean = summed[i]/num_total_samples
            #print(mean.shape)

            # Save means per layer in dict
            inn_model.batch_norm_dict[str(i)] = {}
            inn_model.batch_norm_dict[str(i)]['mean'] = mean

            inn_model.inverter.batchNorm_dict[str(i)] = {}
            inn_model.inverter.batchNorm_dict[str(i)]['mean'] = mean

        #print(inn_model.batch_norm_dict[str(5)]['mean'].shape)

        # Berechne Varianz
        var_summed = []

        for data in lrp_dataloader:
            images = data['image']
            labels = data['label']
            im_path = data['path']
            num_total_samples += len(images)

            bNormInputs = inn_model.get_batch_norm_inputs(images)

            for i in range(len(bNormInputs)):
                #print(bNormInputs[i].shape)
                #var_summed.append(torch.sum(torch.square((bNormInputs[i] - inn_model.batch_norm_dict[str(i)]['mean'])), axis=0))
                var_summed.append((torch.square((bNormInputs[i] - inn_model.batch_norm_dict[str(i)]['mean'])).sum(0)))


        for i in range(len(bNormInputs)):
            #print(summed[i].shape)
            var = var_summed[i]/num_total_samples
            #print(mean.shape)

            # Save means per layer in dict
            inn_model.batch_norm_dict[str(i)]['var'] = var
            inn_model.inverter.batchNorm_dict[str(i)]['var'] = var

        print('hier')
        for i in range(10):
            print('i: ', i)

            print(inn_model.batch_norm_dict[str(i)]['var'].shape)
    # ------------------------------------------------------------------------------------------------------------------



    # Wähle Trainingsdaten ohne transformations
    # erstelle dazu einen neuen dataloader
    lrp_train_dataset = TrafficSignDataset(train_dir, transform=main.test_transform)
    lrp_dataloader = DataLoader(lrp_train_dataset, batch_size=1, shuffle=False)

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

        if labels[0] in class_to_get_lrp:
            # Move data to device
            images, labels = images.to(main.model.device), labels.to(main.model.device)

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

            # Speichere aktuelle Heatmap im entsprechden folder ab
            #rel = np.swapaxes(input_relevance_values[0].detach().numpy(), 0, 2)
            #print(rel.shape)
            rel = np.sum(d.detach().numpy(), axis=0)
            #print(rel.shape)

            #im = Image.fromarray(rel).convert('RGB')

            #im.save(path + str(labels[0].detach().numpy()).zfill(5) + "/" + im_path[0].rsplit('/', 1)[-1], subsampling=0, quality=100)

            # Abspeichern der Relevanzen als .npy file
            if save_lrp:
                fname_to_save_rel = path_rel + "/" + str(labels[0].detach().cpu().numpy()).zfill(5) + "/" + os.path.splitext(im_path[0].rsplit('/', 1)[-1])[0] + ".npy"

                with open(fname_to_save_rel, 'wb') as f:

                    np.save(f, d)

            # Abspeichern der lrp plots als jpgs
            #fname_to_save_im = path_im + "/" + str(labels[0].detach().numpy()).zfill(5) + "/" + os.path.splitext(im_path[0].rsplit('/', 1)[-1])[0] + ".png"
            #print(fname_to_save_im)
            #plt.imshow(d)
            #plt.savefig(fname_to_save_im)

    print('===> LRP finished')

