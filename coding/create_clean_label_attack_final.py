import os
import shutil

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#from Attacks_Poisoning.modelAI import modelAI
from Attacks_Poisoning.InceptionNet3 import InceptionNet3
from Dataset.TrafficSignDataset import TrafficSignDataset
from Attacks_Poisoning.Training_mit_Callbacks_modelAI import set_seed,create_train_transform,create_test_transform
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm

from TrafficSignAI.Datasets.Net import my_Seq_Net, my_2nd_Seq_Net,Net
from os.path import join


class modelAI:
    def __init__(self, net: nn.Module = InceptionNet3, criterion=nn.CrossEntropyLoss(), poisoned_data=True, lr=1e-4,
                 num_classes=43):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = str(net)
        self.net = net().to(self.device)
        self.criterion = criterion
        # self.optimizer = optimizer(net.parameters(), lr=lr) #optimizer = optim.Adam
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        # self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.poisoned_data = poisoned_data
        self.path = []
        self.num_classes = num_classes
        self.BDSR_classwise = np.zeros((self.num_classes,))


    def get_activations_of_last_hidden_layer(self, data_loader):
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

                # print('ID:', id)
                images, labels, poison_labels, idx, path = data
                images = images.to(self.device)

                for blub in range(len(path)):
                    paths.append(path[blub])
                    # paths = paths + path[blub]

                outputs, activations = self.net(images)

                # zweites return statement in forward method von model eingefügt, liefert activations der vorletzten layer
                _, vhs = torch.max(outputs, 1)
                vhs = vhs.detach().cpu().numpy()

                activations = activations.detach().cpu().numpy()
                results_labels.append(labels.detach().cpu().numpy())
                results_poison_labels.append(poison_labels.detach().cpu().numpy())
                results_activations.append(activations)
                results_predictions.append(vhs)

                idx = idx.numpy()
                indices = np.concatenate((indices, idx), axis=None)

            else:

                images, last_labels, last_poison_labels, last_idx, last_path = data
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
                last_idx = last_idx.numpy()

        activations_array = np.asarray(results_activations)
        labels_array = np.asarray(results_labels)
        poison_labels_array = np.asarray(results_poison_labels)
        predictions_array = np.asarray(results_predictions)

        idx = np.concatenate((indices, last_idx), axis=None)

        activations = activations_array.reshape(
            (activations_array.shape[0] * activations_array.shape[1], activations_array.shape[2]))
        labels = labels_array.reshape(labels_array.shape[0] * labels_array.shape[1])
        poison_labels = poison_labels_array.reshape(poison_labels_array.shape[0] * poison_labels_array.shape[1])
        predictions = predictions_array.reshape(predictions_array.shape[0] * predictions_array.shape[1])

        activations = np.concatenate((activations, last_activations), axis=0)
        labels = np.concatenate((labels, last_labels), axis=0)
        poison_labels = np.concatenate((poison_labels, last_poison_labels), axis=0)
        predictions = np.concatenate((predictions, last_preds), axis=0)

        print('ac_shape:', activations.shape)
        print('len_paths:', len(paths))
        return activations, labels, poison_labels, predictions, idx, paths


    def valid_eval(self, test_loader):  # disp_attack_statistics=False, source_class =2, target_class=5):
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

        for data in test_loader:
            # show progress
            images, labels, poison_labels, idx, path = data
            images, labels, poison_labels = images.to(self.device), labels.to(self.device), poison_labels.to(self.device)
            self.optimizer.zero_grad()

            outputs, _ = self.net(images)

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

        # print("anzahl richtiger:" , num_right_total)
        # print(" anzahl falscher:", num_wrong_total)

        # model.save(current_epoch)

        epoch_loss = running_loss / len(test_loader)
        # print(epoch_loss)
        epoch_acc = running_corrects / total
        # print(epoch_acc)
        # if disp_attack_statistics == True and not source_class == None and not target_class == None:
        # if self.poisoned_data:
        #    if num_right_total + num_wrong_total != 0:
        #        print(" Prozentsatz an richtig klassifizierten poisoned images im val set:",
        #              num_right_total / (num_right_total + num_wrong_total))

        # self.log_tensorboard(epoch=current_epoch, loss_train=epoch_loss, accuracy_train=epoch_acc, input=(images, labels), validationType=ValidationType.TRAIN)
        return epoch_loss, epoch_acc


    def test_eval_poisoned(self, test_loader):  # disp_attack_statistics=False, source_class =2, target_class=5):
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

        succesful_backdoors = np.zeros((self.num_classes,))
        unsuccesful_backdoors = np.zeros((self.num_classes))

        for data in test_loader:
            # show progress
            images, labels, poison_labels, idx, path = data
            images, labels, poison_labels = images.to(self.device), labels.to(self.device), poison_labels.to(self.device)
            self.optimizer.zero_grad()

            outputs, _ = self.net(images)

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

            # print(succesful_backdoors)
            # print(unsuccesful_backdoors)
        # model.save(current_epoch)

        epoch_loss = running_loss / len(test_loader)
        # print(epoch_loss)
        epoch_acc = running_corrects / total
        # print(epoch_acc)
        # if disp_attack_statistics == True and not source_class == None and not target_class == None:
        if self.poisoned_data:
            print(total_bd_ca5)
            print(total_bd_nca5)
            self.BDSR_classwise = succesful_backdoors / (succesful_backdoors + unsuccesful_backdoors)
            bdsr = total_bd_ca5 / (total_bd_nca5 + total_bd_ca5)

        else:
            bdsr = 0.0

        print("BackDoorSuccessRate:", bdsr)
        print("Successful Backdoor Activations (Sticker+Source class lead to desired misclassification):", cba_total)
        print("Target class gets misclassified as source class:", cdm_total)
        print("Sticker leads to desired misclassification:", rba_total)

        print("Test_loss:", epoch_loss)
        print("test_acc:", epoch_acc)

        return epoch_loss, epoch_acc, cba_total, cdm_total, rba_total, bdsr


    def test_eval_unpoisoned(self, test_loader):  # disp_attack_statistics=False, source_class =2, target_class=5):
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
            images, labels, poison_labels, idx, path = data
            images, labels, poison_labels = images.to(self.device), labels.to(self.device), poison_labels.to(self.device)
            self.optimizer.zero_grad()

            outputs, _ = self.net(images)

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

        # model.save(current_epoch)

        epoch_loss = running_loss / len(test_loader)
        # print(epoch_loss)
        epoch_acc = running_corrects / total
        # print(epoch_acc)
        # if disp_attack_statistics == True and not source_class == None and not target_class == None:

        print("Test_loss_unpoisoned:", epoch_loss)
        print("test_acc_unpoisoned:", epoch_acc)

        return epoch_loss, epoch_acc, cba_total, cdm_total, rba_total


    def train(self, train_dataloader, current_epoch):
        self.net.train()
        self.scheduler.step(
            current_epoch)  # After 100 steps/epochs the learning rate will be reduce. This provides overfitting and gives a small learning boost.
        torch.set_grad_enabled(True)
        total = 0
        running_loss = 0.0
        running_corrects = 0
        label_array = []
        for data in train_dataloader:
            # show progress
            images, labels, poison_labels, idx, path = data
            images, labels, poison_labels = images.to(self.device), labels.to(self.device), poison_labels.to(self.device)
            self.optimizer.zero_grad()

            outputs, _ = self.net(images)

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
        # print(epoch_loss)
        epoch_acc = running_corrects / total
        # print(epoch_acc)

        # self.log_tensorboard(epoch=current_epoch, loss_train=epoch_loss, accuracy_train=epoch_acc, input=(images, labels), validationType=ValidationType.TRAIN)
        return epoch_loss, epoch_acc


    def save(self, epoch, train_loss, model_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
        }, model_path)
        print("==> Model saved.")


    def load(self, model_path):
        checkpoint = torch.load(model_path)
        if type(checkpoint) is dict:
            start_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # checkpoint = checkpoint(['model_state_dict'])
        # self.net.load_state_dict(checkpoint)
        print("==> Model loaded.")


    def run_training(self, current_epoch, epochs, checkpoint_dir, train_loader, val_loader):
        # Training
        verbose = True
        # dec_precision = 3

        print(f"=>\tStart training AI on {self.device}")
        Abbruch = False
        for epoch in range(current_epoch, current_epoch + epochs):

            # model_path = checkpoint_dir + model_name + "_epoch_" + str(epoch) + ".pth"
            model_path = join(checkpoint_dir, "_epoch_" + str(epoch) + ".pth")
            if Abbruch == True:
                self.save(epoch, train_loss, model_path)
                break

            train_loss, train_acc = self.train(train_loader, current_epoch=epoch)
            if epoch % 50 == 0:  # Save model every 10 epochs
                self.save(epoch, train_loss, model_path)
            if epoch == current_epoch + epochs - 1:
                self.save(epoch, train_loss, model_path)
            if verbose:
                print("=>\t[%d] train loss: %.3f, train accuracy: %.3f" % (epoch, train_loss, train_acc * 100) + "%")
            valid_loss, valid_acc = self.valid_eval(val_loader)
            # test_loss, test_acc = self.test_eval_poisoned(test_loader)

            # AbbruchBedingung: Training wird abgebrochen, falls sich valid_loss in den letzten 20 Epochen nicht verbessert hat:
            if epoch == current_epoch:
                val0 = valid_loss
            if epoch == current_epoch + 1:
                val1 = valid_loss
            if epoch == current_epoch + 2:
                val2 = valid_loss
            if epoch == current_epoch + 3:
                val3 = valid_loss
            if epoch == current_epoch + 4:
                val4 = valid_loss
            if epoch == current_epoch + 5:
                val5 = valid_loss
            if epoch == current_epoch + 6:
                val6 = valid_loss
            if epoch == current_epoch + 7:
                val7 = valid_loss
            if epoch == current_epoch + 8:
                val8 = valid_loss
            if epoch == current_epoch + 9:
                val9 = valid_loss
            if epoch == current_epoch + 10:
                val10 = valid_loss
            if epoch == current_epoch + 11:
                val11 = valid_loss
            if epoch == current_epoch + 12:
                val12 = valid_loss
            if epoch == current_epoch + 13:
                val13 = valid_loss
            if epoch == current_epoch + 14:
                val14 = valid_loss
            if epoch == current_epoch + 15:
                val15 = valid_loss
            if epoch == current_epoch + 16:
                val16 = valid_loss
            if epoch == current_epoch + 17:
                val17 = valid_loss
            if epoch == current_epoch + 18:
                val18 = valid_loss
            if epoch == current_epoch + 19:
                val19 = valid_loss

            if epoch > current_epoch + 19:
                # Update
                val0 = val1
                val1 = val2
                val2 = val3
                val3 = val4
                val4 = val5
                val5 = val6
                val6 = val7
                val7 = val8
                val8 = val9
                val9 = val10
                val10 = val11
                val11 = val12
                val12 = val13
                val13 = val14
                val14 = val15
                val15 = val16
                val16 = val17
                val17 = val18
                val18 = val19
                val19 = valid_loss

                # Abbruchbedingung if val9 = val0 (keine Verbesseung des Validation losses in den letzten 10 Epochen, auf 3 Nachkommastellen beschränken?)
                # oder keine Verbesserung in den letzten 3 Runden, patience = 3 # hat oft so um ca.20 epochen abgebrochen, patience sollte längr sein oder garnicht als Kriterium verwendet werden
                if all(i > val0 for i in
                       [val10, val11, val12, val13, val14, val15, val16, val17, val18, val19, val7, val8, val9, val6, val5,
                        val4, val3, val2, val1]):
                    Abbruch = True

            if verbose:
                print("=>\t[%d] validation loss: %.3f, Validation accuracy: %.3f" % (
                    epoch, valid_loss, valid_acc * 100) + "%")

        return epoch, model_path


    def evaluate_retraining(self, class_to_check, data_loader, T=1):
        # num_classes könnt man irgendwie aus der letzten layer des Netztes abgreifen
        num_classes = 43
        number_of_predictions_of_suspicious_data = np.zeros((num_classes,), dtype=int)

        for data in data_loader:
            images, labels, poison_labels, idx, path = data

            images = images.to(self.device)

            outputs, _ = self.net(images)

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
            print('Dataset is legitimate')
            print(l / p)
            print('l:', l)
            print('p:', p)

        elif lp < T:
            is_poisonous = 1
            print('Dataset is poisonous')
            print(l / p)
            print('l:', l)
            print('p:', p)
        return is_poisonous, lp

def insert_amplitude_backdoor(image, amplitude,p=0):

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
def insert_4corner_amplitude_backdoor(image, amplitude,p=0):
    import operator
    # Get image size:

    num_col = image.size[0]
    num_row = image.size[1]

    # Load pixel map
    pixels = image.load()

    # Add and subtract amplitude on each channel respectively and clip values
    amp = np.asarray([amplitude, amplitude, amplitude])  # [16 16 16]

    # Bottom right
    # von rechts nach links in der untersten Zeile
    pixels[num_col - 1-p, num_row - 1-p] = tuple(np.clip(np.asarray(pixels[num_col - 1-p, num_row - 1-p]) + amp, 0, 255))
    pixels[num_col - 2-p, num_row - 1-p] = tuple(np.clip(np.asarray(pixels[num_col - 2-p, num_row - 1-p]) - amp, 0, 255))
    pixels[num_col - 3-p, num_row - 1-p] = tuple(np.clip(np.asarray(pixels[num_col - 3-p, num_row - 1-p]) + amp, 0, 255))

    pixels[num_col - 1-p, num_row - 2-p] = tuple(np.clip(np.asarray(pixels[num_col - 1-p, num_row - 2-p]) - amp, 0, 255))
    pixels[num_col - 2-p, num_row - 2-p] = tuple(np.clip(np.asarray(pixels[num_col - 2-p, num_row - 2-p]) + amp, 0, 255))
    pixels[num_col - 3-p, num_row - 2-p] = tuple(np.clip(np.asarray(pixels[num_col - 3-p, num_row - 2-p]) - amp, 0, 255))

    pixels[num_col - 1-p, num_row - 3-p] = tuple(np.clip(np.asarray(pixels[num_col - 1-p, num_row - 3-p]) + amp, 0, 255))
    pixels[num_col - 2-p, num_row - 3-p] = tuple(np.clip(np.asarray(pixels[num_col - 2-p, num_row - 3-p]) - amp, 0, 255))
    pixels[num_col - 3-p, num_row - 3-p] = tuple(np.clip(np.asarray(pixels[num_col - 3-p, num_row - 3-p]) - amp, 0, 255))

    # Bottom left
    pixels[0+p, num_row - 1-p] = tuple(np.clip(np.asarray(pixels[0+p, num_row - 1-p]) + amp, 0, 255))
    pixels[1+p, num_row - 1-p] = tuple(np.clip(np.asarray(pixels[1+p, num_row - 1-p]) - amp, 0, 255))
    pixels[2+p, num_row - 1-p] = tuple(np.clip(np.asarray(pixels[2+p, num_row - 1-p]) + amp, 0, 255))

    pixels[0+p, num_row - 2-p] = tuple(np.clip(np.asarray(pixels[0+p, num_row - 2-p]) - amp, 0, 255))
    pixels[1+p, num_row - 2-p] = tuple(np.clip(np.asarray(pixels[1+p, num_row - 2-p]) + amp, 0, 255))
    pixels[2+p, num_row - 2-p] = tuple(np.clip(np.asarray(pixels[2+p, num_row - 2-p]) - amp, 0, 255))

    pixels[0+p, num_row - 3-p] = tuple(np.clip(np.asarray(pixels[0+p, num_row - 3-p]) + amp, 0, 255))
    pixels[1+p, num_row - 3-p] = tuple(np.clip(np.asarray(pixels[1+p, num_row - 3-p]) - amp, 0, 255))
    pixels[2+p, num_row - 3-p] = tuple(np.clip(np.asarray(pixels[2+p, num_row - 3-p]) - amp, 0, 255))

    # Top left
    pixels[0+p, 2+p] = tuple(np.clip(np.asarray(pixels[0+p, 2+p]) + amp, 0, 255))
    pixels[1+p, 2+p] = tuple(np.clip(np.asarray(pixels[1+p, 2+p]) - amp, 0, 255))
    pixels[2+p, 2+p] = tuple(np.clip(np.asarray(pixels[2+p, 2+p]) + amp, 0, 255))

    pixels[0+p, 1+p] = tuple(np.clip(np.asarray(pixels[0+p, 1+p]) - amp, 0, 255))
    pixels[1+p, 1+p] = tuple(np.clip(np.asarray(pixels[1+p, 1+p]) + amp, 0, 255))
    pixels[2+p, 1+p] = tuple(np.clip(np.asarray(pixels[2+p, 1+p]) - amp, 0, 255))

    pixels[0+p, 0+p] = tuple(np.clip(np.asarray(pixels[0+p, 0+p]) + amp, 0, 255))
    pixels[1+p, 0+p] = tuple(np.clip(np.asarray(pixels[1+p, 0+p]) - amp, 0, 255))
    pixels[2+p, 0+p] = tuple(np.clip(np.asarray(pixels[2+p, 0+p]) - amp, 0, 255))

    # Top right
    pixels[num_col-1-p, 2+p] = tuple(np.clip(np.asarray(pixels[num_col-1-p, 2+p]) + amp, 0, 255))
    pixels[num_col-2-p, 2+p] = tuple(np.clip(np.asarray(pixels[num_col-2-p, 2+p]) - amp, 0, 255))
    pixels[num_col-3-p, 2+p] = tuple(np.clip(np.asarray(pixels[num_col-3-p, 2+p]) + amp, 0, 255))

    pixels[num_col-1-p, 1+p] = tuple(np.clip(np.asarray(pixels[num_col-1-p, 1+p]) - amp, 0, 255))
    pixels[num_col-2-p, 1+p] = tuple(np.clip(np.asarray(pixels[num_col-2-p, 1+p]) + amp, 0, 255))
    pixels[num_col-3-p, 1+p] = tuple(np.clip(np.asarray(pixels[num_col-3-p, 1+p]) - amp, 0, 255))

    pixels[num_col-1-p, 0+p] = tuple(np.clip(np.asarray(pixels[num_col-1-p, 0+p]) + amp, 0, 255))
    pixels[num_col-2-p, 0+p] = tuple(np.clip(np.asarray(pixels[num_col-2-p, 0+p]) - amp, 0, 255))
    pixels[num_col-3-p, 0+p] = tuple(np.clip(np.asarray(pixels[num_col-3-p, 0+p]) - amp, 0, 255))

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


def create_clean_label_attack(amplitude = 16,percentage_poison=0.15,im_size=32,s=3,class_to_poison=5,stickerfenster_x=(10,22),stickerfenster_y=(16,24),d=0,eps=300,n_steps=100,insert='amplitude',batch_size=20,disp=False,num_classes=43,projection='l2',step_size=2.0):
    #step_size = 2.0
    #step_size = 1.5 * 1 / 100
    #set_seed(5)
    print('pp:',percentage_poison)
    ## Load unpoisoned model um die schwer klassifizierbaren training samples zu erstellen. Eigentlich nehmen wir hier an, dass dies auf einem unabhängigen, vortrainierten Netz passiert, da der Angreifer keinen Zugriff auf das Training hat.
    # 1) Wir nehmen hier das unpoisoned InceptionNet3 oder
    # 2) vgl. Paper: Wie machen die das denn überhaupt? Unabhängiges Netz wird benutzt.

    # Load pretrained model



    net = InceptionNet3
    model_path = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/InceptionenNet3_test/Setting_s_3_ep_200_pp_0.15_mInception_unpoisoned_Vergleich/_epoch_68.pth"

    model = modelAI(net=net,poisoned_data=False,lr=1e-3)
    model.load(model_path)

    root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Unpoisoned Dataset/"
    # s = 3
    # percentage_poison = 0.1
    # percentage_poison_target_class_testing = 0.33

    # Create poisoned dataset:
    train_dir = root_dir + "Training/"
    valid_dir = root_dir + "Validation/"
    test_dir = root_dir + "Testing/"
    #poison_source_dir_training = train_dir + "00002/"  # Tempo50 Klasse
    #poison_target_dir_training = train_dir + "00005/"  # Tempo80 Klasse
    #poison_source_dir_validation = valid_dir + "00002/"
    #poison_target_dir_validation = valid_dir + "00005/"
    #poison_source_dir_testing = test_dir + "00002/"
    #poison_target_dir_testing = test_dir + "00005/"

    # Erstelle Poisoned Ordner und lösche alte Poisoned Ordner, falls noch welche vorhanden sind:
    root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/"
    path = root_dir + "CleanLabelPoisoned_Git_Dataset/"
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
    poison_dir_training = train_dir + "0000"+str(class_to_poison)+"/" # Tempo80 Klasse

    poison_dir_validation = valid_dir + "0000" + str(class_to_poison)+"/"


    # Load data
    train_dataset = TrafficSignDataset(train_dir, transform=create_test_transform(im_size))
    valid_dataset = TrafficSignDataset(valid_dir, transform=create_test_transform(im_size))
    test_dataset = TrafficSignDataset(test_dir, transform=create_test_transform(im_size))

    # Feed data into Neural Network
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Perturb number of samples of class to poison
    # man könnte den Data loader über alle samples laufen lassen und immer, wenn ein entsprechendes label kommt dann abändern und die Anzahl notieren
    number_of_elements_in_target_class_training = len([name for name in os.listdir(poison_dir_training) if
                                                       os.path.isfile(os.path.join(poison_dir_training, name))])
    number_of_elements_in_target_class_validation = len([name for name in os.listdir(poison_dir_validation) if
                                                       os.path.isfile(os.path.join(poison_dir_validation, name))])
    print(number_of_elements_in_target_class_validation)
    print(number_of_elements_in_target_class_training)


    num_samples_to_poison_train = np.round(np.multiply(percentage_poison,number_of_elements_in_target_class_training)).astype(int)
    print('Number poisoned samples training:',num_samples_to_poison_train)
    num_samples_to_poison_val = np.round(np.multiply(percentage_poison,number_of_elements_in_target_class_validation)).astype(int)
    print('Number poisoned samples validation:',num_samples_to_poison_val)


    # Get input, losses and gradients of num_samples with label class_to_poison

    x_nat_train = []
    y_train = []
    counter_train = 0
    Abbruch = False
    #Get samples and labels
    for data in train_loader:

        if Abbruch:
            break
        # show progress
        images, labels, poison_labels, idx,path = data
        images,labels = images.to(model.device), labels.to(model.device)



        images = images.cpu().numpy()
        labels = labels.cpu().numpy()




        #image = np.rollaxis(image,0,3)

        #imgplot = plt.imshow(image)
        #plt.show()
        for i in range(len(labels)):
            #print(i)
            if labels[i] == class_to_poison:

                # Falls label dem target_label entspricht, füge Bild x_nat hinzu
                counter_train +=1
                #if counter_train == num_samples_to_poison_train-1:
                #    Abbruch = True


                x_nat_train.append(images[i])
                y_train.append(labels[i])
                #print(images.shape)
    x_nat_train = np.asarray(x_nat_train)
    #print('x_nat_shape',x_nat_train.shape)

    y_train = np.asarray(np.asarray(y_train))

    # Wähle zufällige Anzahl an samples der zu poisoneden Klasse aus:
    choice_train = np.random.choice(range(number_of_elements_in_target_class_training),num_samples_to_poison_train,replace=False)
    #print(choice_train)
    x_nat_train = x_nat_train[choice_train]
    y_train = y_train[choice_train]
    #print('x_nat_shape',x_nat_train.shape)


    ## Poison Training Data:

    # Erstelle random id_x und id_y list der Länge number_of_elements to poison_target_class_training
    id_x_training = np.random.choice(stickerfenster_x, num_samples_to_poison_train, replace=True)
    id_y_training = np.random.choice(stickerfenster_y, num_samples_to_poison_train, replace=True)


    # Gehe durch ausgewählt Samples und wähle sie einzeln aus:
    for i, id_x, id_y in zip(range(x_nat_train.shape[0]), id_x_training, id_y_training):
        #x_nat_train
        x_old = torch.from_numpy(x_nat_train[[i]]).to(model.device)
        y_true = torch.from_numpy(y_train[[i]]).to(model.device)



        image = x_old[[0]].cpu().numpy()
        image = image[0]
        image1 = np.rollaxis(image,0,3)

        #imgplot = plt.imshow(image1)
        #plt.show()

        for z in range(n_steps):

            input = torch.tensor(x_old,requires_grad=True)

            # Compute loss wrt inputs x_old
            outputs, _ = model.net(input)
            _, preds = torch.max(outputs, 1)

            loss = model.criterion(outputs, y_true)

            grads = torch.autograd.grad(loss,input)[0]#.cpu().numpy()

            # Gradient descent step
            x_new = x_old + step_size * torch.sign(grads)
            x_new = x_new.cpu().numpy()

            # Projection (vgl. https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py)
            if projection == 'infty':
                x_new = np.clip(x_new, x_nat_train[[i]] - eps, x_nat_train[[i]] + eps)  #.astype(int) #Projection#

            if projection == 'l2':
                # Compute L2 norm of image as the square root of the sum of squared pixel values of difference of original and perturbed image:
                x_diff = x_nat_train[[i]]-x_new
                l2 = np.sqrt(np.sum(np.power(x_diff,2)))
                if l2 <= eps:
                    x_new = x_new
                elif l2 > eps:
                    x_new = x_new / l2
            x_new = np.clip(x_new, 0, 1)  # ensure valid pixel range
            x_old = torch.from_numpy(x_new).to(model.device)

        image = x_old[[0]].cpu().numpy()
        image = image[0]


        #imgplot = plt.imshow(image2)
        #plt.show()

        # Plot two images side by side:

        if disp and i == 1:
            image2 = np.rollaxis(image, 0, 3)
            print('image2_shape:',image2.shape)
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(image1)
            axs[0].set_title('Original Image')
            axs[1].imshow(image2)
            axs[1].set_title('Adversarial Perturbation')

            plt.show()
        # Hier ist das Bild noch richtig dargestellt als Plot, das würde ich gerne abspeichern


        #Speichere das perturbierte Bild im entsprechenden Ordner ab:




        image = np.multiply(np.rollaxis(image, 0, 3),255)
        image = Image.fromarray(image.astype(np.uint8), 'RGB')





        # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
        id_x = int(id_x)
        id_y = int(id_y)

        pixels = image.load()  # create the pixel map
        # Place sticker
        if insert == 'sticker':

            for ii in range(s):
                for jj in range(s):

                    pixels[id_x + ii, id_y+jj] = (245, 255, 0)
                    # print(pixels[id_x + i, id_y + j])
        if insert == 'bw':
            image = insert_3b3_black_and_white(image)

        if insert == 'amplitude':
            image = insert_amplitude_backdoor(image, amplitude=amplitude)
        if insert == 'amplitude4':
            image = insert_4corner_amplitude_backdoor(image, amplitude=amplitude,p=d)

        # Display image with sticker:
        if disp and i == 1:

            #image2 = np.rollaxis(image, 0, 3)
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(image1)
            axs[0].set_title('Original Image')
            axs[1].imshow(image)
            axs[1].set_title('Adversarial Perturbation plus STicker')

            plt.show()



        # Save Image
        image.save(poison_dir_training  + str(i) +"_poison.jpeg", format='JPEG', subsampling=0, quality=100)
        # Delete unpoisoned image in same folder
        os.remove(poison_dir_training + str(i) + ".jpeg")

    print('==> Training Data poisoned.')


    #Get validation Data:
    x_nat_val = []
    y_val = []
    counter_val = 0
    Abbruch = False

    #Get samples and labels
    for data in valid_loader:

        if Abbruch:
            break
        # show progress
        images, labels, poison_labels, idx,path = data
        images,labels = images.to(model.device), labels.to(model.device)



        images = images.cpu().numpy()
        labels = labels.cpu().numpy()




        #image = np.rollaxis(image,0,3)

        #imgplot = plt.imshow(image)
        #plt.show()
        for i in range(len(labels)):
            #print(i)
            if labels[i] == class_to_poison:

                # Falls label dem target_label entspricht, füge Bild x_nat hinzu
                counter_val +=1
                #if counter_val == num_samples_to_poison_val-1:
                #    Abbruch = True

                x_nat_val.append(images[i])
                y_val.append(labels[i])
                #print(images.shape)

    x_nat_val = np.asarray(x_nat_val)
    y_val = np.asarray(np.asarray(y_val))

    # Wähle zufällige Anzahl an samples der zu poisoneden Klasse aus:
    choice_val = np.random.choice(range(number_of_elements_in_target_class_validation), num_samples_to_poison_val,replace=False)

    x_nat_val = x_nat_val[choice_val]
    y_val = y_val[choice_val]

    # Erstelle random id_x und id_y list der Länge number_of_elements to poison_target_class_training
    id_x_val = np.random.choice(stickerfenster_x, num_samples_to_poison_val, replace=True)
    id_y_val = np.random.choice(stickerfenster_y, num_samples_to_poison_val, replace=True)


    # Gehe durch ausgewählt Samples und wähle sie einzeln aus:
    for i, id_x, id_y in zip(range(x_nat_val.shape[0]),id_x_val,id_y_val):

        x_old = torch.from_numpy(x_nat_val[[i]]).to(model.device)
        y_true = torch.from_numpy(y_val[[i]]).to(model.device)



        image = x_old[[0]].cpu().numpy()
        image = image[0]
        image1 = np.rollaxis(image,0,3)

        #imgplot = plt.imshow(image1)
        #plt.show()

        for z in range(n_steps):
            #print(i)
            #input = torch.autograd.Variable(x_old,requires_grad=True)
            # #input.requires_grad = True
            input = torch.tensor(x_old,requires_grad=True)

            # Compute loss wrt inputs x_old
            outputs, _ = model.net(input)
            _, preds = torch.max(outputs, 1)

            loss = model.criterion(outputs, y_true)


            # Gradients should be computed every round
            grads = torch.autograd.grad(loss,input)[0]#.cpu().numpy()
            #print(grads.shape)
            # Gradient descent step
            x_new = x_old + step_size * torch.sign(grads)
            x_new = x_new.cpu().numpy()
            # Projection (vgl. https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py)
            if projection == 'infty':
                x_new = np.clip(x_new, x_nat_train[[i]] - eps, x_nat_train[[i]] + eps)  #.astype(int) #Projection#

            if projection == 'l2':
                # Compute L2 norm of image as the square root of the sum of squared pixel values of difference of original and perturbed image:
                x_diff = x_nat_train[[i]]-x_new
                l2 = np.sqrt(np.sum(np.power(x_diff,2)))
                if l2 <= eps:
                    x_new = x_new
                elif l2 > eps:
                    x_new = x_new / l2




            x_new = np.clip(x_new, 0,1)  # ensure valid pixel range
            x_old = torch.from_numpy(x_new).to(model.device)



        image = x_old[[0]].cpu().numpy()
        image = image[0]


        #imgplot = plt.imshow(image2)
        #plt.show()

        # Plot two images side by side:
        if disp and i == 1:
            image2 = np.rollaxis(image, 0, 3)
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(image1)
            axs[0].set_title('Original Image')
            axs[1].imshow(image2)
            axs[1].set_title('Adversarial Perturbation')

            plt.show()

        #Speichere das perturbierte Bild im entsprechenden Ordner ab:


        image = np.multiply(np.rollaxis(image, 0, 3), 255)
        image = Image.fromarray(image.astype(np.uint8), 'RGB')


        pixels = image.load()  # create the pixel map




        id_x = int(id_x)
        id_y = int(id_y)

        if insert == 'sticker':

            for ii in range(s):
                for jj in range(s):

                    pixels[id_x + ii, id_y+jj] = (245, 255, 0)
                    # print(pixels[id_x + i, id_y + j])

        if insert == 'bw':
            image = insert_3b3_black_and_white(image)

        if insert == 'amplitude':
            image = insert_amplitude_backdoor(image,amplitude=amplitude)

        if insert == 'amplitude4':
            image = insert_4corner_amplitude_backdoor(image,amplitude=amplitude,p=d)



        # Save Image

        image.save(poison_dir_validation  + str(i) +"_poison.jpeg", format='JPEG', subsampling=0, quality=100)

        # Delete unpoisoned image in same folder
        os.remove(poison_dir_validation + str(i) + ".jpeg")

    print('==> Validation Data poisoned.')

    ## Insert backdoor on samples in TESTING DATA:
    # Wähle indices für alle Testing samples außer class_to_poison

    #num_samples_to_poison_test = len(test_dataset) -
    #id_x_val = np.random.choice(stickerfenster_x, num_samples_to_poison_val, replace=True)
    #id_y_val = np.random.choice(stickerfenster_y, num_samples_to_poison_val, replace=True)
    #Insert amplitude trigger on every other sample except the poisoned class
    classes_to_poison = range(num_classes)

    classes_to_poison = np.setdiff1d(classes_to_poison, class_to_poison)


    for i in classes_to_poison:
        if i < 10:
            poison_dir_testing = test_dir + "0000"+str(i) +"/"
        else:
            poison_dir_testing = test_dir + "000" + str(i) + "/"
        size_class_testing = len([name for name in os.listdir(poison_dir_testing) if
                                     os.path.isfile(os.path.join(poison_dir_testing, name))])
        id_x_train = np.random.choice(stickerfenster_x, size_class_testing, replace=True)
        id_y_train = np.random.choice(stickerfenster_y, size_class_testing, replace=True)

        for id,id_x,id_y in zip(range(size_class_testing),id_x_train,id_y_train):
            id = str(id)
            id_x = int(id_x)
            id_y = int(id_y)
            image = Image.open(
                poison_dir_testing + id + ".jpeg")  # '/home/bsi/Dokumente/Dataset_poisoned/Training/00002/1.jpeg')

            pixels = image.load()  # create the pixel map


            if insert == 'sticker':
                #id_x = np.random.choice(stickerfenster_x, size_class_testing, replace=True)
                #id_y = np.random.choice(stickerfenster_y, size_class_testing, replace=True)


                for ii in range(s):
                    for jj in range(s):

                        pixels[id_x + ii, id_y + jj] = (245, 255, 0)
                        # print(pixels[id_x + i, id_y + j])

            if insert == 'bw':
                image = insert_3b3_black_and_white(image)

            if insert == 'amplitude':
                image = insert_amplitude_backdoor(image, amplitude=amplitude)

            if insert == 'amplitude4':
                image = insert_4corner_amplitude_backdoor(image, amplitude=amplitude,p=d)


            # Save Image
            image.save(poison_dir_testing + id + "_poison.jpeg", format='JPEG', subsampling=0, quality=100)
            # Remove unpoisoned image
            os.remove(poison_dir_testing + id + ".jpeg")

    print('==> Testing Data poisoned.')

create_clean_label_attack(amplitude=255,insert='amplitude',eps=300,projection='l2',n_steps=10,step_size=0.015,percentage_poison=0.1,d=0,disp=True)
