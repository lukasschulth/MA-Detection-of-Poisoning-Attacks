import torch
import torch.optim as optim
import numpy as np
from torch import nn
from neural_nets import InceptionNet3
import os
#from TrafficSignMain import TrafficSignMain as main
from TrafficSignDataset import TrafficSignDataset


class modelAi:

    def __init__(self, name_to_save, net: nn.Module = InceptionNet3, criterion=nn.CrossEntropyLoss(), poisoned_data=True, lr=1e-4, num_classes=43, isPretrained=False):
        #super().__init__()

        self.name = name_to_save
        self.best_model_path = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = net().to(self.device)
        self.net_retraining = net().to(self.device)
        self.criterion = criterion
        #self.optimizer = optimizer(net.parameters(), lr=lr) #optimizer = optim.Adam
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer_retraining = optim.Adam(self.net_retraining.parameters(), lr=lr)
        #self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True)
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
        if with_name is not None: # When a name is set, check if the dir contains a file with the specific name.
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
        print(self.device)
        self.net_retraining.train()
        self.net.train()

        #self.scheduler.step(current_epoch)  # After 100 steps/epochs the learning rate will be reduce. This provides overfitting and gives a small learning boost.

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

        # Saving model after each training epoch, independent from val_loss
        #self.save(current_epoch)

        #self.log_tensorboard(epoch=current_epoch, loss_train=epoch_loss, accuracy_train=epoch_acc, input=(images, labels), validationType=ValidationType.TRAIN)
        return epoch_loss, epoch_acc

    def evaluate_test(self, dataloader):
        return self.__evaluate(dataloader, 0)

    def evaluate_valid(self, dataloader, epoch, retraining=False):
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
