import torch

from coding.main_file_lukasschulth import TrafficSignMain

import json
import os
import random
import shutil
from distutils.dir_util import copy_tree
from os.path import join
from random import random

import numpy as np
from sklearn.cluster import KMeans
from TrafficSignDataset import TrafficSignDataset

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

        copy_tree(self.root_dir + "Training", self.checkpoint_dir + "Original_Data/" + "Training/")
        copy_tree(self.root_dir + "Validation", self.checkpoint_dir + "Original_Data/" + "Validation/")
        copy_tree(self.root_dir + "Testing", self.checkpoint_dir + "Original_Data/" + "Testing/")
        print('Daten in extra Ordner gesichert ...')

        if self.main.model.did_saved_model(self.main.model.name):

            last_epochs = self.main.model.load(verbose=False)

        # Lese Daten für Activation Clustering ohne Trafos ein:
        self.main.creating_data_for_ac(train_dir=self.root_dir + 'Training', valid_dir=self.root_dir + 'Validation', test_dir=self.root_dir + 'Testing')

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
        for check_class in range(self.main.model.num_classes):

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
        for check_class in range(self.main.model.num_classes):
           # print('checkCLASS:', check_class)
            self.main.hparams.add_hparam("sil_train_class" + str(check_class), str(sil_scores_train[check_class]))
        for check_class in range(self.main.model.num_classes):
            self.main.hparams.add_hparam("rel_size_train_class" + str(check_class), str(rel_size_scores_train[check_class]))

        with open(output_json_path, 'w') as f:
            json.dump(self.main.hparams.values(), f, indent=2)

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
                self.main.hparams.add_hparam("Detection Accuracy TRAIN_"+str(class_to_check), str(acc_train))

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
                loss, pred_correct = self.main.model.evaluate_valid(dataloader=self.main.valid_dataloader,
                                                               epoch=epoch, retraining=True)  # self.model.evaluate_valid(val_dataloader=self.valid_dataloader)
                if verbose:
                    print("=>\tAccuracy of validation Dataset: %.3f" % (pred_correct * 100) + "% \n")

        print("=>\tFINISHED ReTRAINING")

    def evaluate_retraining(self, class_to_check, T=1):

        # Create data for retraining evaluation:
        #main.creating_data_for_ac(train_dir=root_dir + "Suspicious_Data/Training")
        self.main.creating_data_for_ac(dataset=TrafficSignDataset, train_dir=self.root_dir + "Suspicious_Data/Training/"+str(class_to_check).zfill(5))
        data_loader = self.main.train_dataloader
        # Load retrained model:
        # self.load(self.best_model_path)
        self.main.model.net_retraining.eval()

        # num_classes könnt man irgendwie aus der letzten layer des Netztes abgreifen

        number_of_predictions_of_suspicious_data = np.zeros((self.main.model.num_classes,), dtype=int)

        for data in data_loader:

            images = data['image']
            labels = data['label']
            poison_labels = data['poison_label']
            path = data['path']

            images = images.to(self.main.model.device)

            outputs, _ = self.main.model.net_retraining(images)

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

        self.main.model.net.eval()

        # Für jede einzelne Klasse werden die Predictions für jede andere Klasse notiert.
        number_of_predictions_of_suspicious_data = np.zeros((self.main.model.num_classes, self.main.model.num_classes), dtype=int)

        for data in self.main.train_dataloader:

            images = data['image']
            labels = data['label']

            if 'poison_label' in data:
                poison_labels = data['poison_label']
            if 'path' in data:
                path = data['path']


            images = images.to(self.main.model.device)

            outputs, _ = self.main.model.net_retraining(images)

            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                # Wird ein sample mit label y als Nummer x predicted, wird die Anzahl im jeweiligen Eintrag um eins erhöht.

                number_of_predictions_of_suspicious_data[labels[i]][preds[i]] += 1

        # l gibt pro Klasse an, wie oft sample mit label x als sample mit label x predicted wurde
        l = number_of_predictions_of_suspicious_data.diagonal()

        # Finde das MAximum an predictions, bei denen label und predictions nicht übereinstimmen.
        sorted_array = number_of_predictions_of_suspicious_data.argsort(axis=1)

        lp = []
        for i in range(self.main.model.num_classes):
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
