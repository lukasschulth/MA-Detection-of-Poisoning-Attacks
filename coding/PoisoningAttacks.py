import os
import random
import shutil
from distutils.dir_util import copy_tree

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from coding.TrafficSignDataset import TrafficSignDataset
from coding.TrafficSignMain import TrafficSignMain


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

    def insert_4corner_amplitude_backdoor(self, image, amplitude, p=0):

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

    def insert_3b3_black_and_white(self, image):
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
        #root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/"
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
        random_list2 = random.sample(range(0, size_source_class_validation - 1),
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
            random_list3 = random.sample(range(0, size_source_class_testing - 1),
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
        #root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Unpoisoned Dataset/"
        root_dir = root_dir
        print(root_dir)
        train_dir = root_dir + "Training/"
        valid_dir = root_dir + "Validation/"
        test_dir = root_dir + "Testing/"

        #Trainiere model auf unkorrumpierten Daten
        self.main.creating_data(TrafficSignDataset,train_dir=train_dir,valid_dir=valid_dir)
        self.main.loading_ai(should_train_if_needed=True, should_evaluate=False, isPretrained=False)

        self.main.model.net.eval()

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
        self.main.creating_data_for_ac(TrafficSignDataset, test_dir=test_dir,train_dir=train_dir,valid_dir=valid_dir)

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

            #plt.imshow(image1)
            #plt.show()

            for z in range(n_steps):

                input = torch.tensor(x_old, requires_grad=True)

                # Compute loss wrt inputs x_old
                outputs, _ = self.main.model.net(input)
                _, preds = torch.max(outputs, 1)

                loss = self.main.model.criterion(outputs, y_true)

                grads = torch.autograd.grad(loss, input)[0]  # .cpu().numpy()
                print(grads)
                # Gradient descent step
                x_new = x_old + step_size * torch.sign(grads)
                x_new = x_new.cpu().numpy()

                # Projection (vgl. https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py)
                if projection == 'infty':
                    x_new = np.clip(x_new, x_nat_train[[i]] - eps, x_nat_train[[i]] + eps)  # .astype(int) #Projection#

                if projection == 'l2':
                    # Compute L2 norm of image as the square root of the sum of squared pixel values of difference of original and perturbed image:
                    x_diff = x_nat_train[[i]] - x_new
                    l2 = np.sqrt(np.sum(np.power(x_diff, 2)))
                    if l2 <= eps:
                      x_new = x_new
                    elif l2 > eps:
                      x_new = x_new / l2

                x_new = np.clip(x_new, 0, 255)  # ensure valid pixel range
                x_old = torch.from_numpy(x_new).to(self.main.model.device)

            image = x_old[[0]].cpu().numpy()
            image = image[0]

            image = np.clip(image, 0, 255)
            # Plot two images side by side:

            if disp and i == 1:
                image2 = np.rollaxis(image, 0, 3)
                print('image2_shape:', image2)
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
