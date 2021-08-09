# Ideen: kMeans Clustering, PCA + kMeansClustering
# https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
# https://www.reddit.com/r/learnmachinelearning/comments/92nh4c/how_do_i_load_images_into_pytorch_for_training/e376tzx/
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
# https://medium.com/analytics-vidhya/writing-a-custom-dataloader-for-a-simple-neural-network-in-pytorch-a310bea680af
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import zero_one_loss
from sklearn.decomposition import FastICA, PCA
#import cv2
import os
"""
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
"""


if __name__ == '__main__':
    # Load unpoisoned images and label them as 0
    root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Poisoned_Git_Dataset/Poisoned_Class/"
    root_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Poisoned_Class/"
    root_dir = "./dataset/Poisoned_Class/"
    #images_unpoisoned = load_images_from_folder(root_dir)

    images = []
    labels = []
    reduce = "PCA"

    for filename in os.listdir(root_dir + "unpoisoned"):

        img = Image.open(root_dir + "unpoisoned/" + filename).convert('L') #load image an dconvert to grey scale
        if img is not None:
            image = np.asarray(img).reshape(1,-1)
            images.append(image)
            #images.append(np.asarray(img))
            labels.append(0)

    for filename in os.listdir(root_dir + "poisoned"):
        img = Image.open(root_dir + "poisoned/" + filename).convert('L')    # load image and convert to greyscale
        if img is not None:
            image = np.asarray(img).reshape(1, -1)
            print(image.shape)
            images.append(image)
            #images.append(np.asarray(img))
            labels.append(1)

    X = np.vstack(images)
    y_true = np.asarray(labels)

    print(X.shape)
    #Reduce dimension to 10

    if reduce == 'FastICA':
        projector = FastICA(n_components=10, max_iter=1000, tol=0.005)
    elif reduce == 'PCA':
        projector = PCA(n_components=10)

    X = projector.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    y_pred = kmeans.labels_

    #Evaluation
    loss = zero_one_loss(y_true,y_pred)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true,y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    print(tn, fp, fn, tp)
    print(cm)

    import sklearn
    import imblearn




    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    fnr = 1 - sklearn.metrics.recall_score(y_true, y_pred)

    tpr = imblearn.metrics.sensitivity_score(y_true, y_pred)
    tnr = imblearn.metrics.specificity_score(y_true, y_pred)
    fpr = 1 - tnr

    print('Acc_train:', acc)
    #print('f1_train:', f1_score)

    #print('FNR,MissRate_train:', fnr)
    print('tpr_train:', tpr)
    print('tnr_train:', tnr)
    #print('fpr_train:', fpr)

"""


# Nehme die Poisoned Training samples und führe auf der poisoned class kMeans durch

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'Training': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Validation': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#data_dir = 'data/hymenoptera_data'

data_dir = "/home/bsi/Lukas_Schulth_TrafficSign_Poisoning/Dataset/Git_Dataset/Poisoned_Git_Dataset/"


train_dir = data_dir + "Training"
valid_dir = data_dir + "Validation"
test_dir = data_dir + "Testing"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['Training', 'Validation']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=4)for x in ['Training', 'Validation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['Training', 'Validation']}

class_names = image_datasets['Training'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#Let’s visualize a few training images so as to understand the data augmentations.

def imshow(inp, title=None):
    
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['Training']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

"""
