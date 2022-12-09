'''
Script to generate images and image path with corresponding labels. 

class 'ImageDataset' loads the labels.txt file and split the 
image path from the 5 corresponding labels. If random is set to True
a randomly chosen caption to the image will be chosen. 
The class also loads the feature vector from all the images 
that has been fed through VGG16 in CNN.py. Returns image path,
label (caption), label encoded, and feature vector. 

class 'Images' does almost the same kind of process, but returns an image,
image path and corresponding labels. If transform = True, the images
are normalized and reshaped into a fixed size. 

'''
from torch.utils.data import Dataset
import re
import torch
import numpy as np
import os
import pandas as pd
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pickle import load, dump
from torchvision.io import read_image
from CaptionCoder import tokenizeCaptions


class ImageDataset(Dataset):
    def __init__(self, annotations_file, feature_vector = 'features.p', random = True):
        # Set paths
        self.random = random
        self.img_dir = os.path.join('data', 'images')
        self.annotations_file = os.path.join('data', 'texts', annotations_file)
        self.feature_dir = os.path.join('data', 'features', feature_vector)

        # Load labels
        self.dct = dict()
        with open(self.annotations_file) as f:
            for line in f.readlines():
                line = line.strip()
                img, label = re.split('#\d\t', line)
                self.dct[img] = self.dct.get(img, []) + [label]

        # Generate dataframe of labels
        self.img_labels = pd.DataFrame.from_dict(
            self.dct,
            orient='index',
            columns=['1', '2', '3', '4', '5']
        )
        self.img_labels = self.img_labels.reset_index()
        self.img_labels = self.img_labels.rename(columns={'index': 'img'})

        # Import features
        self.features = load(open(self.feature_dir, "rb"))

    def __len__(self):
        return len(self.img_labels)

    def __getFeatureLen__(self):
        return len(list(self.features.values())[0][0])

    def __getitem__(self, idx):
        # Draw a random number from 1 to 5
        if self.random:
            # The number decides which caption we choose as label
            rand = np.random.randint(low=1, high=6)
            label = self.img_labels.iloc[idx, rand]
            labelEncoded = tokenizeCaptions(label)
            labelEncoded = labelEncoded
        else: 
            labels = self.img_labels.iloc[idx]
            label = [label for label in labels[1:]]
            labelEncoded = [tokenizeCaptions(l) for l in labels[1:]]
            labelEncoded = labelEncoded

        # Loading the features of the given image
        featureVector = self.features.get(self.img_labels.iloc[idx, 0])
        img_path = self.img_labels.iloc[idx][0]

        return img_path,label, torch.LongTensor(labelEncoded), torch.Tensor(featureVector)


class Images(torch.utils.data.Dataset):
    def __init__(self, annotations_file, mean, std, transform = None):
        self.img_dir = os.path.join('data', 'images')
        self.annotations_file = os.path.join('data', 'texts', annotations_file)
        self.transform = transform
        self.mean = mean
        self.std = std

        # Load labels
        self.dct = dict()
        with open(self.annotations_file) as f:
            for line in f.readlines():
                line = line.strip()
                img, label = re.split('#\d\t', line)
                self.dct[img] = self.dct.get(img, []) + [label]

        # Generate dataframe of labels
        self.img_labels = pd.DataFrame.from_dict(
            self.dct,
            orient='index',
            columns=['1', '2', '3', '4', '5']
        )
        self.img_labels = self.img_labels.reset_index()
        self.img_labels = self.img_labels.rename(columns={'index': 'img'})

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        name = self.img_labels.iloc[idx, 0]

        img_path = os.path.join(self.img_dir, name)

        if self.transform:
            preprocess = T.Compose([                          
                T.transforms.Resize(224),                           # Resize image
                T.transforms.Normalize(
                    (0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
                                                                    # Normalize data
            image = read_image(img_path).float()
            image = preprocess(image)
        else:
            preprocess = T.Compose([                   
                T.transforms.Resize(224)])

            image = read_image(img_path).float()
            image = preprocess(image)

        label = self.img_labels.iloc[idx, 1]

        return image, name, label

def calcMeanAndStd():
    image_data = Images('labels.txt')
    image_data_loader = DataLoader(
        image_data,
        # batch size is whole dataset
        batch_size=len(image_data),
        shuffle=False)

    images, _, _ = next(iter(image_data_loader))

    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])

    return mean, std

if __name__ == '__main__':

    
    # Calculate mean and std of all pictures (takes alot of time!!)
    calculateMeanAndStd = False

    if calculateMeanAndStd:
        calculateMeanAndStd()

    # Define the paths for labels and imagss
    annotationsFile = 'labels.txt'

    mean, std = (0.485, 0.456, 0.406),(0.229, 0.224, 0.225)

    # get some images
    dataset = Images(annotationsFile, mean, std)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataiter = iter(data_loader)
    images, names, labels = dataiter.next()

    for image, label in zip(images, labels):  # Run through all samples in a batch
        plt.figure()
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.title(label)
        plt.axis('off')
        plt.show()
