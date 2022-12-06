'''

Script to generate data class

This script assusmes that path is structured as mentioend in readme.

'''
from torch.utils.data import Dataset
import re
import torch
import torch.nn as nn
import numpy as np
import os
import os
import pandas as pd
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pickle import load
from CaptionCoder import tokenizeCaptions

'''
Script to generate data class

This script assusmes that path is structured as mentioend in readme.
'''

class ImageDataset(Dataset):
    def __init__(self, annotations_file):
        # Set paths
        self.img_dir = os.path.join('data', 'images')
        self.annotations_file = os.path.join('data', 'texts', annotations_file)
        self.feature_dir = os.path.join('data', 'features', 'features.p')

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
        rand = np.random.randint(low=1, high=6)

        # The number decides which caption we choose as label
        label = self.img_labels.iloc[idx, rand]
        labelEncoded = tokenizeCaptions(label)

        # Loading the features of the given image
        featureVector = self.features.get(self.img_labels.iloc[idx, 0])

        return label, torch.LongTensor(labelEncoded), torch.Tensor(featureVector)


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
            nn.Sequential(
                                            
                T.transforms.Resize(224),        # Resize image
                T.transforms.Normalize(
                    self.mean,                   # Normalize data
                    self.std))
            image = read_image(img_path).float()
        else:
            image = read_image(img_path)

        label = self.img_labels.iloc[idx, 1]

        return image, name, label

# Calculate mean and std of all pictures (takes alot of time!!)
calculateMeanAndStd = False

if calculateMeanAndStd:
    image_data = Images('labels.txt')
    image_data_loader = DataLoader(
        image_data,
        # batch size is whole dataset
        batch_size=len(image_data),
        shuffle=False)

    images, _, _ = next(iter(image_data_loader))

    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])

if __name__ == '__main__':

    # Define the paths for labels and imagss
    annotationsFile = 'labels.txt'

    mean, std = (117.9994, 113.3671, 102.7541), (70.1257, 68.0825, 71.3111) 

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
