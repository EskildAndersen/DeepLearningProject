'''

Script to generate data class

This script assusmes that path is structured as mentioend in readme.

'''
from torch.utils.data import Dataset
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import os
import pandas as pd
import torchvision.transforms as T
import glob
from torchvision.io import read_image
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pickle import load
from vocabulary import inv_vocab, vocab, max_len
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
        self.feature_dir = os.path.join('data', 'features', 'newfeatures.p')

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
        return len(list(self.features.values())[0])

    def __getitem__(self, idx):
        # Draw a random number from 1 to 5
        rand = np.random.randint(low=1, high=6)

        # The number decides which caption we choose as label
        label = self.img_labels.iloc[idx, rand]
        labelEncoded = tokenizeCaptions(label, vocab=vocab, maxLength=max_len)

        # Loading the features of the given image
        featureVector = self.features.get(self.img_labels.iloc[idx, 0])

        return label, torch.LongTensor(labelEncoded), torch.Tensor(featureVector)


class Images(torch.utils.data.Dataset):
    def __init__(self, annotations_file):
        self.img_dir = os.path.join('data', 'images')
        self.annotations_file = os.path.join('data', 'texts', annotations_file)

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

        # Resize all pictures into the same dimensions. (probably a bit sus)
        self.mean = None
        self.std = None

        self.transform = nn.Sequential(
            # smaller edge of image resized to 256
            T.transforms.Resize(256),
            # get 224x224 crop from random location
            T.transforms.RandomCrop(224),
            # horizontally flip image with probability=0.5
            T.transforms.RandomHorizontalFlip(),
            T.transforms.Normalize(
                (117.9994, 113.3671, 102.7541),  # Normalize data
                (70.1257, 68.0825, 71.3111)))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        name = self.img_labels.iloc[idx, 0]

        img_path = os.path.join(self.img_dir, name)
        image = read_image(img_path).float()

        image = self.transform(image)

        label = self.img_labels.iloc[idx, 1]

        return image, name, label


calculateMeanAndStd = False
if calculateMeanAndStd:
    image_data = Images('labels.txt')
    image_data_loader = DataLoader(
        image_data,
        # batch size is whole dataset
        batch_size=len(image_data),
        shuffle=False)

    images, labels, _ = next(iter(image_data_loader))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # Define the paths for labels and imagss
    annotationsFile = 'labels.txt'

    # Initialzie dataset class and into the loader.
    dataset = ImageDataset(annotationsFile)

    for x, y, z in dataset:
        print(x, y, z)
        break

    # get some images
    dataset = Images(annotationsFile)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataiter = iter(data_loader)
    images, names, labels = dataiter.next()

    for image, label in zip(images, labels):  # Run through all samples in a batch
        plt.figure()
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.title(label)
        plt.axis('off')
        plt.show()
