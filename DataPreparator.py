'''

Script to generate data class

This script assusmes that path is structured as mentioend in readme.

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import os
import pandas as pd
import glob
from torchvision.io import read_image
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep='#\d+\t', header=None, names=['path', 'text']).reset_index(drop=True)
        self.img_dir = img_dir

        # Resize all pictures into the same dimensions. (probably a bit sus)
        self.transform = torchvision.transforms.Resize([500, 500])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        # We define the image path and the corresponding pandas column with the filename:
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)

        # The pandas column containing the labels
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Define the paths for labels and imagss
txtFilePath = 'data/texts/Flickr8k.token.txt'
imgPath = 'data/images/'

# Initialzie dataset class and into the loader. 
Dataset = ImageDataset(txtFilePath, imgPath)
data_loader = DataLoader(Dataset, batch_size=1, shuffle=True)

# get some images
dataiter = iter(data_loader)
images, labels = dataiter.next()

images, labels = dataiter.next()
for image, label in zip(images, labels):  # Run through all samples in a batch
    plt.figure()
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(label)
    plt.axis('off')
    plt.show()

#
