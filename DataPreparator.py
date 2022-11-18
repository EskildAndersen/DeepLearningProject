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
        self.img_labels = pd.read_csv(
            annotations_file, sep='#\d+\t', header=None, names=['path', 'text']).reset_index(drop=True)
        self.img_dir = img_dir

        self.Dict = dict()

        for _, row in self.img_labels.iterrows():
            if row[0] in self.Dict:
                self.Dict[row[0]].append(row[1])
            else:
                self.Dict[row[0]] = [row[1]]

        self.img_labels = pd.DataFrame.from_dict(
            self.Dict, orient='index').reset_index()

        self.img_labels = self.img_labels.rename(columns={'index': 'img_path'})

        # Resize all pictures into the same dimensions. (probably a bit sus)
        self.transform = torchvision.transforms.Resize([224, 224])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        # We define the image path and the corresponding pandas column with the filename:
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)

        # Draw a random number from 1 to 5
        rand = np.random.randint(low=1, high=6, size=1)

        # The number decides which caption we choose as label
        label = self.img_labels.iloc[idx, rand[0]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
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
