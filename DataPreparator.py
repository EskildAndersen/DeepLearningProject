'''
Script to generate data class

This script assusmes that path is structured as mentioend in readme.
'''
import torch
import numpy as np
import os
import re
import pandas as pd
from torchvision.io import read_image
import torchvision
from torch.utils.data import Dataset
from pickle import load
from CaptionCoder import tokenizeCaptions


class ImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        isForVisualization=False
    ):
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

        # Resize all pictures into the same dimensions. (probably a bit sus)
        self.transform = torchvision.transforms.Resize([224, 224])
        self.isForVisualization = isForVisualization

    def __len__(self):
        return len(self.dct)

    def __getFeatureLen__(self):
        return len(list(self.features.values())[0])

    def __getitem__(self, idx):
        # If we want to visualize the image
        if self.isForVisualization:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            # If we want to transform the image
            if self.transform:
                image = self.transform(image)

            return image, label

        # Draw a random number from 1 to 5
        rand = np.random.randint(low=1, high=6)

        # The number decides which caption we choose as label
        label = self.img_labels.iloc[idx, rand]
        labelEncoded = tokenizeCaptions(label)

        # Loading the features of the given image
        featureVector = self.features.get(self.img_labels.iloc[idx, 0])

        return label, torch.LongTensor(labelEncoded), torch.Tensor(featureVector)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    # Define the paths for labels and imagss
    annotationsFile = 'train_labels.txt'

    # Initialzie dataset class and into the loader.
    dataset = ImageDataset(annotationsFile)
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for x, y, z in dataset:
        print(x, y, z)
        break

    # # get some images
    # dataiter = iter(data_loader)
    # images, labels = dataiter.next()

    # images, labels = dataiter.next()
    # for image, label in zip(images, labels):  # Run through all samples in a batch
    #     plt.figure()
    #     plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    #     plt.title(label)
    #     plt.axis('off')
    #     plt.show()
