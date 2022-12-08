'''
Script to utilize trained model for generating captions to our own images..

'''
import os
from torchvision.io import read_image
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from DataPreparator import ImageDataset
import torch.nn as nn

class OwnImages(torch.utils.data.Dataset):
    def __init__(self):
        self.img_dir = os.path.join('data', 'own_images')

    def __len__(self):
        self.count = 0
        for _, _, files in os.walk(self.img_dir):
            self.count += len(files)

        return self.count

    def __getitem__(self, idx):

        name = os.listdir(self.img_dir)[idx]

        img_path = os.path.join(self.img_dir, name)

        image = read_image(img_path)

        return image, img_path

if __name__ == '__main__':

    # get some images
    dataset = OwnImages()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataiter = iter(data_loader)
    images = dataiter.next()

    for image in images:  # Run through all samples in a batch
        plt.figure()
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        #plt.title(label)
        plt.axis('off')
        plt.show()


    # initialize model
    model_path = ''
    model = torch.load(model_path)
    model.eval()

