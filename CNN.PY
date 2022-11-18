import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from DataPreparator import ImageDataset
from json import dump
from pickle import dump, load

model = models.vgg16(weights=True)

model
# CNN finetuned

txtFilePath = 'data/texts/Flickr8k.token.txt'
imgPath = 'data/images/'

# Initialzie dataset class and into the loader.
dataset = ImageDataset(txtFilePath, imgPath)
#data_loader = DataLoader(Dataset, batch_size=64, shuffle=True)
#testData, trainData = torch.utils.data.random_split(dataset, [2000, 6092])
trainLoader = DataLoader(trainData, batch_size=1, shuffle=False)
testLoader = DataLoader(testData, batch_size=1, shuffle=False)
dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

# Quick fix to remove final layers
x = list(model.children())[:-1]
model = x[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

nEpocs = 1  # How many times should we run over the data set?
totaliter = 0
model.train()
model.to(device)

# Extract features!

Features = {}
for epoch in range(nEpocs):  # loop over the dataset multiple times
    # scheduler.step()
    for i, data in enumerate(dataLoader, 0):

        inputs, labels, img_path = data

        outputs = model(inputs.float().to(device))

        Features[img_path[0]] = outputs.flatten().cpu().detach().numpy()

        if i % 100 == 0:

            print(i)
    print('Epoch %d, Train loss: %.3f' % (epoch + 1, running_loss / 100))

    running_loss = 0.0

print('Finished Training')
# .eval()


dump(Features, open("features.p", "wb"))
# pd.DataFrame.from_dict(
#             Features, orient='index').reset_index().to_csv('test.csv')


features = load(open("features.p", "rb"))
