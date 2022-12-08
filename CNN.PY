'''
Script to extract feature vectors from all images in the dataset. 
Feature vectors are then used in the model to be combined
with the word embedding of the captions

'''

import torch
from torch.utils.data import DataLoader
from DataPreparator import Images
from EncoderDecoder import CNNEncoder
from pickle import dump, load


if __name__ == '__main__':
    txtFilePath = 'labels.txt'
    mean, std = (0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
    dataset = Images(txtFilePath, mean, std, transform=False)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CNN = CNNEncoder()
    CNN.to(device)

    # Extract features for all images
    Features = {}
    for i, data in enumerate(dataLoader):
        image, img_path = data[0], data[1]
        features = CNN(image.to(device))
        Features[img_path[0]] = features.cpu().detach().numpy()
        if i % 100 == 0:
            print(i)

    # dump in pickle file.
    dump(Features, open("features.p", "wb"))

    features = load(open("features.p", "rb"))

