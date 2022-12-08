'''
Script to utilize trained model for generating captions to our own images.
class 'OwnImages' loads our own images which are stored inside the 'data' folder
inside 'own_images'. Returns image and image path. 

'''
import os
from torchvision.io import read_image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from EncoderDecoder import CNNEncoder
from DataPreparator import ImageDataset
from pickle import dump, load
from plots import getContextVector, plotAttention
from CaptionCoder import deTokenizeCaptions

class OwnImages(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.img_dir = os.path.join('data', 'own_images')
        self.transform = transform

    def __len__(self):
        self.count = 0
        for _, _, files in os.walk(self.img_dir):
            self.count += len(files)

        return self.count

    def __getitem__(self, idx):

        name = os.listdir(self.img_dir)[idx]

        img_path = os.path.join(self.img_dir, name)
        if self.transform:
            preprocess = T.Compose([                          
                T.transforms.Resize(224)])                          # Resize image
            image = read_image(img_path).float()
            image = preprocess(image)

        else:
            image = read_image(img_path)
    
        return image, img_path


def extractFeaturesOwnImages(device):

    # get some images
    dataset = OwnImages(transform=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = CNNEncoder(device)
        
    Features = {}
    for i, data in enumerate(data_loader):
        image, img_path = data[0], data[1]
        features = model(image.to(device))
        Features[img_path[0]] = features.cpu().detach().numpy()
        if i % 100 == 0:
            print(i)

    # dump in pickle file.
    dump(Features, open("own_features.p", "wb"))



    # dataiter = iter(data_loader)
    # images = dataiter.next()

    # for image in images:  # Run through all samples in a batch
    #     plt.figure()
    #     plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    #     #plt.title(label)
    #     plt.axis('off')
    #     plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractFeaturesOwnImages(device)

    feature_name = 'own_features.p'

    decoder = torch.load('results/models/settings_21_decoder.pt',map_location=device)
    decoder.device = device # since only one gpu quick fix
    decoder.eval()

    train_dataset = ImageDataset(f'test_labels.txt', feature_vector=feature_name)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )

    dataIter = iter(dataloader)

    img_path, labels,input_sentence, input_feature = dataIter.next()

    contextVector, prediction = getContextVector(decoder,input_sentence,input_feature)

    prediction = deTokenizeCaptions(prediction[0],True)

    plotAttention(img_path,contextVector,prediction)

    pass
