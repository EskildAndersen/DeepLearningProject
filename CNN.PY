import torch.nn as nn
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from DataPreparator import Images
from pickle import dump, load

class CNNPreTrain(nn.Module):
    def __init__(self):
        super(CNNPreTrain, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.parameters():
            param.requires_grad_(False)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(vgg16.children())[:-1]
        self.vgg16 = nn.Sequential(*modules)

    def forward(self, images):

        # Feed images through VGG16
        features = self.vgg16(images)
        features = features.permute(0, 2,3,1)
        features = features.view(features.size(0), -1, features.size(-1)).squeeze()

        return features

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """

        for p in self.vgg16.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.vgg16.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


if __name__ == '__main__':
    txtFilePath = 'labels.txt'
    mean, std = (117.9994, 113.3671, 102.7541), (70.1257, 68.0825, 71.3111) 
    dataset = Images(txtFilePath, mean, std, transform=False)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

    # specify dim of image embedding
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CNN = CNNPreTrain()
    CNN.to(device)

    Features = {}
    for i, data in enumerate(dataLoader):
        image, img_path = data[0], data[1]
        features = CNN(image.to(device))
        Features[img_path[0]] = features.cpu().detach().numpy()
        if i % 100 == 0:
            print(i)

    dump(Features, open("features.p", "wb"))

    features = load(open("features.p", "rb"))

