'''
Script to create plot functions for plotting accuracy, loss and attention

'''

import matplotlib.pyplot as plt
import numpy as np
from settings import DEVICE
import torch
from DataPreparator import ImageDataset
from vocabulary import max_len, SOS_token
from CaptionCoder import deTokenizeCaptions
import os
from torchvision.io import read_image

def plotlossNaccuracy(
    loss,
    accTrain,
    accDev
    ):
    fig, axs = plt.subplots(2)
    # loss plot
    axs[0].plot(loss)
    axs[0].set_title('Loss')
    axs[0].set_xlabel('iteration')
    axs[0].set_ylabel('loss value')

    # accuracy plot
    axs[1].plot(accTrain, label = 'Train')
    axs[1].plot(accDev, label = 'Development')
    axs[1].set_title('Accuracy')
    axs[1].set_ylim(0,1)
    xmin, xmax = axs[1].get_xlim()
    meanTrain = accTrain.mean()
    axs[1].hlines(y=meanTrain,xmin=xmin,xmax=xmax,color = 'red',label ='Mean Train')
    axs[1].legend(loc = 'lower right')
    # axs[1].text(0,meanTrain,'Mean train accuracy',ha ='right',va = 'center')
    axs[1].set_xlabel('iteration')
    axs[1].set_ylabel('Accuracy')

    fig.subplots_adjust(left = 0.07, right = 0.93,hspace = 0.3)

    return fig



def getContextVector(   
        decoder,
        input_sentences,
        input_features,
    ): 
        input_features = input_features.to(DEVICE)
        input_sentences = input_sentences.to(DEVICE)

        # reccurent decoder part
        outputs, alphas = decoder(input_sentences, input_features)

        # calculate prediction to use as the next input
        predictions = outputs.argmax(-1)

        return alphas, predictions


def plotAttention(img_path,contextVector,prediction):
    listOfWords = prediction.split()
    image = read_image(os.path.join('data','own_images',img_path[0]))
    dim = contextVector.shape[-1]
    dim = int(np.sqrt(dim))
    axisLen = int(np.ceil(np.sqrt(len(listOfWords))))
    fig,ax = plt.subplots(nrows=3,ncols=3)

    idx = 0
    for i in range(axisLen):
        for j in range(axisLen):
            try:
                context = contextVector[idx].cpu().detach()
                contextToPlot = torch.reshape(context,(38, dim,dim))
                img = ax[i][j].imshow(np.transpose(image.numpy(), (1, 2, 0)))
                ax[i][j].imshow(contextToPlot,cmap = 'gray',
                                alpha=0.6,clim = [0.0,context.max().item()],
                                extent=img.get_extent())
                ax[i][j].set_title(listOfWords[idx])
                idx += 1
            except IndexError:
                break
    
    fig.show()

    pass

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    decoder = torch.load('results/models/settings_21_decoder.pt',map_location=device)
    decoder.device = device # since only one gpu quick fix
    decoder.eval()

    train_dataset = ImageDataset(f'test_labels.txt')
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
    

