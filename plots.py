import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from settings import DEVICE, PAD_token
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
        
        batch_size = input_features.shape[0]

        # initialize for decoder
        prediction = torch.full((batch_size,), SOS_token).to(DEVICE)
        predictions = []
        hidden, cell = decoder.getInitialHidden(batch_size)
        contextList = []
        # reccurent decoder part
        for _ in range(1, max_len):
            input_decoder = prediction
            output, (hidden, cell), (alpha,attnWeights) = decoder(
                input_decoder,
                input_features,
                hidden,
                cell
            )

            # calculate prediction to use as the next input
            prediction = output.argmax(-1)
            predictions.append(prediction)
            contextList.append(alpha)
        # prep final prediction and labels for Bleu calculations
        predictions = torch.stack(predictions, 1)
        contextList = torch.stack(contextList)
        return contextList, predictions

def plotAttention(img_path,contextVector,prediction):
    listOfWords = prediction.split()
    image = read_image(os.path.join('data','images',img_path[0]))
    dim = contextVector.shape[-1]
    dim = int(np.sqrt(dim))
    axisLen = int(np.ceil(np.sqrt(len(listOfWords))))
    fig,ax = plt.subplots(nrows=axisLen,ncols=axisLen)

    idx = 0

    for i in range(axisLen):
        for j in range(axisLen):
            try:
                context = contextVector[idx].cpu().detach()
                contextToPlot = torch.reshape(context,(dim,dim))
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

    # data = pd.read_csv('test.txt', sep = " ", header = None)
    # data.columns = ['Loss']
    # data = data.iloc[::10,:]
    # loss = data['Loss']

    # acc = np.random.random((500))

    # fig = plotlossNaccuracy(loss,acc,acc)

    # fig.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    decoder = torch.load('settings_2_decoder.pt',map_location=device)
    decoder.device = device # since only one gpu quick fix
    decoder.eval()

    train_dataset = ImageDataset(f'train_labels.txt')
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
    

