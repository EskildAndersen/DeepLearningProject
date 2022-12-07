import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from settings import DEVICE, PAD_token
import torch
from DataPreparator import ImageDataset
from vocabulary import max_len
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
        encoder,
        decoder,
        input_sentence,
        input_features,
        device
    ): 
        # send to device
        input_features = input_features.to(device)
        input_sentence = input_sentence.to(device)

        # feed through encoder
        encoder_output = encoder(input_features)
        contextList = []
        # initialize for decoder
        batch_size, encoder_output_size = encoder_output.shape
        predictions = torch.zeros(batch_size, 0,dtype=int).to(device)
        hidden, cell = decoder.getInitialHidden(batch_size, encoder_output_size)
        SOS = input_sentence[:,:1]

        # reccurent decoder part
        for _ in range(1, max_len):
            input_decoder = torch.cat((SOS,predictions),dim = -1)
            outputs, (hidden, cell), (context,attnweight,alignment) = decoder(
                input_decoder,
                encoder_output,
                hidden,
                cell
            )

            contextList.append(context[0,-1,:])
            # calculate prediction to use as the next input
            predictions = outputs.argmax(-1)
        contextList = torch.stack(contextList)
        # prep final prediction and labels for Bleu calculations
        predictions = torch.cat((SOS,predictions),dim = -1)
        return contextList, predictions

def plotAttention(img_path,contextVector,prediction):
    listOfWords = prediction.split()
    image = read_image(os.path.join('data','images',img_path[0]))
    _,dim = contextVector.shape
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

    encoder = torch.load('encoder_model.pt',map_location=device)
    encoder.device = device # since only one gpu quick fix
    encoder.eval()

    decoder = torch.load('decoder_model.pt',map_location=device)
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


    contextVector, prediction = getContextVector(encoder,decoder,input_sentence,input_feature,device)

    prediction = deTokenizeCaptions(prediction[0],True)

    plotAttention(img_path,contextVector,prediction)
    

