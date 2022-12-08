import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



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



def plotAttention():
    pass


if __name__ == '__main__':

    data = pd.read_csv('test.txt', sep = " ", header = None)
    data.columns = ['Loss']
    data = data.iloc[::10,:]
    loss = data['Loss']

    acc = np.random.random((500))

    fig = plotlossNaccuracy(loss,acc,acc)

    fig.show()

    

