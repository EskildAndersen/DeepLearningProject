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
import pandas as pd

def plot_main():

    losses = readData('losses')
    val_acc = readData('dev_acc', 100)
    train_acc = readData('train_acc', 100)

    loss_accuracy_plot(losses, val_acc, train_acc)
    pass


def loss_accuracy_plot(losses, val_acc, train_acc):
    (loss, loss_ax), (acc, acc_axs) = generate_plot()
    loss.show()
    acc.show()
    plot_losses(loss_ax, losses)
    plot_accuracy(acc_axs, val_acc, train_acc)

    loss.savefig(os.path.join('results', 'loss_plt.png'))
    acc.savefig(os.path.join('results', 'acc_plt.png'))



def generate_plot():
    loss_fig = plt.figure(
        # constrained_layout = True,
        figsize=(12, 5)
    )
    loss_fig.suptitle('Loss')
    
    acc_fig = plt.figure(
        # constrained_layout = True,
        figsize=(18, 10)
    )
    acc_fig.suptitle('Accuracy')
    
    loss_fig.set_facecolor('0.95')
    acc_fig.set_facecolor('0.95')
    
    loss_axs = loss_fig.subplots(1, 1)
    loss_fig.subplots_adjust(
        left=0.05, right=0.95,
        bottom=0.1, top=0.91,
        hspace=0.3
    )
    
    acc_axs = acc_fig.subplots(2, 2, sharex = True)
    acc_fig.subplots_adjust(
        left=0.05, right=0.95,
        bottom=0.1, top=0.91,
        hspace=0.19, wspace=0.28
    )

    return (loss_fig, loss_axs), (acc_fig, acc_axs)


def plot_losses(ax: plt.Axes, data: pd.DataFrame, include_batch=False):    
    settings = sorted(list({s for s, _ in data.columns}))
    cols = {col for _, col in data.columns}
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(settings)))
    for i, setting in enumerate(settings):
        _data = data[[(setting, col) for col in cols]]
        _data.columns = [col for col in cols]

        
        if include_batch:
            iter = _data['Iteration']
            n_batches = iter.value_counts().max()
            batch = _data['Batch'].apply(
                lambda b: b/n_batches
            )
            _data = _data.set_index(iter + batch)
            n = 40
            
        else:
            batch = _data['Batch']
            first_batch = batch == 0
            _data = _data.loc[first_batch, :]
            iter = _data['Iteration']
            _data = _data.set_index(iter)
            n = 1

        _data = _data['Loss']
        
        ax.plot(
            _data.index[::n,],
            _data.values[::n,],
            marker='.',
            markersize=0,
            ls='-',
            lw=0.5,
            label=setting,
            color=colors[-(i+1)]
        )

    # ax.set_title('Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    # ax.set_ylim(0.5, 10)
    ax.legend()
    ax.grid(True, 'major', 'x', color = 'grey', linestyle='-', linewidth=0.5)
    ax.grid(True, 'major', 'y', color = 'grey', linestyle='-', linewidth=0.5)
    ax.grid(True, 'minor', 'y', color = 'grey', linestyle='-', linewidth=0.25)


def plot_accuracy(
    axs: plt.Axes,
    val: pd.DataFrame,
    train: pd.DataFrame,
):
    axs = axs.flatten()
    
    linestyles = ['--', ':']
    accuracy_plots(axs, val, linestyles[0])
    bleus, colors = accuracy_plots(axs, train, linestyles[1])

    dummy_colors = []
    for color in colors:
        dummy_colors.append(axs[0].plot([],[], c=color, ls = '-')[0])
    for linestyle in linestyles:
        dummy_colors.append(axs[0].plot([],[], c="black", ls = linestyle)[0])
    legend1 = axs[0].legend(
        [dc for dc in dummy_colors],
        [s for s in bleus] + ['Validation', 'Train'],
        bbox_to_anchor=(1.01, 1.01), loc="upper left"
    )
    axs[0].add_artist(legend1)
    
    pass


def accuracy_plots_no_used(axs: plt.Axes, data: pd.DataFrame, linestyle):
    settings = sorted(list({s for s, _ in data.columns}))
    cols = sorted(list({col for _, col in data.columns}))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(settings)))
    for i, setting in enumerate(settings):
        _data = data[[(setting, col) for col in cols]]
        _data.columns = [col for col in cols]
        _data = _data.set_index(_data['Iteration']).drop('Iteration', axis=1)

        plt_color = colors[i]
        for j, (label, bleu) in enumerate(_data.items()):
            axs[j].plot(
                bleu.index,
                bleu.values,
                marker='',
                markersize=0,
                ls=linestyle,
                color=plt_color,
            )
            
            axs[j].set_title(label)
            if j in {2, 3}:
                axs[j].set_xlabel('Iteration')
            axs[j].set_ylabel('BLEU Score')
            axs[j].set_ylim(0, 0.8)
            axs[j].grid(True, 'major', 'x', color = 'grey', linestyle='-', linewidth=0.5)
            axs[j].grid(True, 'major', 'y', color = 'grey', linestyle='-', linewidth=0.5)
            # axs[j].grid(True, 'minor', 'y', color = 'grey', linestyle='-', linewidth=0.25)
            
    return settings, colors


def accuracy_plots(axs: plt.Axes, data: pd.DataFrame, linestyle):
    settings = sorted(list({s for s, _ in data.columns}))
    cols = sorted(list({col for _, col in data.columns}))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(settings)))
    for j, setting in enumerate(settings):
        _data = data[[(setting, col) for col in cols]]
        _data.columns = [col for col in cols]
        _data = _data.set_index(_data['Iteration']).drop('Iteration', axis=1)

        for i, (_, bleu) in enumerate(_data.items()):
            plt_color = colors[i]
            axs[j].plot(
                bleu.index,
                bleu.values,
                marker='',
                markersize=0,
                ls=linestyle,
                color=plt_color,
            )
            
            axs[j].set_title(setting)
            if j in {2, 3}:
                axs[j].set_xlabel('Iteration')
            axs[j].set_ylabel('BLEU Score')
            axs[j].set_ylim(0, 0.8)
            axs[j].grid(True, 'major', 'x', color = 'grey', linestyle='-', linewidth=0.5)
            axs[j].grid(True, 'major', 'y', color = 'grey', linestyle='-', linewidth=0.5)
            
    return cols[:-1], colors


def readData(type: str, max_iter=None):
    path = os.path.join('results', 'evaluation')

    filenames = os.listdir(path)

    dfs = []

    for filename in filenames:
        if not type in filename:
            continue

        file = os.path.join(path, filename)

        df = pd.read_csv(
            file,
            sep=',',
            decimal='.',
            index_col=None,
        )
        if max_iter:
            df = df.loc[df['Iteration'] < max_iter]
        setting = ' '.join(filename.split('_')[:2])
        multcols = [(setting, col.strip()) for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(multcols)

        dfs.append(df)

    data = pd.concat(dfs, axis=1)

    return data

def getContextVector(   
        decoder,
        input_sentences,
        input_features,
    ): 
        input_features = input_features.to(DEVICE)
        input_sentences = input_sentences.to(DEVICE)

        # reccurent decoder part
        output, alphas = decoder(
            input_sentences,
            input_features,
        )

        # calculate prediction to use as the next input
        prediction = output.argmax(-1)
        return alphas, prediction

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

    decoder = torch.load(os.path.join('results','models','settings_32_decoder.pt'),map_location=device)
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

    plotAttention(img_path,contextVector[0],prediction)
    
    plot_main()
