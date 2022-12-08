import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


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
    axs[1].plot(accTrain, label='Train')
    axs[1].plot(accDev, label='Development')
    axs[1].set_title('Accuracy')
    axs[1].set_ylim(0, 1)
    xmin, xmax = axs[1].get_xlim()
    meanTrain = accTrain.mean()
    axs[1].hlines(y=meanTrain, xmin=xmin, xmax=xmax,
                  color='red', label='Mean Train')
    axs[1].legend(loc='lower right')
    # axs[1].text(0,meanTrain,'Mean train accuracy',ha ='right',va = 'center')
    axs[1].set_xlabel('iteration')
    axs[1].set_ylabel('Accuracy')

    fig.subplots_adjust(left=0.07, right=0.93, hspace=0.3)

    return fig


def plot_main():

    losses = readData('losses')
    val_acc = readData('train_acc')
    train_acc = readData('dev_acc')

    loss_accuracy_plot(losses, val_acc, train_acc)

    pass


def loss_accuracy_plot(losses, val_acc, train_acc):
    fig, axs = generate_plot()
    fig.show()
    plot_losses(axs[0], losses)
    plot_accuracy(axs[1], val_acc, train_acc)

    pass


def generate_plot():
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    fig.subplots_adjust(
        left=0.07, right=0.93,
        bottom=0.07, top=0.93,
        hspace=0.3
    )

    return fig, axs


def plot_losses(ax: plt.Axes, data: pd.DataFrame):
    settings = sorted(list({s for s, _ in data.columns}))
    cols = {col for _, col in data.columns}
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(settings)))
    for i, setting in enumerate(settings):
        _data = data[[(setting, col) for col in cols]]
        _data.columns = [col for col in cols]

        iter = _data['Iteration']
        n_batches = iter.value_counts().max()
        batch = _data['Batch'].apply(
            lambda b: b/n_batches
        )
        _data = _data.set_index(iter + batch)

        _data = _data['Loss']
        n = 10
        ax.plot(
            _data.index[::n,],
            _data.values[::n,],
            marker='.',
            markersize=0,
            ls='-',
            label=setting,
            color=colors[i]
        )

    ax.set_title('Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()

    pass


def plot_accuracy(ax: plt.Axes, val: pd.DataFrame, train: pd.DataFrame):
    settings = sorted(list({s for s, _ in val.columns}))
    cols = sorted(list({col for _, col in val.columns}))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(settings)))
    linestyles = [':', '--', '-.', '-']
    for i, setting in enumerate(settings):
        _data = val[[(setting, col) for col in cols]]
        _data.columns = [col for col in cols]
        _data = _data.set_index(_data['Iteration']).drop('Iteration', axis=1)

        plt_color = colors[i]
        for j, (_, bleu) in enumerate(_data.items()):
            ax.plot(
                bleu.index,
                bleu.values,
                marker='.',
                markersize=0,
                ls=linestyles[j],
                # label=f'{setting} {label}',
                color=plt_color,
            )

    dummy_lines = []
    for linestyle in linestyles:
        dummy_lines.append(ax.plot([],[], c="black", ls = linestyle)[0])
    lines = ax.get_lines()
    legend1 = ax.legend(
        [line for line in lines[3::4]],
        [s for s in settings],
        loc=1
    )
    legend2 = ax.legend(
        [dl for dl in dummy_lines],
        [col for col in _data.columns],
        loc=4
    )
    ax.add_artist(legend1)
    
    ax.set_title('Accuracy')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('BLEU Score')
    
    pass


def readData(type: str, max_iter=50):
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
        df = df.loc[df['Iteration'] < max_iter]
        setting = ' '.join(filename.split('_')[:2])
        multcols = [(setting, col.strip()) for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(multcols)

        dfs.append(df)

    data = pd.concat(dfs, axis=1)

    return data


if __name__ == '__main__':

    plot_main()

    data = pd.read_csv('test.txt', sep=" ", header=None)
    data.columns = ['Loss']
    data = data.iloc[::10, :]
    loss = data['Loss']

    acc = np.random.random((500))

    fig = plotlossNaccuracy(loss, acc, acc)

    fig.show()
