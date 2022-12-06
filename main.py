import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from CaptionCoder import deTokenizeCaptions
from EncoderDecoder import FeatureEncoder, DecoderWithAttention
from DataPreparator import ImageDataset
from vocabulary import max_len, vocab_size
    
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def train(
    input_features,     # (batch_size, feature_len)
    input_sentences,    # (batch_size, sentence_len)
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
):
    global vocab_size

    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()

    loss = 0

    # (batch_size, encoder_output_size)
    encoder_output = encoder(input_features)
    batch_size, encoder_output_size = encoder_output.shape

    hidden, cell = decoder.getInitialHidden(batch_size, encoder_output_size)

    for word_idx in range(1, max_len):
        input_decoder = input_sentences[:, 0:word_idx]
        
        output, (hidden, cell) = decoder(
            input_decoder,
            encoder_output,
            hidden,
            cell
        )

        target = input_sentences[:, 1:word_idx + 1]
        loss += criterion(output.permute(0, 2, 1), target)
        pass

    # Else teachers forcing

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), output[0]


def trainIters(encoder, decoder, optimizer, n_iters, print_every, plot_every, lr):
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    encoder.train()
    decoder.train()

    encoder_optimizer = optimizer(encoder.parameters(), lr=lr)
    decoder_optimizer = optimizer(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = nn.NLLLoss()

    for iter in range(n_iters):
        for i, (_, sentences, features) in enumerate(trainloader):
            if i == 10:
                break
            
            sentences = sentences.to(device)
            features = features.to(device)

            loss, output = train(
                features,
                sentences,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                criterion
            )

            print_loss_total += loss
            plot_loss_total += loss

            if (i+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                output_sentence = output.argmax(1)
                print(f'{print_loss_avg} : {iter}[{i}]')
                print(f'Prediction batch[0]: {deTokenizeCaptions(output_sentence)}')
                print(f'Target batch[0]: {deTokenizeCaptions(sentences[0, 1:])}')
                torch.save(encoder, 'encoder_model.pt')
                torch.save(decoder, 'decoder_model.pt')

            if (i+1) % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    return plot_losses


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def saveLosses(losses, **kwargs):
    file = '+'.join(['_'.join((f'{k}', f'{w}')) for k, w in kwargs.items()])
    
    with open(file, 'w') as f:
        for loss in losses:
            f.write(f'{loss}\n')
    

if __name__ == '__main__1':
    losses = [i for i in range(1000)]
    saveLosses(
        losses,
        batchsize=32,
        lr=0.0005,
        iterations=3
    )


if __name__ == '__main__':
    # Parameters
    batch_size = 32
    hidden_size = 256
    output_size = hidden_size
    learning_rate = 0.0005
    print_every = 6000//(batch_size * 10)
    plot_every = 10
    n_iter = 1
    optimizer = optim.Adam
    number_layers = 5

    # Setting up device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Training dataset
    train_dataset = ImageDataset('train_labels.txt')
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    lenFeature = train_dataset.__getFeatureLen__()

    encoder = FeatureEncoder(
        feature_len=lenFeature,
        output_len=output_size,
        device=device,
    ).to(device)

    decoder = DecoderWithAttention(
        vocab_len=vocab_size,
        sentence_len=max_len,
        hidden_len=hidden_size,
        encoder_output_len=output_size,
        device=device,
        number_layers=number_layers
    ).to(device)

    losses = trainIters(encoder, decoder, optimizer=optimizer, n_iters=n_iter,
               print_every=print_every, plot_every=plot_every, lr=learning_rate)
    
    saveLosses(
        losses, 
        batchsize=batch_size,
        hiddensize=hidden_size,
        lr=learning_rate,
        iterations=n_iter,
        optim=str(optimizer).split('.')[-2],
        numberlayers=number_layers,
    )
