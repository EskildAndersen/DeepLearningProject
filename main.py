import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from CaptionCoder import deTokenizeCaptions
from EncoderDecoder import FeatureEncoder, DecoderWithAttention
from DataPreparator import ImageDataset
from vocabulary import max_len, vocab, inv_vocab, vocab_size,\
    SOS_token, EOS_token, PAD_token


def train(
    input_features,     # (batch_size, feature_len)
    target_sentence,    # (batch_size, sentence_len)
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

    input_sentences = torch.full((batch_size, 1), SOS_token).to(device)

    for word_idx in range(1, max_len):
        output, (hidden, cell) = decoder(
            input_sentences,
            encoder_output,
            hidden,
            cell
        )

        input_sentences = target_sentence[:, 0:word_idx + 1]
        target = input_sentences[:, 1:]
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
                print(
                    f'{print_loss_avg} : {i} - {deTokenizeCaptions(output_sentence)}')
                torch.save(encoder, 'encoder_model.pt')
                torch.save(decoder, 'decoder_model.pt')

            if (i+1) % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0


if __name__ == '__main__':
    # Parameters
    batch_size = 32
    hidden_size = 256
    output_size = hidden_size
    learning_rate = 0.0005

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
    ).to(device)

    trainIters(encoder, decoder, optimizer=optim.Adam, n_iters=3,
               print_every=1, plot_every=100, lr=learning_rate)
