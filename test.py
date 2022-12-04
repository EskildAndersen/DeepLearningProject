import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from CaptionCoder import deTokenizeCaptions
from EncoderDecoder import Encoder, Decoder
from DataPreparator import ImageDataset
from vocabulary import max_len, vocab, inv_vocab, vocab_size,\
    SOS_token, EOS_token, PAD_token

# Parameters
hidden_size = 256
learning_rate = 0.05

# Setting up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Training dataset
train_dataset = ImageDataset('train_labels.txt')

trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=1,
                                          shuffle=True)

lenFeature = train_dataset.__getFeatureLen__()


# Encoder and decoder setup
encoder = Encoder(input_size=max_len, hidden_size=hidden_size,
                  feature_len=lenFeature, vocab_len=vocab_size,
                  padding_index=PAD_token)
decoder = Decoder(input_size=hidden_size,
                  hidden_size=hidden_size, vocab_size=vocab_size)


for x, y, z in trainloader:
    label, input_sentence, input_feature = x, y, z
    break

target_sentence = torch.Tensor(
    F.one_hot(input_sentence.squeeze(0), vocab_size))


def train(input_feature, input_sentence, target_sentence,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=max_len):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)
    loss = 0

    # Encoder loop
    for i in range(1, max_length):
        padding = torch.LongTensor([PAD_token for _ in range(max_length-i)])
        text_in = torch.cat((input_sentence[0][:i], padding))

        output, encoder_hidden = encoder(
            text_in, input_feature, encoder_hidden)
        encoder_outputs[i] = output
        pass

    output = decoder(encoder_outputs)
    # sentence = deTokenizeCaptions(np.array(output_sentence), inv_vocab)

    for o, t in zip(output, target_sentence):
        pass
        loss += criterion(o, t)
        
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def trainIters(encoder, decoder, optimizer, n_iters, print_every, plot_every, lr):
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optimizer(encoder.parameters(), lr=lr)
    decoder_optimizer = optimizer(encoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for iter in range(n_iters):
        for i, (label, sentence, feature) in enumerate(trainloader):
            target = torch.Tensor(
                F.one_hot(sentence.squeeze(0), vocab_size))
            loss = train(feature, sentence, target, encoder, decoder,
                         encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print(f'{print_loss_avg} : {i}')

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0


if __name__ == '__main__':
    trainIters(encoder, decoder, optimizer=optim.Adam, n_iters=3,
               print_every=1, plot_every=100, lr=0.004)
