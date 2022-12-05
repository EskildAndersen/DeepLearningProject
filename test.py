import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from CaptionCoder import deTokenizeCaptions
from EncoderDecoder import Encoder, Decoder
from DataPreparator import ImageDataset
from vocabulary import max_len, vocab, inv_vocab
import random

SOS_token = vocab.get('<SOS>')
EOS_token = vocab.get('<EOS>')
PAD_token = vocab.get('<PAD>')
vocab_size = len(vocab)

Dataset = ImageDataset('labels.txt')

trainloader = torch.utils.data.DataLoader(Dataset,
                                          batch_size=1,
                                          shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
torch.cuda.empty_cache()

lenFeature = Dataset.__getFeatureLen__()


hidden_size = 256
encoder = Encoder(input_size=max_len, hidden_size=hidden_size,
                  feature_len=lenFeature, vocab_len=len(vocab),
                  padding_index=PAD_token, device=device).to(device)
decoder = Decoder(input_size=hidden_size,
                  hidden_size=hidden_size, vocab_size=len(vocab), device=device).to(device)

learning_rate = 0.2
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)


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
        padding = torch.LongTensor(
            [PAD_token for _ in range(max_length-i)]).to(device)
        text_in = torch.cat((input_sentence[0][:i], padding)).to(device)

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


def trainIters(encoder, decoder, n_iters, print_every, plot_every, lr):
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()

    for iter in range(n_iters):
        for i, (label, sentence, feature) in enumerate(trainloader):
            target = torch.Tensor(
                F.one_hot(sentence.squeeze(0), vocab_size))
            loss = train(feature.to(device), sentence.to(device), target.to(device), encoder, decoder,
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
    trainIters(encoder, decoder, n_iters=3,
               print_every=1, plot_every=100, lr=0.004)
