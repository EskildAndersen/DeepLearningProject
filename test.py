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

Dataset = ImageDataset('labels.txt', maxLength=max_len)

trainloader = torch.utils.data.DataLoader(Dataset,
                                          batch_size=1,
                                          shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

lenFeature = Dataset.__getFeatureLen__()


hidden_size = 256
encoder = Encoder(input_size=max_len, hidden_size=hidden_size,
                  feature_len=lenFeature, vocab_len=len(vocab), 
                  padding_index=PAD_token)
decoder = Decoder(input_size=hidden_size,
                  hidden_size=hidden_size, vocab_size=len(vocab))

learning_rate = 0.05
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
#decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

for x, y, z in trainloader:
    label, input_sentence, input_feature = x, y, z
    break


def train(input_feature, input_sentence, target_sentence,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=max_len):
    input_length = 39
    target_length = 39
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)
    loss = 0

    for i in range(1, max_length):
        padding = torch.LongTensor([PAD_token for _ in range(max_length-i)])
        text_in = torch.cat((input_sentence[0][:i], padding))
        # print(len(text_in))

        output, encoder_hidden = encoder(
            text_in, input_feature, encoder_hidden)
        encoder_outputs[i] = output
        pass

    output = decoder(encoder_outputs)
    # output_sentence = output.argmax(axis=1)
    # sentence = deTokenizeCaptions(np.array(output_sentence), inv_vocab)
    
    
    


if __name__ == '__main__':
    train(input_feature, input_sentence, label, encoder,
          decoder, encoder_optimizer, encoder_optimizer, None)
