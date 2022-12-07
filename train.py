from settings import (
    # ENCODER
    ENCODER_OUTPUT_SIZE,
    ENCODER_HIDDEN_1_SIZE,
    ENCODER_HIDDEN_2_SIZE,
    ENCODER_DROP_PROB,
    # DECODER
    DECODER_LSTM_HIDDEN_SIZE,
    NUMBER_OF_LSTM_LAYERS,
    BIDERECTIONAL_LSTM,
    DECODER_DROP_PROB,
    DECODER_PAD_INDEX,
    # GENERAL
    BATCH_SIZE,
    LEARNING_RATE,
    TEATHER_FORCING_PROB,
    OPTIMIZER,
    NUMBER_OF_ITERATIONS,
    DEVICE,
)
from DataPreparator import ImageDataset
from EncoderDecoder import FeatureEncoder, DecoderWithAttention
from vocabulary import max_len, vocab_size
from CaptionCoder import deTokenizeCaptions
from evaluation import evaluate
from HelperFunctions import saveLoss, saveAccuracy

import os
import random
import torch
import torch.nn as nn


def train_batch(
    input_features,     # (batch_size, feature_len)
    input_sentences,    # (batch_size, sentence_len)
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    tf_prob,
):
    global max_len

    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()

    loss = 0

    encoder_output = encoder(input_features)
    batch_size, encoder_output_size = encoder_output.shape

    hidden, cell = decoder.getInitialHidden(batch_size, encoder_output_size)

    use_teacher_forcing = True if random.random() < tf_prob else False
    outputs = None
    for word_idx in range(1, max_len):
        if use_teacher_forcing: # Use target as input
            input_decoder = input_sentences[:, 0:word_idx]
            
        else:   # Use prediction as input
            input_decoder = input_sentences[:, 0:1]
            if not (outputs == None):
                prediction = getPredictions(outputs)
                input_decoder = torch.cat((input_decoder, prediction), dim=-1)

        outputs, (hidden, cell) = decoder(
            input_decoder,
            encoder_output,
            hidden,
            cell
        )

        target = input_sentences[:, 1:word_idx + 1]
        loss += criterion(outputs.permute(0, 2, 1), target)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), outputs[0]


def getPredictions(outputs):
    predictions = outputs.argmax(-1)
    return predictions


def train_loop(
    trainloader,
    encoder,
    decoder,
    optimizer,
    n_iters,
    lr,
    tf_prob,   # Probability of using target as input
    setting_filename,
    print_every=1,
):
    encoder_file = f'{setting_filename}_encoder.pt'
    decoder_file = f'{setting_filename}_decoder.pt'
    encoder_model_path = os.path.join('results', 'models', encoder_file)
    decoder_model_path = os.path.join('results', 'models', decoder_file)

    losses = []
    train_accuracy = []
    validation_accuracy = []

    print_loss_total = 0
    batch_loss_total = 0
    best_avg_update_loss = 0

    encoder.train()
    decoder.train()

    encoder_optimizer = optimizer(encoder.parameters(), lr=lr)
    decoder_optimizer = optimizer(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    def train_iter(iter):
        nonlocal trainloader, encoder, decoder, optimizer, criterion, tf_prob
        nonlocal encoder_model_path, decoder_model_path, print_every
        nonlocal losses, train_accuracy, validation_accuracy
        nonlocal print_loss_total, batch_loss_total, best_avg_update_loss
        
        for i, (_, sentences, features) in enumerate(trainloader):                
            # Send batch to device
            sentences = sentences.to(DEVICE)
            features = features.to(DEVICE)

            loss, output = train_batch(
                    features,
                    sentences,
                    encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    criterion,
                    tf_prob,
                )

            losses.append((iter, i, loss))
            batch_loss_total += loss
            print_loss_total += loss

            if (i+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                prediction = getPredictions(output)
                print(f'{print_loss_avg} : {iter}[{i}]')
                print(f'Prediction: {deTokenizeCaptions(prediction)}')
                print(f'Target: {deTokenizeCaptions(sentences[0, 1:])}')

        # Save model if average batch loss over an iteration has decreased
        isFirstIteration = iter == 0
        avg_update_loss = batch_loss_total / i
        isBetterLoss = best_avg_update_loss > avg_update_loss
        if isFirstIteration or isBetterLoss:
            torch.save(encoder, encoder_model_path)
            torch.save(decoder, decoder_model_path)

            
        train_accuracy.append(evaluate(encoder, decoder, 'train'))
        validation_accuracy.append(evaluate(encoder, decoder, 'dev'))

    for iter in range(n_iters):
        try:
            train_iter(iter)
        except KeyboardInterrupt:
            raise SaveFiles(losses, train_accuracy, validation_accuracy)
        
    return losses, train_accuracy, validation_accuracy


class SaveFiles(Exception):
    pass


def train_main(settings):
    # Initialize
    trainloader = initializeDataLoader(type='train')
    encoder = initializeEncoder()
    decoder = initializeDecoder()

    # Save settings
    
    # Run training loop
    try:
        losses, train_acc, val_acc = train_loop(
            trainloader=trainloader,
            encoder=encoder,
            decoder=decoder,
            optimizer=OPTIMIZER,
            n_iters=NUMBER_OF_ITERATIONS,
            lr=LEARNING_RATE,
            tf_prob=TEATHER_FORCING_PROB,
            setting_filename=settings,
        )
    except SaveFiles as sf:
        losses, train_acc, val_acc = sf.args
        
    finally:
        saveLoss(losses, settings)
        saveAccuracy(train_acc, 'train', settings)
        saveAccuracy(val_acc, 'dev', settings)
        pass


def initializeDecoder():
    model = DecoderWithAttention(
        vocab_len=vocab_size,
        sentence_len=max_len,
        lstm_hidden_len=DECODER_LSTM_HIDDEN_SIZE,
        device=DEVICE,
        number_layers=NUMBER_OF_LSTM_LAYERS,
        bidirectional=BIDERECTIONAL_LSTM,
        drop_prob=DECODER_DROP_PROB,
        pad_token=DECODER_PAD_INDEX,
    ).to(DEVICE)

    return model


def initializeEncoder():
    feature_size = 25088
    model = FeatureEncoder(
        feature_len=feature_size,
        output_len=ENCODER_OUTPUT_SIZE,
        hidden_layer1_len=ENCODER_HIDDEN_1_SIZE,
        hidden_layer2_len=ENCODER_HIDDEN_2_SIZE,
        device=DEVICE,
        drop_prob=ENCODER_DROP_PROB,
    ).to(DEVICE)

    return model


def initializeDataLoader(type: str = 'train'):
    train_dataset = ImageDataset(f'{type}_labels.txt')
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return trainloader


if __name__ == '__main__':
    train_main('test')
