'''
Script for training the auto-encoder model.

Hyperparameters and settings are chosen in settings-file

'''

from settings import (
    # DECODER
    EMBEDDING_DIM,
    ATTENTION_DIM,
    DECODER_DIM,
    DECODER_DROP_PROB,
    DECODER_PAD_INDEX,
    # GENERAL
    BATCH_SIZE,
    LEARNING_RATE,
    TEATHER_FORCING_PROB,
    OPTIMIZER,
    WEIGHT_DECAY,
    LR_STEP,
    LOSS_PAD_INDEX,
    NUMBER_OF_ITERATIONS,
    DEVICE,
)
from DataPreparator import ImageDataset
from EncoderDecoder import DecoderWithAttention
from vocabulary import max_len, vocab_size
from CaptionCoder import deTokenizeCaptions
from evaluation import evaluate
from HelperFunctions import saveLoss, saveAccuracy
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


def train_batch(
    input_features,     # (batch_size, feature_len)
    input_sentences,    # (batch_size, sentence_len)
    decoder,
    decoder_optimizer,
    criterion,
):
    global max_len

    decoder_optimizer.zero_grad()

    loss = 0

    outputs, _ = decoder(input_sentences, input_features)

    targets = input_sentences[:, 1:]

    loss = criterion(outputs.permute(0, 2, 1), targets)

    loss.backward()
    decoder_optimizer.step()
    
    return loss.item(), outputs[0]


def getPredictions(outputs):
    predictions = outputs.argmax(-1)
    return predictions


def train_loop(
    trainloader,
    decoder,
    optimizer,
    n_iters,
    lr,  # Probability of using target as input
    setting_filename,
    print_every=20,
):
    decoder_file = f'{setting_filename}_decoder.pt'
    decoder_model_path = os.path.join('results', 'models', decoder_file)

    losses = []
    train_accuracy = []
    validation_accuracy = []

    best_avg_update_loss = 0

    decoder_optimizer = optimizer(decoder.parameters(), lr=lr, weight_decay =  WEIGHT_DECAY)
    scheduler = StepLR(decoder_optimizer, step_size=LR_STEP, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss(ignore_index=LOSS_PAD_INDEX)
    
    def train_iter(iter):
        nonlocal trainloader, decoder, optimizer, criterion
        nonlocal decoder_model_path, print_every
        nonlocal losses, train_accuracy, validation_accuracy
        nonlocal best_avg_update_loss
        
        decoder.train()
        
        print_loss_total = 0
        batch_loss_total = 0
        
        for i, (_,_, sentences, features) in enumerate(trainloader):
        
            # Send batch to device
            sentences = sentences.to(DEVICE)
            features = features.to(DEVICE)

            loss, output = train_batch(
                    features,
                    sentences,
                    decoder,
                    decoder_optimizer,
                    criterion,
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
                print()

        # Save model if average batch loss over an iteration has decreased
        isFirstIteration = iter == 0
        avg_update_loss = batch_loss_total / i
        isBetterLoss = best_avg_update_loss > avg_update_loss
        if isFirstIteration or isBetterLoss:
            best_avg_update_loss = avg_update_loss
            torch.save(decoder, decoder_model_path)

        
        train_acc = evaluate(decoder, 'train')
        val_acc = evaluate(decoder, 'dev')
        train_accuracy.append(train_acc)
        validation_accuracy.append(val_acc)
        
        print(f'Train evaluation: {train_acc}')
        print(f'Validation evaluation: {val_acc}\n')
        
        scheduler.step()

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
    decoder = initializeDecoder()
    
    # Run training loop
    try:
        losses, train_acc, val_acc = train_loop(
            trainloader=trainloader,
            decoder=decoder,
            optimizer=OPTIMIZER,
            n_iters=NUMBER_OF_ITERATIONS,
            lr=LEARNING_RATE,
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
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        attention_dim=ATTENTION_DIM,
        encoder_dim=512,
        decoder_dim=DECODER_DIM,
        device=DEVICE,
        drop_prob=DECODER_DROP_PROB,
        pad_token=DECODER_PAD_INDEX,
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
