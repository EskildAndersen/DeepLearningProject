'''
Script to initialize hyperparameters and settings to be used in training. 

'''

import torch.optim as optim
import torch
from vocabulary import PAD_token

# Settings specifying a training
## Decoder
EMBEDDING_DIM = 256
ATTENTION_DIM = 256
DECODER_DIM = 512
DECODER_DROP_PROB = 0.3
DECODER_PAD_INDEX = None

## General
BATCH_SIZE = 32
LEARNING_RATE = 0.005
OPTIMIZER = optim.Adam
LR_STEP = 200
NUMBER_OF_ITERATIONS = 300
TEATHER_FORCING_PROB = 1
LOSS_PAD_INDEX = PAD_token

## DEVICE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")