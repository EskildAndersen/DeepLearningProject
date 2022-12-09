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
DECODER_DIM = 256
DECODER_DROP_PROB = 0.5
DECODER_PAD_INDEX = PAD_token

## General
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
OPTIMIZER = optim.Adam
WEIGHT_DECAY = 1e-6 
LR_STEP = 100
NUMBER_OF_ITERATIONS = 100
TEATHER_FORCING_PROB = 1
LOSS_PAD_INDEX = -100 #PAD_token 

## DEVICE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")