import torch.optim as optim
import torch
from vocabulary import PAD_token

# Settings specifying a training
## Encoder
ENCODER_OUTPUT_SIZE = 256
ENCODER_HIDDEN_1_SIZE = 2**12
ENCODER_HIDDEN_2_SIZE = 2**11
ENCODER_DROP_PROB = 0.1

## Decoder
DECODER_LSTM_HIDDEN_SIZE = ENCODER_OUTPUT_SIZE  # MUST BE EQUAL TO ENCODER_OUTPUT_SIZE
NUMBER_OF_LSTM_LAYERS = 5
BIDERECTIONAL_LSTM = False
DECODER_DROP_PROB = 0.1
DECODER_PAD_INDEX = PAD_token

## General
BATCH_SIZE = 32
LEARNING_RATE = 0.005
OPTIMIZER = optim.Adam
NUMBER_OF_ITERATIONS = 300
TEATHER_FORCING_PROB = 1

## DEVICE
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
