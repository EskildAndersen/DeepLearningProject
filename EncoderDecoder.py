import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Så vi skal basically finde ud hidden size som skal encodes. Dernæst hvad gør nn.Encoding præcist?
# nn.GRU = multi-layer gated recurrent unit


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, feature_len, vocab_len, device, padding_idx, dropout_p=0.1):
        super(Encoder, self).__init__()

        self.device = device

        # til venstre
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_len, input_size, padding_idx)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # til højre
        self.Linear1 = nn.Linear(in_features=feature_len, out_features=2500)
        self.Linear2 = nn.Linear(in_features=2500, out_features=2500)
        self.Linear3 = nn.Linear(in_features=2500, out_features=hidden_size)
        self.relu = nn.ReLU()

    def forward(self, input, input2, hidden):

        # til venstre
        embedded = self.embedding(input2.int())  # .view(1, 1, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded)

        # til højre
        featureThing = self.Linear1(input)
        featureThing = self.dropout(featureThing)
        featureThing = self.relu(featureThing)

        featureThing = self.Linear2(featureThing)
        featureThing = self.dropout(featureThing)
        featureThing = self.relu(featureThing)

        featureThing = self.Linear3(featureThing)
        featureThing = self.dropout(featureThing)
        featureThing = self.relu(featureThing)
        # combine shit

        output = torch.cat(
            (output, featureThing), dim=1)  # .view(1, 1, -1)

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

# Her skal vi nok være opmærksom på embedding-leddet igen samt attention layeret.
# Der sker softmax i forward samt torch.bmm -> Performs a batch matrix-matrix product of matrices stored in input and mat2.


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, device, dropout_p=0.1):
        super(Decoder, self).__init__()

        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.vocab_size)
        self.relu = nn.ReLU()
        self.attention = nn.Linear(self.vocab_size, self.vocab_size)

    def forward(self, encoder_outputs):
        pass
        x = self.linear1(encoder_outputs)
        x = self.relu(x)
        x = self.linear2(x)

        return x

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
