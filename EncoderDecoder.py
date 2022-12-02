import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Så vi skal basically finde ud hidden size som skal encodes. Dernæst hvad gør nn.Encoding præcist?
# nn.GRU = multi-layer gated recurrent unit


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, feature_len, vocab_len, padding_index, dropout_p=0.1):
        super(Encoder, self).__init__()

        # til venstre
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_len, input_size, padding_index)
        self.gru = nn.GRU(input_size**2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        # til højre
        self.Linear1 = nn.Linear(in_features=feature_len, out_features=8192)
        self.Linear2 = nn.Linear(in_features=8192, out_features=4096)
        self.Linear3 = nn.Linear(in_features=4096, out_features=hidden_size)
        self.relu = nn.ReLU()

    def forward(self, input, input2, hidden):

        # til venstre
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        rnn, hidden = self.gru(embedded, hidden)

        # til højre
        featureThing = self.Linear1(input2)
        featureThing = self.dropout(featureThing)
        featureThing = self.relu(featureThing)

        featureThing = self.Linear2(featureThing)
        featureThing = self.dropout(featureThing)
        featureThing = self.relu(featureThing)

        featureThing = self.Linear3(featureThing)
        featureThing = self.dropout(featureThing)
        featureThing = self.relu(featureThing)
        
        featureThing = featureThing.unsqueeze(0)

        # combine shit

        output = torch.add(rnn, featureThing, alpha=1)

        return output.squeeze(0).squeeze(0), hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Her skal vi nok være opmærksom på embedding-leddet igen samt attention layeret.
# Der sker softmax i forward samt torch.bmm -> Performs a batch matrix-matrix product of matrices stored in input and mat2.

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.vocab_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, encoder_outputs):
        pass
        x = self.linear1(encoder_outputs)
        x = self.relu(x)
        x = self.linear2(x)
        output = self.softmax(x)
        # output = torch.argmax(output, 1)
        
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
