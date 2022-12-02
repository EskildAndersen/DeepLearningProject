import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Så vi skal basically finde ud hidden size som skal encodes. Dernæst hvad gør nn.Encoding præcist?
# nn.GRU = multi-layer gated recurrent unit


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, feature_len, dropout_p=0.1):
        super(Encoder, self).__init__()

        # til venstre
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
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

        # combine shit

        output = torch.add(rnn, featureThing, alpha=1)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Her skal vi nok være opmærksom på embedding-leddet igen samt attention layeret.
# Der sker softmax i forward samt torch.bmm -> Performs a batch matrix-matrix product of matrices stored in input and mat2.


MAX_LENGTH = 500010413


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size,  device=device)
