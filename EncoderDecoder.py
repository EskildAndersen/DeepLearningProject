import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vocabulary import PAD_token
# Så vi skal basically finde ud hidden size som skal encodes. Dernæst hvad gør nn.Encoding præcist?
# nn.GRU = multi-layer gated recurrent unit


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        feature_len,
        output_len,
        device,
        hidden_len_1=2**12,
        hidden_len_2=2**11,
        dropout=0.1,
    ):
        super(FeatureEncoder, self).__init__()
        self.device = device
        
        self.feature_len = feature_len
        self.output_len = output_len
        self.hidden_len1 = hidden_len_1
        self.hidden_len2 = hidden_len_2

        self.Linear1 = nn.Linear(
            in_features=feature_len,
            out_features=self.hidden_len1
        )
        self.Linear2 = nn.Linear(
            in_features=self.hidden_len1,
            out_features=self.hidden_len2
        )
        self.Linear3 = nn.Linear(
            in_features=self.hidden_len2,
            out_features=self.output_len
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(
        self, 
        features,   # (batch_size, feature_len)
    ):  
        
        denseFeature = self.Linear1(features.squeeze(1))
        denseFeature = self.dropout(denseFeature)
        denseFeature = self.relu(denseFeature)

        denseFeature = self.Linear2(denseFeature)
        denseFeature = self.dropout(denseFeature)
        denseFeature = self.relu(denseFeature)

        denseFeature = self.Linear3(denseFeature)
        denseFeature = self.dropout(denseFeature)
        output = self.relu(denseFeature)  # (batch_size, output_size)
        
        return output
    
    

class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_len,
        sentence_len,
        hidden_len,
        encoder_output_len,
        device,
        dropout=0.1,
        pad_token=PAD_token,
    ):
        super(DecoderWithAttention, self).__init__()
        self.device = device

        self.vocab_size = vocab_len
        self.hidden_size = hidden_len
        self.sentence_len = sentence_len
        self.concat_size = self.hidden_size + encoder_output_len
        self.pad_token = pad_token
        
        self.max_norm = None
        
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.sentence_len,
            padding_idx=self.pad_token,
            max_norm=self.max_norm
        )
        
        self.lstm = nn.LSTM(
            self.sentence_len,
            self.hidden_size,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.att_softmax = nn.Softmax(dim=1)
        self.Linear1 = nn.Linear(
            in_features=self.concat_size,
            out_features=self.hidden_size,
        )
        self.LinearOut = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
        )
        
    def forward(
        self,
        sentences,  # (batch_size, sentence_len)
        encoded_features,    # (batch_size, encoded_output_size)
        encoder_hidden,
        encoder_cell,
    ):
        # Embedding
        embedded = self.embedding(sentences)
        embedded = self.dropout(embedded)
        
        # Attention
        scores = self.get_score(embedded, encoder_hidden)
        alphas = self.att_softmax(scores)
        alphas = alphas.unsqueeze(2)
        context_vector = torch.bmm(alphas, encoded_features)
        
        # "Do Something"
        output = torch.cat((embedded, context_vector), dim=-1)
        output = self.Linear1(output)
        output = self.relu(output)
        
        # LSTM
        output, hidden = self.lstm(output, (encoder_hidden, encoder_cell))
        output = self.LinearOut(output)
        
        return output, hidden

    
    def getInitialHidden(self, batch_size, encoded_output_size):
        hidden = torch.zeros((batch_size, encoded_output_size)).to(self.device)
        cell = torch.zeros((batch_size, encoded_output_size)).to(self.device)

        return hidden, cell
    
    
    def get_score(self, hidden, features):
        return torch.sum(hidden * features, dim=2)
    

class Encoder(nn.Module):
    def __init__(
        self,
        sentence_len,
        feature_len,
        vocab_len,
        hidden_size,
        device,
        dropout_p=0.1,
        lstm_layers=5,
        pad_token=PAD_token,
    ):
        super(Encoder, self).__init__()
        self.device = device

        self.vocab_size = vocab_len
        self.hidden_size = hidden_size
        self.sentence_len = sentence_len
        self.lstm_layers = lstm_layers
        self.attention_dim = hidden_size
        self.max_norm = None

        self.dropout = nn.Dropout(dropout_p)

        # RNN
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.sentence_len,
            padding_idx=pad_token,
            max_norm=self.max_norm
        )
        self.lstm = nn.LSTM(
            self.sentence_len,
            self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True
        )
        self.lstm_cell = nn.LSTMCell(
            self.sentence_len,
            self.hidden_size,
        )

        # Feature extration (dense down)
        self.Linear1 = nn.Linear(in_features=feature_len, out_features=2**12)
        self.Linear2 = nn.Linear(in_features=2**12, out_features=2**11)
        self.Linear3 = nn.Linear(in_features=2**11, out_features=hidden_size)
        self.relu = nn.ReLU()

        # Mapping layer
        self.LinearMap1 = nn.Linear(
            in_features=self.hidden_size*2,
            out_features=2**12
        )
        self.LinearMap2 = nn.Linear(
            in_features=2**12,
            out_features=self.vocab_size
        )
        self.softmax = nn.Softmax(dim=2)

        # Attention
        self.feature_att = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.attention_dim,
        )
        self.sentence_att = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.attention_dim,
        )
        self.att_softmax = nn.Softmax(dim=1)

    def forward(
        self,
        sentences,  # (batch_size, sentence_len)
        features    # (batch_size, feature_len)
    ):
        # batch_size = sentences.shape[0]

        # # RNN CELL
        # embedded = self.embedding(sentences)  # .view(1, 1, -1)
        # embedded = self.dropout(embedded)

        # RNN_outputs = []
        # hidden, cell = self.lstm_init(batch_size)
        # for i in range(self.sentence_len):
        #     words = embedded[:, i, :]
        #     hidden, cell = self.lstm_cell(words, (hidden, cell))
        #     RNN_outputs.append(hidden)

        # RNN_output = torch.stack(RNN_outputs, dim=1)

        # RNN
        embedded = self.embedding(sentences)  # .view(1, 1, -1)
        embedded = self.dropout(embedded)
        # output (batch_size, sentence_len, hidden_size)
        RNN_output, hidden = self.lstm(embedded)

        # Feature extraction
        # features: (batch_size, feature_len)
        denseFeature = self.Linear1(features)
        denseFeature = self.dropout(denseFeature)
        denseFeature = self.relu(denseFeature)

        denseFeature = self.Linear2(denseFeature)
        denseFeature = self.dropout(denseFeature)
        denseFeature = self.relu(denseFeature)

        denseFeature = self.Linear3(denseFeature)
        denseFeature = self.dropout(denseFeature)
        denseFeature = self.relu(denseFeature)  # (batch_size, hidden_size)

        denseFeature = denseFeature.unsqueeze(
            1)  # (batch_size, 1, hidden_size)

        # Attention
        scores = self.get_score(RNN_output, denseFeature)
        alphas = self.att_softmax(scores)
        alphas = alphas.unsqueeze(2)
        context_vector = torch.bmm(alphas, denseFeature)

        # (batch_size, sentence_len,  hidden_size)
        x = torch.cat((RNN_output, context_vector), dim=-1)

        # Linear Mapping
        x = self.LinearMap1(x)
        x = self.relu(x)
        x = self.LinearMap2(x)
        # x = self.relu(x)

        # self.softmax(x)     # (batch_size, sentence_len, vocab_size)
        output = x

        return output

    def lstm_init(self, batch_size):
        hidden = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        cell = torch.zeros((batch_size, self.hidden_size)).to(self.device)

        return hidden, cell

    def get_score(self, hidden, features):
        return torch.sum(hidden * features, dim=2)
