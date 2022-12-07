import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    def __init__(
        self,
        feature_len,
        output_len,
        device,
        hidden_layer1_len=2**12,
        hidden_layer2_len=2**11,
        drop_prob=0.1,
    ):
        super(FeatureEncoder, self).__init__()
        self.device = device

        self.feature_len = feature_len
        self.output_len = output_len
        self.hidden_layer1_len = hidden_layer1_len
        self.hidden_layer2_len = hidden_layer2_len

        self.Linear1 = nn.Linear(
            in_features=feature_len,
            out_features=self.hidden_layer1_len
        )
        self.Linear2 = nn.Linear(
            in_features=self.hidden_layer1_len,
            out_features=self.hidden_layer2_len
        )
        self.Linear3 = nn.Linear(
            in_features=self.hidden_layer2_len,
            out_features=self.output_len
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

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
        lstm_hidden_len,
        device,
        number_layers=5,
        bidirectional=False,
        drop_prob=0.1,
        pad_token=None,
    ):
        super(DecoderWithAttention, self).__init__()
        self.device = device

        self.vocab_size = vocab_len
        self.lstm_hidden_size = lstm_hidden_len
        self.sentence_len = sentence_len
        self.concat_size = self.lstm_hidden_size*2
        self.pad_token = pad_token

        self.number_layers = number_layers
        self.bidirectional = bidirectional

        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.sentence_len,
            padding_idx=self.pad_token,
        )

        self.lstm = nn.LSTM(
            self.sentence_len,
            self.lstm_hidden_size,
            batch_first=True,
            dropout=drop_prob,
            num_layers=self.number_layers,
            bidirectional=self.bidirectional,
        )

        self.attention = self.get_score
        self.att_softmax = nn.Softmax(dim=1)

        self.LinearMap1 = nn.Linear(
            in_features=self.concat_size,
            out_features=self.lstm_hidden_size,
        )
        self.LinearMap2 = nn.Linear(
            in_features=self.lstm_hidden_size,
            out_features=self.vocab_size,
        )

        self.soft_max = nn.LogSoftmax(dim=2)

    def forward(
        self,
        sentences,  # (batch_size, sentence_len)
        # encoder_outputs (batch_size, encoder_output_size)
        encoder_features,
        init_hidden,
        init_cell,
    ):
        # Embedding
        embedded = self.embedding(sentences)
        embedded = self.dropout(embedded)

        # Generate new hidden state for decoder
        lstm_out, (hidden, cell) = self.lstm(
            embedded,
            # Because hidden og cell needs (D * number of layers, batch_size, hidden_size)
            (init_hidden, init_cell)
        )

        # Calculate allignment scores
        allignment_scores = self.attention(
            lstm_out, encoder_features.unsqueeze(1))

        # Softmax allignment_scores to obtain attention weights
        attn_weights = self.att_softmax(allignment_scores)

        # Calculating context vector
        context_vector = torch.bmm(
            attn_weights.unsqueeze(2),
            encoder_features.unsqueeze(1)
        )

        # Calculate fines decoder output
        output = torch.cat((lstm_out, context_vector), dim=-1)

        # Classify concatenated vector to vocab
        output = self.LinearMap1(output)
        output = self.relu(output)
        output = self.LinearMap2(output)

        # Softmax output to get prediction probability
        # output = self.soft_max(output)
        # prediction = prediction.argmax(2).squeeze(-1)

        return output, (hidden, cell)

    def getInitialHidden(self, batch_size, encoded_output_size):
        layers = self.number_layers
        size_0 = layers if not self.bidirectional else layers * 2
        hidden = torch.zeros(
            (
                size_0,
                batch_size,
                encoded_output_size
            )
        ).to(self.device)

        cell = torch.zeros(
            (
                self.number_layers,
                batch_size,
                encoded_output_size
            )
        ).to(self.device)

        return hidden, cell

    def get_score(self, hidden, features):
        return torch.sum(hidden * features, dim=2)
