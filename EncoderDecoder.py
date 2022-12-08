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
        features,   # (batch_size, number_layers, feature_len)
    ):

        denseFeature = self.Linear1(features.squeeze(1))
        denseFeature = self.dropout(denseFeature)
        denseFeature = self.relu(denseFeature)

        denseFeature = self.Linear2(denseFeature)
        denseFeature = self.dropout(denseFeature)
        denseFeature = self.relu(denseFeature)

        denseFeature = self.Linear3(denseFeature)
        denseFeature = self.dropout(denseFeature)
        output = self.relu(denseFeature)  # (batch_size ,output_size)

        return output.unsqueeze(1) # (batch_size, number_layers, output_size)



class Attention(nn.Module):
    def __init__(
        self, decoder_out_dim, attention_dim, encoder_dim=512
    ):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.hidden_map = nn.Linear(decoder_out_dim, attention_dim)
        self.feature_map = nn.Linear(encoder_dim, attention_dim)
        
        self.attention_map = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(
        self, 
        features,
        hidden,
    ):
        mapped_features = self.feature_map(features)
        mapped_hidden = self.hidden_map(hidden)
        
        combined_states = torch.tanh(mapped_features + mapped_hidden.unsqueeze(1))
        attn_scores = self.attention_map(combined_states).squeeze(2)
        
        alphas = self.softmax(attn_scores)
        
        attn_weights = features * alphas.unsqueeze(2)
        attn_weights = attn_weights.sum(dim=1)
        
        return alphas, attn_weights


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        attention_dim,
        encoder_dim,
        decoder_dim,
        device,
        drop_prob=0.1,
        pad_token=None,
    ):
        super(DecoderWithAttention, self).__init__()
        self.device = device

        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.attention = Attention(decoder_dim, attention_dim).to(device)
        self.lstm_cell = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)
        self.LinearMap = nn.Linear(
            in_features=decoder_dim,
            out_features=vocab_size,
        )

    def forward(
        self,
        sentences,  # (batch_size)
        encoder_features, # (batch_size, number_of_layers=49, encoder_output_size=512)
        prev_hidden,    # (batch_size, hidden_size)
        prev_cell,  # (batch_size, hidden_size)
    ):
        # Embedding
        embedded = self.embedding(sentences)
        embedded = self.dropout(embedded)   # (batch_size, embedding_dim)

        # Attention
        alphas, attn_weights = self.attention(encoder_features, prev_hidden)
        
        lstm_input = torch.cat((embedded, attn_weights) , -1)
        hidden, cell = self.lstm_cell(lstm_input, (prev_hidden, prev_cell))
        
        hidden = self.dropout(hidden)
        
        output = self.LinearMap(hidden)
        
        return output, (hidden, cell), (alphas, attn_weights)


    def getInitialHidden(self, batch_size):
        hidden = torch.zeros((batch_size, self.decoder_dim)).to(self.device)
        cell = torch.zeros((batch_size, self.decoder_dim)).to(self.device)

        return hidden, cell


