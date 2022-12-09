import torch
import torch.nn as nn
from vocabulary import max_len
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(
        self
    ):
        super(CNNEncoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.parameters():
            param.requires_grad_(False)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(vgg16.children())[:-1]
        self.vgg16 = nn.Sequential(*modules)

    def forward(
        self, 
        images,
    ):

        # Feed images through VGG16
        features = self.vgg16(images)
        features = features.permute(0, 2,3,1)
        features = features.view(features.size(0), -1, features.size(-1)).squeeze()

        return features

    def fine_tune(
        self, 
        fine_tune=True
    ):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """

        for p in self.vgg16.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.vgg16.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


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
    ):  
        global max_len

        # Embedding
        embedded = self.embedding(sentences)
        embedded = self.dropout(embedded)   # (batch_size, embedding_dim)

        batch_size = sentences.shape[0]

        hidden, cell = self.getInitialHidden(batch_size)


        outputs = []
        alphas = []

        for s in range(max_len-1):

            # Attention
            alpha, attn_weight = self.attention(encoder_features, hidden)
        
            lstm_input = torch.cat((embedded[:, s], attn_weight) , -1)

            hidden, cell = self.lstm_cell(lstm_input, (hidden, cell))
            
            hidden = self.dropout(hidden)
            
            output = self.LinearMap(hidden)

            outputs.append(output)
            alphas.append(alpha)
        
        outputs = torch.stack(outputs, dim = 1)
        alphas = torch.stack(alphas, dim = 1)

        return outputs, alphas


    def getInitialHidden(self, batch_size):
        hidden = torch.zeros((batch_size, self.decoder_dim)).to(self.device)
        cell = torch.zeros((batch_size, self.decoder_dim)).to(self.device)

        return hidden, cell
