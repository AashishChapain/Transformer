import math
import torch
from torch import nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # embedding size of the model
        self.vocab_size = vocab_size # size of our vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # from attention is all you need paper

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        # seq_length is the total tokens (or words) in input sequence
        # d_model is the embedding size
        # so our positional encoder should return vector of shape (seq_len, d_model) i.e pos embedding for each token (or word)

        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # create a div term to divide the position created above
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply the sin function to even postion
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply the cos function to odd position
        pe[:, 1::2] = torch.cos(position * div_term)

        # reshapign the pe vector for the batch
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # this pe vector is saved for later use also
        # we don't have to calculate it each time because this will be same for all upto same seq_len
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x and pe are of same shape i.e. (N, seq_len, d_model) N -> batch size
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
