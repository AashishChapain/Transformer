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
