from math import floor

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..networks.base import BaseModel


class BiGRU(BaseModel):
    def __init__(
        self,
        embed_vecs,
        num_classes,
        rnn_dim=512,
        num_layers=1,
        dropout=0.2,
        activation='tanh',
        **kwargs
    ):
        super(BiGRU, self).__init__(embed_vecs, dropout, activation, **kwargs)

        emb_dim = embed_vecs.shape[1]

        # BiGRU
        self.rnn = nn.GRU(emb_dim, floor(rnn_dim/2), num_layers,
                          bidirectional=True, batch_first=True)

        self.W = nn.Linear(rnn_dim, rnn_dim)
        xavier_uniform_(self.W.weight)

        # context vectors for computing attention
        self.U = nn.Linear(rnn_dim, num_classes)
        xavier_uniform_(self.U.weight)

        # linear output
        self.final = nn.Linear(rnn_dim, num_classes)
        xavier_uniform_(self.final.weight)

    def forward(self, text, lengths):
        x = self.embedding(text) # (batch_size, length, rnn_dim)
        x = self.embed_drop(x) # (batch_size, length, rnn_dim)

        packed_inputs = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(packed_inputs)
        x = pad_packed_sequence(x)[0]
        x = x.permute(1,0,2)

        x = torch.tanh(x)

        # Apply per-label attention
        alpha = torch.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        # Document representations are weighted sums using the attention
        m = alpha.matmul(x) # (batch_size, label size, rnn_dim)

        # Compute a probability for each label
        x = self.final.weight.mul(m).sum(dim=2).add(self.final.bias) # (batch_size, num_classes)

        return {'logits': x, 'attention': alpha}
