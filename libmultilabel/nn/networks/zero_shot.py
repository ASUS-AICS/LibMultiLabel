from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Embedding, GRUEncoder


class ZeroBiGRU(nn.Module):

    def __init__(self, embed_vecs, num_classes, embed_dropout, encoder_dropout,
                 rnn_dim, rnn_layers, label_embedding, output_type):
        super().__init__()
        embed_dim = embed_vecs.shape[1]
        self.embedding = Embedding(embed_vecs, embed_dropout)
        assert rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        self.encoder = GRUEncoder(embed_dim, rnn_dim // 2, rnn_layers, encoder_dropout)
        self.Q = nn.Parameter(label_embedding, requires_grad=False)
        self.attention = LabelAttention(rnn_dim, embed_dim)
        self.output_type = output_type
        if output_type == 'inner':
            self.output = SharedLinearLabelProduct(rnn_dim, embed_dim)
        elif output_type == 'linear':
            self.output = SharedLinear(rnn_dim)
        else:
            raise ValueError(f'Invalid output type: {output_type}')

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, sequence_length, embed_dim)
        x = self.encoder(x, input['length'])  # (batch_size, sequence_length, hidden_dim)
        x = self.attention(x, self.Q) # (batch_size, num_classes, hidden_dim)
        if self.output_type == 'inner':
            x = self.output(x, self.Q)
        else:
            x = self.output(x)
        return {'logits': x}


class LabelAttention(nn.Module):

    def __init__(self, input_size, embed_dim):
        super().__init__()
        self.linear = nn.Linear(input_size, embed_dim)

    def forward(self, x, Q):
        V = torch.tanh(self.linear(x))  # (batch_size, sequence_length, embed_dim)
        A = F.softmax(torch.matmul(Q, V.transpose(1, 2)), -1) # (batch_size, num_classes, sequence_length)
        E = torch.bmm(A, x)  # (batch_size, num_classes, hidden_dim)
        return E


class LabelProduct(nn.Module):
    def __init__(self, label_embedding):
        super().__init__()
        self.Q = nn.Parameter(label_embedding, requires_grad=False)

    def forward(self, x):
        return (self.Q * x).sum(-1)


class SharedLinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.squeeze(self.linear(x), -1)


class SharedLinearLabelProduct(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x, Q):
        x = self.linear(x)
        return (Q * x).sum(-1)
