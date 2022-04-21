from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Embedding, GRUEncoder, LSTMEncoder, CNNEncoder, LabelwiseAttention, LabelwiseMultiHeadAttention, LabelwiseLinearOutput


class CBiGRULWAN(nn.Module):

    def __init__(self, embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim, rnn_layers, label_embedding):
        super(CBiGRULWAN, self).__init__()
        self.embedding = Embedding(embed_vecs, embed_dropout)
        assert rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        self.encoder = GRUEncoder(embed_vecs.shape[1], rnn_dim // 2, rnn_layers, encoder_dropout)
        self.linear = nn.Linear(rnn_dim, embed_vecs.shape[1])
        self.Q = nn.Parameter(label_embedding, requires_grad=False)
        self.output = LabelProduct(label_embedding)

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, sequence_length, embed_dim)
        x = self.encoder(x, input['length'])  # (batch_size, sequence_length, hidden_dim)
        V = torch.tanh(self.linear(x))  # (batch_size, sequence_length, hidden_dim)
        A = F.softmax(torch.matmul(self.Q, V.transpose(1, 2)), -1) # (batch_size, num_classes, sequence_length)
        E = torch.bmm(A, x)  # (batch_size, num_classes, hidden_dim)
        # x = (self.Q * E).sum(-1) # (batch_size, num_classes)
        x = self.output(E)
        return {'logits': x}


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
    def __init__(self, input_dim, label_embedding):
        super().__init__()
        self.Q = nn.Parameter(label_embedding, requires_grad=False)
        output_dim = label_embedding.shape[-1]
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return (self.Q * x).sum(-1)
