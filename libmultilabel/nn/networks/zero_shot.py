from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Embedding, GRUEncoder, LabelwiseAttention, CNNEncoder


class CBiGRULWAN(nn.Module):

    def __init__(self, embed_vecs, num_classes, embed_dropout, encoder_dropout,
                 rnn_dim, rnn_layers):
        super().__init__()
        embed_dim = embed_vecs.shape[1]
        self.embedding = Embedding(embed_vecs, embed_dropout)
        assert rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        self.encoder = GRUEncoder(embed_dim, rnn_dim // 2, rnn_layers, encoder_dropout)
        self.attention = LabelAttention(rnn_dim, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.output = SharedLinearLabelProduct(rnn_dim, embed_dim)

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, sequence_length, embed_dim)
        Q = self.embedding(input['label_desc']) # (num_classes, sequence_length, embed_dim)
        Q = Q.mean(dim=1) # (num_classes, embed_dim)
        Q = self.linear(Q)
        x = self.encoder(x, input['length'])  # (batch_size, sequence_length, num_filter)
        x = self.attention(x, Q) # (batch_size, num_classes, hidden_dim)
        x = self.output(x, Q)
        return {'logits': x}


class CCNNLWAN(nn.Module):

    def __init__(self, embed_vecs, num_classes, embed_dropout, encoder_dropout,
                 filter_sizes, num_filter_per_size, activation):
        super().__init__()
        embed_dim = embed_vecs.shape[1]
        self.embedding = Embedding(embed_vecs, embed_dropout)
        conv_output_size = num_filter_per_size * len(filter_sizes)
        self.encoder = CNNEncoder(embed_dim, filter_sizes,
                                  num_filter_per_size, activation,
                                  encoder_dropout, channel_last=True)
        self.attention = LabelAttention(conv_output_size, embed_dim)

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.output = SharedLinearLabelProduct(conv_output_size, embed_dim)

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, sequence_length, embed_dim)
        Q = self.embedding(input['label_desc']) # (num_classes, sequence_length, embed_dim)
        Q = Q.mean(dim=1) # (num_classes, embed_dim)
        Q = self.linear(Q)
        x = self.encoder(x)  # (batch_size, sequence_length, num_filter)
        x = self.attention(x, Q) # (batch_size, num_classes, hidden_dim)
        x = self.output(x, Q)
        return {'logits': x}


class BiGRUMetaNetwork(nn.Module):
    def __init__(self, embed_dim, output_size, dropout=0):
        super().__init__()
        assert output_size % 2 == 0, """`output_size` should be even."""
        self.hidden_size = output_size // 2
        self.label_encoder = GRUEncoder(embed_dim, self.hidden_size, 1, dropout)
        self.linear = nn.Linear(output_size, output_size)

    def forward(self, label_reps, label_length):
        label_reps = self.label_encoder(label_reps, label_length)  # (num_classes, sequence_length, output_size)
        label_reps = torch.cat([label_reps[:, -1, :self.hidden_size],
                                label_reps[:, 0, -self.hidden_size:]], dim=-1) # (num_classes, output_size)
        label_reps = self.linear(label_reps) # (num_classes, output_size)
        return label_reps


class CNNMetaNetwork(nn.Module):
    def __init__(self, filter_sizes, activation, embed_dim, output_size, dropout=0):
        super().__init__()
        assert len(filter_sizes) == 1, "multi-filter-size is not supported"
        self.label_encoder = CNNEncoder(embed_dim, filter_sizes,
                                        output_size, activation,
                                        dropout, num_pool=1)
        self.linear = nn.Linear(output_size, output_size)

    def forward(self, label_reps, label_length):
        label_reps = self.label_encoder(label_reps)  # (num_classes, output_size, 1)
        label_reps = torch.squeeze(label_reps, 2) # (num_classes, output_size)
        label_reps = self.linear(label_reps) # (num_classes, output_size)
        return label_reps


class MetaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.meta_network = None

    def forward(self, x, label_reps, label_length):
        label_reps = self.meta_network(label_reps, label_length) # (num_classes, feature_size)
        attention = torch.matmul(x, label_reps.T).transpose(1, 2) # (batch_size, num_class, sequence_length)
        attention = F.softmax(attention, -1) # (batch_size, num_class, sequence_length)
        return torch.bmm(attention, x)  # (batch_size, num_classes, feature_size)


class BiGRUMetaAttention(MetaAttention):
    def __init__(self, embed_size, feature_size, dropout=0):
        super().__init__()
        self.meta_network = BiGRUMetaNetwork(embed_size, feature_size, dropout)


class CNNMetaAttention(MetaAttention):
    def __init__(self, filter_sizes, activation, embed_size, feature_size, dropout=0):
        super().__init__()
        self.meta_network = CNNMetaNetwork(filter_sizes, activation, embed_size, feature_size, dropout)


class MetaMLLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.meta_network = None

    def forward(self, x, label_reps, label_length):
        label_reps = self.meta_network(label_reps, label_length) # (num_classes, feature_size)
        return (x * label_reps).sum(dim=-1) # (batch_size, num_classes)


class BiGRUMetaMLLinear(MetaMLLinear):
    def __init__(self, embed_size, feature_size, dropout=0):
        super().__init__()
        self.meta_network = BiGRUMetaNetwork(embed_size, feature_size, dropout)


class CNNMetaMLLinear(MetaMLLinear):
    def __init__(self, filter_sizes, activation, embed_size, feature_size, dropout=0):
        super().__init__()
        self.meta_network = CNNMetaNetwork(filter_sizes, activation, embed_size, feature_size, dropout)


class ZeroMetaBiGRU(nn.Module):

    def __init__(self, embed_vecs, num_classes, embed_dropout, encoder_dropout,
                 rnn_dim, rnn_layers):
        super().__init__()
        embed_dim = embed_vecs.shape[1]
        self.embedding = Embedding(embed_vecs, embed_dropout)
        assert rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        self.doc_encoder = GRUEncoder(embed_dim, rnn_dim // 2, rnn_layers, encoder_dropout)
        self.attention = BiGRUMetaAttention(embed_dim, rnn_dim, encoder_dropout)
        self.output = BiGRUMetaMLLinear(embed_dim, rnn_dim, encoder_dropout)

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, sequence_length, embed_dim)
        label_reps = self.embedding(input['label_desc'])
        label_length = torch.count_nonzero(input['label_desc'], dim=-1)
        x = self.doc_encoder(x, input['length']) # (batch_size, sequence_length, num_filter)
        x = self.attention(x, label_reps, label_length)
        out = self.output(x, label_reps, label_length)
        return {'logits': out}


class ZeroMetaCNN(nn.Module):

    def __init__(self, filter_sizes, num_filter_per_size, activation,
                 embed_vecs, num_classes, embed_dropout, encoder_dropout):
        super().__init__()
        embed_dim = embed_vecs.shape[1]
        self.embedding = Embedding(embed_vecs, embed_dropout)
        self.doc_encoder = CNNEncoder(embed_dim, filter_sizes,
                                      num_filter_per_size, activation,
                                      encoder_dropout, channel_last=True)
        output_size = len(filter_sizes) * num_filter_per_size
        self.attention = CNNMetaAttention(filter_sizes, activation, embed_dim, output_size, encoder_dropout)
        self.output = CNNMetaMLLinear(filter_sizes, activation, embed_dim, output_size, encoder_dropout)

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, sequence_length, embed_dim)
        label_reps = self.embedding(input['label_desc'])
        x = self.doc_encoder(x) # (batch_size, sequence_length, num_filter)
        x = self.attention(x, label_reps, None)
        out = self.output(x, label_reps, None)
        return {'logits': out}


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
        # x = F.normalize(x, dim=-1)
        if len(x.shape) == 3:
            return (Q * x).sum(-1)
        return torch.matmul(x, Q.T)
