from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):
    """
    """

    def __init__(self, embed_vecs, dropout=0.2):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embed_vecs, freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return self.dropout(self.embedding(inputs))


class RNNEncoder(ABC, nn.Module):
    """
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(RNNEncoder, self).__init__()
        self.rnn = self._get_rnn(input_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths, **kwargs):
        self.rnn.flatten_parameters()
        idx = torch.argsort(lengths, descending=True)
        packed_inputs = pack_padded_sequence(
            inputs[idx], lengths[idx].cpu(), batch_first=True)
        outputs, _ = pad_packed_sequence(
            self.rnn(packed_inputs)[0], batch_first=True)
        return self.dropout(outputs[torch.argsort(idx)])

    @abstractmethod
    def _get_rnn(self, input_size, hidden_size, num_layers):
        raise NotImplementedError


class GRUEncoder(RNNEncoder):
    """
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUEncoder, self).__init__(input_size, hidden_size, num_layers,
                                         dropout)

    def _get_rnn(self, input_size, hidden_size, num_layers):
        return nn.GRU(input_size, hidden_size, num_layers,
                      batch_first=True, bidirectional=True)


class LSTMEncoder(RNNEncoder):
    """
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMEncoder, self).__init__(input_size, hidden_size, num_layers,
                                          dropout)

    def _get_rnn(self, input_size, hidden_size, num_layers):
        return nn.LSTM(input_size, hidden_size, num_layers,
                       batch_first=True, bidirectional=True)


class CNNEncoder(nn.Module):
    """
    """

    def __init__(self, input_size, filter_sizes,
                 num_filter_per_size, activation):
        super(CNNEncoder, self).__init__()
        if not filter_sizes:
            raise ValueError(f'CNNEncoder expect non-empty filter_sizes. '
                             f'Got: {filter_sizes}')
        self.convs = nn.ModuleList()
        for filter_size in filter_sizes:
            conv = nn.Conv1d(
                in_channels=input_size,
                out_channels=num_filter_per_size,
                kernel_size=filter_size)
            self.convs.append(conv)
        self.activation = getattr(F, activation)

    def forward(self, inputs, lengths):
        h = inputs.transpose(1, 2)  # (batch_size, input_size, length)
        h_list = [conv(h) for conv in self.convs]  # (batch_size, num_filter, length)
        h = torch.cat(h_list, 1)  # (batch_size, total_num_filter, length)
        h = h.transpose(1, 2)  # (batch_size, length, total_num_filter)
        h = self.activation(h)
        return h


class LabelwiseAttention(nn.Module):
    """
    """

    def __init__(self, hidden_size, num_classes):
        super(LabelwiseAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, inputs):
        attention = self.attention(inputs).transpose(1, 2)  # N, num_classes, L
        attention = F.softmax(attention, -1)
        return attention @ inputs   # N, num_classes, hidden_size


class LabelwiseLinearOutput(nn.Module):
    """
    """

    def __init__(self, hidden_size, num_classes):
        super(LabelwiseLinearOutput, self).__init__()
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        return (self.output.weight * inputs).sum(dim=-1) + self.output.bias
