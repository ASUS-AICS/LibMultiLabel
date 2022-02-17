from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):
    """Embedding layer with dropout

    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
    """

    def __init__(self, embed_vecs, dropout=0.2):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embed_vecs, freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        return self.dropout(self.embedding(input))


class RNNEncoder(ABC, nn.Module):
    """Base class of RNN encoder with dropout

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): The number of recurrent layers.
        dropout (float): The dropout rate of the encoder. Defaults to 0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super(RNNEncoder, self).__init__()
        self.rnn = self._get_rnn(input_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, length, **kwargs):
        self.rnn.flatten_parameters()
        idx = torch.argsort(length, descending=True)
        packed_input = pack_padded_sequence(
            input[idx], length[idx].cpu(), batch_first=True)
        outputs, _ = pad_packed_sequence(
            self.rnn(packed_input)[0], batch_first=True)
        return self.dropout(outputs[torch.argsort(idx)])

    @abstractmethod
    def _get_rnn(self, input_size, hidden_size, num_layers):
        raise NotImplementedError


class GRUEncoder(RNNEncoder):
    """Bi-directional GRU encoder with dropout

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): The number of recurrent layers.
        dropout (float): The dropout rate of the encoder. Defaults to 0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super(GRUEncoder, self).__init__(input_size, hidden_size, num_layers,
                                         dropout)

    def _get_rnn(self, input_size, hidden_size, num_layers):
        return nn.GRU(input_size, hidden_size, num_layers,
                      batch_first=True, bidirectional=True)


class LSTMEncoder(RNNEncoder):
    """Bi-directional LSTM encoder with dropout

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): The number of recurrent layers.
        dropout (float): The dropout rate of the encoder. Defaults to 0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super(LSTMEncoder, self).__init__(input_size, hidden_size, num_layers,
                                          dropout)

    def _get_rnn(self, input_size, hidden_size, num_layers):
        return nn.LSTM(input_size, hidden_size, num_layers,
                       batch_first=True, bidirectional=True)


class CNNEncoder(nn.Module):
    """Multi-filter-size CNN encoder for text classification with max-pooling

    Args:
        input_size (int): The number of expected features in the input.
        filter_sizes (list): Size of convolutional filters.
        num_filter_per_size (int): The number of filters in convolutional layers in each size. Defaults to 128.
        activation (str): Activation function to be used. Defaults to 'relu'.
        dropout (float): The dropout rate of the encoder. Defaults to 0.
        num_pool (int): The number of pools for max-pooling.
                        If num_pool = 0, do nothing.
                        If num_pool = 1, do typical max-pooling.
                        If num_pool > 1, do adaptive max-pooling.
        channel_last (bool): Whether to transpose the dimension from (batch_size, num_channel, length) to (batch_size, length, num_channel)
    """

    def __init__(self, input_size, filter_sizes, num_filter_per_size,
                 activation, dropout=0, num_pool=0, channel_last=False):
        super(CNNEncoder, self).__init__()
        if not filter_sizes:
            raise ValueError(f'CNNEncoder expect non-empty filter_sizes. '
                             f'Got: {filter_sizes}')
        self.channel_last = channel_last
        self.convs = nn.ModuleList()
        for filter_size in filter_sizes:
            conv = nn.Conv1d(
                in_channels=input_size,
                out_channels=num_filter_per_size,
                kernel_size=filter_size)
            self.convs.append(conv)
        self.num_pool = num_pool
        if num_pool > 1:
            self.pool = nn.AdaptiveMaxPool1d(num_pool)
        self.activation = getattr(torch, activation, getattr(F, activation))
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        h = input.transpose(1, 2)  # (batch_size, input_size, length)
        h_list = []
        for conv in self.convs:
            h_sub = conv(h)  # (batch_size, num_filter, length)
            if self.num_pool == 1:
                h_sub = F.max_pool1d(h_sub, h_sub.shape[2])  # (batch_size, num_filter, 1)
            elif self.num_pool > 1:
                h_sub = self.pool(h_sub)  # (batch_size, num_filter, num_pool)
            h_list.append(h_sub)
        h = torch.cat(h_list, 1)  # (batch_size, total_num_filter, *)
        if self.channel_last:
            h = h.transpose(1, 2)  # (batch_size, *, total_num_filter)
        h = self.activation(h)
        return self.dropout(h)


class LabelwiseAttention(nn.Module):
    """Applies attention technique to summarize the sequence for each label
    See `Explainable Prediction of Medical Codes from Clinical Text <https://aclanthology.org/N18-1100.pdf>`_

    Args:
        input_size (int): The number of expected features in the input.
        num_classes (int): Total number of classes.
    """
    def __init__(self, input_size, num_classes):
        super(LabelwiseAttention, self).__init__()
        self.attention = nn.Linear(input_size, num_classes, bias=False)

    def forward(self, input):
        attention = self.attention(input).transpose(1, 2)  # (batch_size, num_classes, seqence_length)
        attention = F.softmax(attention, -1)
        logits = torch.bmm(attention, input)  # (batch_size, num_classes, hidden_dim)
        return logits, attention


class LabelwiseMultiHeadAttention(nn.Module):
    """Labelwise multi-head attention

    Args:
        input_size (int): The number of expected features in the input.
        num_classes (int): Total number of classes.
        num_heads (int): The number of parallel attention heads.
        attention_dropout (float): The dropout rate for the attention. Defaults to 0.0.
    """
    def __init__(self, input_size, num_classes, num_heads, attention_dropout=0.0):
        super(LabelwiseMultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=attention_dropout)
        self.Q = nn.Linear(input_size, num_classes)

    def forward(self, input, attention_mask=None):
        key = value = input.permute(1, 0, 2)  # (sequence_length, batch_size, hidden_dim)
        query = self.Q.weight.repeat(input.size(0), 1, 1).transpose(
            0, 1)  # (num_classes, batch_size, hidden_dim)

        logits, attention = self.attention(query, key, value, key_padding_mask=attention_mask)
        logits = logits.permute(1, 0, 2)  # (num_classes, batch_size, hidden_dim)
        return logits, attention


class LabelwiseLinearOutput(nn.Module):
    """Applies a linear transformation to the incoming data for each label

    Args:
        input_size (int): The number of expected features in the input.
        num_classes (int): Total number of classes.
    """

    def __init__(self, input_size, num_classes):
        super(LabelwiseLinearOutput, self).__init__()
        self.output = nn.Linear(input_size, num_classes)

    def forward(self, input):
        return (self.output.weight * input).sum(dim=-1) + self.output.bias
