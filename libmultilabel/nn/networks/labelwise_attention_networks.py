from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_

from .modules import (
    Embedding,
    GRUEncoder,
    LSTMEncoder,
    CNNEncoder,
    LabelwiseAttention,
    LabelwiseMultiHeadAttention,
    LabelwiseLinearOutput,
    FastLabelwiseAttention,
    MultilayerLinearOutput,
)


class LabelwiseAttentionNetwork(ABC, nn.Module):
    """Base class for Labelwise Attention Network"""

    def __init__(
        self,
        embed_vecs: Tensor,
        num_classes: int,
        embed_dropout: float,
        encoder_dropout: float,
        hidden_dim: int,
        freeze: bool = False,
    ):
        """

        Args:
            embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
            num_classes (int): Total number of classes.
            embed_dropout (float): The dropout rate of the word embedding.
            encoder_dropout (float): The dropout rate of the encoder output.
            hidden_dim (int): The output dimension of the encoder.
            freeze (bool): If True, the tensor does not get updated in the learning process.
                Equivalent to embedding.weight.requires_grad = False. Default: False.
        """
        super().__init__()
        self.embedding = Embedding(embed_vecs, freeze=freeze, dropout=embed_dropout)
        self.encoder = self._get_encoder(embed_vecs.shape[1], encoder_dropout)
        self.attention = self._get_attention()
        self.output = LabelwiseLinearOutput(hidden_dim, num_classes)

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def _get_encoder(self, input_size, dropout):
        raise NotImplementedError

    @abstractmethod
    def _get_attention(self):
        raise NotImplementedError


class RNNLWAN(LabelwiseAttentionNetwork):
    """Base class for RNN Labelwise Attention Network"""

    def forward(self, input):
        # (batch_size, sequence_length, embed_dim)
        x = self.embedding(input["text"])
        # (batch_size, sequence_length, hidden_dim)
        x = self.encoder(x, input["length"])
        x, _ = self.attention(x)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)
        return {"logits": x}


class BiGRULWAN(RNNLWAN):
    """BiGRU Labelwise Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the GRU network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
    """

    def __init__(self, embed_vecs, num_classes, rnn_dim=512, rnn_layers=1, embed_dropout=0.2, encoder_dropout=0):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        super(BiGRULWAN, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return GRUEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        return LabelwiseAttention(self.rnn_dim, self.num_classes)


class BiLSTMLWAN(RNNLWAN):
    """BiLSTM Labelwise Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the LSTM network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
    """

    def __init__(self, embed_vecs, num_classes, rnn_dim=512, rnn_layers=1, embed_dropout=0.2, encoder_dropout=0):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        super(BiLSTMLWAN, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        return LabelwiseAttention(self.rnn_dim, self.num_classes)


class BiLSTMLWMHAN(LabelwiseAttentionNetwork):
    """BiLSTM Labelwise Multihead Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the LSTM network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
        num_heads (int): The number of parallel attention heads. Defaults to 8.
        attention_dropout (float): The dropout rate for the attention. Defaults to 0.0.
    """

    def __init__(
        self,
        embed_vecs,
        num_classes,
        rnn_dim=512,
        rnn_layers=1,
        embed_dropout=0.2,
        encoder_dropout=0,
        num_heads=8,
        attention_dropout=0.0,
    ):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        super(BiLSTMLWMHAN, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        return LabelwiseMultiHeadAttention(self.rnn_dim, self.num_classes, self.num_heads, self.attention_dropout)

    def forward(self, input):
        # (batch_size, sequence_length, embed_dim)
        x = self.embedding(input["text"])
        # (batch_size, sequence_length, hidden_dim)
        x = self.encoder(x, input["length"])
        # (batch_size, num_classes, hidden_dim)
        x, _ = self.attention(x, attention_mask=input["text"] == 0)
        x = self.output(x)  # (batch_size, num_classes)
        return {"logits": x}


class CNNLWAN(LabelwiseAttentionNetwork):
    """CNN Labelwise Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        filter_sizes (list): Size of convolutional filters.
        num_filter_per_size (int): The number of filters in convolutional layers in each size. Defaults to 50.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
        activation (str): Activation function to be used. Defaults to 'tanh'.
    """

    def __init__(
        self,
        embed_vecs,
        num_classes,
        filter_sizes=None,
        num_filter_per_size=50,
        embed_dropout=0.2,
        encoder_dropout=0,
        activation="tanh",
    ):
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filter_per_size = num_filter_per_size
        self.activation = activation
        self.hidden_dim = num_filter_per_size * len(filter_sizes)
        super(CNNLWAN, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, self.hidden_dim)

    def _get_encoder(self, input_size, dropout):
        return CNNEncoder(
            input_size, self.filter_sizes, self.num_filter_per_size, self.activation, dropout, channel_last=True
        )

    def _get_attention(self):
        return LabelwiseAttention(self.hidden_dim, self.num_classes)

    def forward(self, input):
        # (batch_size, sequence_length, embed_dim)
        x = self.embedding(input["text"])
        x = self.encoder(x)  # (batch_size, sequence_length, hidden_dim)
        x, _ = self.attention(x)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)
        return {"logits": x}


class AttentionRNN(RNNLWAN):
    def __init__(
        self,
        embed_vecs,
        num_classes: int,
        rnn_dim: int,
        linear_size: list[int, ...],
        freeze_embed_training: bool = False,
        rnn_layers: int = 1,
        embed_dropout: float = 0.2,
        encoder_dropout: float = 0.5,
    ):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        super().__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim, freeze=freeze_embed_training)
        self.output = MultilayerLinearOutput([self.rnn_dim] + linear_size, 1)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        # return LabelwiseAttention(self.rnn_dim * 2, self.num_classes, init_fn=xavier_uniform_)
        return LabelwiseAttention(self.rnn_dim, self.num_classes, init_fn=xavier_uniform_)

    def forward(self, inputs):
        # N: batch size, L: sequence_length, E: emb_size, v: vocab_size, C: num_classes, H: hidden_dim
        # the index of padding is 0
        lengths = (inputs != 0).sum(dim=1)
        masks = (inputs != 0)[:, : lengths.max()]
        # input : dict["text", (N, L, V), "labels", (N, L, C: csr_matrix)]
        x = self.embedding(inputs)[:, : lengths.max()]  # (N, L, E)
        x = self.encoder(x, lengths)  # (N, L, 2 * H)
        x, _ = self.attention(x, masks)  # (N, C, 2 * H)
        x = self.output(x)  # (N, C)
        return x


class FastAttentionRNN(RNNLWAN):
    def __init__(
        self,
        embed_vecs,
        num_classes: int,
        rnn_dim: int,
        linear_size: list[int],
        freeze_embed_training: bool = False,
        rnn_layers: int = 1,
        embed_dropout: float = 0.2,
        encoder_dropout: float = 0.5,
    ):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        super().__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim, freeze=freeze_embed_training)
        self.output = MultilayerLinearOutput([self.rnn_dim] + linear_size, 1)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        return FastLabelwiseAttention(self.rnn_dim, self.num_classes)

    def forward(self, inputs, candidates):
        # N: num_batches, L: sequence_length, E: emb_size, v: vocab_size, S: sample size, H: hidden_dim
        # the index of padding is 0
        lengths = (inputs != 0).sum(dim=1)
        masks = (inputs != 0)[:, : lengths.max()]
        # input : dict["text", (N, L, V), "labels", (N, L, C: csr_matrix)]
        x = self.embedding(inputs)  # (N, L, E)
        x = self.encoder(x, lengths)  # (N, L, 2 * H)
        x, _ = self.attention(x, masks, candidates)  # (N, S, 2 * H)
        x = self.output(x)  # (N, S)
        return x
