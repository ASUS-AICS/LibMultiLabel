from abc import ABC, abstractmethod

import torch.nn as nn

from .modules import Embedding, GRUEncoder, LSTMEncoder, CNNEncoder, LabelwiseAttention, LabelwiseMultiHeadAttention, LabelwiseLinearOutput


class LabelwiseAttentionNetwork(ABC, nn.Module):
    """Base class for Labelwise Attention Network

    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        embed_dropout (float): The dropout rate of the word embedding.
        encoder_dropout (float): The dropout rate of the encoder output.
        hidden_dim (int): The output dimension of the encoder.
    """

    def __init__(self, embed_vecs, num_classes, embed_dropout, encoder_dropout, hidden_dim):
        super(LabelwiseAttentionNetwork, self).__init__()
        self.embedding = Embedding(embed_vecs, embed_dropout)
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
    """Base class for RNN Labelwise Attention Network
    """

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, sequence_length, embed_dim)
        x = self.encoder(x, input['length'])  # (batch_size, sequence_length, hidden_dim)
        x, _ = self.attention(x)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)
        return {'logits': x}


class BiGRULWAN(RNNLWAN):
    """BiGRU Labelwise Attention Network

    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the GRU network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
        activation (str): Activation function to be used. Defaults to 'tanh'.
    """

    def __init__(
        self,
        embed_vecs,
        num_classes,
        rnn_dim=512,
        rnn_layers=1,
        embed_dropout=0.2,
        encoder_dropout=0
    ):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        super(BiGRULWAN, self).__init__(embed_vecs, num_classes, embed_dropout,
                                        encoder_dropout, rnn_dim)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return GRUEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        return LabelwiseAttention(self.rnn_dim, self.num_classes)


class BiLSTMLWAN(RNNLWAN):
    """BiLSTM Labelwise Attention Network

    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the LSTM network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
        activation (str): Activation function to be used. Defaults to 'tanh'.
    """

    def __init__(
        self,
        embed_vecs,
        num_classes,
        rnn_dim=512,
        rnn_layers=1,
        embed_dropout=0.2,
        encoder_dropout=0
    ):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        super(BiLSTMLWAN, self).__init__(embed_vecs, num_classes, embed_dropout,
                                         encoder_dropout, rnn_dim)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        return LabelwiseAttention(self.rnn_dim, self.num_classes)


class BiLSTMLWMHAN(LabelwiseAttentionNetwork):
    """BiLSTM Labelwise Multihead Attention Network

    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
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
        attention_dropout=0.0
    ):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        super(BiLSTMLWMHAN, self).__init__(embed_vecs, num_classes, embed_dropout,
                                           encoder_dropout, rnn_dim)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2,
                           self.rnn_layers, dropout)

    def _get_attention(self):
        return LabelwiseMultiHeadAttention(self.rnn_dim, self.num_classes, self.num_heads, self.attention_dropout)

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, sequence_length, embed_dim)
        x = self.encoder(x, input['length'])  # (batch_size, sequence_length, hidden_dim)
        x, _ = self.attention(x, attention_mask=input['text'] == 0)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)
        return {'logits': x}


class CNNLWAN(LabelwiseAttentionNetwork):
    """CNN Labelwise Attention Network

    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
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
        activation='tanh'
    ):
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filter_per_size = num_filter_per_size
        self.activation = activation
        hidden_dim = num_filter_per_size * len(filter_sizes)
        super(CNNLWAN, self).__init__(embed_vecs, num_classes, embed_dropout,
                                      encoder_dropout, hidden_dim)

    def _get_encoder(self, input_size, dropout):
        return CNNEncoder(input_size, self.filter_sizes,
                          self.num_filter_per_size, self.activation, dropout,
                          channel_last=True)

    def _get_attention(self):
        return LabelwiseAttention(self.rnn_dim, self.num_classes)

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, sequence_length, embed_dim)
        x = self.encoder(x)  # (batch_size, sequence_length, hidden_dim)
        x, _ = self.attention(x)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)
        return {'logits': x}
