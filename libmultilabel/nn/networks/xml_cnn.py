import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Embedding, CNNEncoder


class XMLCNN(nn.Module):
    """XML-CNN

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the hidden layer output. Defaults to 0.
        filter_sizes (list): Size of convolutional filters.
        hidden_dim (int): Dimension of the hidden layer. Defaults to 512.
        num_filter_per_size (int): The number of filters in convolutional layers in each size. Defaults to 256.
        num_pool (int): The number of pool for dynamic max-pooling. Defaults to 2.
        activation (str): Activation function to be used. Defaults to 'relu'.
    """
    def __init__(
        self,
        embed_vecs,
        num_classes,
        embed_dropout=0.2,
        encoder_dropout=0,
        filter_sizes=None,
        hidden_dim=512,
        num_filter_per_size=256,
        num_pool=2,
        activation='relu'
    ):
        super(XMLCNN, self).__init__()
        self.embedding = Embedding(embed_vecs, embed_dropout)
        self.encoder = CNNEncoder(embed_vecs.shape[1], filter_sizes,
                                  num_filter_per_size, activation,
                                  num_pool=num_pool)
        total_output_size = len(filter_sizes) * num_filter_per_size * num_pool
        self.dropout = nn.Dropout(encoder_dropout)
        self.linear1 = nn.Linear(total_output_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.activation = getattr(torch, activation, getattr(F, activation))

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, length, embed_dim)
        x = self.encoder(x)  # (batch_size, num_filter, num_pool)
        x = x.view(x.shape[0], -1)  # (batch_size, num_filter * num_pool)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return {'logits': x}
