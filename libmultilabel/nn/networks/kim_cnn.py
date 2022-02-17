import torch
import torch.nn as nn

from .modules import Embedding, CNNEncoder


class KimCNN(nn.Module):
    """KimCNN

    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape(vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        filter_sizes (list): The size of convolutional filters.
        num_filter_per_size (int): The number of filters in convolutional layers in each size. Defaults to 128.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
        activation (str): Activation function to be used. Defaults to 'relu'.
    """
    def __init__(
        self,
        embed_vecs,
        num_classes,
        filter_sizes=None,
        num_filter_per_size=128,
        embed_dropout=0.2,
        encoder_dropout=0,
        activation='relu'
    ):
        super(KimCNN, self).__init__()
        self.embedding = Embedding(embed_vecs, embed_dropout)
        self.encoder = CNNEncoder(embed_vecs.shape[1], filter_sizes,
                                  num_filter_per_size, activation,
                                  encoder_dropout, num_pool=1)
        conv_output_size = num_filter_per_size * len(filter_sizes)
        self.linear = nn.Linear(conv_output_size, num_classes)

    def forward(self, input):
        x = self.embedding(input['text'])  # (batch_size, length, embed_dim)
        x = self.encoder(x)  # (batch_size, num_filter, 1)
        x = torch.squeeze(x, 2) # (batch_size, num_filter)
        x = self.linear(x)  # (batch_size, num_classes)
        return {'logits': x}
