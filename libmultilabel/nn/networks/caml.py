from math import floor

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from ..networks.base import BaseModel


class CAML(BaseModel):
    """CAML (Convolutional Attention for Multi-Label classification)
    Follows the work of Mullenbach et al. [https://aclanthology.org/N18-1100.pdf]

    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        filter_sizes (list): Size of convolutional filters.
        num_filter_per_size (int): Number of filters in convolutional layers in each size. Defaults to 50.
        dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        activation (str): Activation function to be used. Defaults to 'tanh'.
    """
    def __init__(
        self,
        embed_vecs,
        num_classes,
        filter_sizes=None,
        num_filter_per_size=50,
        dropout=0.2,
        activation='tanh'
    ):
        super(CAML, self).__init__(embed_vecs, dropout, activation)
        if not filter_sizes and len(filter_sizes) != 1:
            raise ValueError(f'CAML expect 1 filter size. Got filter_sizes={filter_sizes}')
        filter_size = filter_sizes[0]

        # Initialize conv layer
        self.conv = nn.Conv1d(embed_vecs.shape[1], num_filter_per_size, kernel_size=filter_size, padding=int(floor(filter_size/2)))
        xavier_uniform_(self.conv.weight)

        """Context vectors for computing attention with
        (in_features, out_features) = (num_filter_per_size, num_classes)
        """
        self.U = nn.Linear(num_filter_per_size, num_classes)
        xavier_uniform_(self.U.weight)

        # Final layer: create a matrix to use for the #labels binary classifiers
        self.final = nn.Linear(num_filter_per_size, num_classes)
        xavier_uniform_(self.final.weight)

    def forward(self, input):
        # Get embeddings and apply dropout
        x = self.embedding(input['text'])  # (batch_size, length, embed_dim)
        x = self.embed_drop(x)
        x = x.transpose(1, 2) # (batch_size, embed_dim, length)

        """ Apply convolution and nonlinearity (tanh). The shapes are:
            - self.conv(x): (batch_size, num_filte_per_size, length)
            - x after transposing the first and the second dimension and applying
              the activation function: (batch_size, length, num_filte_per_size)
        """
        x = torch.tanh(self.conv(x).transpose(1, 2))

        """Apply per-label attention. The shapes are:
           - U.weight: (num_classes, num_filte_per_size)
           - matrix product of U.weight and x: (batch_size, num_classes, length)
           - alpha: (batch_size, num_classes, length)
        """
        alpha = torch.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        # Document representations are weighted sums using the attention
        m = alpha.matmul(x) # (batch_size, num_classes, num_filter_per_size)

        # Compute a probability for each label
        x = self.final.weight.mul(m).sum(dim=2).add(self.final.bias) # (batch_size, num_classes)

        return {'logits': x, 'attention': alpha}
