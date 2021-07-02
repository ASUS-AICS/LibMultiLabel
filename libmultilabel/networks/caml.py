from math import floor

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from ..networks.base import BaseModel


class CAML(BaseModel):
    def __init__(self, config, embed_vecs):
        super(CAML, self).__init__(config, embed_vecs)

        if len(config.filter_sizes) != 1:
            raise ValueError(f'CAML expect 1 filter size. Got filter_sizes={config.filter_sizes}')
        filter_size = config.filter_sizes[0]

        num_filter_per_size = config.num_filter_per_size

        # Initialize conv layer
        self.conv = nn.Conv1d(embed_vecs.shape[1], num_filter_per_size, kernel_size=filter_size, padding=int(floor(filter_size/2)))
        xavier_uniform_(self.conv.weight)

        # Context vectors for computing attention
        self.U = nn.Linear(num_filter_per_size, config.num_classes)
        xavier_uniform_(self.U.weight)

        # Final layer: create a matrix to use for the #labels binary classifiers
        self.final = nn.Linear(num_filter_per_size, config.num_classes)
        xavier_uniform_(self.final.weight)

    def forward(self, text):
        # Get embeddings and apply dropout
        # x: (batch_size, embedding size, document length)
        x = self.embedding(text)
        x = self.embed_drop(x)
        x = x.transpose(1,2)

        # Apply convolution and nonlinearity (tanh)
        # x: (batch_size, document length, num_filte_per_size)
        x = torch.tanh(self.conv(x).transpose(1,2))

        # Apply per-label attention
        # alpha: (batch_size, label size, document length)
        alpha = torch.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)

        # Document representations are weighted sums using the attention
        # m: (batch_size, label size, num_filter_per_size)
        m = alpha.matmul(x)

        # Compute a probability for each label
        # x: (batch_size, label size)
        x = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return {'logits': x, 'attention': alpha}
