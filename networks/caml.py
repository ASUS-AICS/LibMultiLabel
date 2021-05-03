import os
from math import floor

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from networks.base import BaseModel


class CAML(BaseModel):
    def __init__(self, config, embed_vecs):
        super(CAML, self).__init__(config, embed_vecs)

        if len(config.filter_sizes) != 1:
            raise ValueError(f'CAML expect 1 filter size. Got filter_sizes={config.filter_sizes}')
        filter_size = config.filter_sizes[0]

        num_filter_per_size = config.num_filter_per_size

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(embed_vecs.shape[1], num_filter_per_size, kernel_size=filter_size, padding=int(floor(filter_size/2)))
        xavier_uniform_(self.conv.weight)

        # Context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_per_size, config.num_classes)
        xavier_uniform_(self.U.weight)

        # Final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_per_size, config.num_classes)
        xavier_uniform_(self.final.weight)

    def forward(self, text):
        # Get embeddings and apply dropout
        x = self.embedding(text)
        x = self.embed_drop(x)
        x = x.transpose(1,2)

        # Apply convolution and nonlinearity (tanh / relu)
        x = self.activation(self.conv(x).transpose(1,2))

        # Apply attention
        #    batch * text_length * 1200
        # -> batch * text_length * num_class (matmul)
        # -> batch * num_class (softmax)
        alpha = torch.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)

        # Document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)

        # similarity
        x = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return {'logits': x, 'attention': alpha}
