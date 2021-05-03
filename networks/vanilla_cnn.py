import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from networks.base import BaseModel


class VanillaCNN(BaseModel):
    def __init__(self, config, embed_vecs):
        super(VanillaCNN, self).__init__(config, embed_vecs)

        if len(config.filter_sizes) != 1:
            raise ValueError(f'VanillaCNN expect 1 filter size. Got filter_sizes={config.filter_sizes}')
        filter_size = config.filter_sizes[0]

        num_filter_per_size = config.num_filter_per_size

        self.conv = nn.Conv1d(embed_vecs.shape[1], num_filter_per_size, kernel_size=filter_size)
        xavier_uniform_(self.conv.weight)

        self.linear = nn.Linear(num_filter_per_size, config.num_classes)
        xavier_uniform_(self.linear.weight)


    def forward(self, text):
        h = self.embedding(text) # (batch_size, length, embed_dim)
        h = self.embed_drop(h)
        h = h.transpose(1, 2) # (batch_size, embed_dim, length)

        h = self.conv(h) # (batch_size, num_filter, length)
        h = F.max_pool1d(self.activation(h), kernel_size=h.size()[2]) # (batch_size, num_filter, 1)
        h = h.squeeze(dim=2) # batch_size, num_filter

        h = self.linear(h)
        return {'logits': h}

