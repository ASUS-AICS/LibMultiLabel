import math

import torch
import torch.nn as nn

from ..networks.base import BaseModel


class XMLCNN(BaseModel):
    def __init__(self, config, embed_vecs):
        super(XMLCNN, self).__init__(config, embed_vecs)
        assert config.seed is None, ("nn.AdaptiveMaxPool1d doesn't have a "
                                     "deterministic implementation but seed is"
                                     "specified. Please do not specify seed.")

        emb_dim = embed_vecs.shape[1]

        self.convs = nn.ModuleList()
        for filter_size in config.filter_sizes:
            conv = nn.Conv1d(
                in_channels=emb_dim,
                out_channels=config.num_filter_per_size,
                kernel_size=filter_size)
            self.convs.append(conv)

        # Automatically set stride and kernel_size for pooling by output_size
        # stride = floor(input_size / output_size)
        # kernel_size = input_size - (output_size - 1) * stride
        self.pool = nn.AdaptiveMaxPool1d(output_size=config.num_pool)

        total_output_size = len(config.filter_sizes) * config.num_filter_per_size * config.num_pool
        self.dropout2 = nn.Dropout(config.dropout2)
        self.linear1 = nn.Linear(total_output_size, config.hidden_dim)
        self.linear2 = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, text):
        h = self.embedding(text) # (batch_size, length, embed_dim)
        h = self.embed_drop(h)
        h = h.transpose(1, 2) # (batch_size, embed_dim, length)

        h_list = []
        for conv in self.convs:
            h_sub = conv(h) # (batch_size, num_filter, length)
            h_sub = self.pool(h_sub) # (batch_size, num_filter, num_pool)
            h_sub = h_sub.view(h_sub.shape[0], -1) # (batch_size, num_filter * num_pool)
            h_list.append(h_sub)

        if len(self.convs) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
        # Max-pooling and monotonely increasing non-linearities commute. Here
        # we apply the activation function after max-pooling for better
        # efficiency.
        h = self.activation(h) # (batch_size, total_num_filter)

        # linear output
        h = self.activation(self.linear1(h))
        h = self.dropout2(h)
        h = self.linear2(h)
        return {'logits': h}
