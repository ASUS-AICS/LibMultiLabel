import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import BaseModel


def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
    # refer to https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    a = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
    l_out = math.floor(a / stride + 1)
    return l_out


class XMLCNN(BaseModel):
    def __init__(self, config, embed_vecs):
        super(XMLCNN, self).__init__(config, embed_vecs)
        assert config.fixed_length is True

        self.filter_sizes = config.filter_sizes
        num_filter_per_size = config.num_filter_per_size
        emb_dim = embed_vecs.shape[1]
        pool_size = config.pool_size

        self.convs = nn.ModuleList()
        self.poolings = nn.ModuleList()

        total_output_size = 0
        for filter_size in self.filter_sizes:
            conv = nn.Conv1d(
                in_channels=emb_dim,
                out_channels=num_filter_per_size,
                kernel_size=filter_size)

            # Dynamic Max-Pooling
            conv_output_size = out_size(config.max_seq_length, filter_size)
            pool = nn.MaxPool1d(pool_size, stride=pool_size)
            total_output_size += num_filter_per_size * out_size(conv_output_size, pool_size, stride=pool_size)
            self.convs.append(conv)
            self.poolings.append(pool)

        self.dropout2 = nn.Dropout(config.dropout2)
        self.linear1 = nn.Linear(total_output_size, config.hidden_dim)
        self.linear2 = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, text):
        h = self.embedding(text) # (batch_size, length, embed_dim)
        h = self.embed_drop(h)
        h = h.transpose(1, 2) # (batch_size, embed_dim, length)

        h_list = []
        for i in range(len(self.filter_sizes)):
            h_sub = self.convs[i](h) # (batch_size, num_filter, length')
            h_sub = self.poolings[i](h_sub) # (batch_size, num_filter, length'')
            h_sub = h_sub.view(h_sub.shape[0], -1) # (batch_size, num_filter * length'')
            h_list.append(h_sub)

        if len(self.filter_sizes) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
        h = self.activation(h) # (batch_size, total_num_filter)

        # linear output
        h = self.activation(self.linear1(h))
        h = self.dropout2(h)
        h = self.linear2(h)
        return {'logits': h}
