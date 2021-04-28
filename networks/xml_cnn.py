import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import BaseModel


def out_size(l_in, kernel_size, channels, padding=0, dilation=1, stride=1):
    a = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
    b = int(a / stride)
    return (b + 1) * channels


class XMLCNN(BaseModel):
    def __init__(self, config, embed_vecs):
        super(XMLCNN, self).__init__(config, embed_vecs)
        assert config.fixed_length is True

        self.filter_sizes = config.filter_sizes
        emb_dim = embed_vecs.shape[1]
        num_filter_maps = config.num_filter_maps
        strides = config.strides
        d_max_pool_p = config.d_max_pool_p

        self.convs = nn.ModuleList()
        self.poolings = nn.ModuleList()

        for filter_size, p, stride in zip(self.filter_sizes, d_max_pool_p, strides):
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=num_filter_maps,
                kernel_size=(filter_size, emb_dim),
                stride=(stride, emb_dim))

            # Dynamic Max-Pooling
            conv_out_size = out_size(config.max_seq_length, filter_size, num_filter_maps, stride=stride)
            assert conv_out_size % p == 0
            pool_size = conv_out_size // p
            pool = nn.MaxPool1d(pool_size, stride=pool_size)
            self.convs.append(conv)
            self.poolings.append(pool)
        conv_output_size = sum(d_max_pool_p)

        self.dropout2 = nn.Dropout(config.dropout2)
        self.linear1 = nn.Linear(conv_output_size, config.hidden_dim)
        self.linear2 = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, text):
        h = self.embedding(text) # (batch_size, length, embed_dim)
        h = self.embed_drop(h)
        h = h.view(h.shape[0], 1, h.shape[1], h.shape[2]) # (batch_size, 1, length, embed_dim)

        h_list = []
        for i in range(len(self.filter_sizes)):
            h_sub = self.convs[i](h) # (batch_size, num_filter, H, 1)
            h_sub = h_sub.view(h_sub.shape[0], 1, h_sub.shape[1] * h_sub.shape[2]) # (batch_size, 1, num_filter * H)
            h_sub = self.poolings[i](h_sub) # (batch_size, 1, P)
            h_sub = h_sub.view(h_sub.shape[0], -1) # (batch_size, P)
            h_list.append(h_sub)

        if len(self.filter_sizes) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
        h = self.activation(h) # (batch_size, N * num_filter)

        # linear output
        h = self.activation(self.linear1(h))
        h = self.dropout2(h)
        h = self.linear2(h)
        return {'logits': h}
