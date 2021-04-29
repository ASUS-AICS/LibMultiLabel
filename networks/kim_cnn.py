import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import BaseModel


class KimCNN(BaseModel):
    def __init__(self, config, embed_vecs):
        super(KimCNN, self).__init__(config, embed_vecs)

        self.filter_sizes = config.filter_sizes
        emb_dim = embed_vecs.shape[1]
        num_filter_maps = config.num_filter_maps
        strides = config.strides

        self.convs = nn.ModuleList()

        for filter_size, stride in zip(self.filter_sizes, strides):
            conv = nn.Conv1d(
                in_channels=emb_dim,
                out_channels=num_filter_maps,
                kernel_size=filter_size,
                stride=stride)
            self.convs.append(conv)
        conv_output_size = num_filter_maps * len(self.filter_sizes)

        self.linear = nn.Linear(conv_output_size, config.num_classes)

    def forward(self, text):
        h = self.embedding(text) # (batch_size, length, embed_dim)
        h = self.embed_drop(h)
        h = h.view(h.shape[0], h.shape[2], h.shape[1]) # (batch_size, embed_dim, length)

        h_list = []
        for conv in self.convs:
            h_sub = conv(h) # (batch_size, num_filter, length)
            h_sub = F.max_pool1d(h_sub, kernel_size=h_sub.size()[2]) # (batch_size, num_filter, 1)
            h_sub = h_sub.view(h_sub.shape[0], -1) # (batch_size, num_filter)
            h_list.append(h_sub)

        if len(self.filter_sizes) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
        h = self.activation(h) # (batch_size, total_num_filter)

        # linear output
        h = self.linear(h)
        return {'logits': h}
