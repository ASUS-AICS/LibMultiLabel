import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks.base import BaseModel


class KimCNN(BaseModel):
    def __init__(
        self,
        embed_vecs,
        num_classes,
        filter_sizes=None,
        num_filter_per_size=128,
        **kwargs
    ):
        super(KimCNN, self).__init__(embed_vecs, **kwargs)

        self.filter_sizes = filter_sizes
        emb_dim = embed_vecs.shape[1]

        self.convs = nn.ModuleList()

        for filter_size in self.filter_sizes:
            conv = nn.Conv1d(
                in_channels=emb_dim,
                out_channels=num_filter_per_size,
                kernel_size=filter_size)
            self.convs.append(conv)
        conv_output_size = num_filter_per_size * len(self.filter_sizes)

        self.linear = nn.Linear(conv_output_size, num_classes)

    def forward(self, text):
        h = self.embedding(text) # (batch_size, length, embed_dim)
        h = self.embed_drop(h)
        h = h.transpose(1, 2) # (batch_size, embed_dim, length)

        h_list = []
        for conv in self.convs:
            h_sub = conv(h) # (batch_size, num_filter, length - kernel_size + 1)
            h_sub = F.max_pool1d(h_sub, kernel_size=h_sub.size()[2]) # (batch_size, num_filter, 1)
            h_sub = h_sub.view(h_sub.shape[0], -1) # (batch_size, num_filter)
            h_list.append(h_sub)

        # Max-pooling and monotonely increasing non-linearities commute. Here
        # we apply the activation function after max-pooling for better
        # efficiency.
        if len(self.filter_sizes) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
        h = self.activation(h) # (batch_size, total_num_filter)

        # linear output
        h = self.linear(h)
        return {'logits': h}
