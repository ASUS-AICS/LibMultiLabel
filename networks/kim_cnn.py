import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from networks.base import BaseModel


class KimCNN(BaseModel):
    def __init__(self, config, embed_vecs):
        super(KimCNN, self).__init__(config, embed_vecs)

        num_filter_maps = config.num_filter_maps
        filter_size = config.filter_size

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(embed_vecs.shape[1], num_filter_maps, kernel_size=filter_size)
        xavier_uniform_(self.conv.weight)

        # linear output
        self.fc = nn.Linear(num_filter_maps, config.num_classes)
        xavier_uniform_(self.fc.weight)

    def forward(self, text):
        # embedding
        x = self.embedding(text)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # conv/max-pooling
        c = self.conv(x)
        x = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2])
        x = x.squeeze(dim=2)

        # linear output
        x = self.fc(x)
        return {'logits': x}
