from math import floor

import torch.nn as nn

from ..networks.base import BaseModel


class BiGRU(BaseModel):
    def __init__(self, config, embed_vecs):
        super(BiGRU, self).__init__(config, embed_vecs)

        rnn_dim = config.rnn_dim
        emb_dim = embed_vecs.shape[1]

        # BiGRU
        self.rnn = nn.GRU(emb_dim, floor(rnn_dim/2), config.num_layers, bidirectional=True)

        # linear output
        self.final = nn.Linear(rnn_dim, config.num_classes)

    def forward(self, text):
        # let doc length be first as GRU's default is batch_first=False
        x = self.embedding(text).transpose(0,1) # (length, batch_size, embed_dim)

        out, hidden = self.rnn(x)

        batch_size = x.shape[1]

        # get last hidden states of forward/backward directions
        last_h = hidden[-2:]

        # (batch_size, rnn_dim = 2* hidden_dim = 2*floor(rnn_dim/2))
        last_h = last_h.transpose(0,1).contiguous().view(batch_size, -1)

        # linear output
        last_h = self.final(last_h)

        return {'logits': last_h}
