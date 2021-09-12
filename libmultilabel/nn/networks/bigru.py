import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..networks.base import BaseModel


class BiGRU(BaseModel):
    """BiGRU (Bidirectional Gated Recurrent Unit)

    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the GRU network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): Number of recurrent layers. Defaults to 1.
        dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        activation (str): Activation function to be used. Defaults to 'tanh'.
    """
    def __init__(
        self,
        embed_vecs,
        num_classes,
        rnn_dim=512,
        rnn_layers=1,
        dropout=0.2,
        activation='tanh',
        **kwargs
    ):
        super(BiGRU, self).__init__(embed_vecs, dropout, activation, **kwargs)
        assert rnn_dim%2 == 0, """`rnn_dim` should be even."""

        # BiGRU
        emb_dim = embed_vecs.shape[1]
        self.rnn = nn.GRU(emb_dim, rnn_dim//2, rnn_layers,
                          bidirectional=True, batch_first=True)

        # context vectors for computing attention
        self.U = nn.Linear(rnn_dim, num_classes)
        xavier_uniform_(self.U.weight)

        # linear output
        self.final = nn.Linear(rnn_dim, num_classes)
        xavier_uniform_(self.final.weight)

    def forward(self, input):
        text, length, indices = self.sort_data_by_length(input['text'], input['length'])
        x = self.embedding(text) # (batch_size, length, rnn_dim)
        x = self.embed_drop(x) # (batch_size, length, rnn_dim)

        packed_inputs = pack_padded_sequence(x, length, batch_first=True)
        x, _ = self.rnn(packed_inputs)
        x = pad_packed_sequence(x)[0]
        x = x.permute(1, 0, 2)

        x = torch.tanh(x)

        # Apply per-label attention
        alpha = torch.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        # Document representations are weighted sums using the attention
        m = alpha.matmul(x) # (batch_size, num_classes, rnn_dim)

        # Compute a probability for each label
        x = self.final.weight.mul(m).sum(dim=2).add(self.final.bias) # (batch_size, num_classes)

        return {'logits': x[indices], 'attention': alpha[indices]}

    def sort_data_by_length(self, data, length):
        """Sort data by lengths. If data is not sorted before calling `pack_padded_sequence`,
        under `enforce_sorted=False`, the setting of `use_deterministic_algorithms` must be False.
        To keep `use_deterministic_algorithms` True, we sort the input here and return indices for
        restoring the original order.

        Args:
            data (torch.Tensor): Batch of sequences with shape (batch_size, length)
            length (list): List of text lengths before padding.

        Returns:
            data (torch.Tensor): Sequences sorted by lengths in descending order.
            length (torch.Tensor): Lengths sorted in descending order.
            indices (torch.Tensor): The indexes of the elements in the original data.
        """
        length = torch.as_tensor(length, dtype=torch.int64)
        length, sorted_indices = torch.sort(length, descending=True)
        sorted_indices = sorted_indices.to(data.device)
        data = data.index_select(0, sorted_indices)

        data_size = sorted_indices.size(-1)
        indices = torch.empty(data_size, dtype=torch.long)
        indices[sorted_indices] = torch.arange(data_size)

        return data, length, indices
