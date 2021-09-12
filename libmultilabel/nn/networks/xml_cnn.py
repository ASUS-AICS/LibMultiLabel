import torch
import torch.nn as nn

from ..networks.base import BaseModel


class XMLCNN(BaseModel):
    """XML-CNN

        Args:
            embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
            num_classes (int): Total number of classes.
            dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
            dropout2 (float): Optional specification of the second dropout. Defaults to 0.2.
            filter_sizes (list): Size of convolutional filters.
            hidden_dim (int): Dimension of the hidden layer. Defaults to 512.
            num_filter_per_size (int): Number of filters in convolutional layers in each size. Defaults to 256.
            num_pool (int): Number of pool for dynamic max-pooling. Defaults to 2.
            activation (str): Activation function to be used. Defaults to 'relu'.
        """
    def __init__(
        self,
        embed_vecs,
        num_classes,
        dropout=0.2,
        dropout2=0.2,
        filter_sizes=None,
        hidden_dim=512,
        num_filter_per_size=256,
        num_pool=2,
        activation='relu'
    ):
        super(XMLCNN, self).__init__(embed_vecs, dropout, activation)
        if not filter_sizes:
            raise ValueError(
                f'XMLCNN expect filter_sizes. Got filter_sizes={filter_sizes}')

        emb_dim = embed_vecs.shape[1]

        self.convs = nn.ModuleList()
        for filter_size in filter_sizes:
            conv = nn.Conv1d(
                in_channels=emb_dim,
                out_channels=num_filter_per_size,
                kernel_size=filter_size)
            self.convs.append(conv)

        # Automatically set stride and kernel_size for pooling by output_size
        # stride = floor(input_size / output_size)
        # kernel_size = input_size - (output_size - 1) * stride
        self.pool = nn.AdaptiveMaxPool1d(output_size=num_pool)

        total_output_size = len(filter_sizes) * num_filter_per_size * num_pool
        self.dropout2 = nn.Dropout(dropout2)
        self.linear1 = nn.Linear(total_output_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input):
        h = self.embedding(input['text'])  # (batch_size, length, embed_dim)
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
