import torch
import torch.nn as nn

from .modules import Embedding, LSTMEncoder


class LAAT(nn.Module):
    """LAAT Vu.

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        filter_sizes (list): Size of convolutional filters.
        num_filter_per_size (int): The number of filters in convolutional layers in each size. Defaults to 50.
        dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
    """

    def __init__(
        self,
        embed_vecs,
        num_classes,
        rnn_dim=1024,
        num_layers=1,
        d_a=512,
        embed_dropout=0.3,
        encoder_dropout=0,
        freeze_embed=True,
    ):
        super(LAAT, self).__init__()

        self.embedding = Embedding(
            embed_vecs, dropout=embed_dropout, freeze_embed=freeze_embed)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim

        # Initialize rnn layer (H: 2u * N)
        self.encoder = LSTMEncoder(
            input_size=embed_vecs.shape[1],
            hidden_size=rnn_dim//2, num_layers=num_layers, dropout=encoder_dropout)

        self.W = nn.Linear(rnn_dim, d_a, bias=False)

        """Context vectors for computing attention with
        (in_features, out_features) = (d_a, num_classes)
        """
        self.Q = nn.Linear(d_a, num_classes, bias=False)

        # Final layer: create a matrix to use for the #labels binary classifiers
        self.output = nn.Linear(rnn_dim, num_classes, bias=True)

    def forward(self, input):
        # Get embeddings and apply dropout
        x = self.embedding(input['text'])  # (batch_size, length, embed_dim)
        H = self.encoder(x, input['length'])  # (batch_size, length, rnn_dim)

        # Equation (4) Z = tanh(WH), W: (d_a * 2u), H: (2u * N), Z: (d_a * N)
        Z = torch.tanh(self.W(H))  # (batch_size, length, d_a)

        # Equation (5) A = softmax(UZ), A: (batch_size, class_num, length)
        #     Q: (|L| * d_a), Z: (d_a * N), A: |L| * N
        # alpha = self.Q(Z)
        # alpha = torch.softmax(alpha, 1).transpose(1, 2)
        alpha = torch.softmax(self.Q.weight.matmul(Z.transpose(1, 2)), dim=2)

        # Document representations are weighted sums using the attention
        E = alpha.matmul(H)

        # Compute a probability for each label
        logits = self.output.weight.mul(E).sum(dim=2).add(
            self.output.bias)  # (batch_size, num_classes)

        return {'logits': logits, 'attention': alpha}
