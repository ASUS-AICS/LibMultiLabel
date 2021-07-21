import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """Base Model for process different inputs

    Args:
        embed_vecs (FloatTensor): Embedding vectors for initialization.
        dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        activation (str): Activation function to be used. Defaults to 'relu'.
    """

    def __init__(
        self,
        embed_vecs,
        dropout=0.2,
        activation='relu',
    ):
        super().__init__()
        self.embedding = nn.Embedding(len(embed_vecs), embed_vecs.shape[1], padding_idx=0)
        self.embedding.weight.data = embed_vecs.clone()
        self.embed_drop = nn.Dropout(p=dropout)
        # TODO Put the activation function to model files: https://github.com/ASUS-AICS/LibMultiLabel/issues/42
        self.activation = getattr(F, activation)
