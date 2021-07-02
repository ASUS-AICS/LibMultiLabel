import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    '''Base Model for process different inputs

    Args:
        config (ArgDict): config of the experiment
        embed_vecs (FloatTensor): embedding vectors for initialization
    '''

    def __init__(self, config, embed_vecs):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(len(embed_vecs), embed_vecs.shape[1], padding_idx=0)
        self.embedding.weight.data = embed_vecs.clone()
        self.embed_drop = nn.Dropout(p=config.dropout)
        # TODO Put the activation function to model files: https://github.com/ASUS-AICS/LibMultiLabel/issues/42
        self.activation = getattr(F, config.activation)
