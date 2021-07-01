# [Proposal] https://github.com/huggingface/transformers/blob/0d1f67e651220bffef1441fa7589620e426ba958/src/transformers/models/bert/configuration_bert.py#L51
from pytorch_lightning.utilities.parsing import AttributeDict


class CAMLConfig(AttributeDict):
    def __init__(
        self,
        activation='tanh',
        dropout=0.2,
        filter_sizes=[10],
        num_filter_per_size=50,
        **kwargs
    ):
        self.model_name = 'CAML'
        self.activation = activation
        self.dropout = dropout
        self.filter_sizes = filter_sizes
        self.num_filter_per_size = num_filter_per_size


class KimCNNConfig(AttributeDict):
    def __init__(
        self,
        activation='relu',
        dropout=0.2,
        filter_sizes=[2, 4, 8],
        num_filter_per_size=128,
        **kwargs
    ):
        self.model_name = 'KimCNN'
        self.activation = activation
        self.dropout = dropout
        self.filter_sizes = filter_sizes
        self.num_filter_per_size = num_filter_per_size


class XMLCNNConfig(AttributeDict):
    def __init__(
        self,
        activation='relu',
        dropout=0.2,
        dropout2=0.2,
        hidden_dim=512,
        filter_sizes=[2, 4, 8],
        num_filter_per_size=256,
        num_pool=2,
        seed=1337,
        **kwargs
    ):
        self.model_name = 'XMLCNN'
        self.activation = activation
        self.dropout = dropout
        self.dropout2 = dropout2
        self.hidden_dim = hidden_dim
        self.filter_sizes = filter_sizes
        self.num_filter_per_size = num_filter_per_size
        self.num_pool = num_pool
        self.seed = seed # TODO discuss if we need seed here


def get_model_config(model_name, config=None):
    # type checking for pytorch_lightning `AttributeDict`
    config = dict() if not config else dict(config)

    if model_name == 'CAML':
        return CAMLConfig(**config)
    elif model_name == 'KimCNN':
        return KimCNNConfig(**config)
    elif model_name == 'XMLCNN':
        return XMLCNNConfig(**config)
    else:
        raise ValueError(f'Invalid model name: {model_name}')
