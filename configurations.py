# [Proposal] https://github.com/huggingface/transformers/blob/0d1f67e651220bffef1441fa7589620e426ba958/src/transformers/models/bert/configuration_bert.py#L51

class CAMLConfig():
    def __init__(
        self,
        activation='tanh',
        dropout=0.2,
        filter_sizes=[10],
        num_filter_per_size=50
    ):
        self.model_name = 'CAML'
        pass


class KimCNNConfig():
    def __init__(
        self,
        activation='relu',
        dropout=0.2,
        filter_sizes=[2, 4, 8],
        num_filter_per_size=128
    ):
        self.model_name = 'KimCNN'
        pass


class XMLCNNConfig():
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

    ):
        self.model_name = 'XMLCNN'
        pass
