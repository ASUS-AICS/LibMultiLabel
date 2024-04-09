import torch.nn as nn

from .bert import BERT
from .bert_attention import BERTAttention
from .caml import CAML
from .kim_cnn import KimCNN
from .xml_cnn import XMLCNN
from .labelwise_attention_networks import BiGRULWAN
from .labelwise_attention_networks import BiLSTMLWAN
from .labelwise_attention_networks import BiLSTMLWMHAN
from .labelwise_attention_networks import CNNLWAN
from .labelwise_attention_networks import AttentionXML_0, AttentionXML_1


def get_init_weight_func(init_weight):
    def init_weight_func(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            getattr(nn.init, init_weight + "_")(m.weight)

    return init_weight_func
