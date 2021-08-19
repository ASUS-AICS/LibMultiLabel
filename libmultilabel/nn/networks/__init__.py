import torch.nn as nn

from .bigru import BiGRU
from .caml import CAML
from .kim_cnn import KimCNN
from .xml_cnn import XMLCNN


def get_init_weight_func(init_weight):
    def init_weight_func(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            getattr(nn.init, init_weight+ '_')(m.weight)
    return init_weight_func
