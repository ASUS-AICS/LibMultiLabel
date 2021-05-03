import torch.nn as nn

from networks.caml import CAML
from networks.kim_cnn import KimCNN
from networks.vanilla_cnn import VanillaCNN
from networks.xml_cnn import XMLCNN


def get_init_weight_func(config):
    def init_weight(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            getattr(nn.init, config.init_weight+ '_')(m.weight)
    return init_weight
