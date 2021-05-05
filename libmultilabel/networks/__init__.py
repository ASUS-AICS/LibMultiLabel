import torch.nn as nn

from libmultilabel.networks.caml import CAML
from libmultilabel.networks.kim_cnn import KimCNN
from libmultilabel.networks.xml_cnn import XMLCNN
from libmultilabel.networks.cnn import VanillaConv


def get_init_weight_func(config):
    def init_weight(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            getattr(nn.init, config.init_weight+ '_')(m.weight)
    return init_weight
