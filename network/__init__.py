from network.caml import CamlConvAttnPool
from network.cnn import VanillaConv


def get_network(config, embed_vecs):
    if config.model_name == 'caml':
        return CamlConvAttnPool(config, embed_vecs)

    elif config.model_name == 'cnn':
        return VanillaConv(config, embed_vecs)
