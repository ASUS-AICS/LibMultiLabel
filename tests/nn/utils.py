from .components import datasets, stoi, embed_vecs, classes, network


test_components = [datasets(),
                   stoi(),
                   embed_vecs(),
                   classes(),
                   network()]

test_components_mapping = {c.get_name(): c for c in test_components}


def get_names():
    return [c.get_name() for c in test_components]


def get_components_from_trainer(trainer):
    return [c.get_from_trainer(trainer) for c in test_components]


def compare(name, a, b):
    return test_components_mapping[name].compare(a, b)
