class datasets(object):
    def __init__(self):
        pass

    def get_name(self):
        return "datasets"

    def get_from_trainer(self, trainer):
        return trainer.datasets

    def compare(self, a, b):
        return a == b


class token_to_id(object):
    def __init__(self):
        pass

    def get_name(self):
        return "token_to_id"

    def get_from_trainer(self, trainer):
        return trainer.model.word_dict.get_stoi()

    def compare(self, a, b):
        return a == b


class embed_vecs(object):
    def __init__(self):
        pass

    def get_name(self):
        return "embed_vecs"

    def get_from_trainer(self, trainer):
        return trainer.model.embed_vecs

    def compare(self, a, b):
        return (a == b).all()


class classes(object):
    def __init__(self):
        pass

    def get_name(self):
        return "classes"

    def get_from_trainer(self, trainer):
        return trainer.model.classes

    def compare(self, a, b):
        return a == b


class network(object):
    def __init__(self):
        pass

    def get_name(self):
        return "network"

    def get_from_trainer(self, trainer):
        return trainer.model.network.state_dict()

    def compare(self, a, b):
        for aa, bb in zip(a, b):
            if not (aa == bb):
                return False
        return True
