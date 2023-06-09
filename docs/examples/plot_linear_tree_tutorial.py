"""
Handling Data with Many Labels
==============================

For the case that the amount of labels is very large,
the training time of the standard ``train_1vsrest`` method may be unpleasantly long.
The ``train_tree`` method in LibMultiLabel can vastly improve the training time on such data sets.

To illustrate this speedup, we will use the `EUR-Lex dataset <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#EUR-Lex>`_,
which contains 3,956 labels.
In this example, the data is downloaded under the directory ``data/eur-lex``.
"""

import math
import libmultilabel.linear as linear
import time

datasets = linear.load_dataset("txt", "data/eurlex/train.txt", "data/eurlex/test.txt")
preprocessor = linear.Preprocessor()
datasets = preprocessor.fit_transform(datasets)


training_start = time.time()
# the standard one-vs-rest method for multi-label problems
ovr_model = linear.train_1vsrest(datasets["train"]["y"], datasets["train"]["x"])
training_end = time.time()
print("Training time of one-versus-rest: {:10.2f}".format(training_end - training_start))

training_start = time.time()
# the train_tree method for fast training on data with many labels
tree_model = linear.train_tree(datasets["train"]["y"], datasets["train"]["x"])
training_end = time.time()
print("Training time of tree-based: {:10.2f}".format(training_end - training_start))

######################################################################
# On a machine with an AMD-7950X CPU,
# the ``train_1vsrest`` function took `578.30` seconds,
# while the ``train_tree`` function only took `144.37` seconds.
#
# .. note::
#
#   The ``train_tree`` function in this tutorial is based on the work of :cite:t:`SK20a`.
#
# ``train_tree`` achieves this speedup by approximating ``train_1vsrest``. To check whether the approximation
# performs well, we'll compute some metrics on the test set.

ovr_preds = linear.predict_values(ovr_model, datasets["test"]["x"])
tree_preds = linear.predict_values(tree_model, datasets["test"]["x"])

target = datasets["test"]["y"].toarray()

ovr_score = linear.compute_metrics(ovr_preds, target, ["P@1", "P@3", "P@5"])
print("Score of 1vsrest:", ovr_score)

tree_score = linear.compute_metrics(tree_preds, target, ["P@1", "P@3", "P@5"])
print("Score of tree:", tree_score)

######################################################################
#  :math:`P@K`, a ranking-based criterion, is a metric often used for data with a large amount of labels.
#
# .. code-block::
#
#   Score of 1vsrest: {'P@1': 0.833117723156533, 'P@3': 0.6988357050452781, 'P@5': 0.585666235446313}
#   Score of tree: {'P@1': 0.8217335058214748, 'P@3': 0.692539887882708, 'P@5': 0.578835705045278}
#
# For this data set, ``train_tree`` gives a slightly lower :math:`P@K`, but has a significantly faster training time.
# Typcially, the speedup of ``train_tree`` over ``train_1vsrest`` increases with the amount of labels.
#
# For even larger data sets, we may not be able to store the entire ``preds`` and ``target`` in memory at once.
# In this case, the metrics can be computed in batches.


def metrics_in_batches(model):
    batch_size = 256
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(["P@1", "P@3", "P@5"], num_classes=datasets["test"]["y"].shape[1])

    for i in range(num_batches):
        preds = linear.predict_values(model, datasets["test"]["x"][i * batch_size : (i + 1) * batch_size])
        target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
        metrics.update(preds, target)

    return metrics.compute()


print("Score of 1vsrest:", metrics_in_batches(ovr_model))
print("Score of tree:", metrics_in_batches(tree_model))

######################################################################
#
# .. bibliography::
