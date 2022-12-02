import json

Linear_quickstart = dict()
Linear_quickstart["HEAD"] = """
import libmultilabel.linear as linear
"""
Linear_quickstart["load_data_set"] = """
preprocessor = linear.Preprocessor(data_format='txt')
data_sets = preprocessor.load_data('data/rcv1/train.txt', 'data/rcv1/test.txt')
"""
Linear_quickstart["TAIL"] = """
model = linear.train_1vsrest(data_sets['train']['y'], data_sets['train']['x'], '')

preds = linear.predict_values(model, data_sets['test']['x'])

metrics = linear.get_metrics(metric_threshold=0,
                             monitor_metrics=['Macro-F1', 'Micro-F1', 'P@1', 'P@3', 'P@5'],
                             num_classes=data_sets['test']['y'].shape[1])

target = data_sets['test']['y'].toarray()

metrics.update(preds, target)
print(metrics.compute())
"""

with open("linear_component.json", "w") as F:
    json.dump(Linear_quickstart, F)
