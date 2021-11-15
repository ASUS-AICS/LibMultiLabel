import libmultilabel.linear as linear

preprocessor = linear.Preprocessor(data_format='txt')

datasets = preprocessor.load_data('data/rcv1/train.txt',
                                  'data/rcv1/test.txt')
model = linear.train_1vsrest(datasets['train']['y'],
                             datasets['train']['x'],
                             '')

preds = linear.predict_values(model, datasets['test']['x'])
target = datasets['test']['y'].toarray()

metrics = linear.get_metrics(metric_threshold=0,
                             monitor_metrics=['P@1', 'P@3', 'P@5'],
                             num_classes=datasets['test']['y'].shape[1])
metrics.update(preds, target)
print(metrics.compute())