import libmultilabel.linear as linear

preprocessor = linear.Preprocessor(data_format='txt')

datasets = preprocessor.load_data('data/rcv1/train.txt',
                                  'data/rcv1/test.txt')

model = linear.train_1vsrest(datasets['train']['y'],
                             datasets['train']['x'],
                             '')

preds = linear.predict_values(model, datasets['test']['x'])

metrics = linear.get_metrics(metric_threshold=0,
                             monitor_metrics=['Macro-F1', 'Micro-F1', 'P@1', 'P@3', 'P@5'],
                             num_classes=datasets['test']['y'].shape[1])

target = datasets['test']['y'].toarray()

metrics.update(preds, target)
print(metrics.compute())
