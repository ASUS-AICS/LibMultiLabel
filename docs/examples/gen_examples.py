NN_quickstart = dict()
NN_quickstart["HEAD"] = """
from libmultilabel.nn.data_utils import *
from libmultilabel.nn.nn_utils import *
from transformers import AutoTokenizer

set_seed(1337)
device = init_device()  # use gpu by default
"""
NN_quickstart["load_data_set_KimCNN"] = """
data_sets = load_datasets('data/rcv1/train.txt', 'data/rcv1/test.txt', tokenize_text=True)
"""
NN_quickstart["load_data_set_BERT"] = """
data_sets = load_datasets('data/rcv1/train.txt', 'data/rcv1/test.txt', tokenize_text=False)
"""
NN_quickstart["build_label"] = """
classes = load_or_build_label(data_sets)
"""
NN_quickstart["KimCNN_part"] = """
word_dict, embed_vecs = load_or_build_text_dict(dataset=data_sets['train'], embed_file='glove.6B.300d')
tokenizer = None
add_special_tokens = False

model_name = 'KimCNN'
network_config = {
    'embed_dropout': 0.2,
    'encoder_dropout': 0.2,
    'filter_sizes': [2, 4, 8],
    'num_filter_per_size': 128
}
learning_rate = 0.0003
"""
NN_quickstart["BERT_part"] = """
word_dict, embed_vecs = None, None 
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
add_special_tokens = True

model_name='BERT'
network_config = {
    'dropout': 0.1,
    'lm_weight': 'bert-base-uncased',
}
learning_rate = 0.00003
"""
NN_quickstart["TAIL"] = """
model = init_model(
    model_name=model_name,
    network_config=network_config,
    classes=classes,
    word_dict=word_dict,
    embed_vecs=embed_vecs,
    learning_rate=learning_rate,
    monitor_metrics=['Micro-F1', 'Macro-F1', 'P@1', 'P@3', 'P@5']
)

trainer = init_trainer(checkpoint_dir='runs/NN-example', epochs=15, val_metric='P@5')

loaders = dict()
for split in ['train', 'val', 'test']:
    loaders[split] = get_dataset_loader(
        data=data_sets[split],
        word_dict=word_dict,
        classes=classes,
        device=device,
        max_seq_length=512,
        batch_size=8,
        shuffle=True if split == 'train' else False,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens
    )

trainer.fit(model, loaders['train'], loaders['val'])
trainer.test(model, dataloaders=loaders['test'])
"""

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

load_Hugging_Face_data_sets = dict()
load_Hugging_Face_data_sets["HEAD"] = """
import pandas
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

data_sets = dict()
data_sets['train'] = load_dataset('tweet_eval', 'emoji', split='train')
data_sets['val'] = load_dataset('tweet_eval', 'emoji', split='validation')
data_sets['test'] = load_dataset('tweet_eval', 'emoji', split='test')

for tag in ['train', 'val', 'test']:
    data_sets[tag] = pandas.DataFrame(data_sets[tag], columns=['label', 'text'])
    data_sets[tag] = data_sets[tag].reset_index()
"""
load_Hugging_Face_data_sets["NN_part"] = """
for tag in ['train', 'val', 'test']:
    data_sets[tag]['label'] = data_sets[tag]['label'].astype(str).map(lambda s: s.split())
    data_sets[tag] = data_sets[tag].to_dict('records')

for tag in ['train', 'val']:
    num_no_label_data = sum(1 for d in data_sets[tag] if len(d['label']) == 0)
    if num_no_label_data > 0:
        data_sets[tag] = [d for d in data_sets[tag] if len(d['label']) > 0]
"""
load_Hugging_Face_data_sets["Linear_part"] = """
for tag in ['train', 'val', 'test']:
    data_sets[tag]['label'] = data_sets[tag]['label'].map(str)
    data_sets[tag] = data_sets[tag].to_dict('list')

data_sets['train']['label'] += data_sets['val']['label']
data_sets['train']['text'] += data_sets['val']['text']
data_sets['train']['index'] += list(map( lambda x: x + data_sets['train']['index'][-1] + 1, data_sets['val']['index'] ))

classes = set(data_sets['train']['label'] + data_sets['test']['label'])
classes = sorted([cls for cls in classes], key=int)

vectorizer = TfidfVectorizer()
binarizer = MultiLabelBinarizer(sparse_output=True, classes=classes)   
vectorizer.fit(data_sets['train']['text'])
binarizer.fit(data_sets['train']['label'])
for tag in ['train', 'test']:
    data_sets[tag]['x'] = vectorizer.transform(data_sets[tag]['text'])
    data_sets[tag]['y'] = binarizer.transform(data_sets[tag]['label']).astype('d')

num_labels = data_sets['train']['y'].getnnz(axis=1)
num_no_label_data = np.count_nonzero(num_labels == 0)
if num_no_label_data > 0:
    if remove_no_label_data:
        data_sets['train']['x'] = data_sets['train']['x'][num_labels > 0]
        data_sets['train']['y'] = data_sets['train']['y'][num_labels > 0]
"""

def gen_HuggingFace_example(model="Linear"):
    code = load_Hugging_Face_data_sets["HEAD"]
    if model == "Linear":
        code += load_Hugging_Face_data_sets["Linear_part"]
        code += gen_Linear_quickstart(include_load_data=False)
    elif model == "NN":
        code += load_Hugging_Face_data_sets["NN_part"]
        code += gen_NN_quickstart(NN_model="KimCNN", include_load_data=False)
    return code

def gen_Linear_quickstart(include_load_data=True):
    code = Linear_quickstart["HEAD"]
    if include_load_data:
        code += Linear_quickstart["load_data_set"]
    code += Linear_quickstart["TAIL"]
    return code

def gen_NN_quickstart(NN_model="KimCNN", include_load_data=True):
    code = NN_quickstart["HEAD"]
    if NN_model == "KimCNN":
        if include_load_data:
            code += NN_quickstart["load_data_set_KimCNN"]
        code += NN_quickstart["build_label"]
        code += NN_quickstart["KimCNN_part"]
    elif NN_model == "BERT":
        if include_load_data:
            code += NN_quickstart["load_data_set_BERT"]
        code += NN_quickstart["build_label"]
        code += NN_quickstart["BERT_part"]
    code += NN_quickstart["TAIL"]
    return code

if __name__ == "__main__":
    with open("linear_quickstart.py", "w") as F:
        code = gen_Linear_quickstart()
        F.write(code)

    with open("kimcnn_quickstart.py", "w") as F:
        code = gen_NN_quickstart(NN_model="KimCNN")
        F.write(code)

    with open("bert_quickstart.py", "w") as F:
        code = gen_NN_quickstart(NN_model="BERT")
        F.write(code)

    with open("dataset_example_nn.py", "w") as F:
        code = gen_HuggingFace_example(model="NN")
        F.write(code)

    with open("dataset_example_linear.py", "w") as F:
        code = gen_HuggingFace_example(model="Linear")
        F.write(code)
