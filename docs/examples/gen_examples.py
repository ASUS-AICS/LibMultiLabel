import json

with open("linear_component.json") as F:
    Linear_quickstart = json.load(F)

with open("nn_component.json") as F:
    NN_quickstart = json.load(F)

with open("dataset_component.json") as F:
    load_Hugging_Face_data_sets = json.load(F)

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

def gen_HuggingFace_example(model="Linear"):
    code = load_Hugging_Face_data_sets["HEAD"]
    if model == "Linear":
        code += load_Hugging_Face_data_sets["Linear_part"]
        code += gen_Linear_quickstart(include_load_data=False)
    elif model == "NN":
        code += load_Hugging_Face_data_sets["NN_part"]
        code += gen_NN_quickstart(NN_model="KimCNN", include_load_data=False)
    return code

code = gen_Linear_quickstart()
print(code)
"""
# Generate codes
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
"""
