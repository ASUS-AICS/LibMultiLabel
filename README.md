# LibMultiLabel — a Library for Multi-class and Multi-label Text Classification

LibMultiLabel is a library for binary, multi-class, and multi-label classification. It has the following functionalities

- end-to-end services from raw texts to final evaluation/analysis
- support for common neural network architectures and linear classifiers
- easy hyper-parameter selection

This is an on-going development so many improvements are still being made. Comments are very welcome.

## Environments
- Python: 3.7+
- CUDA: 10.2 (if training neural networks by GPU)
- Pytorch 1.12+

If you have a different version of CUDA, follow the installation instructions for PyTorch LTS at their [website](https://pytorch.org/).

## Documentation
See the documentation here: https://www.csie.ntu.edu.tw/~cjlin/libmultilabel

## Label Descriptions for Zero-shot Models
Step 1: save the label descriptions for each data sets from following links:
- [EURLEX57K](https://www.csie.ntu.edu.tw/~d09922007/EURLEX57K_label_descriptions.txt)
- [MICMIC](https://www.csie.ntu.edu.tw/~d09922007/ICD9CM_aug.txt)

Step 2: specify the path to `label_file` in the configuration file
