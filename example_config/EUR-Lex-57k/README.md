# EURLEX57K

Download: [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#EURLEX57K)  
Note: This dataset is called **EURLex-4.3K** in [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html).

## Benchmark
The best model is selected by **RP@5** on validation set then evaluated on test set.

| Method | Reference |    Macro-F1     |     Micro-F1     |       P@1        |       P@5        |       **RP@5**       |      nDCG@5      |  Cfg | Time |
|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|     BERT-BASE-LWAN      | [Chalkidis et al. 2020](http://aclanthology.lst.uni-saarland.de/2020.emnlp-main.607/) |        26.1981      |     72.7849      |     89.9833      |     67.8533      |     78.4742      |     80.9834         | [Cfg](./bert_lwan.yml) | 9 hrs |
|     CNN-LWAN      | [Mullenbach et al. 2018](https://arxiv.org/abs/1802.05695) |     26.3245      |     71.6865      |     89.8167      |     67.3100      |     78.0261      |     80.5446      | [Cfg](./cnn_lwan.yml) | 2 hrs |
|     BiGRU-LWAN      | [Chalkidis et al. 2019](https://aclanthology.org/P19-1636/) |     25.4260      |     71.3199      |     90.7333      |     67.0800      |     77.7506      |     80.5999      | [Cfg](./bigru_lwan.yml) | 3 hrs 20 mins |
