# EURLEX57K

Download: [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#EURLEX57K)  
Note: This dataset is called **EURLex-4.3K** in [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html).

## Benchmark
The best model is selected by **RP@5** on validation set then evaluated on test set.

| Method |     Macro-F1     |     Micro-F1     |       P@1        |       P@5        |       **RP@5**       |      nDCG@5      |  Cfg |
|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|     CNN-LWAN      |     26.3245      |     71.6865      |     89.8167      |     67.3100      |     78.0261      |     80.5446      | [Cfg](./example_config/EUR-Lex-57k/cnn_lwan.yml) |
|     BiGRU-LWAN      |     25.4260      |     71.3199      |     90.7333      |     67.0800      |     77.7506      |     80.5999      | [Cfg](./example_config/EUR-Lex-57k/bigru_lwan.yml) |
