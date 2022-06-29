# EUR-Lex

Download: [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#EUR-Lex)
Note: This dataset is called **EURLex-4K** in [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html).

## Benchmark
The best model is selected by **RP@5** on validation set then evaluated on test set.

| Method | Reference |     Macro-F1     |     Micro-F1     |     P@1     |     P@5     |     **RP@5**     |     nDCG@5     | Cfg | Time |
|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|     Kim-CNN     | [Chen et al. 2022](https://www.csie.ntu.edu.tw/~cjlin/papers/xmlcnn/xml_cnn_study.pdf) |     17.7251     |     52.4463     |     78.1113     |     53.5731     |     58.1216     |     62.5291     | [Cfg](./kim_cnn.yml) | 20 mins |
|     CNN-LWAN     | [Mullenbach et al. 2018](https://aclanthology.org/N18-1100/) |     22.3065     |     55.3257     |     79.7930     |     56.1811     |     61.1440     |     65.1978     | [Cfg](./cnn_lwan.yml) | 20 mins |
|     BiGRU-LWAN     | [Chalkidis et al. 2019](https://aclanthology.org/P19-1636/) |     21.0723     |     57.1339     |     81.3713     |     56.7141     |     61.5429     |     65.9885     | [Cfg](./bigru_lwan.yml) | 50 mins |
