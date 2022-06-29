# Wiki10-31K

Download: [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#Wiki10-31K)

## Benchmark
The best model is selected by **RP@15** on validation set then evaluated on test set.

| Method | Reference |     P@1     |     P@3     |     P@5     |     P@15     |     RP@8     |     RP@15     |     nDCG@15     | Cfg | Time |
|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|     Kim-CNN     | [Chen et al. 2022](https://www.csie.ntu.edu.tw/~cjlin/papers/xmlcnn/xml_cnn_study.pdf) |     81.3029     |     67.3519     |     57.4758     |     34.2654     |     47.7563     |     37.2417     |     45.6148     | [Cfg](./kim_cnn.yml) | 20 mins |
|     CNN-LWAN     | [Mullenbach et al. 2018](https://aclanthology.org/N18-1100/) |     83.1923     |     71.9518     |     62.6209     |     38.2215     |     52.7204     |     41.4342     |     49.7467     | [Cfg](./cnn_lwan.yml) | 1 hr |
|     BiGRU-LWAN     | [Chalkidis et al. 2019](https://aclanthology.org/P19-1636/) |     84.2201     |     73.2618     |     64.4649     |     39.8982     |     54.5826     |     43.3372     |     51.5495     |[Cfg](./bigru_lwan.yml) | 2 hrs |
