# MIMIC-III

Download: [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)  
Note: We follow [Mullenbach et al., 2018](https://arxiv.org/abs/1802.05695) to process the data. The detail can be found in their [repository](https://github.com/jamesmullenbach/caml-mimic).
## Benchmark
The best model is selected by **RP@15** on validation set then evaluated on test set.

| Method | Reference |     Macro-F1     |     Micro-F1     |     P@8       |     P@15     |     RP@8     |     RP@15     |     nDCG@15     | Cfg | Time |
|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|     Kim-CNN     | [Chen et al. 2022](https://www.csie.ntu.edu.tw/~cjlin/papers/xmlcnn/xml_cnn_study.pdf) |     3.9854     |     46.0679     |     64.9058     |     49.7904     |     66.3422     |     57.4185     |     63.4506     | [Cfg](./kim_cnn.yml) | 40 mins |
|     CNN-LWAN     | [Mullenbach et al. 2018](https://aclanthology.org/N18-1100/) |     6.9354     |     52.9829     |     72.4088     |     57.4970     |     74.0617     |     66.3423     |     71.3641     | [Cfg](./cnn_lwan.yml) | 2 hrs 50 mins |
|     BiGRU-LWAN     | [Chalkidis et al. 2019](https://aclanthology.org/P19-1636/) |     7.8685     |     55.8535     |     72.7647     |     58.0012     |     74.4926     |     66.8688     |     71.8933     |[Cfg](./bigru_lwan.yml) | 17 hrs 20 mins |
