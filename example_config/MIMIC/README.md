# MIMIC-III

Download: [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)  
Note: We follow [Mullenbach et al., 2018](https://arxiv.org/abs/1802.05695) to process the data. The detail can be found in their [repository](https://github.com/jamesmullenbach/caml-mimic).
## Benchmark
The best model is selected by **RP@15** on validation set then evaluated on test set.

| Method |     Macro-F1     |     Micro-F1     |       P@15       |       P@8        |      RP@15       |       RP@8       |     nDCG@15      | Cfg | Time | 
|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
|      BiGRU-LWAN      |      7.8685      |     55.8535      |     58.0012      |     72.7647      |     66.8688      |     74.4926      |     71.8933      |[Cfg](./bigru_lwan.yml) | 17 hrs 20 mins |
|      CNN-LWAN      |      6.9354      |     52.9829      |     57.4970      |     72.4088      |     66.3423      |     74.0617      |     71.3641      | [Cfg](./cnn_lwan.yml) | 2 hrs 50 mins |
