## Machine Translation: WMT'15 English-German

Single models only, no ensembles. Results are listed in chronological order.

| Model Name & Reference | Settings / Notes| Training Time | Test Set BLEU |
| --- | --- | --- | --- |
| tf-seq2seq | [Configuration](https://github.com/google/seq2seq/blob/master/example_configs/nmt_large.yml) | ~4 days on 8 NVidia K80 GPUs | newstest2014: **22.19** </br> newstest2015: **25.23** | [Model]() <br/> [Data]() |
| [Gehring, et al. (2016-11)](https://arxiv.org/abs/1611.02344) <br/> Deep Convolutional 15/5 | | --- | newstest2014: - <br/> newstest2015: **24.3** | --- |
| [Wu et al. (2016-09)](https://arxiv.org/abs/1609.08144) <br/> GNMT | 8 encoder/decoder layers, 1024 LSTM units, 32k shared wordpieces (similar to BPE); residual between layers connections; lots of other tricks; newstest2012 and newstest2013 as validation sets. | --- |  newstest2014:&nbsp;**24.61** <br/>newstest2015: -|
| [Zhou et al. (2016-06)](https://arxiv.org/abs/1606.04199) <br/> Deep-Att | | --- | newstest2014: **20.6** <br/> newstest2015: - | --- |
| [Chung, et al. (2016-03)](https://arxiv.org/abs/1603.06147v4) <br/> BPE-Char | **Character-level decoder with BPE encoder.** Based on Bahdanau attention model; Bidirectional encoder with 512 GRU units; 2-layer GRU decoder with 1024 units; Adam; batch size 128; gradient clipping at norm 1; Moses Tokenizer; limit sequences to 50 symbols in source and 100 symbols and 500 characters in target. | --- |  newstest2014: **21.5** </br> newstest2015: **23.9** | --- | 
| [Sennrich et al. (2015-8)](https://arxiv.org/abs/1508.07909) <br/> BPE | **Authors propose BPE for subword unit nsegmentation as a pre/post-processing step to handle open vocabulary**;  Base model is based on [Bahndanau's paper](https://arxiv.org/abs/1409.0473). Bidirectional encoder; GRU; 1000 hidden units; 1000 attention units; 620-dimensional word embeddings; single-layer; beam search width 12; Adadelta with batch size 80; Using [Groundhog](https://github.com/sebastien-j/LV_groundhog); | | newstest2014: - <br/>newstest2015: **20.5** | --- |
| [Luong et al. (2015-08)](https://arxiv.org/abs/1508.04025) | **Novel local/global attention mechanism;** 50k vocabulary; 4 layers in encoder and decoder; unidirectional encoder; gradient clipping at norm 5;  1028 LSTM units, 1028-dimensional embeddings; (somewhat complicated) SGD decay schedule; dropout 0.2; UNK replace;| --- | newstest2014: **20.9** <br/> newstest2015: - | --- |
| [Jean et al. (2014-12)](https://arxiv.org/abs/1412.2007) <br/> RNNsearch-LV | **Authors propose a new sampling-based approach to incorporate a larger vocabulary**; Base model is based on [Bahndanau's paper](https://arxiv.org/abs/1409.0473). Bidirectional encoder; GRU; 1000 hidden units; 1000 attention units; 620-dimensional word embeddings; single-layer; beam search width 12; | --- | newstest2014: **19.4** <br/> newstest2015: - | --- |


## Machine Translation: WMT'17

Coming soon.


## Text Summarization: Gigaword

Coming soon.


## Image Captioning: MSCOCO

Coming soon.


## Conversational Modeling

Coming soon.