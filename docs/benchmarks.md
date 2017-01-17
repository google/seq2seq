## Machine Translation: WMT'16 English-German

Single models only, no ensembles. Results are listed in chronological order.

| Model Name & Reference | Settings / Notes| Training Time | Test Set BLEU | Downloads |
| --- | --- | --- | --- | --- |
| 2-Layer LSTM Attention Model + BPE | [Hyperparameters]() | --- | newstest2014: - </br> newstest2015: - | [Model]() <br/> [Data]() | --- |
| [Gehring, et al. (2016-11)](https://arxiv.org/abs/1611.02344) <br/> Deep Convolutional 15/5 | | --- | newstest2014: - <br/> newstest2015: **24.3** | --- |  --- |
| [Wu et al. (2016-09)](https://arxiv.org/abs/1609.08144) <br/> GNMT | 8 encoder/decoder layers, 1024 units, 32k shared wordpieces. newstest2012 and newstest2013 as validation sets. Paper mentions 5M training examples, how is this possible if WMT only has ~4.5M? | --- |  newstest2014:&nbsp;**24.61** <br/>newstest2015: -| --- |
| [Zhou et al. (2016-06)](https://arxiv.org/abs/1606.04199) <br/> Deep-Att |  | --- | newstest2014: **20.6** <br/> newstest2015: - | --- | --- |
| [Chung, et al. (2016-03)](https://arxiv.org/abs/1603.06147v4) <br/> BPE-Char-Biscale | Bidirectional encoder with 512 hidden units, 2-layer char-level decoder with 1024 units per layer. GRU.  | --- |  newstest2014: **21.3** </br> newstest2015: **23.9** | --- |    --- |
| [Sennrich et al. (2015-8)](https://arxiv.org/abs/1508.07909) <br/> BPE | 1000 units, 620d word embedding, 1000 attention units, using [Groundhog](https://github.com/sebastien-j/LV_groundhog), 90k shared BPE | | newstest2014: - <br/>newstest2015: **22.8** | --- | --- |
| [Luong et al. (2015-08)](https://arxiv.org/abs/1508.04025) | 4 layers, 1028 units, 1028d embeddings | --- | newstest2014: **20.9** <br/> newstest2015: - | --- | --- |
| [Jean et al. (2014-12)](https://arxiv.org/abs/1412.2007) <br/> RNNsearch-LV | 4 layers, 1028 units, 1028d embeddings | --- | newstest2014: **19.4** <br/> newstest2015: - | --- | --- |



## Machine Translation: WMT'16 German-English

Coming soon.


## Machine Translation: WMT'16 English-French

Coming soon.


## Machine Translation: WMT'16 French-English

Coming soon.


## Text Summarization

Coming soon.


## Conversational Modeling

Coming soon.