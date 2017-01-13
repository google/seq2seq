## Machine Translation: WMT'16 English-German

Single models only, no ensembles.

| Description | Settings / Notes| Training Time | Test Set BLEU | Downloads |
| --- | --- | --- | --- | --- |
| 2-Layer LSTM Attention Model + BPE | [Hyperparameters]() | --- | newstest2014: - </br> newstest2015: - | [Model]() <br/> [Data]() | --- |
| [We et al. (2016) - GNMT](https://arxiv.org/abs/1609.08144) | 8 encoder/decoder layers, 1024 units, 32k shared wordpieces. newstest2012 and newstest2013 as validation sets. Paper mentions 5M training examples, how if WMT only has ~4.5M? | --- |  newstest2014:&nbsp;**24.61** <br/>newstest2015: -| --- |
| [Chung, et al. (2016) - BPE-Char](https://arxiv.org/abs/1603.06147v4) | Bidir encoder 512 hidden units, 2-layer (?) 1024 unit decoder, GRU cells | --- |  newstest2014: **21.3** </br> newstest2015: **23.9** | --- |    --- |
| [Gehring, et al. (2016) - Convolutional](https://arxiv.org/abs/1611.02344) | Deep Convolutional Encoder 15/5| --- | newstest2014: - <br/> newstest2015: **24.3** | --- |  --- |
| [Sennrich et al. (2016) - BPE](https://arxiv.org/abs/1508.07909) | 1000 units, 620d word embedding, 1000 attention units, using [Groundhog](https://github.com/sebastien-j/LV_groundhog), 90k shared BPE | | newstest2014: - <br/>newstest2015: **22.8** | --- | --- |
| [Jean et al. (2015)](https://arxiv.org/abs/1508.04025) | 4 layers, 1028 units, 1028d embeddings | --- | newstest2014: **20.9** <br/> newstest2015: - | --- | --- |
| [Luong et al. (2015)](https://arxiv.org/abs/1508.04025) | 4 layers, 1028 units, 1028d embeddings | --- | newstest2014: **20.9** <br/> newstest2015: - | --- | --- |


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