## Machine Translation: WMT'16 English-German

| Description | Settings / Notes| Training Time | Test Set | Downloads |
| --- | --- | --- | --- | --- |
| 2-Layer LSTM Attention Model + BPE | [Hyperparameters]() | N/A | 0.00 PPL <br/> 0.00 BLEU | [Model]() <br/> [Data]() | --- |
| [We et al. (2016) - GNMT](https://arxiv.org/abs/1609.08144) | 8 encoder/decoder layers, 1024 units, 32k shared wordpieces. newstest2012 and newstest2013 as validation sets. Paper mentions 5M training examples, how if WMT only has ~4.5M? | --- |  24.61 BLEU (newstest2014) | --- |
| [Chung, et al. (2016) - BPE-Char](https://arxiv.org/abs/1603.06147v4) | Bidir encoder 512 hidden units, 2-layer (?) 1024 unit decoder, GRU cells | --- |  23.9 BLEU (newstest2015)<br/>21.3 BLEU (newstest2014) | --- |    --- |
| [Gehring, et al. (2016) - Convolutional](https://arxiv.org/abs/1611.02344) | Deep Convolutional Encoder 15/5| --- |  24.3 BLEU (newstest2015) | --- |  --- |
| [Sennrich et al. (2016) - BPE](https://arxiv.org/abs/1508.07909) | 1000 units, 620d word embedding, 1000 attention units, using [Groundhog](https://github.com/sebastien-j/LV_groundhog), 90k shared BPE | | 22.8 BLEU (newstest2015) | --- | --- |
| [Jean et al. (2015)](https://arxiv.org/abs/1508.04025) | 4 layers, 1028 units, 1028d embeddings | --- | 20.9 BLEU (newstest2014) | --- | --- |
| [Luong et al. (2015)](https://arxiv.org/abs/1508.04025) | 4 layers, 1028 units, 1028d embeddings | --- | 20.9 BLEU (newstest2014) | --- | --- |
