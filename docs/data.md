## Available Datasets

We provide data generation scripts to generate standard datasets.

| Dataset | Description | Training/Dev/Test Size | Vocabulary | Download |
| --- | --- | --- | --- | --- |
| WMT'16 EN-DE | Data for the [WMT'16 Translation Task](http://www.statmt.org/wmt16/translation-task.html) English to German. Training data is combined from Europarl v7, Common Crawl, and News Commentary v11. Development data sets include `newstest[2010-2015]`. `newstest2016` should serve as test data. All SGM files were converted to plain text.  | 4.56M/3K/2.6K | 32k BPE| [Generate](https://github.com/google/seq2seq/blob/master/bin/data/wmt16_en_de.sh) <br/> [Download](https://drive.google.com/open?id=0B_bZck-ksdkpM25jRUN2X2UxMm8) |
| WMT'17 All Pairs | Data for the [WMT'17 Translation Task](http://www.statmt.org/wmt17/translation-task.html). | Coming soon. | Coming soon. | [Coming soon]() |
| Toy Copy | A toy dataset where the target sequence is equal to the source sequence. The model must learn to copy the source sequence. | 10k/1k/1k | 20 | [Generate](https://github.com/google/seq2seq/blob/master/bin/data/toy.sh) |
| Toy Reverse | A toy dataset where the target sequence is equal to the reversed source sequence. The model must learn to reverse the source sequence. | 10k/1k/1k | 20 | [Generate](https://github.com/google/seq2seq/blob/master/bin/data/toy.sh) |

## Creating your own data

To create your own data, we recommend taking a look at the data generation scripts above. A typical data preprocessing pipeline looks as follows:

1. Generate data in parallel text format
2. Tokenize your data
3. Create fixed vocabularies for your source and target data
4. Learn and apply subword units to handle rare and unknown words