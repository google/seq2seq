## Pre-processed datasets

[See the data generation notebooks](https://github.com/dennybritz/seq2seq/tree/master/notebooks) for details on how this data was generated.

| Dataset | Description | Training/Dev/Test Examples | Vocab Size | URL |
| --- | --- | --- | --- | --- |
| WMT'16 EN-DE | Data for the [WMT'16 Translation Task](http://www.statmt.org/wmt16/translation-task.html) English to German. Training data is combined from Europarl v7, Common Crawl, and News v11. Development data is newstest2013. Test data is newstest2015.  | 4.56M/3K/2.6K | 50k | [Download](https://drive.google.com/open?id=0B_bZck-ksdkpdmlvajhSbS1JTXc) |
| Toy Copy | A toy dataset where the target sequence is equal to the source sequence. Thus, the network must learn to "copy" the source sequence.  | 10k/1k/1k | 20 | [Download](https://drive.google.com/open?id=0B_bZck-ksdkpX0FFbHFRbGY3UTQ) |
| Toy Reverse | A toy dataset where the target sequence is equal to the reversed source sequence. | 10k/1k/1k | 20 | [Download](https://drive.google.com/open?id=0B_bZck-ksdkpR2Z1ZWRQZEZDVHM) |


## Training/Dev data in Parallel Text Format

The input pipeline expect parallel tokenized data in raw text format. That is, you need `sources.txt` and a `targets.txt` file that contain corresponding sentences, aligned line-by-line. Each line corresponds to one input/output example. These words/tokens in these files must be separated by spaces.

## Generating Vocabulary

A vocabulary file is a raw text file that contains one token per line. The total number of lines is the size of the vocabulary, and each token is mapped to its line number. The special words `UNK`, `SEQUENCE_START` and `SEQUENCE_END` are not part of the vocabulary file, and correspond to `vocab_size + 1`, `vocab_size + 2`, and `vocab_size + 3` respectively

Given a raw text file of tokens separated by spaces you can generate a vocabulary file using the [`generate_vocab.py`](https://github.com/dennybritz/seq2seq/blob/master/seq2seq/scripts/generate_vocab.py) script:

```shell
./seq2seq/scripts/generate_vocab.py \
  --input_file /data/source.txt \
  --output_file /data/source_vocab \
  --min_frequency 1 \
  --max_vocab_size 50000
```

The resulting vocabulary file contains one word per line.


## Subword Unit Preprocessing

In order to deal with an open vocabulary, rare words can be split into
subword units as proposed in [1]. This improves the model's translation
performance particularly on rare words. The authors propose to use
Byte Pair Encoding (BPE), a simple compression algorithm, for splitting
words into subwords. Starting from characters, BPE iteratively replaces the
most frequent pair of symbols with a new symbol. The final symbol vocabulary
is equal to the size of the initial vocabulary plus the number of merge
operations, which is the only hyperparameter of the method.
To apply BPE as a pre-processing step to your raw text input,
follow the below steps:

1. Download the open-source implementation of the paper
   from [here](https://github.com/rsennrich/subword-nmt).
2. Learn the BPE encoding: `cd subword-nmt`.
   `./learn_bpe.py -s {num_operations} -i {train_file} -o {codes_file}`.
   `num_operations` is the number of merge operations. Default is `10000`.
3. Apply the BPE encoding to the training and test files:
  `./apply_bpe.py -c {codes_file} -i {input_file} -o {output_file}`.

The resulting BPE-processed files can be used as-is in place of the raw text
files for training the NMT model.

References:

[1] Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of
Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of
the Association for Computational Linguistics (ACL 2016).
Retrieved from http://arxiv.org/abs/1508.07909


## Training/Dev data in TFRecords format (Old)

The input pipeline expects data in [TFRecord](https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html#file-formats) format consisting of [`tf.Example`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) protocol buffers. Each `Example` record contains the following fields:

- `pair_id (string)` (optional) is a dataset-unique id for this example.
- `source_len (int64)` is the length of the source sequence.
- `target_len (int64)` is the length of the target sequence.
- `source_tokens (string)` is a list of source tokens.
- `target_tokens (string)` is a list of targets tokens.

Given a parallel corpus, i.e. source and target files aligned by line such as [those from WMT](http://www.statmt.org/wmt16/translation-task.html), we provide a [script](https://github.com/dennybritz/seq2seq/blob/master/seq2seq/scripts/generate_examples.py) to generate a corresponding TFRecords file:

```bash
./seq2seq/scripts/generate_examples.py \
  --source_file /path/to/data/train.de.txt \
  --target_file /path/to/data/train.en.txt \
  --output_file /path/to/data/train.tfrecords
```