## Ready-to-use Datasets

We prepared the following datasets to help you get started. You can either download the data or generate it yourself
using the data generation scripts in the `tools/data` folder.

| Dataset | Description | Training/Dev/Test Size | Vocabulary | URL |
| --- | --- | --- | --- | --- |
| WMT'16 EN-DE | Data for the [WMT'16 Translation Task](http://www.statmt.org/wmt16/translation-task.html) English to German. Training data is combined from Europarl v7, Common Crawl, and News Commentary v11. Development data sets include `newstest[2010-2015]`. `newstest2016` should serve as test data. All SGM files were converted to plain text. Refer to the [`tools/data/wmt_16_en_de.sh`](https://github.com/dennybritz/seq2seq/blob/master/bin/data/wmt16_en_de.sh) script for more details.  | 4.56M/3K/2.6K | 50k Words <br/> 50k BPE| [Download](https://drive.google.com/open?id=0B_bZck-ksdkpdmlvajhSbS1JTXc) |
| Toy Copy | A toy dataset where the target sequence is equal to the source sequence. The model must learn to copy the source sequence. You can generate this dataset using [`bin/data/toy.sh`](https://github.com/dennybritz/seq2seq/blob/master/bin/data/toy.sh). | 10k/1k/1k | 20 | [Download](https://drive.google.com/open?id=0B_bZck-ksdkpX0FFbHFRbGY3UTQ) |
| Toy Reverse | A toy dataset where the target sequence is equal to the reversed source sequence. You can generate this dataset using [`bin/data/toy.sh`](https://github.com/dennybritz/seq2seq/blob/master/bin/data/toy.sh). | 10k/1k/1k | 20 | [Download](https://drive.google.com/open?id=0B_bZck-ksdkpR2Z1ZWRQZEZDVHM) |

## Creating your own data

If you want to use your own data you need to bring it into the right format. A typical data preprocessing pipeline looks as follows:

1. Generate data in parallel text format
2. Tokenize your data
3. Create a fixed vocabulary for your source and target data
4. (Optional) Use Subword Units to handle rare or unknown words
5. (Optional) Learn an alignment dictionary to handle unknown words

### Parallel Text Format

The input pipeline expect parallel data in raw text format that is tokenized. You need a `sources` and `targets` file that contain corresponding sequences, aligned line-by-line. Each line corresponds to one input/output example. The tokens (words) in these files must be separated by spaces. For example, in Machine Translation you typically have one file with sentences in the source language, and one file with the corresponding translations.

### Tokenization

To get good results it is crucial to [tokenize your text](http://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html). Tokenization can make a big difference in the final results and there are many tools available, such as:

- Use the Moses [`tokenizer.perl`](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) script.
- Use a library such a [spaCy](https://spacy.io/docs/usage/processing-text), [nltk](http://www.nltk.org/api/nltk.tokenize.html) or [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml).

For example, to use the Moses tokenizer:

```bash
# Clone from Github
git clone https://github.com/moses-smt/mosesdecoder.git

# Tokenize English (en) data
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -threads 8 < english_data > english_data.tok

# Tokenize German (de) data
mosesdecoder/scripts/tokenizer/tokenizer.perl -l de -threads 8 < german_data > german_data.tok
```

### Generating Vocabulary

A vocabulary file is a raw text file that contains one token per line. The total number of lines is the size of the vocabulary, and each token is mapped to its line number. The special words `UNK`, `SEQUENCE_START` and `SEQUENCE_END` are not part of the vocabulary file, and correspond to `vocab_size + 1`, `vocab_size + 2`, and `vocab_size + 3` respectively

Given a raw text file of tokens separated by spaces you can generate a vocabulary file using the [`generate_vocab.py`](https://github.com/dennybritz/seq2seq/blob/master/seq2seq/bin/tools/generate_vocab.py) script:

```shell
./bin/tools/generate_vocab.py \
  --input_file /data/source.txt \
  --output_file /data/source_vocab \
  --min_frequency 1 \
  --max_vocab_size 50000
```

The resulting vocabulary file contains one word per line.

### Subword Units (BPE)

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


### Dictionary Alignments

TODO


