### Input Files

To train a model you need to follow files. See [Data](https://github.com/dennybritz/seq2seq/wiki/Data) for more details on how to generate or download each of them.

- Training data: Two parallel (line by line aligned) text files that are tokenized, i.e. have words separated by spaces.
- Development data: Same format as the training data, but used for validation.
- Source vocabulary file: A file with one word per line that defines the source vocabulary.
- Target vocabulary file: Same format as the source vocabulary, but for the target language.

### Running Training

From the root directory, run:

```shell
python -m seq2seq.training.train \
--train_source=data/train.sources.txt \
--train_target=data/train.targets.txt \
--dev_source=data/dev.sources.txt \
--dev_target=data/dev.targets.txt \
--vocab_source=data/vocab_source \
--vocab_target=data/vocab_target \
--model AttentionSeq2Seq \
--batch_size=16 \
--hparams="rnn_cell.num_layers=2,
  rnn_cell.type=GRUCell,
  rnn_cell.num_units=128,
  source.max_seq_len=40,
  target.max_seq_len=40" \
--output_dir=/path/to/model/dir \
--buckets=10,15,20,25,30,35
```

Here, `train_source`, `train_target` `dev_source`, `dev_source`, `vocab_source` and `vocab_target` are the input files described above. 

`model` is the name of some class defined in `seq2seq.models`. Currently, the available models are:

- `BasicSeq2Seq` - Uses a unidirectional RNN encoder, passes state from encoder to decoder, and uses no attention mechanism.
- `AttentionSeq2Seq` - Uses a bidirectional RNN encoder and an attention mechanism.

Refer to the source code comments for more details of these details.

`hparams` are model-specific Hyperparameters. Refer to the [`default_params`](https://github.com/dennybritz/seq2seq/blob/master/seq2seq/models/attention_seq2seq.py#L25) of model classes for a list of all available hyperparameters.

`buckets` is an optional argument that you can use to speed up training by bucketing training examples into batches of roughly equal length. `10,15,20,25,30,35` would result in 8 buckets: Sequences of length `<10`, `10..15`, ..., and `>35`.

### Monitoring Training

In addition to looking at the training output, you can use Tensorboard to monitor progress:

```shell
tensorboard --logdir=/path/to/model/dir
```

