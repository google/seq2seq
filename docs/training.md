For a concrete of how to run the training script, refer to the [Neural Machine Translation Tutorial](nmt/).

## Configuring Training

Also see [Configuration](concepts/#configuration). The configuration for input data, models, and training parameters is done via [YAML](https://en.wikipedia.org/wiki/YAML). You can pass YAML strings directly to the training script, or create configuration files and pass their paths to the script. These two approaches are technically equivalent. However, large YAML strings can become difficult to manage so we recommend the latter one. For example, the following two are equivalent:

1\. Pass FLAGS directly:

```shell
python -m bin.train \
  --model AttentionSeq2Seq \
  --model_params "
    embedding.dim: 256
    encoder.class: seq2seq.encoders.BidirectionalRNNEncoder
    encoder.params:
      rnn_cell:
        cell_class: GRUCell"
```


2\. Define `config.yml`

```yaml
model: AttentionSeq2Seq
model_params:
  embedding.dim: 256
  encoder.class: seq2seq.encoders.BidirectionalRNNEncoder
  encoder.params:
    rnn_cell:
      cell_class: GRUCell
```

... and pass FLAGS via config:

```shell
python -m bin.train --config_paths config.yml
```


Multiple configuration files are merged recursively, in the order they are passed. This means you can have separate configuration files for model hyperparameters, input data, and training options, and mix and match as needed.

For a concrete examples of configuration files, refer to the [example configurations](https://github.com/google/seq2seq/tree/master/example_configs) and [Neural Machine Translation Tutorial](NMT/).


## Monitoring Training

In addition to looking at the output of the training script, Tensorflow write summaries and training logs into the specified `output_dir`. Use [Tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/) to visualize training progress.

```shell
tensorboard --logdir=/path/to/model/dir
```

## Distributed Training

Distributed Training is supported out of the box using `tf.learn`. Cluster Configurations can be specified using the `TF_CONFIG` environment variable, which is parsed by the [`RunConfig`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/estimators/run_config.py). Refer to the [Distributed Tensorflow](https://www.tensorflow.org/how_tos/distributed/) Guide for more information.


## Training script Reference

The [train.py](https://github.com/google/seq2seq/blob/master/bin/train.py) script has many more options.

| Argument | Default | Description |
| --- | --- | --- |
| config_paths | `""` | Path to a YAML configuration file defining FLAG values. Multiple files can be separated by commas. Files are merged recursively. Setting a key in these files is equivalent to setting the FLAG value with the same name. |
| hooks | `"[]"` | YAML configuration string for the training hooks to use. |
| metrics | `"[]"` | YAML configuration string for the training metrics to use. |
| model | `""` | Name of the model class. Can be either a fully-qualified name, or the name of a class defined in `seq2seq.models`. |
| model_params | `"{}"` | YAML configuration string for the model parameters. |
| input_pipeline_train | `"{}"` | YAML configuration string for the training data input pipeline. |
| input_pipeline_dev | `"{}"` | YAML configuration string for the development data input pipeline. |
| buckets | `None` | Buckets input sequences according to these length. A comma-separated list of sequence length buckets, e.g. `"10,20,30"` would result in 4 buckets: `<10, 10-20, 20-30, >30`. `None` disables bucketing. |
| batch_size | `16` | Batch size used for training and evaluation. |
| output_dir | `None` | The directory to write model checkpoints and summaries to. If None, a local temporary directory is created. |
| train_steps | `None` | Maximum number of training steps to run. If None, train forever. |
| eval_every_n_steps | `1000` | Run evaluation on validation data every N steps. |
| tf_random_seed | `None` | Random seed for TensorFlow initializers. Setting this value allows consistency between reruns. |
| save_checkpoints_secs | `600` | Save checkpoints every N seconds. Can not be specified with `save_checkpoints_steps`. |
| save_checkpoints_steps | `None` | Save checkpoints every N steps. Can not be specified with `save_checkpoints_secs`. |
| keep_checkpoint_max | `5` | Maximum number of recent checkpoint files to keep. As new files are created, older files are deleted. If None or 0, all checkpoint files are kept. |
| keep_checkpoint_every_n_hours | `4` | In addition to keeping the most recent checkpoint files, keep one checkpoint file for every N hours of training. |

