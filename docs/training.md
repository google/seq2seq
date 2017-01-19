### Input Files

In order to train a model, you need the following files. Refer to [Data](https://github.com/dennybritz/seq2seq/wiki/Data) for more details on each of these.

- Training data: Two parallel (aligned line by line) tokenized text files with tokens separated by spaces.
- Development data: Same format as the training data, but used for validation.
- Source vocabulary: Defines the source vocabulary. A raw text file that contains one word per line.
- Target vocabulary: Defines the target vocabulary. A raw text file that contains one word per line.

### Running Training

To train a new model, run the training script below (also see [Getting Started](getting_started.md)):

```shell
./bin/train.py \
  --train_source $HOME/nmt_data/toy_reverse/train/sources.txt \
  --train_target $HOME/nmt_data/toy_reverse/train/targets.txt \
  --dev_source $HOME/nmt_data/toy_reverse/dev/sources.txt \
  --dev_target $HOME/nmt_data/toy_reverse/dev/targets.txt \
  --vocab_source $HOME/nmt_data/toy_reverse/train/vocab.sources.txt \
  --vocab_target $HOME/nmt_data/toy_reverse/train/vocab.targets.txt \
  --model AttentionSeq2Seq \
  --batch_size 32 \
  --train_epochs 5 \
  --hparams "embedding.dim=512,optimizer.name=Adam" \
  --output_dir ${TMPDIR}/nmt_toy_reverse
```

### Passing Hyperparameters

[Model hyperparameters](models.md) can be passed via the `hparams` flags. This flag is a string of the form
`"param1=value1,param2=value2,..."`. Whitespace between parameters pairs is ignored and you can have line breaks in your string. For complex parameters specifications, like cell specificatons, we recommend using a separate configuration file (see below).


### Passing a configuration file

An alternative to passing arguments to the training script is to define a configuration file in YAML format and pass it via the `config_path` flags. For example, the trian command above would be expressed as follows in a configuration file:

```yaml
train_source: /home/nmt_data/toy_reverse/train/sources.txt
train_target: /home/nmt_data/toy_reverse/train/targets.txt
dev_source: /home/nmt_data/toy_reverse/dev/sources.txt
dev_target: /home/nmt_data/toy_reverse/dev/targets.txt
vocab_source: /home/nmt_data/toy_reverse/train/vocab.sources.txt
vocab_target: /home/nmt_data/toy_reverse/train/vocab.targets.txt
model: AttentionSeq2Seq
batch_size: 32
train_epochs: 5
hparams:
  embedding.dim: 512
  optimizer.name: Adam
output_dir: /tmp/nmt_toy_reverse
```

Note that environment variables in configuration files are not yet supported. Flags defined as both command line arguments and configuration file values are overwritten by configuration file values. Command line flags not present in the configuration file will be merged. Parameters that   as JSON objects on the command like (e.g. `cell_spec`) can be defined as key-value pairs directly in the YAML file.


### Distributed Training

Distributed Training is supported out of the box using `tf.learn`. Cluster Configurations can be specified using the `TF_CONFIG` environment variable, which is parsed by the [`RunConfig`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/estimators/run_config.py). Refer to the [Distributed Tensorflow](https://www.tensorflow.org/how_tos/distributed/) Guide for more information.


### Monitoring Training

In addition to looking at the output of the training script, Tensorflow write summaries and training logs into the specified `output_dir`. Use [Tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/) to visualize training progress.

```shell
tensorboard --logdir=/path/to/model/dir
```

### Training script arguments

The [train.py](https://github.com/dennybritz/seq2seq/blob/master/bin/train.py) script has many more options. Bold arguments are required.

| Argument | Default | Description |
| --- | --- | --- |
| **train_source** | --- | Path to the training data source sentences. A raw text file with tokens separated by spaces. |
| **train_target** | --- | Path to the training data target sentences. A raw text file with tokens separated by spaces. |
| **dev_source** | --- | Path to the development data source sentences. Same format as training data. |
| **dev_target** | --- | Path to the development data target sentences. Same format as training data.|
| **vocab_source** | --- | Path to the source vocabulary. A raw text file with one word per line. |
| **vocab_target** | --- | Path to the target vocabulary. A raw text file with one word per line. |
| **delimiter** | `" "` | Split input files into tokens on this delimiter. Defaults to `" "` (space). |
| config_path | --- | Path to a YAML configuration file defining FLAG values and hyperparameters. See below. |
| model | `AttentionSeq2Seq` | The model class to use. Refer to the documentation for all available models. |
| buckets | `None` | Buckets input sequences according to these length. A comma-separated list of sequence length buckets, e.g. `"10,20,30"` would result in 4 buckets: `<10, 10-20, 20-30, >30`. `None` disables bucketing. |
| batch_size | `16` | Batch size used for training and evaluation. |
| hparams | `None` | A comma-separated list of hyeperparameter values that overwrite the model defaults, e.g. `"optimizer.name=Adam,optimizer.learning_rate=0.1"`. Refer to the Models section and the TensorFlow documentation for a detailed list of available hyperparameters. |
| output_dir | `None` | The directory to write model checkpoints and summaries to. If None, a local temporary directory is created. |
| train_steps | `None` | Maximum number of training steps to run. If None, train forever. |
| train_epochs | `None` | Maximum number of training epochs over the data. If None, train forever. |
| eval_every_n_steps | `1000` | Run evaluation on validation data every N steps. |
| sample_every_n_steps | `500` | Sample and print sequence predictions every N steps during training. |
| tf_random_seed | `None` | Random seed for TensorFlow initializers. Setting this value allows consistency between reruns. |
| save_checkpoints_secs | `600` | Save checkpoints every N seconds. Can not be specified with `save_checkpoints_steps`. |
| save_checkpoints_steps | `None` | Save checkpoints every N steps. Can not be specified with `save_checkpoints_secs`. |
| keep_checkpoint_max | `5` | Maximum number of recent checkpoint files to keep. As new files are created, older files are deleted. If None or 0, all checkpoint files are kept. |
| keep_checkpoint_every_n_hours | `4` | In addition to keeping the most recent checkpoint files, keep one checkpoint file for every N hours of training. |

