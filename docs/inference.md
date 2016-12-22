To make predictions with a trained model you need the following:

- A trained model directory, `model_dir`: The directory should contain the model checkpoint and a `hparams.txt` file with the hyperparameters used during training. This is the `output_dir` you specify in the training script.
- A `source_text` file in the same format as the [training input data](https://github.com/dennybritz/seq2seq/wiki/Data), one tokenized input example per line.
- Vocabulary files for both source and targets, `vocab_source` and `vocab_target`. Same as used during training.


Given the above, you use `scripts/infer.py` as follows:

```shell
./seq2seq/scripts/infer.py \
  --source sources.txt \
  --model_dir /tmp/model \
  --vocab_source vocab_source.txt \
  --vocab_target vocab_target.txt > out.txt
```