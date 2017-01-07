## Performing Inference

After you have trained your mode you can use the `bin/infer.py` script to make predictions with your model. For example, from the [Getting Started Guide](getting-started.md):

```bash
./bin/infer.py \
  --source $HOME/nmt_data/toy_reverse/test/sources.txt \
  --vocab_source $HOME/nmt_data/toy_reverse/train/vocab.sources.txt \
  --vocab_target $HOME/nmt_data/toy_reverse/train/vocab.targets.txt \
  --model AttentionSeq2Seq \
  --model_dir ${TMPDIR}/nmt_toy_reverse \
  > ${TMPDIR}/nmt_toy_reverse/predictions.txt
```

The inference script reads the model hyperparameters from the `hparams.txt` file in the model directory, so
you do not need to pass them explicitly. By default, the latest model checkpoint found in `model_dir` is used, but you can also pass a specific checkpoint (e.g. `${TMPDIR}/nmt_toy_reverse/model.ckpt-1562`) via
the `checkpoint_path` flag.

## Beam Search

To perform beam search you can pass the `beam_width` flag to specify the number of beams to use. When using beam search
your batch size will be set to 1 and the `beam_width` will be used as an implicit batch size. Beam search can thus become very expensive with large beam widths.


## UNK token replacement using a Copy Mechanism

Rare words (such as place and people names) are often not included in the target vocabulary and result in `UNK` tokens in the output predictions. An easy strategy to improve predictions is to replace each `UNK` token with the word in the source sequence it is best aligned with. Alignments are typically calculated using an attention mechanism which produces `source_length` alignment scores for each target token.

If you trained a model, such as `AttentionSeq2Seq`, that generates such attention scores you can use them to perform UNK token replacement by passing the `unk_replace` flag to the inference script.


```bash
./bin/infer.py \
  --source $HOME/nmt_data/toy_reverse/test/sources.txt \
  --vocab_source $HOME/nmt_data/toy_reverse/train/vocab.sources.txt \
  --vocab_target $HOME/nmt_data/toy_reverse/train/vocab.targets.txt \
  --model AttentionSeq2Seq \
  --model_dir ${TMPDIR}/nmt_toy_reverse \
  --unk_replace \
  > ${TMPDIR}/nmt_toy_reverse/predictions.txt
```


## UNK token replacement using a mapping

A slightly more sophisticated approach to UNK token replacement is to use a mapping instead of copying words from the source. For example, "Munich" is translated as "München" in German, but if you were to simply copy "Munich" from the source you would never get this translation right.

You can use [fast_align](https://github.com/clab/fast_align) to generate such a mapping basedon the
conditional probabilities of target given source.

```bash
# Download and build fast_align
git clone https://github.com/clab/fast_align.git
mkdir fast_align/build && cd fast_align/build
cmake ../ && make

# Bring you data into a format that fast_align understands
paste \
  $HOME/nmt_data/toy_reverse/train/sources.txt \
  $HOME/nmt_data/toy_reverse/train/targets.txt \
  | sed "s/$(printf '\t')/ ||| /g" > $HOME/nmt_data/toy_reverse/train/source_targets.fastalign

# Learn alignments
./fast_align \
  -i $HOME/nmt_data/toy_reverse/train/source_targets.fastalign \
  -v -p $HOME/nmt_data/toy_reverse/train/source_targets.cond \
  > $HOME/nmt_data/toy_reverse/train/source_targets.align
```

The output file specified by the `-p` argument will contain conditional probabilities
for `p(target | source)` in the format of `<source>\t<target>\t<prob>`. These can
be used to do smarter UNK token replacement:

```bash
./bin/infer.py \
  --source $HOME/nmt_data/toy_reverse/test/sources.txt \
  --vocab_source $HOME/nmt_data/toy_reverse/train/vocab.sources.txt \
  --vocab_target $HOME/nmt_data/toy_reverse/train/vocab.targets.txt \
  --model AttentionSeq2Seq \
  --model_dir ${TMPDIR}/nmt_toy_reverse \
  --unk_replace \
  --unk_mapping $HOME/nmt_data/toy_reverse/train/source_targets.cond \
  > ${TMPDIR}/nmt_toy_reverse/predictions.txt
```





