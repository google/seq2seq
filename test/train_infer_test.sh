#! /usr/bin/env bash

TEST_DIR="${TMPDIR}/seq2seq"

# Clar out data
rm -rf $TEST_DIR
mkdir -p $TEST_DIR

# Create dummy vocabulary
echo -e "a\nb\nc" > $TEST_DIR/vocab

# Create dummy train data
echo "a a a a
a a b b
b b c c" | tee $TEST_DIR/train.sources $TEST_DIR/train.targets > /dev/null

# Create dummy dev data
echo "a a a a
b b b b
c c c c" | tee $TEST_DIR/dev.sources $TEST_DIR/dev.targets > /dev/null

# Run training for 100 steps
./bin/train.py \
  --train_source $TEST_DIR/train.sources \
  --train_target $TEST_DIR/train.targets \
  --dev_source $TEST_DIR/dev.sources \
  --dev_target $TEST_DIR/dev.sources \
  --vocab_source $TEST_DIR/vocab \
  --vocab_target $TEST_DIR/vocab \
  --model AttentionSeq2Seq \
  --batch_size 2 \
  --train_steps 100 \
  --output_dir ${TEST_DIR}/out

# Make sure that the model checkpoint exists
if [ ! -f "${TEST_DIR}/out/model.ckpt-100.data-00000-of-00001" ]; then
  echo "Expected model checkpoint not found. Failing test."
  exit 1;
fi;

# Run inference on the model checkpoint
./bin/infer.py \
  --source $TEST_DIR/dev.sources \
  --vocab_source $TEST_DIR/vocab \
  --vocab_target $TEST_DIR/vocab \
  --model AttentionSeq2Seq \
  --model_dir ${TEST_DIR}/out \
  --checkpoint_path "${TEST_DIR}/out/model.ckpt-100" \
  --batch_size 2 > ${TEST_DIR}/predictions.txt

# Make sure that the predictions are non-empty
if [ ! -s "${TEST_DIR}/predictions.txt" ]; then
  echo "Expected ${TEST_DIR}/predictions.txt to contain data."
  exit 1;
fi;

echo "Success."

exit 0
