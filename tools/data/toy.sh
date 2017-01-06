#! /usr/bin/env bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

OUTPUT_DIR=${OUTPUT_DIR:-$HOME/nmt_data/toy_copy}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

DATA_TYPE=${DATA_TYPE:-copy}
echo "Using type=${DATA_TYPE}. To change this set DATA_TYPE to 'copy' or 'reverse'"

OUTPUT_DIR_TRAIN="${OUTPUT_DIR}/train"
OUTPUT_DIR_DEV="${OUTPUT_DIR}/dev"
OUTPUT_DIR_TEST="${OUTPUT_DIR}/test"

mkdir -p $OUTPUT_DIR

# Write train, dev and test data
${BASE_DIR}/bin/generate_toy_data.py  \
  --type copy \
  --num_examples 10000 \
  --vocab_size 20 \
  --max_len 20 \
  --output_dir ${OUTPUT_DIR_TRAIN}

${BASE_DIR}/bin/generate_toy_data.py  \
  --type copy \
  --num_examples 1000 \
  --vocab_size 20 \
  --max_len 20 \
  --output_dir ${OUTPUT_DIR_DEV}

${BASE_DIR}/bin/generate_toy_data.py  \
  --type copy \
  --num_examples 1000 \
  --vocab_size 20 \
  --max_len 20 \
  --output_dir ${OUTPUT_DIR_TEST}

# Create Vocabulary
${BASE_DIR}/bin/generate_vocab.py \
  --input_file ${OUTPUT_DIR_TRAIN}/sources.txt \
  --output_file ${OUTPUT_DIR_TRAIN}/vocab.txt

# Creating zip file
ARCHIVE_PATH="${OUTPUT_DIR}/toy_copy.tar.gz"
tar -cvzf ${ARCHIVE_PATH} \
  -C ${OUTPUT_DIR} train/ dev/ test/
echo "Wrote ${ARCHIVE_PATH}"
