#! /usr/bin/env bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

DATA_TYPE=${DATA_TYPE:-copy}
echo "Using type=${DATA_TYPE}. To change this set DATA_TYPE to 'copy' or 'reverse'"

OUTPUT_DIR=${OUTPUT_DIR:-$HOME/nmt_data/toy_${DATA_TYPE}}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_TRAIN="${OUTPUT_DIR}/train"
OUTPUT_DIR_DEV="${OUTPUT_DIR}/dev"
OUTPUT_DIR_TEST="${OUTPUT_DIR}/test"

mkdir -p $OUTPUT_DIR

# Write train, dev and test data
${BASE_DIR}/bin/tools/generate_toy_data.py  \
  --type ${DATA_TYPE} \
  --num_examples 10000 \
  --vocab_size 20 \
  --max_len 20 \
  --output_dir ${OUTPUT_DIR_TRAIN}

${BASE_DIR}/bin/tools/generate_toy_data.py  \
  --type ${DATA_TYPE} \
  --num_examples 1000 \
  --vocab_size 20 \
  --max_len 20 \
  --output_dir ${OUTPUT_DIR_DEV}

${BASE_DIR}/bin/tools/generate_toy_data.py  \
  --type ${DATA_TYPE} \
  --num_examples 1000 \
  --vocab_size 20 \
  --max_len 20 \
  --output_dir ${OUTPUT_DIR_TEST}

# Create Vocabulary
${BASE_DIR}/bin/tools/generate_vocab.py \
  --input_file ${OUTPUT_DIR_TRAIN}/sources.txt \
  --output_file ${OUTPUT_DIR_TRAIN}/vocab.sources.txt
${BASE_DIR}/bin/tools/generate_vocab.py \
  --input_file ${OUTPUT_DIR_TRAIN}/targets.txt \
  --output_file ${OUTPUT_DIR_TRAIN}/vocab.targets.txt

# Create character vocabulary
${BASE_DIR}/bin/tools/generate_char_vocab.py \
  < ${OUTPUT_DIR_TRAIN}/sources.txt \
  > ${OUTPUT_DIR_TRAIN}/vocab.sources.char.txt
${BASE_DIR}/bin/tools/generate_char_vocab.py \
  < ${OUTPUT_DIR_TRAIN}/targets.txt \
  > ${OUTPUT_DIR_TRAIN}/vocab.targets.char.txt

# Creating zip file
ARCHIVE_PATH="${OUTPUT_DIR}/toy_copy.tar.gz"
tar -cvzf ${ARCHIVE_PATH} \
  -C ${OUTPUT_DIR} train/ dev/ test/
echo "Wrote ${ARCHIVE_PATH}"
