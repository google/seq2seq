#! /usr/bin/env bash

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The argument is the dat directory with all story files
# Downloaded from http://cs.nyu.edu/~kcho/DMQA/
DATA_DIR=$1

# Directory to write processed dataset to
OUTPUT_DIR=$2

# seq2seq root directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"

mkdir -p $OUTPUT_DIR
echo "Writing to $OUTPUT_DIR"

for story in $(find $DATA_DIR/ -name *.story); do
  $BASE_DIR/bin/data/cnn_daily_mail_summarization/process_story.py \
    < $story \
    >> ${OUTPUT_DIR}/stories_and_summaries.txt
done

# Split processed files into stories and summaries
cut -f 1 ${OUTPUT_DIR}/stories_and_summaries.txt > ${OUTPUT_DIR}/data.stories
cut -f 2 ${OUTPUT_DIR}/stories_and_summaries.txt > ${OUTPUT_DIR}/data.summaries

# Split into train/dev/test
# First 1000 lines are dev, next 1000 lines are test, the rest is train
tail -n +2000 ${OUTPUT_DIR}/data.stories > ${OUTPUT_DIR}/train.stories
tail -n +2000 ${OUTPUT_DIR}/data.summaries > ${OUTPUT_DIR}/train.summaries
head -n 1000 ${OUTPUT_DIR}/data.stories > ${OUTPUT_DIR}/dev.stories
head -n 1000 ${OUTPUT_DIR}/data.summaries > ${OUTPUT_DIR}/dev.summaries
head -n 2000 ${OUTPUT_DIR}/data.stories | tail -n +1000  > ${OUTPUT_DIR}/test.stories
head -n 2000 ${OUTPUT_DIR}/data.summaries | tail -n +1000 > ${OUTPUT_DIR}/test.summaries

# Use google/sentencepiece to learn vocabulary
# Follow installation instructions at https://github.com/google/sentencepiece
spm_train \
  --input=${OUTPUT_DIR}/train.stories,${OUTPUT_DIR}/train.summaries \
  --model_prefix=${OUTPUT_DIR}/bpe \
  --vocab_size=32000 \
  --model_type=bpe

for data in train dev test; do
  spm_encode --model=${OUTPUT_DIR}/bpe.model --output_format=piece \
    < ${OUTPUT_DIR}/${data}.summaries \
    > ${OUTPUT_DIR}/${data}.bpe.summaries
  spm_encode --model=${OUTPUT_DIR}/bpe.model --output_format=piece \
    < ${OUTPUT_DIR}/${data}.stories \
    > ${OUTPUT_DIR}/${data}.bpe.stories
done
