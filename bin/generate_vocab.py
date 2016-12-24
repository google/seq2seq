#! /usr/bin/env python
#pylint: disable=invalid-name
"""
Generate vocabulary for a tokenized text file.
"""

import argparse
import collections

parser = argparse.ArgumentParser(
    description="Generate vocabulary for a tokenized text file.")
parser.add_argument(
    "--input_file", type=str, help="path to the input file", required=True)
parser.add_argument(
    "--output_file",
    type=str,
    help="path to the vocabulary file",
    required=True)
parser.add_argument(
    "--min_frequency",
    dest="min_frequency",
    type=int,
    default=0,
    help="Minimum frequency of a word to be included in the vocabulary.")
parser.add_argument(
    "--max_vocab_size",
    dest="max_vocab_size",
    type=int,
    help="Maximum number of words in the vocabulary")
parser.add_argument(
    "--downcase",
    dest="downcase",
    type=bool,
    help="If set to true, downcase all text before processing.",
    default=False)

args = parser.parse_args()

# Counter for all words in the vocabulary
cnt = collections.Counter()

with open(args.input_file) as f:
  for line in f:
    if args.downcase:
      line = line.lower()
    tokens = line.strip().split(" ")
    tokens = [_ for _ in tokens if len(_) > 0]
    cnt.update(tokens)

print("Found {} unique words in the vocabulary.".format(len(cnt)))

# Filter words below the frequency threshold
if args.min_frequency > 0:
  filtered_words = [(w, c) for w, c in cnt.most_common()
                    if c > args.min_frequency]
  cnt = collections.Counter(dict(filtered_words))

print("Found {} unique words with frequency > {}.".format(
    len(cnt), args.min_frequency))

# Sort words by 1. frequency 2. lexically to break ties
word_with_counts = cnt.most_common()
word_with_counts = sorted(
    word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

# Take only max-vocab
if args.max_vocab_size is not None:
  word_with_counts = word_with_counts[:args.max_vocab_size]

with open(args.output_file, "w") as f:
  for word, count in word_with_counts:
    f.write("{}\n".format(word))

print("Wrote vocab of size {}: {}".format(
    len(word_with_counts), args.output_file))
