#! /usr/bin/env python
"""
Functions to generate various toy datasets.
"""

import argparse
import os
import numpy as np
from sklearn.cross_validation import train_test_split

PARSER = argparse.ArgumentParser(description="Generates toy datasets.")
PARSER.add_argument(
    "--vocab_size", type=int, default=100, help="size of the vocabulary")
PARSER.add_argument(
    "--num_examples", type=int, default=10000, help="number of examples")
PARSER.add_argument(
    "--min_len", type=int, default=5, help="minimum sequence length")
PARSER.add_argument(
    "--max_len", type=int, default=40, help="maximum sequence length")
PARSER.add_argument(
    "--dev_split",
    type=float,
    default=0.1,
    help="Fraction of data to use for the dev set")
PARSER.add_argument(
    "--type",
    type=str,
    default="copy",
    choices=["copy", "reverse"],
    help="Type of dataet to generate. One of \"copy\" or \"reverse\"")
PARSER.add_argument(
    "--output_dir",
    type=str,
    help="path to the output directory",
    required=True)
ARGS = PARSER.parse_args()

VOCABULARY = list([str(x) for x in range(ARGS.vocab_size)])


def make_copy(num_examples, min_len, max_len):
  """
  Generates a dataset where the target is equal to the source.
  Sequence lengths are chosen randomly from [min_len, max_len].

  Args:
    num_examples: Number of examples to generate
    min_len: Minimum sequence length
    max_len: Maximum sequence length

  Returns:
    An iterator of (source, target) string tuples.
  """
  for _ in range(num_examples):
    turn_length = np.random.choice(np.arange(min_len, max_len + 1))
    source_tokens = np.random.choice(
        list(VOCABULARY), size=turn_length, replace=True)
    target_tokens = source_tokens
    yield " ".join(source_tokens), " ".join(target_tokens)


def make_reverse(num_examples, min_len, max_len):
  """
  Generates a dataset where the target is equal to the source reversed.
  Sequence lengths are chosen randomly from [min_len, max_len].

  Args:
    num_examples: Number of examples to generate
    min_len: Minimum sequence length
    max_len: Maximum sequence length

  Returns:
    An iterator of (source, target) string tuples.
  """
  for _ in range(num_examples):
    turn_length = np.random.choice(np.arange(min_len, max_len + 1))
    source_tokens = np.random.choice(
        list(VOCABULARY), size=turn_length, replace=True)
    target_tokens = source_tokens[::-1]
    yield " ".join(source_tokens), " ".join(target_tokens)


def write_parallel_text(sources, targets, output_prefix):
  """
  Writes two files where each line corresponds to one example
    - [output_prefix].sources.txt
    - [output_prefix].targets.txt

  Args:
    sources: Iterator of source strings
    targets: Iterator of target strings
    output_prefix: Prefix for the output file
  """
  source_filename = os.path.abspath(output_prefix + ".sources.txt")
  target_filename = os.path.abspath(output_prefix + ".targets.txt")

  with open(source_filename, "w") as source_file:
    for record in sources:
      source_file.write(record + "\n")
  print("Wrote {}".format(source_filename))

  with open(target_filename, "w") as target_file:
    for record in targets:
      target_file.write(record + "\n")
  print("Wrote {}".format(target_filename))


def main():
  """Main function"""

  if ARGS.type == "copy":
    generate_fn = make_copy
  elif ARGS.type == "reverse":
    generate_fn = make_reverse

  # Generate dataset
  examples = list(generate_fn(ARGS.num_examples, ARGS.min_len, ARGS.max_len))
  examples_train, examples_dev = train_test_split(
      examples, test_size=ARGS.dev_split)
  os.makedirs(ARGS.output_dir, exist_ok=True)

  # Write train data
  train_sources, train_targets = zip(*examples_train)
  write_parallel_text(train_sources, train_targets,
                      os.path.join(ARGS.output_dir, "train"))

  # Write dev data
  dev_sources, dev_targets = list(zip(*examples_dev))
  write_parallel_text(dev_sources, dev_targets,
                      os.path.join(ARGS.output_dir, "dev"))


if __name__ == "__main__":
  main()
