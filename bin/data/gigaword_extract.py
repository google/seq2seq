#! /usr/bin/env python

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

"""Processes the Gigaword Corpus (https://catalog.ldc.upenn.edu/LDC2011T07).
Generates source and target files where sources correspond to the first N
sentences of each article, and targets corresponds to article headlines.

Usage:

bin/data/gigaword_summary < GIGAWORD \
  -f gigaword_data
  -o output/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import argparse

from bs4 import BeautifulSoup

PARSER = argparse.ArgumentParser(
    description="Processes the Gigaword corpus.")
PARSER.add_argument(
    "-f", "--file", type=str, required=True,
    help="path to the Gigaword SGML file")
PARSER.add_argument(
    "-o", "--output_dir", type=str, required=True,
    help="path to the output directory")
PARSER.add_argument(
    "-n", "--num_sentences", type=int, required=False, default=2,
    help="Use the first N sentences as source text")
ARGS = PARSER.parse_args()

def gigaword_iter(path, n_sentences=2):
  """Creates an iterator that yields (source, target) tuples.
  """
  soup = BeautifulSoup(open(path), "html.parser")
  for doc in soup.find_all("doc"):
    # Skip docs without headline
    if doc.headline is None:
      continue
    # Find first N sentences
    sentences = doc.find_all("p")[:n_sentences]
    if not sentences:
      continue
    sentences = [_.text.strip().replace("\n", " ") for _ in sentences]
    sentences = [_.replace("  ", " ") for _ in sentences]
    headline = doc.headline.text.replace("\n", " ").strip()
    # Returns sentencs and headline
    yield " ".join(sentences), headline

def main():
  """The entrypoint for the script"""

  if not os.path.exists(ARGS.output_dir):
    os.makedirs(ARGS.output_dir)

  sources_filename = os.path.join(ARGS.output_dir, "sources.txt")
  targets_filename = os.path.join(ARGS.output_dir, "targets.txt")
  sources_file = open(sources_filename, "w")
  targets_file = open(targets_filename, "w")

  records = gigaword_iter(ARGS.file, ARGS.num_sentences)
  for i, (source, target) in enumerate(records, 1):
    sources_file.write(source + "\n")
    targets_file.write(target + "\n")
    if i % 1000 == 0:
      sys.stderr.write(".")
    if i % 100000 == 0:
      sys.stderr.write("\n")
  sys.stderr.write("\n")

  sources_file.close()
  targets_file.close()

if __name__ == "__main__":
  main()
