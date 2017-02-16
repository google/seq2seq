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

""" Generate beam search visualization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import json
import shutil
from string import Template
import numpy as np

import networkx as nx
from networkx.readwrite import json_graph

PARSER = argparse.ArgumentParser(
    description="Generate beam search visualizations")
PARSER.add_argument(
    "-d", "--data", type=str, required=True,
    help="path to the beam search data file")
PARSER.add_argument(
    "-o", "--output_dir", type=str, required=True,
    help="path to the output directory")
PARSER.add_argument(
    "-v", "--vocab", type=str, required=False,
    help="path to the vocabulary file")
ARGS = PARSER.parse_args()


HTML_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Beam Search</title>
    <link rel="stylesheet" type="text/css" href="tree.css">
    <script src="http://d3js.org/d3.v3.min.js"></script>
  </head>
  <body>
    <script>
      var treeData = $DATA
    </script>
    <script src="tree.js"></script>
  </body>
</html>""")


def _add_graph_level(graph, level, parent_ids, names, scores):
  """Adds a levelto the passed graph"""
  for i, parent_id in enumerate(parent_ids):
    new_node = (level, i)
    parent_node = (level - 1, parent_id)
    graph.add_node(new_node)
    graph.node[new_node]["name"] = names[i]
    graph.node[new_node]["score"] = str(scores[i])
    graph.node[new_node]["size"] = 100
    # Add an edge to the parent
    graph.add_edge(parent_node, new_node)

def create_graph(predicted_ids, parent_ids, scores, vocab=None):
  def get_node_name(pred):
    return vocab[pred] if vocab else str(pred)

  seq_length = predicted_ids.shape[0]
  graph = nx.DiGraph()
  for level in range(seq_length):
    names = [get_node_name(pred) for pred in predicted_ids[level]]
    _add_graph_level(graph, level + 1, parent_ids[level], names, scores[level])
  graph.node[(0, 0)]["name"] = "START"
  return graph


def main():
  beam_data = np.load(ARGS.data)

  # Optionally load vocabulary data
  vocab = None
  if ARGS.vocab:
    with open(ARGS.vocab) as file:
      vocab = file.readlines()
    vocab = [_.strip() for _ in vocab]
    vocab += ["UNK", "SEQUENCE_START", "SEQUENCE_END"]

  if not os.path.exists(ARGS.output_dir):
    os.makedirs(ARGS.output_dir)

  # Copy required files
  shutil.copy2("./bin/tools/beam_search_viz/tree.css", ARGS.output_dir)
  shutil.copy2("./bin/tools/beam_search_viz/tree.js", ARGS.output_dir)

  for idx in range(len(beam_data["predicted_ids"])):
    predicted_ids = beam_data["predicted_ids"][idx]
    parent_ids = beam_data["beam_parent_ids"][idx]
    scores = beam_data["scores"][idx]

    graph = create_graph(
        predicted_ids=predicted_ids,
        parent_ids=parent_ids,
        scores=scores,
        vocab=vocab)

    json_str = json.dumps(
        json_graph.tree_data(graph, (0, 0)),
        ensure_ascii=False)

    html_str = HTML_TEMPLATE.substitute(DATA=json_str)
    output_path = os.path.join(ARGS.output_dir, "{:06d}.html".format(idx))
    with open(output_path, "w") as file:
      file.write(html_str)
    print(output_path)


if __name__ == "__main__":
  main()