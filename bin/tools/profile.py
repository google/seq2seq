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

""" Script to generates model profiling information
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import six

#pylint: disable=E0611
from google.protobuf import text_format

import tensorflow as tf
from tensorflow.contrib.tfprof import model_analyzer
from tensorflow.contrib.tfprof.python.tools.tfprof import tfprof_logger
from tensorflow import gfile
from tensorflow.tools.tfprof import tfprof_log_pb2
from tensorflow.python.framework import op_def_registry # pylint: disable=E0611
from tensorflow.python.framework.ops import RegisterShape # pylint: disable=E0611
from tensorflow.python.framework import common_shapes # pylint: disable=E0611

# Import custom ops
from seq2seq.decoders.attention import att_sum_bahdanau, att_sum_dot


tf.flags.DEFINE_string("model_dir", None, "path to model directory")

FLAGS = tf.flags.FLAGS
CUSTOM_OP_FUNCTIONS = [att_sum_bahdanau, att_sum_dot]

def _register_function_ops(func_list):
  """Registers custom ops in the default graph. This is needed
  Because our checkpoint is saved with ops that are not part of Tensorflow."""
  op_dict = op_def_registry.get_registered_ops()
  for func in func_list:
    #pylint: disable=W0212
    func._create_definition_if_needed()
    op_def = func._definition.signature
    op_dict[op_def.name] = op_def
    RegisterShape(op_def.name)(common_shapes.unknown_shape)

def load_metadata(model_dir):
  """Loads RunMetadata, Graph and OpLog from files
  """
  # Import RunMetadata
  run_meta_path = os.path.join(model_dir, "metadata/run_meta")
  run_meta = tf.RunMetadata()
  if gfile.Exists(run_meta_path):
    with gfile.GFile(run_meta_path, "rb") as file:
      run_meta.MergeFromString(file.read())
    print("Loaded RunMetadata from {}".format(run_meta_path))
  else:
    print("RunMetadata does not exist a {}. Skipping.".format(run_meta_path))

  # Import Graph
  graph_def_path = os.path.join(model_dir, "graph.pbtxt")
  graph = tf.Graph()
  if gfile.Exists(graph_def_path):
    with graph.as_default():
      _register_function_ops(CUSTOM_OP_FUNCTIONS)
      graph_def = tf.GraphDef()
      with gfile.GFile(graph_def_path, "rb") as file:
        text_format.Parse(file.read(), graph_def)
      tf.import_graph_def(graph_def, name="")
      print("Loaded Graph from {}".format(graph_def_path))
  else:
    print("Graph does not exist a {}. Skipping.".format(graph_def_path))

  # Import OpLog
  op_log_path = os.path.join(model_dir, "metadata/tfprof_log")
  op_log = tfprof_log_pb2.OpLog()
  if gfile.Exists(op_log_path):
    with gfile.GFile(op_log_path, "rb") as file:
      op_log.MergeFromString(file.read())
      print("Loaded OpLog from {}".format(op_log_path))
  else:
    print("OpLog does not exist a {}. Skipping.".format(op_log_path))

  return run_meta, graph, op_log


def merge_default_with_oplog(graph, op_log=None, run_meta=None):
  """Monkeypatch. There currently is a bug in tfprof_logger that
    prevents it from being used with Python 3. So we override the method
    manually until the fix comes in.
  """
  tmp_op_log = tfprof_log_pb2.OpLog()
  # pylint: disable=W0212
  logged_ops = tfprof_logger._get_logged_ops(graph, run_meta)
  if not op_log:
    tmp_op_log.log_entries.extend(logged_ops.values())
  else:
    all_ops = dict()
    for entry in op_log.log_entries:
      all_ops[entry.name] = entry
    for op_name, entry in six.iteritems(logged_ops):
      if op_name in all_ops:
        all_ops[op_name].types.extend(entry.types)
        if entry.float_ops > 0 and all_ops[op_name].float_ops == 0:
          all_ops[op_name].float_ops = entry.float_ops
      else:
        all_ops[op_name] = entry
    tmp_op_log.log_entries.extend(all_ops.values())
  return tmp_op_log


def param_analysis_options(output_dir):
  """Options for model parameter analysis
  """
  options = model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS.copy()
  options["select"] = ["params", "bytes"]
  options["order_by"] = "params"
  options["account_type_regexes"] = ["Variable"]
  if output_dir:
    options["dump_to_file"] = os.path.join(output_dir, "params.txt")
  return "scope", options


def micro_anaylsis_options(output_dir):
  """Options for microsecond analysis
  """
  options = model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS.copy()
  options["select"] = ["micros", "device"]
  options["min_micros"] = 1000
  options["account_type_regexes"] = [".*"]
  options["order_by"] = "micros"
  if output_dir:
    options["dump_to_file"] = os.path.join(output_dir, "micro.txt")
  return "graph", options


def flops_analysis_options(output_dir):
  """Options for FLOPS analysis
  """
  options = model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS.copy()
  options["select"] = ["float_ops", "micros", "device"]
  options["min_float_ops"] = 1
  options["order_by"] = "float_ops"
  options["account_type_regexes"] = [".*"]
  if output_dir:
    options["dump_to_file"] = os.path.join(output_dir, "flops.txt")
  return "scope", options


def device_analysis_options(output_dir):
  """Options for device placement analysis
  """
  options = model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS.copy()
  options["select"] = ["device", "float_ops", "micros"]
  options["order_by"] = "name"
  options["account_type_regexes"] = [".*"]
  if output_dir:
    options["dump_to_file"] = os.path.join(output_dir, "device.txt")
  return "scope", options


def main(_argv):
  """Main functions. Runs all anaylses."""
  # pylint: disable=W0212
  tfprof_logger._merge_default_with_oplog = merge_default_with_oplog

  FLAGS.model_dir = os.path.abspath(os.path.expanduser(FLAGS.model_dir))
  output_dir = os.path.join(FLAGS.model_dir, "profile")
  gfile.MakeDirs(output_dir)

  run_meta, graph, op_log = load_metadata(FLAGS.model_dir)

  param_arguments = [
      param_analysis_options(output_dir),
      micro_anaylsis_options(output_dir),
      flops_analysis_options(output_dir),
      device_analysis_options(output_dir),
  ]

  for tfprof_cmd, params in param_arguments:
    model_analyzer.print_model_analysis(
        graph=graph,
        run_meta=run_meta,
        op_log=op_log,
        tfprof_cmd=tfprof_cmd,
        tfprof_options=params)

    if params["dump_to_file"] != "":
      print("Wrote {}".format(params["dump_to_file"]))


if __name__ == '__main__':
  tf.app.run()
