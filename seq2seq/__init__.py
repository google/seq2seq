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
"""
seq2seq library base module
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seq2seq.graph_module import GraphModule

from seq2seq import contrib
from seq2seq import data
from seq2seq import decoders
from seq2seq import encoders
from seq2seq import global_vars
from seq2seq import graph_utils
from seq2seq import inference
from seq2seq import losses
from seq2seq import metrics
from seq2seq import models
from seq2seq import test
from seq2seq import training
