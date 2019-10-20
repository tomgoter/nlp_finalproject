# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load raw data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import pandas as pd

from absl import flags

import tensorflow as tf

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

class GoTProcessor(object):

  def get_train_examples(self, raw_data_dir):
    """See base class."""
    examples = self._create_examples(
        pd.read_pickle(os.path.join(raw_data_dir, "train", "train.pkl")),
                        "train")
    print("Number of examples created: {}\nNumber expected: {}".format(len(examples),self.get_train_size()))
    assert len(examples) == self.get_train_size()
    return examples

  def get_dev_examples(self, raw_data_dir):
    """See base class."""
    return self._create_examples(
        pd.read_pickle(os.path.join(raw_data_dir, "dev", "dev.pkl")),
                        "dev")
    
  def get_test_examples(self, raw_data_dir):
    """See base class."""
    return self._create_examples(
        pd.read_pickle(os.path.join(raw_data_dir, "test", "test.pkl")),
                        "test")

  def get_unsup_examples(self, raw_data_dir, unsup_set):
    """See base class."""
    if unsup_set == "unsup_in":
      return self._create_examples(
          pd.read_pickle(os.path.join(raw_data_dir, "train", "train.pkl")),
          "unsup_in", skip_unsup=False)
    else:
      pass

  def _create_examples(self, df, set_type, skip_unsup=True,
                       only_unsup=False):
    """Creates examples for the training and dev sets."""
    examples = []
    for i, row in df.iterrows():
      if skip_unsup and row[1] == "unsup":
        continue
      if only_unsup and row[1] != "unsup":
        continue
      guid = "%s-%d".format(set_type, i)
      text_a = row[0]
      text_b = None
      label = row[1]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_train_size(self):
    return 15001

  def get_dev_size(self):
    return 2500

  def get_labels(self):
    """See base class."""
    return [str(i) for i in range(1, 6)]


def get_processor(task_name):
  """get processor."""
  task_name = task_name.lower()
  processors = {
      "got": GoTProcessor,
  }
  processor = processors[task_name]()
  return processor






