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
"""build datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import os
import string

import numpy as np
import tensorflow as tf


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  # Moved to io class
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example


def get_sup_feature_specs(max_seq_len):
 
  """
  This function creates a dictionary which maps feature names to 
  Fixed Length Features of the appropriate dimensions.
  """
  feature_specs = collections.OrderedDict()
  feature_specs["input_ids"] = tf.io.FixedLenFeature([max_seq_len], tf.int64)
  feature_specs["input_mask"] = tf.io.FixedLenFeature([max_seq_len], tf.int64)
  feature_specs["input_type_ids"] = tf.io.FixedLenFeature([max_seq_len], tf.int64)
  feature_specs["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
  return feature_specs

def get_sup_feature_specs_eval(max_seq_len):
 
  """
  This function creates a dictionary which maps feature names to 
  Fixed Length Features of the appropriate dimensions.
  """
  feature_specs = collections.OrderedDict()
  feature_specs["input_ids"] = tf.io.FixedLenFeature([max_seq_len], tf.int64)
  feature_specs["input_mask"] = tf.io.FixedLenFeature([max_seq_len], tf.int64)
  feature_specs["input_type_ids"] = tf.io.FixedLenFeature([max_seq_len], tf.int64)
  feature_specs["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
  return feature_specs


def get_unsup_feature_specs(max_seq_len):
  """
  This function creates a dictionary which maps feature names to 
  Fixed Length Features of the appropriate dimensions.
  """
  feature_specs = collections.OrderedDict()
  feature_specs["ori_input_ids"] = tf.io.FixedLenFeature(
        [max_seq_len], tf.int64)
  feature_specs["ori_input_mask"] = tf.io.FixedLenFeature(
        [max_seq_len], tf.int64)
  feature_specs["ori_input_type_ids"] = tf.io.FixedLenFeature(
        [max_seq_len], tf.int64)
  feature_specs["aug_input_ids"] = tf.io.FixedLenFeature(
        [max_seq_len], tf.int64)
  feature_specs["aug_input_mask"] = tf.io.FixedLenFeature(
        [max_seq_len], tf.int64)
  feature_specs["aug_input_type_ids"] = tf.io.FixedLenFeature(
        [max_seq_len], tf.int64)
  return feature_specs


def get_aug_files(data_base_path, aug_ops, aug_copy):
  """get aug files."""
  tf.logging.info("Getting Augmented Data Files")
  sub_policy_list = aug_ops.split("+")
  total_data_files = []
  for sub_policy in sub_policy_list:
    sub_policy_data_files = []
    
    exist_copy_num = {}
    for copy_dir in tf.io.gfile.listdir(os.path.join(
        data_base_path, sub_policy)):
      tf.logging.info("Looking at {}".format(copy_dir))
      if copy_dir[-5:] != 'Store':
        copy_num = int(copy_dir.strip("/"))
        tf.logging.info("Using copy number {}".format(copy_num))
        if copy_num >= aug_copy:
          continue
        exist_copy_num[copy_num] = 1
        tf.logging.info("exist_copy_num: ".format(exist_copy_num))
        data_record_path = os.path.join(
            data_base_path, sub_policy, copy_dir, "tf_examples.tfrecord*")
        data_files = tf.contrib.slim.parallel_reader.get_data_files(
            data_record_path)
        sub_policy_data_files += data_files
        tf.logging.info("Current number of files to process: {}".format(len(sub_policy_data_files)))
    if len(exist_copy_num) < aug_copy * 0.9:
      tf.logging.info("not enough copies for aug op: {:s}".format(aug_ops))
      tf.logging.info("found files: {:s}".format(
          " ".join(sub_policy_data_files)))
      tf.logging.info("found copy: {:d} / desired copy: {:d}".format(
          len(exist_copy_num), aug_copy))
    assert len(exist_copy_num) > aug_copy * 0.9
    total_data_files += sub_policy_data_files
  np.random.shuffle(total_data_files)
  tf.logging.info("Number of data files to process: {}".format(len(total_data_files)))
  return total_data_files


def get_training_dataset(total_data_files, batch_size, is_training,
                         shuffle_buffer_size,feature_specs):
  """
  Simplified version of original function. Handles files serially. Not a big
  deal because we only ever load in <4 files
  """
  d = tf.data.TFRecordDataset(total_data_files)
  tf.logging.debug("{}".format(d))
  d = d.shuffle(buffer_size=shuffle_buffer_size)
  d = d.map(lambda record: _decode_record(record, feature_specs))
  d = d.batch(batch_size=batch_size, drop_remainder=is_training)
  tf.logging.debug("Returning batch data {}".format(d))
  return d


def get_evaluation_dataset(total_data_files, batch_size, feature_specs):
  """build non-repeat dataset from files."""
  tf.logging.debug("{}".format(feature_specs))
  d = tf.data.TFRecordDataset(total_data_files)
  d = d.map(lambda record: _decode_record(record, feature_specs))
  d = d.batch(batch_size=batch_size, drop_remainder=True)
  tf.logging.debug("Returning evaluation batch data {}".format(d))
  return d


def evaluation_input_fn_builder(data_base_path, task, prefetch_size=1000, options=None,max_seq_len=None):

  total_data_files = tf.contrib.slim.parallel_reader.get_data_files(
      os.path.join(data_base_path, "tf_examples.tfrecord*"))
  tf.logging.info("loading eval {} data from these files: {:s}".format(
      task, " ".join(total_data_files)))

  def input_fn(params):
    batch_size = params["eval_batch_size"]

    if task == "clas":
      dataset = get_evaluation_dataset(
          total_data_files,
          batch_size,
          get_sup_feature_specs(max_seq_len))
    else:
      assert False

    dataset = dataset.prefetch(prefetch_size)

    return dataset

  return input_fn


def training_input_fn_builder(
    sup_data_base_path=None,
    unsup_data_base_path=None,
    aug_ops=None,
    aug_copy=None,
    unsup_ratio=None,
    num_threads=8,
    shuffle_buffer_size=100000,
    prefetch_size=1000, 
    max_seq_len=None):
  
  # Generate list of input files from which to grab our records
  sup_total_data_files = tf.contrib.slim.parallel_reader.get_data_files(
      os.path.join(sup_data_base_path, "tf_examples.tfrecord*"))
  
  # Print the files out to the screen
  tf.logging.info("loading training data from these files: {:s}".format(
      " ".join(sup_total_data_files)))

  # Unsupervised Data input requirements
  if unsup_ratio is not None and unsup_ratio > 0:
    assert aug_ops is not None and aug_copy is not None, \
        "Require aug_ops, aug_copy to load augmented unsup data."
    assert unsup_data_base_path is not None and unsup_data_base_path != "", \
        "Require unsup_data_base_path to load unsup data. Get {}.".format(
            unsup_data_base_path)

    unsup_total_data_files = get_aug_files(
        unsup_data_base_path, aug_ops, aug_copy)

  is_training = True

  def input_fn(params):
    """The `input_fn` for our Esimator."""
    sup_batch_size = params["train_batch_size"]
    total_batch_size = 0
    tf.logging.info("Supervised batch size: %d", (sup_batch_size))

    dataset_list = []

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if sup_data_base_path is not None:
      tf.logging.info("Getting training examples")
      
      # Get training dataset returns a batch of decoded data
      sup_dst = get_training_dataset(
          sup_total_data_files,
          sup_batch_size,
          is_training,
          shuffle_buffer_size,
          get_sup_feature_specs(max_seq_len))
      tf.logging.info("Got a batch of training data of size: {}".format(sup_batch_size))
      total_batch_size += sup_batch_size
      dataset_list.append(sup_dst)

      ## only consider unsupervised data when supervised data is considered
      if unsup_data_base_path is not None and unsup_ratio > 0:
        unsup_dst = get_training_dataset(
            unsup_total_data_files,
            sup_batch_size * unsup_ratio,
            is_training,
            shuffle_buffer_size,
            get_unsup_feature_specs(max_seq_len))
        total_batch_size += sup_batch_size * unsup_ratio * 2
        dataset_list.append(unsup_dst)
        tf.logging.info("unsup batch size: %d", (sup_batch_size * unsup_ratio))

    tf.logging.info("total sample in a batch: %d", (total_batch_size))

    def flatten_input(*features):
      """Merging multiple feature dicts resulted from zipped datasets."""
      result = {}
      for feature in features:
        for key in feature:
          assert key not in result
          result[key] = feature[key]

      return result

    if len(dataset_list) > 1:
      d = tf.data.Dataset.zip(tuple(dataset_list))
      d = d.map(flatten_input)
    else:
      d = dataset_list[0]

    # Prefetching creates a buffer to make sure there is always data to
    # read in the event of network latency variance.
    d = d.prefetch(prefetch_size)

    # Estimator supports returning a dataset instead of just features.
    # It will call `make_one_shot_iterator()` and such.
    return d
  return input_fn


