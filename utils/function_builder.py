"""doc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf
from xlnet import modeling, xlnet



def construct_scalar_host_call(
    monitor_dict,
    model_dir,
    prefix="",
    reduce_fn=None):
  """
  Construct host calls to monitor training progress on TPUs.
  """

  metric_names = list(monitor_dict.keys())

  def host_call_fn(global_step, *args):
    """actual host call function."""
    step = global_step[0]
    with tf.contrib.summary.create_file_writer(
        logdir=model_dir, filename_suffix=".host_call").as_default():
      with tf.contrib.summary.always_record_summaries():
        for i, name in enumerate(metric_names):
          if reduce_fn is None:
            scalar = args[i][0]
          else:
            scalar = reduce_fn(args[i])
          with tf.contrib.summary.record_summaries_every_n_global_steps(
              100, global_step=step):
            tf.contrib.summary.scalar(prefix + name, scalar, step=step)

        return tf.contrib.summary.all_summary_ops()

  global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [tf.reshape(monitor_dict[key], [1]) for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors


def two_stream_loss(options, features, labels, mems, is_training):
  """Pretraining loss with two-stream attention Transformer-XL."""

  #### Unpack input
  mem_name = "mems"
  mems = mems.get(mem_name, None)

  inp_k = tf.transpose(features["input_k"], [1, 0])
  inp_q = tf.transpose(features["input_q"], [1, 0])

  seg_id = tf.transpose(features["seg_id"], [1, 0])

  inp_mask = None
  perm_mask = tf.transpose(features["perm_mask"], [1, 2, 0])

  if options['num_predict'] is not None:
    # [num_predict x tgt_len x bsz]
    target_mapping = tf.transpose(features["target_mapping"], [1, 2, 0])
  else:
    target_mapping = None

  # target for LM loss
  tgt = tf.transpose(features["target"], [1, 0])

  # target mask for LM loss
  tgt_mask = tf.transpose(features["target_mask"], [1, 0])

  # construct xlnet config and save to model_dir
  xlnet_config = xlnet.XLNetConfig(options)
  xlnet_config.to_json(os.path.join(options['model_dir'], "config.json"))

  # construct run config from FLAGS
  run_config = xlnet.create_run_config(is_training, False, options)

  xlnet_model = xlnet.XLNetModel(
      xlnet_config=xlnet_config,
      run_config=run_config,
      input_ids=inp_k,
      seg_ids=seg_id,
      input_mask=inp_mask,
      mems=mems,
      perm_mask=perm_mask,
      target_mapping=target_mapping,
      inp_q=inp_q)

  output = xlnet_model.get_sequence_output()
  new_mems = {mem_name: xlnet_model.get_new_memory()}
  lookup_table = xlnet_model.get_embedding_table()

  initializer = xlnet_model.get_initializer()

  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    # LM loss
    lm_loss = modeling.lm_loss(
        hidden=output,
        target=tgt,
        n_token=xlnet_config.n_token,
        d_model=xlnet_config.d_model,
        initializer=initializer,
        lookup_table=lookup_table,
        tie_weight=True,
        bi_data=run_config.bi_data,
        use_tpu=run_config.use_tpu)

  #### Quantity to monitor
  monitor_dict = {}

  if options['use_bfloat16']:
    tgt_mask = tf.cast(tgt_mask, tf.float32)
    lm_loss = tf.cast(lm_loss, tf.float32)

  total_loss = tf.reduce_sum(lm_loss * tgt_mask) / tf.reduce_sum(tgt_mask)
  monitor_dict["total_loss"] = total_loss

  return total_loss, new_mems, monitor_dict


def get_loss(options, features, labels, mems, is_training):
  """Pretraining loss with two-stream attention Transformer-XL."""
  if options['use_bfloat16']:
    with tf.tpu.bfloat16_scope():
      return two_stream_loss(options, features, labels, mems, is_training)
  else:
    return two_stream_loss(options, features, labels, mems, is_training)


def get_classification_loss(
    options, features, n_class, is_training):
  """Loss for downstream classification tasks."""

  bsz_per_core = tf.shape(features["input_ids"])[0]

  inp = tf.transpose(features["input_ids"], [1, 0])
  seg_id = tf.transpose(features["segment_ids"], [1, 0])
  inp_mask = tf.transpose(features["input_mask"], [1, 0])
  label = tf.reshape(features["label_ids"], [bsz_per_core])

  xlnet_config = xlnet.XLNetConfig(json_path=options['model_config_path'])
  run_config = xlnet.create_run_config(is_training, True, options)

  xlnet_model = xlnet.XLNetModel(
      xlnet_config=xlnet_config,
      run_config=run_config,
      input_ids=inp,
      seg_ids=seg_id,
      input_mask=inp_mask)

  summary = xlnet_model.get_pooled_out(options['summary_type'],
                                       options['use_summ_proj'])

  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

    if options['cls_scope'] is not None and options['cls_scope']:
      cls_scope = "classification_{}".format(options['cls_scope'])
    else:
      cls_scope = "classification_{}".format(options['task_name'].lower())

    per_example_loss, logits = modeling.classification_loss(
        hidden=summary,
        labels=label,
        n_class=n_class,
        initializer=xlnet_model.get_initializer(),
        scope=cls_scope,
        return_logits=True)

    total_loss = tf.reduce_mean(per_example_loss)

    return total_loss, per_example_loss, logits
