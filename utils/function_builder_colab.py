"""doc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf
from xpath import modeling, xlnet_colab
from upath import model_utils



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
  xlnet_config = xlnet_colab.XLNetConfig(options)
  xlnet_config.to_json(os.path.join(options['model_dir'], "config.json"))

  # construct run config from FLAGS
  run_config = xlnet_colab.create_run_config(is_training, False, options)

  xlnet_model = xlnet_colab.XLNetModel(
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

  xlnet_config = xlnet_colab.XLNetConfig(json_path=options['model_config_file'])
  run_config = xlnet_colab.create_run_config(is_training, True, options)

  xlnet_model = xlnet_colab.XLNetModel(
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

def get_uda_classification_loss(
    options, features, n_class, is_training, global_step):
  """Loss for downstream classification tasks."""


  tsa = options['tsa']
  unsup_ratio = options['unsup_ratio']
  num_train_steps = options['num_train_steps']
  uda_softmax_temp = options['uda_softmax_temp']
  uda_confidence_thresh = options['uda_confidence_thresh']

  bsz_per_core = tf.shape(features["input_ids"])[0]

  inp = tf.transpose(features["input_ids"], [1, 0])
  seg_id = tf.transpose(features["segment_ids"], [1, 0])
  inp_mask = tf.transpose(features["input_mask"], [1, 0])
  labels = tf.reshape(features["label_ids"], [bsz_per_core])

  num_sample = features["input_ids"].shape[0].value
  tf.logging.info("Batch Size {}".format(num_sample))
  if is_training:
    assert num_sample % (1 + 2 * unsup_ratio) == 0
    sup_batch_size = num_sample // (1 + 2 * unsup_ratio)
    unsup_batch_size = sup_batch_size * unsup_ratio
  else:
    sup_batch_size = num_sample
    unsup_batch_size = 0

  xlnet_config = xlnet_colab.XLNetConfig(json_path=options['model_config_file'])
  run_config = xlnet_colab.create_run_config(is_training, True, options)

  xlnet_model = xlnet_colab.XLNetModel(
      xlnet_config=xlnet_config,
      run_config=run_config,
      input_ids=inp,
      seg_ids=seg_id,
      input_mask=inp_mask)

  summary = xlnet_model.get_pooled_out(options['summary_type'],
                                       options['use_summ_proj'])

  if options['cls_scope'] is not None and options['cls_scope']:
    cls_scope = "classification_{}".format(options['cls_scope'])
  else:
    cls_scope = "classification_{}".format(options['task_name'].lower())

  clas_logits = modeling.uda_logits(
      hidden=summary,
      labels=labels,
      n_class=n_class,
      initializer=xlnet_model.get_initializer(),
      scope=cls_scope)

  log_probs = tf.nn.log_softmax(clas_logits, axis=-1)
  tf.logging.info(log_probs)
  correct_label_probs = None

  with tf.variable_scope("sup_loss"):
    sup_log_probs = log_probs[:sup_batch_size]
    one_hot_labels = tf.one_hot(labels, depth=n_class, dtype=tf.float32)
    tgt_label_prob = one_hot_labels

    per_example_loss = -tf.reduce_sum(tgt_label_prob * sup_log_probs, axis=-1)
    loss_mask = tf.ones_like(per_example_loss, dtype=per_example_loss.dtype)
    correct_label_probs = tf.reduce_sum(
        one_hot_labels * tf.exp(sup_log_probs), axis=-1)

    if tsa:
      logging.info("Applying TSA")
      # Starting threshold is just the inverse number of labels.
      tsa_start = 1. / num_labels
      tsa_threshold = model_utils.get_tsa_threshold(
          tsa, global_step, num_train_steps,
          tsa_start, end=1)

      larger_than_threshold = tf.greater(
          correct_label_probs, tsa_threshold)
      loss_mask = loss_mask * (1 - tf.cast(larger_than_threshold, tf.float32))
    else:
      tsa_threshold = 1

    loss_mask = tf.stop_gradient(loss_mask)
    per_example_loss = per_example_loss * loss_mask
    sup_loss = (tf.reduce_sum(per_example_loss) /
                tf.maximum(tf.reduce_sum(loss_mask), 1))

  unsup_loss_mask = None
  if is_training and unsup_ratio > 0:
    with tf.variable_scope("unsup_loss"):
      ori_start = sup_batch_size
      ori_end = ori_start + unsup_batch_size
      aug_start = sup_batch_size + unsup_batch_size
      aug_end = aug_start + unsup_batch_size

      ori_log_probs = log_probs[ori_start : ori_end]
      aug_log_probs = log_probs[aug_start : aug_end]
      unsup_loss_mask = 1
      if options['uda_softmax_temp'] != -1:
        tgt_ori_log_probs = tf.nn.log_softmax(
            clas_logits[ori_start : ori_end] / options['uda_softmax_temp'],
            axis=-1)
        tgt_ori_log_probs = tf.stop_gradient(tgt_ori_log_probs)
      else:
        tgt_ori_log_probs = tf.stop_gradient(ori_log_probs)

      if options['uda_confidence_thresh'] != -1:
        largest_prob = tf.reduce_max(tf.exp(ori_log_probs), axis=-1)
        unsup_loss_mask = tf.cast(tf.greater(
            largest_prob, options['uda_confidence_thresh']), tf.float32)
        unsup_loss_mask = tf.stop_gradient(unsup_loss_mask)

      per_example_kl_loss = model_utils.kl_for_log_probs(
          tgt_ori_log_probs, aug_log_probs) * unsup_loss_mask
      unsup_loss = tf.reduce_mean(per_example_kl_loss)

  else:
    unsup_loss = 0.

  return (sup_loss, unsup_loss, clas_logits[:sup_batch_size],
          per_example_loss, loss_mask,
          tsa_threshold, unsup_loss_mask, correct_label_probs)
