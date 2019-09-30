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

bert_path=../bert_pretrained/bert_base
bert_vocab_file=${bert_path}/vocab.txt
bert_ckpt=${bert_path}/bert_model.ckpt
sup_size=20
sub_set=train


python main.py \
  --use_tpu=False \
  --do_train=True \
  --do_eval=True \
  --sup_train_data_dir=../Data/proc_data/GoT/${sub_set}_${sup_size} \
  --eval_data_dir=Data/proc_data/GoT/dev \
  --bert_config_file=pretrained_models/bert_base/bert_config.json \
  --vocab_file=${bert_vocab_file} \
  --init_checkpoint=${bert_ckpt} \
  --task_name=GoT \
  --model_dir=ckpt/base \
  --num_train_steps=3000 \
  --learning_rate=3e-05 \
  --num_warmup_steps=300 \
  $@
