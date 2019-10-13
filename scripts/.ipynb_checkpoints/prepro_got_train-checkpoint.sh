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
#!/bin/bash
bert_vocab_file=./bert_pretrained/bert_base/vocab.txt
sup_size=20
sub_set=train
max_seq_length=128

echo Supervised set size of $sup_size
echo Running on $sub_set set

# Preprocess supervised training set
python preprocess.py \
  --raw_data_dir=./Data \
  --output_base_dir=./Data/proc_data/GoT/${sub_set}_${sup_size} \
  --data_type=sup \
  --sub_set=${sub_set} \
  --sup_size=${sup_size} \
  --max_seq_length=${max_seq_length}
  --vocab_file=$bert_vocab_file \
  $@
