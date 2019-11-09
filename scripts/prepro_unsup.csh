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
#!/bin/csh

# Set up cases to process data for
set bert_vocab_file = ./bert_pretrained/bert_base/vocab.txt
set sub_set =  ( 'unsup' 'unsup' 'unsup' 'unsup' 'unsup' 'unsup' 'unsup' 'unsup' 'unsup' )
set probs =    (    0.1     0.2     0.3     0.4     0.5     0.6     0.7     0.8     0.9  )
foreach i      (      1       2       3       4       5       6       7       8       9  )

  # Run the cases
  echo Running on ${sub_set[${i}]} set

  # Preprocess unlabeled set
python preprocess.py \
  --raw_data_dir=./Data \
  --output_base_dir=./Data/proc_data/GoT/${sub_set[${i}]} \
  --data_type=${sub_set[${i}]} \
  --sub_set=unsup_in \
  --aug_ops=tf_idf-${probs[${i}]} \
  --aug_copy_num=0 \
  --vocab_file=$bert_vocab_file
end
