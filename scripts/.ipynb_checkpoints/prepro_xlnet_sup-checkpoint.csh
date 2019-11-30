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
set sup_size = (     20     200    2000    5000   12000     0      0 )
set sub_set =  ( 'train' 'train' 'train' 'train' 'train' 'dev' 'test')
foreach i      ( `seq 1 7`)             

  # Run the cases
  echo Running on ${sub_set[${i}]} set

  # Flag for training versus non-training
  if (${sub_set[${i}]} == 'train') then
      # Preprocess supervised training set
      echo Supervised set size of ${sup_size[${i}]}
    python preprocess.py \
      --raw_data_dir=./Data \
      --output_base_dir=./Data/proc_data/GoT_xlnet/${sub_set[${i}]}_${sup_size[${i}]} \
      --data_type=sup \
      --sub_set=${sub_set[${i}]} \
      --sup_size=${sup_size[${i}]} \
      --xlnet=True \
      --spiece_model_file=./xlnet_pretrained/xlnet_base/spiece.model \
      --overwrite_data=True \
      \$@
  else
    # Preprocess supervised evaluation set
    python preprocess.py \
      --raw_data_dir=./Data \
      --output_base_dir=./Data/proc_data/GoT_xlnet/${sub_set[${i}]} \
      --data_type=sup \
      --sub_set=${sub_set[${i}]} \
      --xlnet=True \
      --spiece_model_file=./xlnet_pretrained/xlnet_base/spiece.model \
      --overwrite_data=True \
      \$@
  endif
end
