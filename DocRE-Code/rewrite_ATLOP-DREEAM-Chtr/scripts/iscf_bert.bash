#!/bin/bash
MODEL_DIR=$1
MODEL_EVI_DIR=$2
SPLIT=$3

echo python run_iscf.py --data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--load_path ${MODEL_EVI_DIR} \
--eval_mode single \
--test_file ${SPLIT}.json \
--test_batch_size 8 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class 97

echo python run_iscf.py --data_dir dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--load_path ${MODEL_DIR} \
--results_path ${MODEL_EVI_DIR} \
--eval_mode fushion \
--test_file ${SPLIT}.json \
--test_batch_size 32 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class 97