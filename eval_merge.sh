# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add HR, LR, and attention visualization
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

TEST_ROOT1=$1
TEST_ROOT2=$2
MODEL_PATH1=$3
MODEL_PATH2=$4
WORLD_SIZE=$5
QMETHOD=$6
PORT_NUM=$7

if [ -z "$PORT_NUM" ]; then
  PORT_NUM=29703
  echo "port_num is null -> set to $PORT_NUM"
fi

CONFIG_FILE1="${TEST_ROOT1}/*${TEST_ROOT1: -1}.json"
CONFIG_FILE2="${TEST_ROOT2}/*${TEST_ROOT2: -1}.json"
# CHECKPOINT_FILE="${TEST_ROOT}/iter_40000.pth"
SHOW_DIR="${TEST_ROOT}/preds"
echo 'Config File1:' $CONFIG_FILE1
echo 'Config File2:' $CONFIG_FILE2
echo 'Predictions Output Directory:' $SHOW_DIR

echo "model path1: " $MODEL_PATH1 ", model path2: " $MODEL_PATH2 ", qmethod: " $QMETHOD ", world size (n_gpus): " $WORLD_SIZE ", port num: " $PORT_NUM
python -m evals.eval_merge ${CONFIG_FILE1} ${CONFIG_FILE2} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1 --model_path1 $MODEL_PATH1 --model_path2 $MODEL_PATH2 --world_size $WORLD_SIZE --qmethod $QMETHOD --port_num $PORT_NUM
     