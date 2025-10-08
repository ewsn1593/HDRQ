# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add HR, LR, and attention visualization
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

TEST_ROOT=$1
MODEL_PATH=$2
WORLD_SIZE=$3

CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
CHECKPOINT_FILE="${TEST_ROOT}/iter_40000.pth"
SHOW_DIR="${TEST_ROOT}/preds"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR

echo "model path: " $MODEL_PATH "/ world size (n_gpus): " $WORLD_SIZE
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=2333 -m evals.example6 ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1 --launcher pytorch
# python -m evals.example3_single ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1
python -m evals.example3 ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1 --model_path $MODEL_PATH --world_size $WORLD_SIZE
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=2333 -m evals.example ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1

# ./example.sh work_dirs/fp_gtaHR2csHR_hrda_r101_a5271 ckpt/CS/HDRQ_PTQTestR101_W6A6_fixed_CS_seed1005.pt 4
# CUDA_VISIBLE_DEVICES=4,5,6,7 ./example.sh work_dirs/fp_gtaHR2iddHR_hrda_r101_b958a ckpt/IDD/HDRQ_PTQTestR101_W6A6_fixed_IDD_seed1005.pt 4