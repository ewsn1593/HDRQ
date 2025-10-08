# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add HR, LR, and attention visualization
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

TEST_ROOT=$1
W_BIT=$2
A_BIT=$3
DATASET_NAME=$4
SEED=$5
NOTE=$6 # Additioanl text
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
CHECKPOINT_FILE="${TEST_ROOT}/iter_40000.pth"
SHOW_DIR="${TEST_ROOT}/preds"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR

echo "W"$W_BIT"A"$A_BIT
echo ./log/${W_BIT}${A_BIT}test_brecq_${DATASET_NAME}_seed${SEED}_${NOTE}.txt
python -m tools.test_brecq ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1 --act_quant --n_bits_w ${W_BIT} --n_bits_a ${A_BIT} --save_ds ${DATASET_NAME} --seed ${SEED} 2>&1 | tee ./log/${W_BIT}${A_BIT}test_brecq_${DATASET_NAME}_seed${SEED}_${NOTE}.txt
# [Lagecy] python -m tools.test_brecq ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1 --n_bits_w 2 --n_bits_a 4 --act_quant

# Example commands (GTA -> CS, BRECQ, W8A8, seed: 2025)
# ./test_brecq.sh work_dirs/fp_gtaHR2csHR_hrda_r101_a5271 8 8 cs 2025 fixed

# Example commands (GTA -> IDD, BRECQ, W8A8, seed: 2025)
# ./test_brecq.sh work_dirs/fp_gtaHR2iddHR_hrda_r101_b958a 8 8 idd 2025 fixed