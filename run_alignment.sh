#!/usr/bin/env bash

# Usage: ./run_alignment.sh <trial_number>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <trial_number>"
    exit 1
fi

TRIAL_NUM=$1
TRIAL_NAME="trial_${TRIAL_NUM}"

BASE_DIR="/media/emmah/PortableSSD/Arclab_data/thread_meat_3_21"

NPY_FILE="${BASE_DIR}/thread_meat_3_21_collected/${TRIAL_NAME}.npy"
# PNG_FILE="${BASE_DIR}/thread_meat_3_21_collected/${TRIAL_NAME}_left.png" choose between png with marker or without marker
PNG_FILE="${BASE_DIR}/thread_meat_3_21_collected/${TRIAL_NAME}_label_left.png"
THREAD_FILE="${BASE_DIR}/thread_meat_3_21_collected/${TRIAL_NAME}_spline.npy"
NEEDLE_POS_FILE="${BASE_DIR}/needle_pose/${TRIAL_NAME}_needle_pose.pkl"

python alignment_run.py \
    --npy_file "$NPY_FILE" \
    --png_file "$PNG_FILE" \
    --thread "$THREAD_FILE" \
    --needle_pos "$NEEDLE_POS_FILE"