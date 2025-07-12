#!/usr/bin/env bash

# Usage: ./run_alignment.sh <trial_number>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <trial_number>"
    exit 1
fi

TRIAL_NUM=$1
TRIAL_NAME="trial_${TRIAL_NUM}"

# BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="/media/emmah/PortableSSD/Arclab_data/thread_meat_3_21"

THREAD_FILE="${BASE_DIR}/thread_meat_3_21_collected/${TRIAL_NAME}_spline.npy"
GOAL_FILE="${BASE_DIR}/thread_meat_3_21_collected/grasp/${TRIAL_NAME}_2025_06_17_14_38_15.npy"

# reliability
THREAD_SPECS_FILE="${BASE_DIR}/thread_meat_3_21_collected/${TRIAL_NAME}_spline_specs.pkl"

python thread_gripping_sim.py \
    --thread_specs "$THREAD_SPECS_FILE" \
    -t "$THREAD_FILE" \
    --goal_file "$GOAL_FILE"