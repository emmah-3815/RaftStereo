#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <trial_number>"
    exit 1
fi

# Pad the trial number with leading zeros to 3 digits
TRIAL_NUM=$(printf "%01d" $1)
TRIAL_NAME="trial_$TRIAL_NUM"

BASE_DIR="/media/emmah/PortableSSD/Arclab_data"
# PARENT_FOLDER="6_12_25"
PARENT_FOLDER="7_12_25"

# Define paths
# BAG_FILE="${BASE_DIR}/${PARENT_FOLDER}/${TRIAL_NAME}.bag" # ros 1
BAG_FILE="${BASE_DIR}/${PARENT_FOLDER}/${TRIAL_NAME}/" # ros 2
IMAGE_DIR="${BASE_DIR}/${PARENT_FOLDER}/${TRIAL_NAME}/"
FRAME=0
FRAME_NAME="_$(printf "%03d" $FRAME)"

while true
do
    echo "-----------------------------------"
    echo "Select an option:"
    echo "1) Run rosbag_npy_generator.py"
    echo "2) Run fit_eval_exp.py"
    echo "3) Run alignment_run.py"
    echo "4) Exit"
    echo "-----------------------------------"

    read -p "Enter choice [1-4]: " choice

    case $choice in
        1)
            python rosbag_npy_generator.py \
                --bag_file "$BAG_FILE" \
                --image_dir "$IMAGE_DIR" \
                --topic_l /stereo/left/rectified_downscaled_image \
                --topic_r /stereo/right/rectified_downscaled_image \
                --output_name "$TRIAL_NAME" \
                --restore_ckpt /media/emmah/PortableSSD/Arclab_data/models/raftstereo-realtime.pth \
                --shared_backbone \
                --n_downsample 3 \
                --n_gru_layers 2 \
                --slow_fast_gru \
                --valid_iters 7 \
                --corr_implementation reg \
                --mixed_precision \
                --save_numpy
            ;;

        2)
            # segmenter
            read -p "Enter segmenter name (e.g. sam, hand, unet): " SEGMENTER
            python ~/ARClab/thread_reconstruction/src/fit_eval_exp.py \
                --parent_folder "$PARENT_FOLDER" \
                --trial "$TRIAL_NUM" \
                --frame "$FRAME" \
                --segmenter "$SEGMENTER"
            ;;

        3)
            NPY_FILE="${IMAGE_DIR}npy/${TRIAL_NAME}${FRAME_NAME}.npy"
            # PNG_FILE="${BASE_DIR}/thread_meat_3_21_collected/${TRIAL_NAME}_left.png" choose between png with marker or without marker
            PNG_FILE="${IMAGE_DIR}left_rgb/${TRIAL_NAME}${FRAME_NAME}.png"
            THREAD_FILE="${IMAGE_DIR}${TRIAL_NAME}${FRAME_NAME}_spline.npy"
            NEEDLE_POS_FILE="${IMAGE_DIR}${TRIAL_NAME}${FRAME_NAME}_needle_pose.pkl"

            # reliability
            THREAD_SPECS_FILE="${IMAGE_DIR}${TRIAL_NAME}${FRAME_NAME}_spline_specs.pkl"

            # masks
            # MEAT_MASK_FILE="${BASE_DIR}/thread_meat_3_21_collected/${TRIAL_NAME}_left_mask.png"
            THREAD_MASK_FILE="${IMAGE_DIR}left_rgb/${TRIAL_NAME}${FRAME_NAME}_mask.png"
            NEEDLE_MASK_FILE="${IMAGE_DIR}left_rgb/${TRIAL_NAME}${FRAME_NAME}_n_mask.png"

            if [ -e "$THREAD_MASK_FILE" ]; then
                echo "Thread mask exists: $THREAD_MASK_FILE"
                MASK_ARGS="--thread_mask_file $THREAD_MASK_FILE "
            else
                echo "Thread mask not found. Setting thread mask to none."
            fi

            if [ -e "$NEEDLE_MASK_FILE" ]; then
                echo "Needle mask exists: $NEEDLE_MASK_FILE"
                MASK_ARGS+="--needle_mask_file $NEEDLE_MASK_FILE"

            else
                echo "Needle mask not found. Setting needle mask to none."
            fi


            python alignment_run.py \
                --npy_file "$NPY_FILE" \
                --png_file "$PNG_FILE" \
                --thread "$THREAD_FILE" \
                --thread_specs_file "$THREAD_SPECS_FILE" \
                $MASK_ARGS
                # --thread_mask_file "$THREAD_MASK_FILE" \
                # --needle_mask_file "$NEEDLE_MASK_FILE" \
                # --needle_pos "$NEEDLE_POS_FILE" \
            ;;

        4)
            echo "Exiting. Goodbye!"
            break
            ;;

        *)
            echo "Invalid choice. Please select 1, 2, or 3."
            ;;
    esac

    echo ""  # blank line for readability
done
