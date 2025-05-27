from alignment_constraints import ReconstructAlign
import argparse
import os

Constraints = ReconstructAlign()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_file', help="path_to_npy_file") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_RAFT_output_1/frame_000001.npy")
    parser.add_argument('--png_file', help="path_to_png_file_for_visualization") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_left_rgb/frame_000001.png")
    parser.add_argument('--meat_mask_file', help="path_to_meat_mask") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png")
    parser.add_argument('--use_default_meat_mask', help="if no meat mask is provided, use the default?", action='store_true', default=False)
    parser.add_argument('--thread', help='path to thread array data') # , default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')
    parser.add_argument('--mask_erode', help='choose to erode mask for less chance of flying points', default=True)
    parser.add_argument('--calib', help="camera calibration yaml file", default=os.path.dirname(__file__) + "/assets/camera_calibration_fei.yaml")
    parser.add_argument('--needle', help='path to needle obj file') # , default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')
    parser.add_argument('--needle_pos', help='path to needle pos pkl file') # , default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')


    args = parser.parse_args()

    Constraints.init_camera_params(args.calib)
    Constraints.init_object_params(args.mask_erode)

    npy_file = args.npy_file
    png_file = args.png_file
    mask_file = args.meat_mask_file if args.meat_mask_file is not None else None
    thread_file = args.thread
    needle_file = args.needle
    needle_pos_file = args.needle_pos

    Constraints.add_meat(npy_file, png_file, mask_file)
    Constraints.add_thread(thread_file)
    Constraints.add_needle(needle_file, needle_r=8.2761)
    Constraints.load_needle_pos(needle_pos_file)
    Constraints.add_origin()

    # move needle to the recorded position
    Constraints.needle, Constraints.needle_bound = Constraints.transform(Constraints.needle_pos, Constraints.needle, Constraints.needle_bound, quat=True)


    Constraints.meat, Constraints.thread_hl = Constraints.KNN_play(Constraints.meat, Constraints.thread, neighbors=10)
    meat_neighborhoods, _, thread_points = Constraints.KNN_neighborhoods(Constraints.meat, Constraints.thread)

    
    # add additional objects and visualize
    objects = [Constraints.thread_hl]
    Constraints.visualize_objects(objects)

    change = [0, 0, 0, 0, 0, 0]
    Constraints.meat,Constraints.meat_bound = Constraints.transform(change, Constraints.meat, Constraints.meat_bound)

    # rerun thread highlights incase thread was moved
    Constraints.meat, Constraints.thread_hl = Constraints.KNN_play(Constraints.meat, Constraints.thread, neighbors=10)

    # add additional objects and visualize
    objects = [Constraints.thread_hl]
    Constraints.visualize_objects(objects)


    # # connect thread and needle
    # Constraints.needle, Constraints.needle_bound = Constraints.needle_thread_conn(Constraints.needle, Constraints.needle_bound, Constraints.thread, Constraints.thread_bound)
  
    # # visualize three
    # Constraints.visualize_objects(objects)


# sample commands
'''
python alignment_run.py --npy_file /media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/trial_21/npy/frame_000000.npy \
    --png_file /media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/trial_21/left_rgb/frame_000000.png \
        --thread /media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/thread_meat_3_21_collected/trial_21_spline.npy

threads:
trial 20 doesn't have great thread reconstruction
trial 30 doesn't have a good depth reconstruction

trial 22 has the right shape, but the height is off, so the optimal selection point is going to be wrong
trial 23 is pretty accurate, could use to select best point
trial 24 has an accurate good grasping point, but the thread-needle end points downwards (wrong)
trial 25 good reconstruction in half of the thread, but the thread-needle end points downwards(wrong)
trial 26 has same issue as 
'''