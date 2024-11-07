from alignment_constraints import ReconstructAlign
import argparse
import numpy as np
import pdb


Constraints = ReconstructAlign()

# camera parameters
fx, fy, cx1, cy = 882.996114514, 882.996114514, 445.06146749, 190.24049547
cx2 = 445.061467
baseline = 5.8513759749420302 # mm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--npy_file', help="path_to_npy_file", default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_RAFT_output_1/frame_000001.npy")
    parser.add_argument('--png_file', help="path_to_png_file_for_visualization", default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_left_rgb/frame_000001.png")
    parser.add_argument('--meat_mask_file', help="path_to_meat_mask", default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png")
    parser.add_argument('--thread', help='path to thread array data', default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')
    parser.add_argument('--mask_erode', help='choose to erode mask for less chance of flying points', default=True)

    args = parser.parse_args()

    Constraints.init_camera_params([fx, fy, cx1, cy], cx2, baseline)
    Constraints.init_object_params(args.mask_erode)
    Constraints.add_meat(args.npy_file, args.png_file, args.meat_mask_file)
    Constraints.add_thread(args.thread)
    Constraints.meat, Constraints.spheres_one = Constraints.KNN_play(Constraints.meat, Constraints.thread)
    meat_neighborhoods, thread_points = Constraints.KNN_neighborhoods(Constraints.meat, Constraints.thread)
    # pdb.set_trace()

    dis = Constraints.norm_of_neighborhoods(meat_neighborhoods, thread_points)
    print("distance between meat and thread nodes", dis)


    objects = [Constraints.spheres_one]
    Constraints.visualize_objects(objects)

    Constraints.thread_trans = Constraints.align_objects(Constraints.meat, Constraints.thread, Constraints.meat_bound.center, Constraints.thread_bound.center)
    Constraints.meat, Constraints.spheres_two = Constraints.KNN_play(Constraints.meat, Constraints.thread_trans)

    meat_neighborhoods, thread_points = Constraints.KNN_neighborhoods(Constraints.meat, Constraints.thread_trans)
    # pdb.set_trace()

    dis_trans = Constraints.norm_of_neighborhoods(meat_neighborhoods, thread_points)
    print("distance between meat and thread nodes after trans", dis_trans)

    print("dis norm: ", np.linalg.norm(dis))
    print("dis_trans norm:",np.linalg.norm(dis_trans))
    
    objects = [Constraints.spheres_one, Constraints.spheres_two, Constraints.thread_trans]
    Constraints.visualize_objects(objects)





