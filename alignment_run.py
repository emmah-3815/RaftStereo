from alignment_constraints import ReconstructAlign
import argparse
import os
import pdb
from pathlib import Path

Constraints = ReconstructAlign()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--npy_file', help="path_to_npy_file") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_RAFT_output_1/frame_000001.npy")
    parser.add_argument('--png_file', help="path_to_png_file_for_visualization") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_left_rgb/frame_000001.png")
    parser.add_argument('--meat_mask_file', help="path_to_meat_mask") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png")
    parser.add_argument('--thread_mask_file', help="path_to_thread_mask") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png")
    parser.add_argument('--needle_mask_file', help="path_to_needle_mask") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png")
    parser.add_argument('--use_default_meat_mask', help="if no meat mask is provided, use the default?", action='store_true', default=False)
    parser.add_argument('--thread', help='path to thread array data') # , default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')
    parser.add_argument('--mask_erode', help='choose to erode mask for less chance of flying points', default=True)
    parser.add_argument('--rect_img', help="non-rectified images are 1080 by 1920, rectified are 480 by 640", default=True)
    parser.add_argument('--downloads', help="use downloaded file", action='store_true')
    parser.add_argument('--calib', help="camera calibration yaml file", default=os.path.dirname(__file__) + "/assets/camera_calibration_fei.yaml")
    parser.add_argument('--needle', help='path to needle obj file') # , default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')
    parser.add_argument('--needle_pos', help='path to needle pos pkl file') # , default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')
    parser.add_argument('--thread_specs_file', help='path to thread specs file') # , default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')
    parser.add_argument('--trial_number', type=int, help="trial number like 1-24")

    args = parser.parse_args()

    Constraints.init_camera_params(args.calib)
    Constraints.init_object_params(args.mask_erode)

    # use downloaded file  in the raft stereo directory without the external ssd
    if args.downloads:
        npy_file = "/home/emmah/ARClab/RAFT-Stereo/alignment_dataset/trial_9_frame_000001.npy"
        png_file = "/home/emmah/ARClab/RAFT-Stereo/alignment_dataset/trial_9_frame_000001.png"
        mask_file = "/home/emmah/ARClab/RAFT-Stereo/alignment_dataset/trial_9_mask_frame0001.png"
        thread_file = "/home/emmah/ARClab/RAFT-Stereo/alignment_dataset/thread_frame_000000.npy"
        needle_file = "/home/emmah/ARClab/RAFT-Stereo/assets/Needle_R_01146.obj"
        needle_pos_file = "/home/emmah/ARClab/RAFT-Stereo/alignment_dataset/trial_20_needle_pose.pkl"
        trial_number = 9
        Constraints.add_meat(npy_file, png_file, mask_file)
        Constraints.add_thread(thread_file)
        Constraints.add_needle(needle_file)
        Constraints.load_needle_pos(needle_pos_file)
        Constraints.add_origin()
    else:
        npy_file = args.npy_file if args.npy_file is not None else "/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_RAFT_output_1/frame_000001.npy"
        png_file = args.png_file if args.png_file is not None else"/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_left_rgb/frame_000001.png"
        mask_file = args.meat_mask_file if args.meat_mask_file is not None else None
        meat_mask_file = "/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png" \
                    if args.use_default_meat_mask and mask_file == None else None
        thread_mask_file = args.thread_mask_file if args.thread_mask_file is not None else None
        needle_mask_file = args.needle_mask_file if args.needle_mask_file is not None else None
     
        thread_file = args.thread if args.thread is not None else "/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy"
        needle_file = args.needle if args.needle is not None else "/media/emmah/PortableSSD/Arclab_data/Needle_R_01146.obj"
        needle_pos_file = args.needle_pos if args.needle_pos is not None else f"/home/emmah/ARClab/RAFT_Stereo/alignment_dataset/trial_20_needle_pose.pkl"

        # reliability
        thread_specs_file = args.thread_specs_file if args.thread_specs_file is not None else None

        trial_number = args.trial_number

        # pdb.set_trace()
        Constraints.add_meat(npy_file, png_file, meat_mask_file, thread_mask_file, needle_mask_file)
        Constraints.add_thread(thread_file)
        Constraints.add_needle(needle_file, needle_r=8.2761)
        Constraints.load_needle_pos(needle_pos_file)
        Constraints.load_thread_specs(thread_specs_file)
        Constraints.add_origin()
        Constraints.add_lower_bound_spline()
        Constraints.add_upper_bound_spline()

    # mark the origin with a sphere
    # origin = Constraints.create_spheres_at_points([[0, 0, 0]])

    # move needle to the recorded position
    Constraints.needle, Constraints.needle_bound = Constraints.transform(Constraints.needle_pos, Constraints.needle, Constraints.needle_bound, quat=True)


    Constraints.meat, Constraints.thread_hl = Constraints.KNN_play(Constraints.meat, Constraints.thread, neighbors=10)
    meat_neighborhoods, _, thread_points = Constraints.KNN_neighborhoods(Constraints.meat, Constraints.thread, 10)
    # pdb.set_trace()

    # distance between thread and meat nodes
    dis = Constraints.norm_of_neighborhoods(meat_neighborhoods, thread_points)
    # print("distance between meat and thread nodes", dis)
    
    change = [0, 0, 0, 0, 0, 0]
    print(f"original distance is {Constraints.thread_transformation_dis(change, Constraints.meat, Constraints.thread, Constraints.meat_bound, Constraints.thread_bound)}")
    # objects = [Constraints.thread_hl, origin]
    objects = [Constraints.thread_hl, Constraints.lower_bound_3d, Constraints.upper_bound_3d]
    # objects = [Constraints.lower_bound_3d]
    print("thread normal calcs original")
    Constraints.thread_normal_calcs(change, Constraints.meat, Constraints.thread)

    # visualize one
    Constraints.visualize_objects(objects)

    # check if first item of thread is at needle
    Constraints.flip_thread(thread_file, thread_specs_file)
    # Constraints.thread, Constraints.thread_bound = Constraints.thread_meat_orient(Constraints.meat, Constraints.thread, Constraints.meat_bound, Constraints.thread_bound)

    # depth alignment (runs twice to get closer to the meat)
    change = Constraints.depth_solver(Constraints.meat, Constraints.thread)
    Constraints.meat,Constraints.meat_bound = Constraints.transform(change, Constraints.meat, Constraints.meat_bound)
    change = Constraints.depth_solver(Constraints.meat, Constraints.thread)
    Constraints.meat, Constraints.meat_bound = Constraints.transform(change, Constraints.meat, Constraints.meat_bound)

    # redraw spheres and neighbors
    Constraints.meat, Constraints.thread_hl = Constraints.KNN_play(Constraints.meat, Constraints.thread)

    Constraints.rely_spheres = Constraints.paint_reliability(Constraints.thread)
    grasp_points, Constraints.grasp_spheres = Constraints.grasp(Constraints.meat, Constraints.thread)
    goal_H_cams = Constraints.goal_H_cam_gen(grasp_points)
    # Constraints.thread_trans = Constraints.align_objects(Constraints.meat, Constraints.thread, Constraints.meat_bound.center, Constraints.thread_bound.center)
    # Constraints.meat, Constraints.spheres_two = Constraints.KNN_play(Constraints.meat, Constraints.thread_trans)

    # meat_neighborhoods, _, thread_points = Constraints.KNN_neighborhoods(Constraints.meat, Constraints.thread_trans)
    # pdb.set_trace()

    # slsqp optimization
    '''
    change = Constraints.slsqp_solver(Constraints.meat, Constraints.thread)
    Constraints.thread_trans = Constraints.thread_transform(change, Constraints.meat, Constraints.thread)
    print(f"after moving {change}, distance is {Constraints.thread_transformation_dis(change, Constraints.meat, Constraints.thread)}")
    Constraints.meat, Constraints.spheres_two = Constraints.KNN_play(Constraints.meat, Constraints.thread_trans)
    print("thread normal calcs after trans")
    Constraints.thread_normal_calcs(change, Constraints.meat, Constraints.thread_trans)
    '''





    # manual translation and rotation
    '''
    user = None
    Constraints.thread_old = Constraints.thread
    while user != 's':
        user = input("input change separated by spaces ").split()
        if len(user) != 6:
            continue
        change = np.array(user)
        Constraints.thread_trans = Constraints.thread_transform(change, Constraints.meat, Constraints.thread)
        print(f"after moving {change}, distance is {Constraints.thread_transformation_dis(change, Constraints.meat, Constraints.thread)}")
        print(f"after moving {change}, normals is {Constraints.thread_normal_calcs(change, Constraints.meat, Constraints.thread)}")
        Constraints.meat, Constraints.spheres_two = Constraints.KNN_play(Constraints.meat, Constraints.thread_trans)
        objects = [Constraints.thread_hl, Constraints.spheres_two, Constraints.thread_trans, Constraints.thread_old]
        Constraints.visualize_objects(objects)
        Constraints.thread_old = Constraints.thread_trans
    '''

    # transform test + distance calc test
    # change = [x, y, z, rz, ry, rz] confirmed moves the thread approprately 
    # and calculates thread node to neighbor avg distance
    # change = [-10.43597845298734, 4.913627692600272, -2.169965977054915, 0, 0, 0]
    # change = [-30, 14, -11, 0, 0, 0]
    # Constraints.thread_trans = Constraints.thread_transform(change, Constraints.meat, Constraints.thread)
    # print(f"after moving {change}, distance is {Constraints.thread_transformation_dis(change, Constraints.meat, Constraints.thread)}")
    # Constraints.meat, Constraints.spheres_two = Constraints.KNN_play(Constraints.meat, Constraints.thread_trans)


    # calculate distance of meat and thread
    # dis_trans = Constraints.norm_of_neighborhoods(meat_neighborhoods, thread_points)
    # print("distance between meat and thread nodes after trans", dis_trans)

    # print("dis norm: ", np.linalg.norm(dis))
    # print("dis_trans norm:",np.linalg.norm(dis_trans))
    
    # visualize two
    # objects = [Constraints.thread_hl, Constraints.spheres_two, Constraints.thread_trans]
    # objects = [Constraints.thread_hl]
    objects = [Constraints.rely_spheres, Constraints.grasp_spheres, Constraints.upper_bound_3d]
    Constraints.visualize_objects(objects)

    save_grasp = input("save graspings points? y ")
    if save_grasp == 'y':
        save_path = Path(thread_specs_file).parent
        Constraints.save_with_date(grasp_points, trial_number, save_path)

    Constraints.needle, Constraints.needle_bound = Constraints.needle_thread_conn(Constraints.needle, Constraints.needle_bound, Constraints.thread, Constraints.thread_bound)

    # visualize three
    Constraints.visualize_objects(objects)


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
\
'''