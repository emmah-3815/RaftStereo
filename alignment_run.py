from alignment_constraints import ReconstructAlign
import argparse
import os

Constraints = ReconstructAlign()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--npy_file', help="path_to_npy_file") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_RAFT_output_1/frame_000001.npy")
    parser.add_argument('--png_file', help="path_to_png_file_for_visualization") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_left_rgb/frame_000001.png")
    parser.add_argument('--meat_mask_file', help="path_to_meat_mask") # , default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png")
    parser.add_argument('--use_default_meat_mask', help="if no meat mask is provided, use the default?", action='store_true', default=False)
    parser.add_argument('--thread', help='path to thread array data') # , default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')
    parser.add_argument('--mask_erode', help='choose to erode mask for less chance of flying points', default=True)
    parser.add_argument('--rect_img', help="non-rectified images are 1080 by 1920, rectified are 480 by 640", default=True)
    parser.add_argument('--downloads', help="use downloaded file", action='store_true')
    parser.add_argument('--calib', help="camera calibration yaml file", default=os.path.dirname(__file__) + "/assets/camera_calibration_fei.yaml")
    parser.add_argument('--needle', help='path to needle obj file') # , default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')


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
        Constraints.add_meat(npy_file, png_file , mask_file)
        Constraints.add_thread(thread_file)
        Constraints.add_needle(needle_file)
        Constraints.add_sudo_origin()
    else:
        npy_file = args.npy_file if args.npy_file is not None else "/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_RAFT_output_1/frame_000001.npy"
        png_file = args.png_file if args.png_file is not None else"/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_left_rgb/frame_000001.png"
        mask_file = args.meat_mask_file if args.meat_mask_file is not None else None
        mask_file = "/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png" \
                    if args.use_default_meat_mask and mask_file == None else None
        thread_file = args.thread if args.thread is not None else "/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy"
        needle_file = args.needle if args.needle is not None else "/media/emmah/PortableSSD/Arclab_data/Needle_R_01146.obj"
        # pdb.set_trace()
        Constraints.add_meat(npy_file, png_file , mask_file)
        Constraints.add_thread(thread_file)
        Constraints.add_needle(needle_file)
        Constraints.add_sudo_origin()


    #mark the origin with a sphere
    origin = Constraints.create_spheres_at_points([[0, 0, 0]])

    # move needle to the correct position
    needle_pos = 9.85004, -27.545028, 299.91204, \
                 0.14580808, 0.368426940, -0.54304734, 0.737861926
    Constraints.needle_align(needle_pos, quat=True)

    Constraints.meat, Constraints.spheres_one = Constraints.KNN_play(Constraints.meat, Constraints.thread)
    meat_neighborhoods, _, thread_points = Constraints.KNN_neighborhoods(Constraints.meat, Constraints.thread)
    # pdb.set_trace()

    # distance between thread and meat nodes
    dis = Constraints.norm_of_neighborhoods(meat_neighborhoods, thread_points)
    # print("distance between meat and thread nodes", dis)
    
    change = [0, 0, 0, 0, 0, 0]
    print(f"original distance is {Constraints.thread_transformation_dis(change, Constraints.meat, Constraints.thread)}")
    objects = [Constraints.spheres_one, origin]
    print("thread normal calcs original")
    Constraints.thread_normal_calcs(change, Constraints.meat, Constraints.thread)

    Constraints.visualize_objects(objects)

    # Constraints.thread_trans = Constraints.align_objects(Constraints.meat, Constraints.thread, Constraints.meat_bound.center, Constraints.thread_bound.center)
    # Constraints.meat, Constraints.spheres_two = Constraints.KNN_play(Constraints.meat, Constraints.thread_trans)

    # meat_neighborhoods, _, thread_points = Constraints.KNN_neighborhoods(Constraints.meat, Constraints.thread_trans)
    # pdb.set_trace()

    # slsqp optimization
    change = Constraints.slsqp_solver(Constraints.meat, Constraints.thread)
    Constraints.thread_trans = Constraints.thread_transform(change, Constraints.meat, Constraints.thread)
    print(f"after moving {change}, distance is {Constraints.thread_transformation_dis(change, Constraints.meat, Constraints.thread)}")
    Constraints.meat, Constraints.spheres_two = Constraints.KNN_play(Constraints.meat, Constraints.thread_trans)
    print("thread normal calcs after trans")
    Constraints.thread_normal_calcs(change, Constraints.meat, Constraints.thread_trans)
    
    # depth alignment
    change = Constraints.depth_solver(Constraints.meat, Constraints.thread)



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
        objects = [Constraints.spheres_one, Constraints.spheres_two, Constraints.thread_trans, Constraints.thread_old]
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
    
    objects = [Constraints.spheres_one, Constraints.spheres_two, Constraints.thread_trans]
    Constraints.visualize_objects(objects)




# sample commands
'''
python alignment_run.py --npy_file /media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/trial_21/npy/frame_000000.npy \
    --png_file /media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/trial_21/left_rgb/frame_000000.png \
        --thread /media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/thread_meat_3_21_collected/trial_21_spline.npy

threads:
trial 20 doesn't have great thread reconstruction
trial 30 doesn't have a good depth reconstruction
'''