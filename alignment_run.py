from alignment_constraints import ReconstructAlign
import argparse
import numpy as np
import pdb


Constraints = ReconstructAlign()

def camera_params(rect=True):
    if rect==True:
        # camera parameters on rectified images
        fx, fy, cx1, cy = 882.996114514, 882.996114514, 445.06146749, 190.24049547
        cx2 = 445.061467
        baseline = 5.8513759749420302 # mm

    else:
        # camera parameters on non-rectified images
        fx, fy, cx1, cy = 1.6796e+03, 1.6681e+03, 839.1909, 496.6793
        cx2 = 1.0265e+03
        baseline = 6.6411 # mm
    return fx, fy, cx1, cy, cx2, baseline



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--npy_file', help="path_to_npy_file", default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_RAFT_output_1/frame_000001.npy")
    parser.add_argument('--png_file', help="path_to_png_file_for_visualization", default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_left_rgb/frame_000001.png")
    parser.add_argument('--meat_mask_file', help="path_to_meat_mask", default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png")
    parser.add_argument('--thread', help='path to thread array data', default='/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy')
    parser.add_argument('--mask_erode', help='choose to erode mask for less chance of flying points', default=True)
    parser.add_argument('--rect_img', help="non-rectified images are 1080 by 1920, rectified are 480 by 640", default=True)


    args = parser.parse_args()

    Constraints.init_camera_params(camera_params(rect=args.rect_img))
    Constraints.init_object_params(args.mask_erode)
    Constraints.add_meat(args.npy_file, args.png_file , args.meat_mask_file)
    Constraints.add_thread(args.thread)
    Constraints.meat, Constraints.spheres_one = Constraints.KNN_play(Constraints.meat, Constraints.thread)
    meat_neighborhoods, _, thread_points = Constraints.KNN_neighborhoods(Constraints.meat, Constraints.thread)
    # pdb.set_trace()

    # dis = Constraints.norm_of_neighborhoods(meat_neighborhoods, thread_points)
    # print("distance between meat and thread nodes", dis)
    
    change = [0, 0, 0, 0, 0, 0]
    print(f"original distance is {Constraints.thread_transformation_dis(change, Constraints.meat, Constraints.thread)}")
    objects = [Constraints.spheres_one]
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




