import os
import time
import copy
import argparse
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy

from read_dvrk_msg.ros_dvrk import ROSdVRK
from psm_control.psm_control import PsmControl

import utils.dvrk_utils as dvrk_utils

import pdb

def spinNode(node): 
    rclpy.spin(node)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--psm_id', type=int, default=2, 
            choices=[1, 2], 
            help='ID of the PSM that holds the needle')
    args = parser.parse_args()

    rclpy.init(args = None)

    # from Lucas's calibration results
    pose_cam_base = np.load('data/cam_pose_psm2.npz')
    H_cam_base = dvrk_utils.posquat2H(
        pos = pose_cam_base['pos'], 
        quat = pose_cam_base['quat'],  # wxyz
    )

    # from gravity_calibration results:
    pose_cam_world = np.load('assets/cam_pose_world.npz')
    H_cam_world = dvrk_utils.posquat2H(
        pos =   pose_cam_world['pos'], 
        quat =  pose_cam_world['quat'],  # wxyz
    )

    # Read ROS messages

    ros_dvrk = ROSdVRK(
        control = True, 
        needle = True, 
        img = True, 
        mono_img = True, 
    )

    # create a thread for spinning
    spin_thread = threading.Thread(
        target = spinNode, 
        args = (ros_dvrk,), 
    )
    spin_thread.start()
    time.sleep(1.)

    # control PSM
    psm_control = PsmControl(ros_dvrk)
    psm_control.openGripper(args.psm_id)


    # get initial pose of the ee:
    # Record the initial pose in the base frame
    initial_ros_msg = ros_dvrk.getSyncMsg()
    initial_pose_base = initial_ros_msg['pose_base_ee{}'.format(args.psm_id)]  # (qw, qx, qy, qz, x, y, z)

    # Convert to a homogeneous transformation (base frame)
    initial_H_base_ee = dvrk_utils.posquat2H(
        pos=initial_pose_base[-3:],
        quat=initial_pose_base[:4]
    )

    # Convert the initial pose to the camera frame using H_cam_base
    initial_H_cam_ee = np.matmul(H_cam_base, initial_H_base_ee)
    initial_pos_cam_ee, initial_quat_cam_ee = dvrk_utils.matrix2PosQuat(initial_H_cam_ee)
    initial_pose_cam_ee = np.concatenate([initial_quat_cam_ee, initial_pos_cam_ee])



    # get the needle pose

    ros_msg = ros_dvrk.getSyncMsg()
    pose_cam_needle = ros_msg['pose_cam_needle'] # (qw, qx, qy, qz, x, y, z)

    H_cam_needle = dvrk_utils.posquat2H(
        pos = pose_cam_needle[-3:], 
        quat = pose_cam_needle[:4], 
    )
    H_cam_needle = dvrk_utils.getHCamGoal(
        H_cam_needle = H_cam_needle, 
        needle_r = 0.01146, 
        pick = True, 
    )

    # set the object's pos to be above its actual pos
    if H_cam_needle[2,3] > 0: 
        H_cam_needle[:3,3] -= 0.01 * H_cam_needle[:3,2]
    else: 
        H_cam_needle[:3,3] += 0.01 * H_cam_needle[:3,2]

    # TODO: depends on the grasping configuration!
    # (ee, needle): (x, x), (y, z)
    H_needle_ee = np.array([
        [1., 0.,  0., 0.], 
        [0., 0., -1., 0.], 
        [0., 1.,  0., 0.],
        [0., 0.,  0., 1.], 
    ])

    goal_H_cam_ee = np.matmul(H_cam_needle, H_needle_ee)
    goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee)
    goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])

    # move the ee to be above the needle
    psm_control.controlPoseReeInCam(
        psm_id = args.psm_id, 
        goal_pose_cam_ree = goal_pose_cam_ee, 
        H_cam_base = H_cam_base, 
    )

    # move the ee downward to pick up the needle




    pdb.set_trace()
    # ros_msg = ros_dvrk.getSyncMsg()
    # pose_base_ee = ros_msg['pose_base_ee{}'.format(args.psm_id)] # (qw, qx, qy, qz, x, y, z)
    # goal_H_base_ee = dvrk_utils.posquat2H(
    #     pos = pose_base_ee[-3:], 
    #     quat = pose_base_ee[:4], 
    # )

    # goal_H_cam_ee = np.matmul(H_cam_base, goal_H_base_ee)
    # # move 2cm along the ee's y axis

    # goal_H_cam_ee[:3,3] += 0.02 * goal_H_cam_ee[:3,1]
    # goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee)
    # goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])

    # psm_control.controlPoseReeInCam(
    #     psm_id = args.psm_id, 
    #     goal_pose_cam_ree = goal_pose_cam_ee, 
    #     H_cam_base = H_cam_base, 
    # )

    '''optionally try moving directly down to pick up the needle'''
    # Get the current end‐effector pose (in the base frame) and convert it to the camera frame.
    ros_msg = ros_dvrk.getSyncMsg()
    p_ee = ros_msg['pose_cam_ee{}'.format(args.psm_id)][-3:]  # x, y, z
   
    print(f"Current end-effector position in camera frame: {p_ee}")
    # Get the needle position in the camera frame.
    # (H_cam_needle was computed earlier)
    p_needle = H_cam_needle[:3, 3]
    print(f"Needle position in camera frame: {p_needle}")
    # Extract the gravity (world z) direction in the camera frame.
    # H_cam_world expresses the world (gravity) frame relative to the camera.
    R_cam_world = H_cam_world[:3, :3]
    gravity_direction_in_cam = R_cam_world[:, 2]  # the world z-axis

    # Compute the distance between the needle and EE along the gravity direction.
    distance_along_gravity = 250 * np.linalg.norm(p_needle - p_ee)

    print(f"Distance along gravity: {distance_along_gravity:.4f} m")
    # Now, move the end-effector by a desired displacement along this gravity direction.
    # For example, moving 0.005 m (1.5 cm) toward the needle along gravity:
    goal_H_cam_ee[:3, 3] -= distance_along_gravity * gravity_direction_in_cam

    # Convert back to pose representation and command the robot.
    goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee)
    goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])

    psm_control.controlPoseReeInCam(
        psm_id=args.psm_id, 
        goal_pose_cam_ree=goal_pose_cam_ee, 
        H_cam_base=H_cam_base, 
    )





    # close the gripper
    pdb.set_trace()
    psm_control.closeGripper(args.psm_id, True)

    # move the ee upward to lift the needle

    #pdb.set_trace()
    # ros_msg = ros_dvrk.getSyncMsg()
    # pose_base_ee = ros_msg['pose_base_ee{}'.format(args.psm_id)] # (qw, qx, qy, qz, x, y, z)
    # goal_H_base_ee = dvrk_utils.posquat2H(
    #     pos = pose_base_ee[-3:], 
    #     quat = pose_base_ee[:4], 
    # )
    # goal_H_cam_ee = np.matmul(H_cam_base, goal_H_base_ee)
    # # move -5cm along the ee's y axis
    # goal_H_cam_ee[:3,3] += -0.05 * goal_H_cam_ee[:3,1]
    # goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee)
    # goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])

    # psm_control.controlPoseReeInCam(
    #     psm_id = args.psm_id, 
    #     goal_pose_cam_ree = goal_pose_cam_ee, 
    #     H_cam_base = H_cam_base, 
    # )

    ros_msg = ros_dvrk.getSyncMsg()
    pose_base_ee = ros_msg['pose_base_ee{}'.format(args.psm_id)]  # (qw, qx, qy, qz, x, y, z)
    goal_H_base_ee = dvrk_utils.posquat2H(
        pos=pose_base_ee[-3:], 
        quat=pose_base_ee[:4],
    )
    goal_H_cam_ee = np.matmul(H_cam_base, goal_H_base_ee)

    # Compute the gravity direction in the camera frame.
    # H_cam_world expresses the world (gravity) frame relative to the camera.
    # Its rotation matrix's third column is the world z-axis (i.e. the gravity direction).
    R_cam_world = H_cam_world[:3, :3]
    gravity_direction_in_cam = R_cam_world[:, 2]  # Already normalized

    # Now, move the end-effector by a desired displacement along this gravity direction.
    # For example, moving 0.005 m (1.5 cm) toward the needle along gravity:
    goal_H_cam_ee[:3, 3] += 0.025 * gravity_direction_in_cam

    # Convert back to pose representation and command the robot.
    goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee)
    goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])

    psm_control.controlPoseReeInCam(
        psm_id=args.psm_id, 
        goal_pose_cam_ree=goal_pose_cam_ee, 
        H_cam_base=H_cam_base, 
    )



    # Command the robot to return to the initial pose in the camera frame
    pdb.set_trace()
    psm_control.controlPoseReeInCam(
    psm_id=args.psm_id,
    goal_pose_cam_ree=initial_pose_cam_ee,
    H_cam_base=H_cam_base,
    )

    pdb.set_trace()
    psm_control.openGripper(args.psm_id)


    spin_thread.join()
    ros_dvrk.destroy_node()
    rclpy.shutdown()
