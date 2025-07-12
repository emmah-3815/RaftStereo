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
from scipy.spatial.transform import Rotation as R
import pdb

def spinNode(node): 
    rclpy.spin(node)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--psm_id', type=int, default=2, 
            choices=[1, 2], 
            help='ID of the PSM that holds the object')
    args = parser.parse_args()

    rclpy.init(args = None)

    # from Lucas's calibration results
    pose_cam_base = np.load('data/cam_pose_psm2.npz')
    H_cam_base = dvrk_utils.posquat2H(
        pos = pose_cam_base['pos'], 
        quat = pose_cam_base['quat'],  # wxyz
    )

    # Read ROS messages

    ros_dvrk = ROSdVRK(
        control = True, 
        needle = True, 
        img = True, 
        mono_img = True, 
        aruco_marker= True,
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



    # get the object pose

    ros_msg = ros_dvrk.getSyncMsg()
    pose_cam_object = ros_msg['pose_cam_aruco'] # (qw, qx, qy, qz, x, y, z)

    H_cam_object  = dvrk_utils.posquat2H(
        pos = pose_cam_object [-3:], 
        quat = pose_cam_object [:4], 
    )

    # H_cam_object  = dvrk_utils.getHCamGoal(
    #     H_cam_needle = H_cam_object, 
    #     needle_r = 0.01146, 
    #     pick = True, 
    # )


    # set the object's pos to be above its actual pos
    if H_cam_object[2,3] > 0: 
        H_cam_object[:3,3] += 0.03 * H_cam_object[:3,2]
    else: 
        H_cam_object[:3,3] -= 0.03 * H_cam_object[:3,2]
    


    # TODO: depends on the grasping configuration!
    # (ee, object): (x, x), (y, z)



    H_object_ee = np.array([
        [1., 0.,  0., 0.], 
        [0., 0., -1., 0.], 
        [0., 1.,  0., 0.],       # rotate 90 degrees around x axis
        [0., 0.,  0., 1.], 
    ])

    H_object_ee_a = np.array([
    [-1.,  0.,  0., 0.],
    [0., 1.,  0., 0.],                  # align with the object's z axis
    [0.,  0., -1., 0.],  
    [0.,  0.,  0., 1.],
    ])

    H_object_ee = np.matmul(H_object_ee_a,  H_object_ee )
    # R_desired = R.from_euler('y', -90, degrees=True).as_matrix()
    # H_object_ee = np.eye(4)
    # H_object_ee[:3, :3] = R_desired



    goal_H_cam_ee = np.matmul(H_cam_object, H_object_ee)
    goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee)
    goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])

    # move the ee to be above the object
    psm_control.controlPoseReeInCam(
        psm_id = args.psm_id, 
        goal_pose_cam_ree = goal_pose_cam_ee, 
        H_cam_base = H_cam_base, 
    )

    # move the ee downward to pick up the object

    pdb.set_trace()
    ros_msg = ros_dvrk.getSyncMsg()
    pose_base_ee = ros_msg['pose_base_ee{}'.format(args.psm_id)] # (qw, qx, qy, qz, x, y, z)
    goal_H_base_ee = dvrk_utils.posquat2H(
        pos = pose_base_ee[-3:], 
        quat = pose_base_ee[:4], 
    )

    goal_H_cam_ee = np.matmul(H_cam_base, goal_H_base_ee)
    # move 2cm along the ee's y axis

    goal_H_cam_ee[:3,3] += 0.040 * goal_H_cam_ee[:3,1]
    goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee)
    goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])

    psm_control.controlPoseReeInCam(
        psm_id = args.psm_id, 
        goal_pose_cam_ree = goal_pose_cam_ee, 
        H_cam_base = H_cam_base, 
    )


    # close the gripper
    pdb.set_trace()
    psm_control.closeGripper(args.psm_id, True)

    # move the ee upward to lift the object

    #pdb.set_trace()
    ros_msg = ros_dvrk.getSyncMsg()
    pose_base_ee = ros_msg['pose_base_ee{}'.format(args.psm_id)] # (qw, qx, qy, qz, x, y, z)
    goal_H_base_ee = dvrk_utils.posquat2H(
        pos = pose_base_ee[-3:], 
        quat = pose_base_ee[:4], 
    )
    goal_H_cam_ee = np.matmul(H_cam_base, goal_H_base_ee)
    # move -5cm along the ee's y axis
    goal_H_cam_ee[:3,3] += -0.05 * goal_H_cam_ee[:3,1]
    goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee)
    goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])

    psm_control.controlPoseReeInCam(
        psm_id = args.psm_id, 
        goal_pose_cam_ree = goal_pose_cam_ee, 
        H_cam_base = H_cam_base, 
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
