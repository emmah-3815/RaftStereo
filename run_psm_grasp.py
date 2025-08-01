import os
import time
import copy
import argparse
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy

# import sys

# Print the list of paths in sys.path
# for path in sys.path:
#     print(path)

from read_dvrk_msg.ros_dvrk import ROSdVRK
from psm_control.psm_control import PsmControl

from psm_control.psm_control import utils as dvrk_utils

import pdb
import thread_grasp_constraints as grasp_constr

def spinNode(node): 
    rclpy.spin(node)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--psm_id', type=int, default=2, 
            choices=[1, 2], 
            help='ID of the PSM that holds the needle')
    parser.add_argument('-g', '--grasp_thread', type=int, default=1, 
            choices=[0, 1], 
            help='grasp needle or confirm robot pose in camera frame')
    parser.add_argument('-s', "--straighten", action='store_true',)
    args = parser.parse_args()

    print(f"psm id: {args.psm_id}")
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
        needle_tracking = True,
        joints = True, 
    ) # just subscribe to all of them

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

    # straighten end effector 
    if args.straighten:
        print("Straightening the end effector...")
        # get the current joint angles of the PSM
        joints = ros_dvrk.getSyncMsg()['psm{}_joints'.format(args.psm_id)]
        straighten_joints = copy.deepcopy(joints)
        straighten_joints[-3:] = 0  # set the last three joints to 0
        psm_control.controlArmByJoints(arm_id=args.psm_id, goal_joints=straighten_joints)

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
    print(f"Initial end-effector{args.psm_id} pose matrix in camera frame: \n{initial_H_cam_ee.tolist()}")
    initial_pos_cam_ee, initial_quat_cam_ee = dvrk_utils.matrix2PosQuat(initial_H_cam_ee)
    initial_pose_cam_ee = np.concatenate([initial_quat_cam_ee, initial_pos_cam_ee])



    if args.grasp_thread==True:
        print("Thread grasping mode enabled.")
        # get the thread grasping pose
        while True:
            grasp_input = input("input the grasp point transformation matrix: \n")
            goal_H_cam_tip = np.array(eval(grasp_input))
            if goal_H_cam_tip.shape == (4, 4):
                break
            else:
                print("Please input a valid 4x4 transformation matrix.")


        H_tip_offset_ee = np.eye(4)
        tip_offset  = -0.005
        # accoutn for the offset of the grasp point (tip) from the end effector (ee) in the camera frame
        H_tip_offset_ee = np.array([
            [1., 0.,  0., 0.], 
            [0., 1.,  0., tip_offset], 
            [0., 0.,  1., 0.],
            [0., 0.,  0., 1.], 
        ])

        goal_H_cam_ee = np.matmul(H_tip_offset_ee, goal_H_cam_tip)  # rotate 90 degrees around x axis and move the tip offset
        goal_H_cam_ee = grasp_constr.orient_goal(initial_H_cam_ee, goal_H_cam_ee)        

        approach_H_cam_ee = copy.copy(goal_H_cam_ee)
        approach_len = 0.015
        approach_H_cam_ee[:3, 3] -= approach_H_cam_ee[:3, 1] * approach_len # based on Neelay's execute_grasp


        print(f"Goal end-effector{args.psm_id} pose matrix in camera frame: \n{goal_H_cam_ee.tolist()}")

        # Check current state of robot and grasp goal
        ros_msg = ros_dvrk.getSyncMsg()
        pose_cam_ee = ros_msg['pose_cam_ee{}'.format(args.psm_id)]

        curr_H_cam_ee = dvrk_utils.posquat2H(
            pos=pose_cam_ee[-3:],
            quat=pose_cam_ee[:4]
        )

        print(f"Current end-effector{args.psm_id} pose matrix in camera frame: \n{curr_H_cam_ee.tolist()}")

        
        goal_H_cam_ee_traj = np.array([approach_H_cam_ee, goal_H_cam_ee])
    
        for goal_H_cam_ee_step in goal_H_cam_ee_traj:
            goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee_step)
            goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])

            # move the ee
            pdb.set_trace()
            psm_control.controlPoseReeInCam(
                psm_id = args.psm_id, 
                goal_pose_cam_ree = goal_pose_cam_ee, 
                H_cam_base = H_cam_base, 
            )

            # check current state of robot and step-wise grasp goal, should match
            ros_msg = ros_dvrk.getSyncMsg()
            p_ee = ros_msg['pose_cam_ee{}'.format(args.psm_id)][-3:]  # x, y, z
        
            print(f"Current end-effector position in camera frame: \n{p_ee}")
            print(f"Goal end-effector position in camera frame: \n{goal_pos_cam_ee}")

            # Check current state of robot and grasp goal
            ros_msg = ros_dvrk.getSyncMsg()
            pose_cam_ee = ros_msg['pose_cam_ee{}'.format(args.psm_id)]
            print(f"Goal end-effector{args.psm_id} pose matrix in camera frame: \n{goal_H_cam_ee.tolist()}")
            print(f"Current end-effector{args.psm_id} pose matrix in camera frame: \n{curr_H_cam_ee.tolist()}")


            # pause after each step
            pdb.set_trace()
            pause = input("Press Enter to continue to the next step...")

    if args.grasp_thread==False:
        print("Thread grasping mode disabled.")

        # Check current state of robot 
        ros_msg = ros_dvrk.getSyncMsg()
        pose_cam_ee = ros_msg['pose_cam_ee{}'.format(args.psm_id)] # rx, ry, rz, rw, x, y, z

        print(f"Current end-effector pos in camera frame: {pose_cam_ee.tolist()}")

        # get the thread grasping pose
        while True:
            pos_input = input(f"input psm{args.psm_id} goal pos in camera frame: \n")
            goal_pose_cam_ee = np.array(eval(pos_input))
            # pdb.set_trace()
            if goal_pose_cam_ee.shape == (7,):
                break
            else:
                print("Please input a valid 1x7 [quat, pos] array.")

        # Confirm input
        print(f"Thread Grasp Goal in camera frame: {goal_pose_cam_ee}")

        pdb.set_trace()
        # move the ee
        psm_control.controlPoseReeInCam(
            psm_id = args.psm_id, 
            goal_pose_cam_ree = goal_pose_cam_ee, 
            H_cam_base = H_cam_base, 
        )

        # check current state of robot and step-wise grasp goal, should match
        ros_msg = ros_dvrk.getSyncMsg()
        pose_ee = ros_msg['pose_cam_ee{}'.format(args.psm_id)] # quat, pos
    
        print(f"Current end-effector pose in camera frame: {pose_ee}")
        print(f"Goal end-effector pose in camera frame: {goal_pose_cam_ee}")


    # move the ee downward to pick up the needle

    pdb.set_trace()

    # close the gripper
    pdb.set_trace()
    psm_control.closeGripper(args.psm_id, True)

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