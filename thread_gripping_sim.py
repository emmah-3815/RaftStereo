import os
import time
import copy
import pickle
import pdb
import argparse
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt

import rclpy

def load_thread(thread_file_path:str, bounds_file_path:str):
    global thread_init, plot_init, gripper_init
    # self.thread = np.load(thread_file_path, allow_pickle=True)

    thread_data = np.load(thread_file_path, allow_pickle=True)

    with open(bounds_file_path, 'rb') as f:
        data = pickle.load(f)

    thread_reliability = data.get('reliability')
    thread_lower = data.get('lower_constr')
    thread_upper = data.get('upper_constr')

    thread_data = thread_data * 1  # * 1000 scale up to match meat point cloud size
    n = thread_data.shape[0]
    thread = thread_data
    thread_init = True
    return thread, thread_upper, thread_lower

def load_quiver(pos):
    # origin = pos[:3]
    # r = R.from_quat(pos[3:])
    # # xhat = np.array([1, 0, 0])
    # # yhat = np.array([0, 1, 0])
    # # zhat = np.array([0, 0, 1])
    # r_mat = r.as_matrix()

    # Sarah's version
    origin = pos[:3, 3]
    r_mat = pos[:3, :3]

    # pdb.set_trace()
    x = origin + r_mat[:, 0]
    y = origin + r_mat[:, 1]
    z = origin + r_mat[:, 2]

    # return origin, x, y, z
    return origin, r_mat[:, 0], r_mat[:, 1], r_mat[:, 2]

def load_goal(goal_file, thresh=10):
    goals = np.load(goal_file, allow_pickle=True) # array of points along the thread suitable for grasping
    goal_pos = []
    normal_base = []
    for i, _ in enumerate(goals[:-2]):
        origin = goals[i+1]
        v0 = origin - goals[i] 
        v1 = origin - goals[i+2]
        if np.linalg.norm(v0) > thresh or np.linalg.norm(v1) > thresh: # if distance between origin and previous point or next point is too large, continue
            continue

        # Sarah's version
        # pdb.set_trace()
        xhat = np.cross(v0, v1)
        xhat = xhat / np.linalg.norm(xhat)

        yhat = (v0 + v1)*0.5 #vector pointing from origin to mid point between adjacent points
        # yhat = yhat - np.dot(yhat, xhat) * xhat
        yhat = yhat / np.linalg.norm(yhat)
        
        zhat = np.cross(xhat, yhat)
        zhat = zhat / np.linalg.norm(zhat)

        # yhat = np.cross(zhat, xhat)

        goal_H_cam_tip = np.eye(4)

        goal_H_cam_tip[:3, 0] = xhat
        goal_H_cam_tip[:3, 1] = yhat
        goal_H_cam_tip[:3, 2] = zhat
        goal_H_cam_tip[:3, 3] = origin

        # R_mat = np.column_stack((xhat, yhat, zhat))
        # rot = R.from_matrix(R_mat)
        # quat = rot.as_quat()

        # goal_pos.append([*origin, *quat])
        goal_pos.append(goal_H_cam_tip)
        normal_base.append([v0, v1])

    goal_pos = np.asarray(goal_pos)
    # pdb.set_trace()
    return goal_pos



def gen_trajectory(start, goal, steps=50):
    start_pos = start[:3]
    start_r = R.from_quat(start[3:])

    # goal_pos = goal[:3]
    # goal_r = R.from_quat(goal[3:])

    # Sarah's version
    goal_pos = goal[:3, 3]
    goal_r = R.from_matrix(goal[:3, :3])

    combined = R.from_quat([start_r.as_quat(), goal_r.as_quat()])

    slerp = Slerp([0, 1], combined)
    rot_traj = slerp(np.linspace(0, 1, steps))

    pos_traj = np.linspace(start_pos, goal_pos, steps)

    trajectory = []
    # for pos, rot in zip(pos_traj, rot_traj):
    #     quat = rot.as_quat()
    #     traj = np.concatenate((pos, quat))
    #     trajectory.append(traj)

    for pos, rot in zip(pos_traj, rot_traj):
        traj = np.eye(4)
        traj[:3, :3] = rot.as_matrix()
        traj[:3, 3] = pos
        trajectory.append(traj)

    trajectory = np.asarray(trajectory)
    return trajectory #x y z, qw, qx, qy, qz or 4 by 4 matrix with xhat; yhat; zhat; origin

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def live_plot(thread, thread_upper, thread_lower, goal_pts, trajectory):
    # set up plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plt.ion()

    # normal base vectors


    # thread
    ax.plot(thread[:, 0], thread[:, 1], thread[:, 2])
    ax.scatter(goal_pts[:, 0], goal_pts[:, 1], goal_pts[:, 2])

    # bounds
    ax.plot(thread_upper[:, 0], thread_upper[:, 1], thread_upper[:, 2], c='turquoise')
    ax.plot(thread_lower[:, 0], thread_lower[:, 1], thread_lower[:, 2], c='turquoise')
    set_axes_equal(ax)

    # mark goal
    # origin, x, y, z = load_quiver(goal)
    # goal_x = ax.quiver(*origin, *x, length=10.0, normalize=True, color='darkred')
    # goal_y = ax.quiver(*origin, *y, length=10.0, normalize=True, color='darkgreen')
    # goal_z = ax.quiver(*origin, *z, length=10.0, normalize=True, color='darkblue')

    # mark start pos
    origin, x, y, z = load_quiver(trajectory[0])
    gripper_x = ax.quiver(*origin, *x, length=5.0, normalize=True, color='lightcoral')
    gripper_y = ax.quiver(*origin, *y, length=5.0, normalize=True, color='palegreen')
    gripper_z = ax.quiver(*origin, *z, length=5.0, normalize=True, color='royalblue')

    plt.show()

    for traj in trajectory:
        origin, x, y, z = load_quiver(traj)
        # xhat0 = goals[i] - origin 
        # xhat1 = goals[i+2] - origin        
        gripper_x.remove()
        gripper_y.remove()
        gripper_z.remove()
        gripper_x = ax.quiver(*origin, *x, length=5.0, normalize=True, color='lightcoral')
        gripper_y = ax.quiver(*origin, *y, length=5.0, normalize=True, color='palegreen')
        gripper_z = ax.quiver(*origin, *z, length=5.0, normalize=True, color='royalblue')

        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    # live trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread_file', '-t', help="path to thread file")
    parser.add_argument('--thread_specs')
    parser.add_argument('--goal_file')
    # parser.add_argument('--gripper_pos', '-g', help="pos of gripper x y z qw qx qy qz")
    args = parser.parse_args()

    thread, upper, lower = load_thread(args.thread_file, args.thread_specs)

    # gripper_pos = args.gripper_pos
    gripper_pos = [0, 0, 0, 0, 0, 0, 1]

    goals_og = np.load(args.goal_file, allow_pickle=True) # array of points along the thread suitable for grasping

    goals = load_goal(args.goal_file)
    goal = goals[1]
    # goal = np.array([10, 10, 200, 0.1571893, 0.2847014, 0.0913176, 0.9412214])
    for goal in goals:
        # trajectory = gen_trajectory(gripper_pos, goal, steps=50)
        # live_plot(thread, upper, lower, goals_og, trajectory)
        live_plot(thread, upper, lower, goals_og, [goal])
