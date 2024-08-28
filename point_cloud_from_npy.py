import sys
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from imageio.v3 import imread
import argparse
import pdb
import cv2 as cv
import matplotlib.pyplot as plt
import open3d


# camera parameters
fx, fy, cx1, cy = 882.996114514, 882.996114514, 445.06146749, 190.24049547
cx2 = 445.061467
baseline = 5.8513759749420302 # mm

def visualize_point_cloud(objects):
    open3d.visualization.draw_geometries(objects,
                                #   zoom=1
                                #   ,
                                #   front=[-0.0, -0.0, -0.1],
                                #   lookat=[2.6172, 2.0475, 1.532],
                                #   up=[-0.0694, -0.5, 0.2024]
                                  )


def load_thread(args):
    thread_file_path = '/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy'
    thread_data = np.load(thread_file_path)
    # thread_data = np.load(args.thread_file)
    n = thread_data.shape[0]
    vectors = [[i, i+1] for i in range(n)]
    colors = [[1, 0, 0] for i in range(n-1)]

    thread = open3d.geometry.LineSet()
    thread.points = open3d.utility.Vector3dVector(thread_data)
    thread.lines = open3d.utility.Vector2iVector(vectors)
    thread.colors = open3d.utility.Vector3dVector(colors)

    # pdb.set_trace()
    # open3d.visualization.draw_geometries([thread])
    return thread

def generate_point_cloud(args):
    disp_path = args.npy_file
    disp = np.load(disp_path)
    image = imread(args.png_file)

    # inverse-project
    depth = (fx * baseline) / (-disp + (cx2 - cx1))
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth

    # Remove flying points
    flying_mask = np.ones((H, W), dtype=bool)
    flying_mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
    flying_mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False


    # mask meat only
    if args.meat_mask_file is not None:
        meat_mask = imread(args.meat_mask_file)
        if args.mask_erode is True:
            kernel = np.ones((5,5),np.uint8)
            meat_mask = cv.erode(meat_mask, kernel, iterations=1)
            # imgplot = plt.imshow(meat_mask)
            # plt.show()

        meat_mask = meat_mask > 0
        mask = meat_mask * flying_mask
    else:
        mask = flying_mask


    points = points_grid.transpose(1,2,0)[mask]
    colors = image[mask].astype(np.float64) / 255

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    open3d.geometry.PointCloud.estimate_normals(pcd)
    open3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd)


    # pdb.set_trace()

    return pcd

if __name__ == '__main__':
    ''' python point_cloud_from_npy.py \
    --png_file=/media/emmah/PortableSSD/Arclab_data/trial_9_left_rgb/frame_000001.png \
    --npy_file=/media/emmah/PortableSSD/Arclab_data/trial_9_RAFT_output_1/frame_000001.npy \
    --meat_mask=/media/emmah/PortableSSD/Arclab_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png 
    '''



    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--npy_file', help="path_to_npy_file", default="/media/emmah/PortableSSD/Arclab_data/trial_9_RAFT_output_1/frame_000001.npy")
    parser.add_argument('--png_file', help="path_to_png_file_for_visualization", default="/media/emmah/PortableSSD/Arclab_data/trial_9_left_rgb/frame_000001.png")
    parser.add_argument('--meat_mask_file', help="path_to_meat_mask", default="/media/emmah/PortableSSD/Arclab_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png")
    parser.add_argument('--mask_erode', help='choose to erode mask for less chance of flying points', default=True)

    args = parser.parse_args()


    thread = load_thread(args)
    pcd = generate_point_cloud(args)
    visualize_point_cloud([pcd, thread])
