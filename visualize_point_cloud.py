import sys
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from imageio.v3 import imread
import argparse
import pdb
import cv2 as cv
import matplotlib.pyplot as plt
import open3d as o3d
import copy

# camera parameters on rectified images
fx, fy, cx1, cy = 882.996114514, 882.996114514, 445.06146749, 190.24049547
cx2 = 445.061467
baseline = 5.8513759749420302 # mm

'''
# camera parameters on non-rectified images
fx, fy, cx1, cy = 1.6796e+03, 1.6681e+03, 839.1909, 496.6793
cx2 = 1.0265e+03 # camera right? x position
baseline = 6.6411 # mm, distance between cameras'
'''

'''
python visualize_point_cloud.py --npy_file /media/emmah/PortableSSD/Arclab_data/test_objects_3_7_25/icebreakers_1/raft_output/frame_000001.npy --png_file /media/emmah/PortableSSD/Arclab_data/test_objects_3_7_25/icebreakers_1/left_rgb/frame_000001.png
'''


def visualize_point_cloud(objects):
    # Create a visualization object and window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for object in objects:
        # down sample point cloud
        # if o3d.geometry.Geometry.get_geometry_type(object).value == 1: # type point cloud is 1
        #     object = o3d.geometry.PointCloud.random_down_sample(object, 0.3)
        vis.add_geometry(object)
    vis.run()

def generate_point_cloud(args):
    disp_path = args.npy_file
    disp = np.load(disp_path)
    image = imread(args.png_file)

    # inverse-project
    depth = (fx * baseline) / (-disp + (cx2 - cx1))
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth


    mask = np.ones((H, W), dtype=bool)
    # # Remove flying points
    # flying_mask = np.ones((H, W), dtype=bool)
    # flying_mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
    # flying_mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False


    # # mask meat only
    # if args.meat_mask_file is not None:
    #     meat_mask = imread(args.meat_mask_file)
    #     if args.mask_erode is True:
    #         kernel = np.ones((5,5),np.uint8)
    #         meat_mask = cv.erode(meat_mask, kernel, iterations=1)
    #         # imgplot = plt.imshow(meat_mask)
    #         # plt.show()

    #     meat_mask = meat_mask > 0
    #     mask = meat_mask * flying_mask
    # else:
    #     mask = flying_mask


    points = points_grid.transpose(1,2,0)[mask]
    colors = image[mask].astype(np.float64) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.geometry.PointCloud.estimate_normals(pcd)
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd)

    # down sample point cloud
    # pcd = o3d.geometry.PointCloud.random_down_sample(pcd, 0.1)

    return pcd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--npy_file', help="path_to_npy_file")
    parser.add_argument('--png_file', help="path_to_png_file_for_visualization")
    parser.add_argument('--pcd', help="point_cloud data")    
    parser.add_argument('--generate', help="generate point cloud from npy_file", action='store_true')


    args = parser.parse_args()
    if args.generate:
        pcd = generate_point_cloud(args)
    else:
        assert args.pcd is not None, "pcd file is empty"
        
        pcd = args.pcd

    visualize_point_cloud([
                           pcd,
                           ])


## for 3_21 dataset
'''
python visualize_point_cloud.py --generate --png_file \
/media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/trial_26/left_rgb/frame_000000.png \
--npy_file /media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/trial_26/npy/frame_000000.npy 
'''