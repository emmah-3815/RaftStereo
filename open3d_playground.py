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


# camera parameters
fx, fy, cx1, cy = 882.996114514, 882.996114514, 445.06146749, 190.24049547
cx2 = 445.061467
baseline = 5.8513759749420302 # mm

def visualize_point_cloud(objects):
    # Create a visualization object and window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for object in objects:
        # down sample point cloud
        # if o3d.geometry.Geometry.get_geometry_type(object).value == 1: # type point cloud is 1
        #     object = o3d.geometry.PointCloud.random_down_sample(object, 0.3)
        vis.add_geometry(object)
    # o3d.visualization.draw_geometries(objects,
    #                               zoom=1
    #                               ,
    #                               front=[-0.0, -0.0, -0.1],
    #                               lookat=[0, 0, 0],
    #                               up=[-0.0694, -0.5, 0.2024]
    #                               )
    # print("thread points in visualize:", np.asarray(objects[0].points), "array length", np.asarray(objects[0].points).size)
    vis.run()


def load_thread(args):
    thread_file_path = '/media/emmah/PortableSSD/Arclab_data/paper_singular_needle/frame_000000.npy'
    thread_data = np.load(thread_file_path)
    thread_data = thread_data * 1000
    # thread_data = np.load(args.thread_file)
    n = thread_data.shape[0]
    vectors = [[i, i+1] for i in range(n)]
    colors = [[0, 1, 0] for i in range(n-1)]

    thread = o3d.geometry.LineSet()
    thread.points = o3d.utility.Vector3dVector(thread_data)
    thread.lines = o3d.utility.Vector2iVector(vectors)
    thread.colors = o3d.utility.Vector3dVector(colors)

    # pdb.set_trace()
    # o3d.visualization.draw_geometries([thread]S)
    # print("thread points in load thread function:", np.asarray(thread.points) , "array length", np.asarray(thread.points).size)
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.geometry.PointCloud.estimate_normals(pcd)
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd)

    # down sample point cloud
    # pcd = o3d.geometry.PointCloud.random_down_sample(pcd, 0.1)

    return pcd

def generate_planes(args, pcd):
    ## find most common plane ##
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
    #                                         ransac_n=3,
    #                                         num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # return inlier_cloud, outlier_cloud

    # returns points in clusters based on density
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(
    #         pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # return pcd

    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    downpcd_farthest = pcd.farthest_point_down_sample(1000) #2000 or below significant points
    # key_points = np.arange(0, len(pcd.points), 2).tolist() #(start, stop, step)
    # pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    downpcd_farthest.paint_uniform_color([0, 1, 0])
    # print("Find its 200 nearest neighbors, and paint them blue.")
    # [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
    # np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

    return downpcd_farthest




def generate_bounding_box(point_cloud):
    bounding_box = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(point_cloud)
    aligned_bounding_box = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(point_cloud)
    bounding_box.color = [0, 1, 0]
    bounding_vertices = np.asarray(o3d.geometry.OrientedBoundingBox.get_box_points(bounding_box))
    # print("bounding box center:", bounding_box.center)
    # print("bounding box vertices", bounding_vertices)


    return bounding_box

def create_geometry_at_points(points):
    geometries = o3d.geometry.TriangleMesh()
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5) #create a small sphere to represent point
        sphere.translate(point) #translate this sphere to point
        geometries += sphere
    geometries.paint_uniform_color([1.0, 0.0, 0.0])
    return geometries

def align_objects(first, second, first_center, second_center):
    first_pts = np.asarray(first.points)
    second_pts = np.asarray(second.points)
    first_closest = first_pts[first_pts[:,2].argsort()[1]] # highest is closes
    second_furthest = second_pts[second_pts[:,2].argsort()[-1]]
    print("first closest", first_closest, "second futhest", second_furthest)
    highlights = create_geometry_at_points([first_closest, second_furthest])  

    tz = first_center[2] - second_furthest[2]
    tx,ty= first_center[:2] - second_center[:2]
    # translation = [tx, ty, -tz]
    translation = [tx, ty, tz]
    print("translation", translation)

    transformed_second = copy.deepcopy(second).translate(translation)

    return transformed_second, highlights

def reconstruct_mesh(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]))
    return mesh

def write_to_file(data, path):
    with open(path, 'w') as file:
        file.write(data)


if __name__ == '__main__':
    ''' python open3d_playground.py \
    --png_file=/media/emmah/PortableSSD/Arclab_data/trial_9_left_rgb/frame_000001.png \
    --npy_file=/media/emmah/PortableSSD/Arclab_data/trial_9_RAFT_output_1/frame_000001.npy \
    --meat_mask=/media/emmah/PortableSSD/Arclab_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png 
    '''



    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--npy_file', help="path_to_npy_file", default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_RAFT_output_1/frame_000001.npy")
    parser.add_argument('--png_file', help="path_to_png_file_for_visualization", default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_left_rgb/frame_000001.png")
    parser.add_argument('--meat_mask_file', help="path_to_meat_mask", default="/media/emmah/PortableSSD/Arclab_data/trial_9_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/frame0001.png")
    parser.add_argument('--mask_erode', help='choose to erode mask for less chance of flying points', default=True)

    args = parser.parse_args()


    thread = load_thread(args)
    # print("thread points in main:", np.asarray(thread.points), "array length", np.asarray(thread.points).size)
    pcd = generate_point_cloud(args)

    meat_bound_box = generate_bounding_box(pcd)
    thread_bound_box = generate_bounding_box(thread)

    meat_clusters= generate_planes(args, pcd)

    thread_trans, highlights = align_objects(pcd, thread, meat_bound_box.center, thread_bound_box.center)
    thread_trans_bound_box = generate_bounding_box(thread_trans)
    origin = create_geometry_at_points([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)])
    
    # reconstuct meat mesh and export obj file
    meat_small = o3d.geometry.PointCloud.random_down_sample(pcd, 0.01)
    meat_mesh = reconstruct_mesh(meat_small)
    mesh_file = "/media/emmah/PortableSSD/Arclab_data/meat_mesh.obj"
    # o3d.io.write_triangle_mesh(mesh_file, meat_mesh, write_vertex_normals=False, write_vertex_colors=False, print_progress=True)
    # tried to run it in py_pbd softbody, but the mesh failes to tetrahedralize

    meat_point_cloud_file = "/media/emmah/PortableSSD/Arclab_data/meat_thread_data_9_26/trial_08_meat_pcd.pcd"
    # o3d.io.write_point_cloud(meat_point_cloud_file, pcd)




    visualize_point_cloud([
                        #    thread,
                        #    thread_trans,
                        #    pcd,
                        #    highlights,
                           origin,
                           meat_bound_box, 
                           meat_clusters,
                        #    thread_bound_box,
                        #    thread_trans_bound_box,
                           ])
