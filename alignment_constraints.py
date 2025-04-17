import numpy as np
import open3d as o3d
from imageio.v3 import imread
import cv2 as cv
import copy
import pdb
from scipy.optimize import Bounds, minimize



class ReconstructAlign:
    def __init__(self) -> None:
        # bounding boxes position and orientation
        self.needle_bound = None
        self.thread_bound = None
        self.meat_bound = None

        # sets of points
        self.needle = None
        self.thread = None
        self.meat = None
        self.spheres = None

        # initilization checks
        self.needle_init = False
        self.thread_init = False
        self.meat_init = False
        self.spheres_init = False

        self.n_t_contact = None
        self.n_m_contact = None
        self.t_m_contact = None
        self.vis_objects = []

    def init_camera_params(self, params) -> None:
        # params = fx, fy, cx1, cy, cx2, baseline
        # camera parameters
        self.fx, self.fy, self.cx1, self.cy = params[0], params[1], params[2], params[3]
        self.cx2 = params[4]
        self.baseline = params[5]

    def init_object_params(self, mask_erode=True) -> None:
        self.mask_erode = mask_erode # meat mask erode

    def init_objects(self):
        assert(self.needle_init == True)
        assert(self.thread_init == True)
        assert(self.meat_init == True)


    def add_needle(self, needle, bounding_box):
        assert(self.needle_init == False)
        self.needle = needle
        self.needle_bound = bounding_box
        # add addtional modifications here


        self.needle_init = True

    def add_thread(self, thread_file_path:str) -> o3d.geometry.LineSet:
        assert(self.thread_init == False)
        self.thread = np.load(thread_file_path, allow_pickle=True)

        thread_data = np.load(thread_file_path, allow_pickle=True)
        thread_data = thread_data * 1  # * 1000 scale up to match meat point cloud size
        n = thread_data.shape[0]
        vectors = [[i, i+1] for i in range(n-1)]
        colors = [[0, 0, 0] for i in range(n-1)]

        self.thread = o3d.geometry.LineSet()
        self.thread.points = o3d.utility.Vector3dVector(thread_data)
        self.thread.lines = o3d.utility.Vector2iVector(vectors)
        self.thread.colors = o3d.utility.Vector3dVector(colors)

        self.thread_bound = self.generate_bounding_box(self.thread)
        self.thread_init = True




    def add_meat(self, meat_npy_path, meat_png_path, meat_mask_file=None):
        assert(self.meat_init == False)

        disp = np.load(meat_npy_path)
        image = imread(meat_png_path)

        # inverse-project
        depth = (self.fx * self.baseline) / (-disp + (self.cx2 - self.cx1))
        H, W = depth.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        points_grid = np.stack(((xx-self.cx1)/self.fx, (yy-self.cy)/self.fy, np.ones_like(xx)), axis=0) * depth

        # Remove flying points
        flying_mask = np.ones((H, W), dtype=bool)
        # flying_mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
        # flying_mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False


        # mask meat only

        if meat_mask_file == "None":
            meat_mask_file = None

        if meat_mask_file != None:
            meat_mask = imread(meat_mask_file)
            if self.mask_erode is True:
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

        self.meat = o3d.geometry.PointCloud()
        self.meat.points = o3d.utility.Vector3dVector(points)
        self.meat.colors = o3d.utility.Vector3dVector(colors)
        o3d.geometry.PointCloud.estimate_normals(self.meat)
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(self.meat)

        self.meat_bound = self.generate_bounding_box(self.meat)
        # down sample point cloud
        # pcd = o3d.geometry.PointCloud.random_down_sample(pcd, 0.1)
        self.meat_init = True

    def generate_bounding_box(self, geometry:o3d.geometry.PointCloud) -> o3d.geometry.OrientedBoundingBox:
        bounding_box = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(geometry)
        # aligned_bounding_box = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(point_cloud)
        bounding_box.color = [1, 0, 0]
        # bounding_box_vertices = np.asarray(o3d.geometry.OrientedBoundingBox.get_box_points(bounding_box))
        return bounding_box

    def align_objects(self, first, second, first_center, second_center): #  meat, thread, meat_bound.center, thread_bound.center
        first_pts = np.asarray(first.points)
        second_pts = np.asarray(second.points)
        first_closest = first_pts[first_pts[:,2].argsort()[1]] # highest is closes
        second_furthest = second_pts[second_pts[:,2].argsort()[-1]]
        # print("first closest", first_closest, "second futhest", second_furthest)
        # highlights = self.create_spheres_at_points([first_closest, second_furthest])  

        tz = first_center[2] - second_furthest[2] + 5
        tx,ty= first_center[:2] - second_center[:2]
        # translation = [tx, ty, -tz]
        translation = [tx, ty, tz]
        print("translation", translation)

        transformed_second = copy.deepcopy(second).translate(translation)

        return transformed_second
    
    def needle_thread_contact(self, threshhold):
        # measure difference between end points of thread and mount point of needle
        # return distance -> minimize
        return

    def needle_meat_contact(self, threshold):
        # calc distance between needle and meat for all points
        # if distance < threshold, consider in contact
        # return number of contacts -> maximize or >contact threshold
        # or
        # consider the needle's bounding box, since it is rigid, 
        # consider the angle between the needle and meat as a percentage of contact. 
        # Best case if alignment of meat normal vector with needle z vector?
        return
    
    def thread_meat_contact(self, threshold):
        # calc distance between thread and meat for all points
        # if distance < threshold, consider in contact
        # return number of contacts -> maximize or >contact threshold
        p1 = np.array(self.thread.points)[:, np.newaxis, :]
        p2 = np.array(self.meat.points)[np.newaxis, :, :]
        
        contact_pts = (np.linalg.norm(p1 - p2, axis=2) < threshold).nonzero()


        return contact_pts

    def intersection_constraint(self):
        # group thread points to corrosponding meat surface(s) (based on proximity of position)
        # for each group, check if thread is above the normal(s) of the meat
        
        # group thead points closest to needle (based on proximity of position)
        # need to generate normal for needle?
        # for the thread points closest to needle, check if thread is outside/above the normal(s) of the needle
        # might be uncessary considering the needle_thread_contact constraint

        # extract the needle bounding box vertices
        # group vertices to closest meat surface(s)
        # check if the vertices are above meat normal

        return


    def distance(self, meat, thread, suture):
        # calculate abs(distance between three objects)
        distance = None
        return distance

    def nearby_objects(self, primary, secondary, radius):
        # if primary != #points:
        p1 = np.array(primary.points)[:, np.newaxis, :]
        p2 = np.array(secondary.point)[np.newaxis, :, :]
        
        close_pts = (np.linalg.norm(p1 - p2, axis=2) < radius).nonzero()

    def KNN_play(self, pcd, thread): # pcd and thread as o3d.geometry objects
        ## KNN search 200(neighbors) closest points on pcd with respect to each node on thread
        color = 0
        red = 0
        blue = 1
        neighbors = 3
        pcd = pcd
        key_points = thread.points
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        spheres = o3d.geometry.TriangleMesh()
        for point in key_points:
            # sphere = self.create_spheres_at_points([point], radius=0.5, color=[color, 0, 1])
            sphere = self.create_spheres_at_points([point], radius=0.5, color=[red, 0, blue])
            spheres += sphere
            # print("Finding ", point, "'s ", neighbors, "nearest neighbors, and painting them blue")
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point, neighbors)
            # np.asarray(pcd.colors)[idx[0:], :] = [color, 0, 1] # set idx first index to 1 if want to skip the closest point
            # color += 1/len(key_points)
            np.asarray(pcd.colors)[idx[0:], :] = [red, 0, blue] # set idx first index to 1 if want to skip the closest point
            red = not red
            blue = not blue
            color = red
        return pcd, spheres

    
    def KNN_neighborhoods(self, pcd, thread):
        key_points = thread.points
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        neighbors = 1
        pcd_neighbors = []
        pcd_neighbors_normal = []
        pcd_idx = []
        for point in key_points:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point, neighbors)
            pcd_neighbors.append(np.asarray(pcd.points)[idx, :])
            pcd_neighbors_normal.append(np.asarray(pcd.normals)[idx,:])
            pcd_idx.append(idx)

        # print(pcd_neighbors)
        return np.asarray(pcd_neighbors), np.asarray(pcd_neighbors_normal), np.asarray(key_points)

    def norm_of_neighborhoods(self, pcd_neighbors, thread_points):
        dis = []
        for i in range(len(np.asarray(thread_points))):
            p1 = np.array(thread_points[i])[:, np.newaxis,  np.newaxis]
            p2 = np.array(pcd_neighbors[i])[np.newaxis, :, :]
            norm = (np.linalg.norm(p1 - p2, axis=2))
            dis.append(norm)

        return dis
    def norm_of_thread_to_neighbors(self, pcd_neighbors, thread_points):
        # computes the distance of each thread point to its respective pcd_neighbor, takes the average when pcd_neighbor contains more than one point (k>1)
        dis = np.mean(np.mean(np.absolute(np.array(thread_points)[:, np.newaxis] - np.array(pcd_neighbors)), axis=2), axis=1)
    
        return dis

    def thread_transform(self, x, pcd, thread_input):
        thread = copy.copy(thread_input)
        R_center = self.generate_bounding_box(thread).center
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(x[3:])
        T = x[:3]
        thread_translate = thread.translate(T)
        thread_transform = thread_translate.rotate(R, center=R_center)
        return thread_transform


    def thread_transformation_dis(self, x, pcd, thread_input):
        # x is translation upon the origin and rotation based on an axis angle method
        thread = copy.copy(thread_input)
        pcd_neighbors_og, pcd_neighbors_n_og, key_points_og = self.KNN_neighborhoods(pcd, thread)
        R_center = self.generate_bounding_box(thread).center
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(x[3:])
        T = x[:3]
        thread_translate = thread.translate(T)
        thread_transform = thread_translate.rotate(R, center=R_center)
        pcd_neighbors_trans, pcd_neighbors_n_trans, key_points_trans = self.KNN_neighborhoods(pcd, thread_transform)

        dis = self.norm_of_thread_to_neighbors(pcd_neighbors_trans, key_points_trans)
        dis_sum = np.sum(dis)
        return dis_sum
    
    def thread_normal_const(self, x, pcd, thread_input): # checks if the thread is above meat using normal vectors
        thread = copy.copy(thread_input)
        R_center = self.generate_bounding_box(thread).center
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(x[3:])
        T = x[:3]
        thread_translate = thread.translate(T)
        thread_transform = thread_translate.rotate(R, center=R_center)
        pcd_neighbors_trans, pcd_neighbors_n_trans, key_points_trans = self.KNN_neighborhoods(pcd, thread_transform)
        pcd_normals = np.average(pcd_neighbors_n_trans, axis=1)
        thread_knn_normals = np.average(pcd_neighbors_trans, axis=1) - key_points_trans # thread nodes to its nearest point vector
        # array of meat to thread vectors
        normals_ontop = np.diag(np.dot(pcd_normals, thread_knn_normals.T))


        # methods = ["sigmoid", "sign", "tanh"]
        method = "tanh"
        print(f"using alignment method {method}")

        # sigmoid method
        if method == "sigmoid":
            def sigmoid(x):
                return 1/(1+np.exp(-x))
            signs = sigmoid(normals_ontop) - 0.5 # sigmoid 
            # signs = signs < 0
            sum_normals = np.mean(signs)

        # sign method
        elif method == "sign":
            signs = np.sign(normals_ontop)
            sum_normals = 1 - np.mean(signs)

        # tanh method
        elif method == "tanh":
            signs = np.tanh(normals_ontop) # tanh cuts in even more
            sum_normals = np.mean(signs)

        sum_normals = np.mean(signs)

        return sum_normals
        
    def thread_normal_calcs(self, x, pcd, thread_input): # checks if the thread is above meat using normal vectors
        thread = copy.copy(thread_input)
        R_center = self.generate_bounding_box(thread).center
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(x[3:])
        T = x[:3]
        thread_translate = thread.translate(T)
        thread_transform = thread_translate.rotate(R, center=R_center)
        pcd_neighbors_trans, pcd_neighbors_n_trans, key_points_trans = self.KNN_neighborhoods(pcd, thread_transform)
        pcd_normals = np.average(pcd_neighbors_n_trans, axis=1)
        thread_knn_normals = np.average(pcd_neighbors_trans, axis=1) - key_points_trans # thread nodes to its nearest point vector

        # array of meat to thread vectors
        normals_ontop = np.diag(np.dot(pcd_normals, thread_knn_normals.T))


        # methods = ["sigmoid", "sign", "tanh"]
        method = "tanh"

        # sigmoid method
        if method == "sigmoid":
            def sigmoid(x):
                return 1/(1+np.exp(-x))
            signs = sigmoid(normals_ontop) - 0.5 # sigmoid 
            # signs = signs < 0
            sum_normals = np.mean(signs)
            normals = signs

        # sign method
        elif method == "sign":
            signs = np.sign(normals_ontop)
            sum_normals = 1 - np.mean(signs)
            normals = signs

        # tanh method
        elif method == "tanh":
            signs = np.tanh(normals_ontop) # tanh cuts in even more
            sum_normals = np.mean(signs)
            normals = signs



        # signs = np.sign(normals_ontop) # sign, non-differentiable, works if constraint is ineq
        # addition = normals_ontop*(normals_ontop < 0) * 10
        # ontop = np.add(normals_ontop, addition)
        # sum_normals = np.mean(normals_ontop - ontop)


        return sum_normals, normals

    

    def slsqp_solver(self, pcd, thread):
        # objective function is to minimize the distance between the thread nodes and their knn neighbors on the meat, 
        # with the input to the objective function being the transformation parameters, and output being the distance 
        # closest manual value -60 10 20 0.9 0 0 (x, y, z, rx, ry, rz)

        # bounds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
        bounds = ((-100, 100), (-100, 1000), (-100, 100), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2))
       
        eq_cons = {'type': 'ineq', 'fun' : self.thread_normal_const, 'args': (pcd, thread)}
        
        x0 = np.random.rand(6) * 10e-7  
        res = minimize(self.thread_transformation_dis, x0, method='SLSQP', args=(pcd, thread),
                    options={'ftol': 1e-9, 'disp': True, 'maxiter': 400}, constraints=[eq_cons],
                    bounds=bounds)

        print("slsqp results", res.x)
        print("normal constraint results", self.thread_normal_calcs(res.x, pcd, thread))
        return res.x

    # def thread_top_meat_constraint(self, pcb, thread_points, alignment):

    #     N = [] # normal vectors of meat closest to the thread nodes
    #     x = [] # thread nodes 
    #     x_align = alignment(x)
    #     eq_cons = {'type' : 'eq',
    #                'fun' : lambda x: np.array([np.sign(np.dot(N.T, alignment(x)))-1]),
    #                'jac' : lambda x: np.array([[alignment(x)]])}
    #     return eq_cons
    
    # def thread_form_constraint(self, pcb, thread_points, alignment):



    
    def create_spheres_at_points(self, points, radius=0.5, color=[1, 0, 0]):
        spheres = o3d.geometry.TriangleMesh()
        for point in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius) #create a small sphere to represent point
            sphere.translate(point) #translate this sphere to point
            sphere.paint_uniform_color(color)
            spheres += sphere
        # geometries.paint_uniform_color([1.0, 0.0, 0.0]) # paint all red
        return spheres
    
    def visualize_objects(self, objects=[]): # allow externally added objects
        # Create a visualization object and window
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()

        self.vis_objects = objects
        if self.thread_init == True: self.vis_objects.append(self.thread)
        if self.meat_init == True: self.vis_objects.append(self.meat) 
        if self.needle_init == True: self.vis_objects.append(self.needle)
        if self.thread_bound is not None: self.vis_objects.append(self.thread_bound)
        if self.meat_bound is not None: self.vis_objects.append(self.meat_bound)
        if self.needle_bound is not None: self.vis_objects.append(self.needle_bound)

        
        for object in self.vis_objects:
            ## down sample point cloud ##
            # if o3d.geometry.Geometry.get_geometry_type(object).value == 1: # type point cloud is 1
            #     object = o3d.geometry.PointCloud.random_down_sample(object, 0.3)
            vis.add_geometry(object)
        
        opt.show_coordinate_frame = True
        # red is x, green is y, blue is z
        vis.run()


'''
point1 = [(177.4150316259643, 80.06868164738896, 82.32650976966762),
          (162.04772459076872, 80.29781125179436, 153.403148072541),
          (153.403148072541, 82.32650976966762, 80.29781125179436),
          (165.2980351177718, 84.62501223785691, 83.98251402600364),
          (-168.7499999978935, 83.98251402600364, 81.58268534624698),
          (-168.74999999845744, 81.58268534624698, 81.58268534624698)]

point2 = [(157.50739908371222, 77.7317781763434, 153.403148072541),
          (-168.75000000105112, 76.80636446168572, 81.58268534624698),
          (-179.99260091805525, 77.73177817832965, 81.58268534624698),
          (177.41503162596428, 80.06868164738897, 82.32650976966762),
          (-168.7500000015425, 81.58268534399583, 81.58268534624698),
          (-154.91503162867107, 80.06868164543069, 153.403148072541)]

import numpy as np

# set tolerance to control how close values need to be 
# to be considered the same
tolerance = 10**(-6) 

p1 = np.array(point1)[:, np.newaxis, :]
p2 = np.array(point2)[np.newaxis, :, :]
print(p1, "\n", p2)
print((np.linalg.norm(p1 - p2, axis=2) < tolerance).nonzero())

should print
(array([0, 5]), array([3, 4]))
meaning 0 and 3 are similar, 5 and 4 and similar

import numpy as np

# set tolerance to control how close values need to be 
# to be considered the same
tolerance = 10**(-6) 

p1 = np.array(point1)[:, np.newaxis, :]
p2 = np.array(point2)[np.newaxis, :, :]
print(np.linalg.norm(p1 - p2, axis=2))
close_index = (np.linalg.norm(p1 - p2, axis=2) < tolerance).nonzero()
print(close_index)
close_list = np.array([i.tolist() for i in close_index])
print(close_list)
p1_list = close_list[:,0]
print(p1_list)










'''