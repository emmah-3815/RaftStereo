import numpy as np
import open3d as o3d
print(o3d.__version__)

from imageio.v3 import imread
import cv2 as cv
import copy
import pdb
from scipy.optimize import Bounds, minimize
import pickle
import scipy.interpolate as interp




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
        self.origin = None

        # initilization checks
        self.needle_init = False
        self.thread_init = False
        self.meat_init = False
        self.spheres_init = False
        self.origin_init = False

        self.n_t_contact = None
        self.n_m_contact = None
        self.t_m_contact = None
        self.vis_objects = []
        self.needle_r = None
        self.thread_reliability = None
        self.lower_bound_3d = None
        self.upper_bound_3d = None


    def init_camera_params(self, calib) -> None:
        # camera parameters on non-rectified images
        # fx, fy, cx1, cy = 1.6796e+03, 1.6681e+03, 839.1909, 496.6793
        # cx2 = 1.0265e+03
        # baseline = 6.6411 # mm

        # fei's yaml camera parameters
        # fx, fy, cx1, cy = 1.02588223e+03, 1.02588223e+03, 1.67919017e+02, 2.34152707e+02
        # cx2 = 1.67919017e+02
        # baseline = 6.820275085 # mm
        
        cv_file = cv.FileStorage(calib, cv.FILE_STORAGE_READ)
        K1 = cv_file.getNode("K1").mat()
        T = cv_file.getNode("T").mat()
        self.fx = K1[0][0]
        self.fy = K1[1][1]
        self.cx1 = K1[0][2]
        self.cy = K1[1][2]
        self.cx2 = self.cx1 # the same value in fei's yaml
        self.baseline = T[0][0] * -1



    def init_object_params(self, mask_erode=True) -> None:
        self.mask_erode = mask_erode # meat mask erode

    def init_objects(self):
        assert(self.needle_init == True)
        assert(self.thread_init == True)
        assert(self.meat_init == True)


    def add_needle(self, needle_obj_path:str=None, needle_r=None, sudo_needle=True):
        assert(self.needle_init == False)
        assert((needle_obj_path!=None) or (sudo_needle==True))
        # add addtional modifications here
        # instead of adding the needle, use a set of points to indicate needle instead, so that the coordinate frame is given to match with the camera
        if sudo_needle:
            n = 20
            theta = np.linspace(np.pi/2, np.pi*3/2, n) # orietation supposedly based on paper
            # theta = np.linspace(-np.pi/2, np.pi/2, n)
            z_lin = np.linspace(0, 0.01, n)

            points = np.array((needle_r*np.cos(theta), needle_r*np.sin(theta), z_lin))
            points = np.transpose(points)
            vectors = [[i, i+1] for i in range(n-1)]
            # colors = [[0.1, 0.1, 0.1] for i in range(n-1)]

            # pdb.set_trace()
            self.needle = o3d.geometry.LineSet()
            self.needle.points = o3d.utility.Vector3dVector(points)
            self.needle.lines = o3d.utility.Vector2iVector(vectors)
            # self.needle.colors = o3d.utility.Vector3dVector(colors)
            
            self.needle = self.needle_mesh(self.needle, radius=0.5)
            self.needle_r = needle_r #mm
            self.needle_bound = self.generate_bounding_box(self.needle)

        else:
            needle = o3d.io.read_triangle_mesh(needle_obj_path)
            self.needle = needle
            self.needle = self.needle.scale(1000.0, center=(0, 0, 0))
            self.needle_bound = self.generate_bounding_box(self.needle)

        self.needle_init = True
        
    def needle_mesh(self, lineset, radius=0.1):
        from scipy.spatial.transform import Rotation as Rot
        color = (1, 0.8, 0.8)

        mesh = o3d.geometry.TriangleMesh()
        pts = np.asarray(lineset.points)
        for i, j in np.asarray(lineset.lines):
            p1, p2 = pts[i], pts[j]
            d = p2 - p1
            l = np.linalg.norm(d)
            z = np.array([0, 0, 1])

            if l == 0:
                continue
            if i == 0:
                cyl = o3d.geometry.TriangleMesh.create_cylinder(radius/2, l)
            else:
                cyl = o3d.geometry.TriangleMesh.create_cylinder(radius, l)
            R = Rot.align_vectors([d], [z])[0].as_matrix()
            cyl.rotate(R, center=d/2)
            cyl.translate(p1)
            mesh += cyl
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh
    
    def add_thread(self, thread_file_path:str) -> o3d.geometry.LineSet:
        assert(self.thread_init == False)
        # self.thread = np.load(thread_file_path, allow_pickle=True)

        thread_data = np.load(thread_file_path, allow_pickle=True)
        thread_data = thread_data * 1  # * 1000 scale up to match meat point cloud size
        n = thread_data.shape[0]
        vectors = [[i, i+1] for i in range(n-1)]
        colors = [[0, 0, 0] for i in range(n-1)]

        self.thread = o3d.geometry.LineSet()
        # pdb.set_trace()
        self.thread.points = o3d.utility.Vector3dVector(thread_data)
        self.thread.lines = o3d.utility.Vector2iVector(vectors)
        self.thread.colors = o3d.utility.Vector3dVector(colors)

        self.thread_bound = self.generate_bounding_box(self.thread)
        self.thread_init = True

    def load_thread_specs(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        self.thread_reliability = data.get('reliability')
        thread_lower_constr = data.get('lower_constr')
        thread_upper_constr = data.get('upper_constr')

        self.thread_lower = thread_lower_constr
        self.thread_upper = thread_upper_constr

        # bound_idx = np.linspace(0, 1, len(thread_lower_constr))
        # lower_f = interp.interp1d(bound_idx, thread_lower_constr)
        # upper_f = interp.interp1d(bound_idx, thread_upper_constr)
        # self.thread_lower = lower_f(np.linspace(0, 1, len(self.thread.points))) # get lower const at key points
        # self.thread_upper = upper_f(np.linspace(0, 1, len(self.thread.points))) # get upper const at key points

    def flip_thread(self, thread_file_path:str):
        assert(self.thread_init == True)
        user = input("Flip thread? y/n ")
        if user == "y":
            self.thread_init = False
            thread_data = np.load(thread_file_path, allow_pickle=True)[::-1]
            with open(thread_file_path, "wb") as f:
                print("saving flipped spline")
                np.save(f, thread_data)
            self.add_thread(thread_file_path)

    def add_lower_bound_spline(self):
        thread = self.thread
        bound = self.thread_lower
        # pdb.set_trace()
        # assert(len(np.asarray(thread.points)) == len(np.asarray(bound)))
        if bound.shape[1] != 3:
            bound = np.transpose(bound)
            assert(bound.shape[1] == 3)
        bound_points = bound        
        
        n = bound.shape[0]
        vectors = [[i, i+1] for i in range(n-1)]
        colors = [[0, 0, 1] for i in range(n-1)]

        # bound_points = copy.copy(np.asarray(thread.points))
        # bound_points[:, 2] = bound
        # pdb.set_trace()

        self.lower_bound_3d = o3d.geometry.LineSet()
        self.lower_bound_3d.points = o3d.utility.Vector3dVector(bound_points)
        self.lower_bound_3d.lines = o3d.utility.Vector2iVector(vectors)
        self.lower_bound_3d.colors = o3d.utility.Vector3dVector(colors)

        # self.lower_bound_3d_bound = self.generate_bounding_box(self.lower_bound_3d)

    def add_upper_bound_spline(self):
        thread = self.thread
        bound = self.thread_upper
        # pdb.set_trace()
        # assert(len(np.asarray(thread.points)) == len(np.asarray(bound)))
        if bound.shape[1] != 3:
            bound = np.transpose(bound)
            assert(bound.shape[1] == 3)
        bound_points = bound
        
        n = bound.shape[0]
        vectors = [[i, i+1] for i in range(n-1)]
        colors = [[1, 0, 1] for i in range(n-1)]

        # bound_points = copy.copy(np.asarray(thread.points))
        # bound_points[:, 2] = bound
        self.upper_bound_3d = o3d.geometry.LineSet()
        self.upper_bound_3d.points = o3d.utility.Vector3dVector(bound_points)
        self.upper_bound_3d.lines = o3d.utility.Vector2iVector(vectors)
        self.upper_bound_3d.colors = o3d.utility.Vector3dVector(colors)

        # self.lower_bound_3d_bound = self.generate_bounding_box(self.lower_bound_3d)

    def add_meat(self, meat_npy_path, meat_png_path, meat_mask_file=None, thread_mask_file=None, needle_mask_file=None) -> o3d.geometry.PointCloud:
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
        flying_mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
        flying_mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False


        meat_mask = imread(meat_mask_file).any(axis=2) if meat_mask_file is not None else np.ones((H, W), dtype=bool)
        thread_mask = imread(thread_mask_file)==0 if thread_mask_file is not None else np.ones((H, W), dtype=bool) # equal to zero to flip mask
        needle_mask = imread(needle_mask_file)==0 if needle_mask_file is not None else np.ones((H, W), dtype=bool) # equal to zero to flip mask
        # pdb.set_trace()

        # combine all masks
        mask = meat_mask * thread_mask * needle_mask

        if self.mask_erode is True:
            kernel_size = 3
            kernel = np.ones(kernel_size,np.uint8)
            mask = cv.erode(mask.astype(np.uint8), kernel, iterations=1)
            # imgplot = plt.imshow(meat_mask)
            # plt.show()

        mask = mask > 0
        mask = mask * (np.array(flying_mask))

        points = points_grid.transpose(1,2,0)[mask]
        colors = image[mask].astype(np.float64) / 255

        self.meat = o3d.geometry.PointCloud()
        self.meat.points = o3d.utility.Vector3dVector(points)
        self.meat.colors = o3d.utility.Vector3dVector(colors)
        cl, ind = self.meat.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        self.meat = self.meat.select_by_index(ind)

        o3d.geometry.PointCloud.estimate_normals(self.meat)
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(self.meat)

        self.meat_bound = self.generate_bounding_box(self.meat)
        # down sample point cloud
        # pcd = o3d.geometry.PointCloud.random_down_sample(pcd, 0.1)
        self.meat_init = True

    def add_origin(self):
        assert(self.origin_init == False)
        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.6, origin=[0, 0, 0]).scale(50.0, center=(0, 0, 0))
        self.origin_init = True


    def generate_bounding_box(self, geometry:o3d.geometry.PointCloud) -> o3d.geometry.OrientedBoundingBox:
        bounding_box = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(geometry)
        # bounding_box = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(geometry)
        bounding_box.color = [1, 0, 0]
        # bounding_box_vertices = np.asarray(o3d.geometry.OrientedBoundingBox.get_box_points(bounding_box))
        return bounding_box
    
    def depth_solver(self, pcd, thread):
        thread_pts = thread.points

        # obtain the mean position of the meat neighborhoods
        pcd_neighbors_og, pcd_neighbors_n_og, key_points_og = self.KNN_neighborhoods(pcd, thread, neighbors=100)
        neighbor_z_means = np.max(pcd_neighbors_og[:, :, 2], axis=1)

        # find the lowest point of the thread with respect to the closest neighborhood on the meat
        z_dis = neighbor_z_means -  np.array(thread_pts)[:, 2]
        tz = np.min(z_dis)
        translation = [0, 0, -tz, 0, 0, 0] # only z translation changes
        
        return translation
    
    '''
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

    '''
    def KNN_play(self, pcd, thread, neighbors=3): # pcd and thread as o3d.geometry objects
        ## KNN search 200(neighbors) closest points on pcd with respect to each node on thread
        red = 0
        blue = 1
        first_pt = True
        key_points = thread.points
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        spheres = o3d.geometry.TriangleMesh()
        for point in key_points:
            # sphere = self.create_spheres_at_points([point], radius=0.5, color=[color, 0, 1])
            if first_pt:
                sphere = self.create_spheres_at_points([point], radius=0.5, color=[0, 1, 0])
                first_pt = False
            else:
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

    def paint_reliability(self, thread):
        assert(self.thread_reliability is not None)
        # reliability /= np.max(reliability) 

        assert(len(thread.points) == self.thread_reliability.size) # same amount of points

        points = thread.points
        spheres = o3d.geometry.TriangleMesh()
        for i, point in enumerate(points):
            sphere = self.create_spheres_at_points([point], radius=0.5, color=[1-self.thread_reliability[i], self.thread_reliability[i], 0])
            spheres += sphere
        return spheres



    def KNN_neighborhoods(self, pcd, thread, neighbors=1):
        key_points = thread.points
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
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

    def transform(self, x, object_in, object_box, quat=False): #rotate and translate based on center of object bounding box
        object = copy.copy(object_in)
        
        if quat: # if rotation is a quaternion
            R = o3d.geometry.get_rotation_matrix_from_quaternion(x[3:])
        else:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(x[3:])

        # rotate
        # R_center = self.generate_bounding_box(object).center # if regenerating bounding box, but can flip coordinate directions
        R_center = object_box.center
        T = x[:3]
        object_transform = object.rotate(R, center=R_center)
        object_box_transform = object_box.rotate(R, center=R_center)

        # translate
        object_transform = object_transform.translate(T)
        object_box_transform = object_box_transform.translate(T)
        return object_transform, object_box_transform

    def load_needle_pos(self, pos_file):
        with open(pos_file, 'rb') as f:
            data = pickle.load(f)

        self.needle_pos = np.array([data.get('x')*1000, data.get('y')*1000, data.get('z')*1000, data.get('qw'), data.get('qx'), data.get('qy'), data.get('qz')])

        # rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(needle_pos[3:])
        # euler_angles = o3d.geometry.get_euler_angles_from_matrix(rotation_matrix)
        # self.needle_pos = np.array((needle_pos[:3], euler_angles))




    def needle_align(self, x, quat=False): # uses the stored needle points
        # pdb.set_trace()
        T = x[:3]
        if quat: # if rotation is a quaternion
            R = o3d.geometry.get_rotation_matrix_from_quaternion(x[3:])
        else:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(x[3:])
        self.needle.translate(T)
        # R_center = self.generate_bounding_box(self.needle).center
        self.needle_bound.translate(T)
        R_center = self.needle_bound.center
        self.needle.rotate(R, center=R_center)
        self.needle_bound.rotate(R, center=R_center)


    def needle_thread_conn(self, needle, needle_bound, thread, thread_box):
        r = self.needle_r
        needle_conn_pt = np.array([-r, -r/2, 0]) # point of connection from center of needle
        # needle_bound = self.generate_bounding_box(needle) # use the given box instead
        needle_R = needle_bound.R # needle rotation matrix
        needle_center = needle_bound.center # position of needle center
        oriented_conn_pt = np.matmul(needle_R, needle_conn_pt) # oriented connection point
        T = thread.points[0] - needle_center - oriented_conn_pt # how much to translate needle to match connection point to thread
        needle.translate(T)
        needle_bound = needle_bound.translate(T)

        y_axis = thread.points[0] - thread.points[1] # tangent vector to thread
        y_axis = y_axis / np.linalg.norm(y_axis)
        R = self.align_vector_to_vector(needle_R[:, 1], y_axis) # rotation matrix to rotate needle to match thread
        needle.rotate(R, center=thread.points[0])
        needle_bound.rotate(R, center=thread.points[0])

        return needle, needle_bound

    def align_vector_to_vector(self, v1, v2):
        """
        Returns a rotation matrix that aligns v1 to v2 using Open3D's get_rotation_matrix_from_axis_angle.

        Parameters:
            v1, v2: 3D vectors

        Returns:
            3x3 rotation matrix (numpy array)
        """
        v1 = np.asarray(v1) / np.linalg.norm(v1)
        v2 = np.asarray(v2) / np.linalg.norm(v2)
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)

        if np.isclose(dot, 1.0):
            return np.eye(3)  # No rotation needed
        if np.isclose(dot, -1.0):
            # 180-degree rotation around any orthogonal vector
            ortho = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
            axis = np.cross(v1, ortho)
            axis = axis / np.linalg.norm(axis)
            return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)

        angle = np.arccos(dot)
        axis = cross / np.linalg.norm(cross)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


    def thread_transformation_dis(self, x, pcd, thread, pcd_bound, thread_bound):
        # x is translation upon the origin and rotation based on an axis angle method
        pcd_neighbors_og, pcd_neighbors_n_og, key_points_og = self.KNN_neighborhoods(pcd, thread)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(x[3:])
        T = x[:3]
        
        # translation
        thread_trans = thread.translate(T)
        thread_bound_trans = thread_bound.translate(T)

        # rotation
        # R_center = self.generate_bounding_box(thread).center
        R_center = thread_bound_trans.center
        thread_trans = thread_trans.rotate(R, center=R_center)
        thread_bound_trans = thread_bound_trans.rotate(R, center=R_center)
        pcd_neighbors_trans, pcd_neighbors_n_trans, key_points_trans = \
            self.KNN_neighborhoods(pcd, thread_trans)

        dis = self.norm_of_thread_to_neighbors(pcd_neighbors_trans, key_points_trans)
        dis_sum = np.sum(dis)
        return dis_sum
    
    def thread_normal_const(self, x, pcd, thread_input, pcd_bound, thread_bound): # checks if the thread is above meat using normal vectors
        thread = copy.copy(thread_input)
        # R_center = self.generate_bounding_box(thread).center

        R = o3d.geometry.get_rotation_matrix_from_axis_angle(x[3:])
        T = x[:3]

        # translation
        thread_trans = thread.translate(T)
        thread_bound_trans = thread_bound.translate(T)

        # rotation
        R_center = thread_bound_trans.center
        thread_transform = thread_trans.rotate(R, center=R_center)
        pcd_neighbors_trans, pcd_neighbors_n_trans, key_points_trans = \
            self.KNN_neighborhoods(pcd, thread_transform)
        pcd_normals = np.average(pcd_neighbors_n_trans, axis=1)
        thread_knn_normals = np.average(pcd_neighbors_trans, axis=1) - key_points_trans # thread nodes to its nearest point vector
        # array of meat to thread vectors

        normals_ontop = np.diag(np.dot(pcd_normals, thread_knn_normals.T)) * -1


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
            normals = 1 - signs
            sum_error = np.sum(normals) - len(normals)
            sum_normals = np.mean(signs)

            # with the shift it still means that the thread solves for the cloestest point disregarding being on top of the thread

        sum_normals = np.sum(signs)

        return sum_error
        
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
        normals_ontop = np.diag(np.dot(pcd_normals, thread_knn_normals.T)) * -1


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
            normals = 1 - signs
            sum_error = np.sum(normals) - len(normals)
            sum_normals = np.mean(signs)
            # normals = signs



        # signs = np.sign(normals_ontop) # sign, non-differentiable, works if constraint is ineq
        # addition = normals_ontop*(normals_ontop < 0) * 10
        # ontop = np.add(normals_ontop, addition)
        # sum_normals = np.mean(normals_ontop - ontop)

        return sum_error, normals

    def thread_meat_orient(self, pcd, thread, pcd_bound, thread_bound):

        R = -np.linalg.inv(pcd_bound.R).T

        thread.rotate(R, center=thread_bound.center)
        thread_bound.rotate(R, center=thread_bound.center)
        return thread, thread_bound


    def slsqp_solver(self, pcd, thread):
        # objective function is to minimize the distance between the thread nodes and their knn neighbors on the meat, 
        # with the input to the objective function being the transformation parameters, and output being the distance 
        # closest manual value -60 10 20 0.9 0 0 (x, y, z, rx, ry, rz)

        # bounds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))
        bounds = ((-100, 100), (-100, 1000), (-100, 100), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2))

        # bounds for no rotation
        bounds = ((-100, 100), (-100, 1000), (-100, 100), (0, 0), (0, 0), (0, 0))
       
        eq_cons = {'type': 'ineq', 'fun' : self.thread_normal_const, 'args': (pcd, thread)}
        
        x0 = np.random.rand(6) * 10e-7  
        res = minimize(self.thread_transformation_dis, x0, method='SLSQP', args=(pcd, thread),
                    options={'ftol': 1e-9, 'disp': True, 'maxiter': 400}, constraints=[eq_cons],
                    bounds=bounds)

        print("slsqp results", res.x)
        print("normal constraint results", self.thread_normal_calcs(res.x, pcd, thread))
        return res.x

    def grasp(self, pcd, thread, rely_thresh=0.5, dis_thresh=2, velo_thresh=0.7):
        reliability = self.thread_reliability
        assert(len(thread.points) == reliability.size)
        assert(self.thread_init and self.meat_init and reliability is not None)
        
        points = np.asarray(thread.points)
        candidates = np.ones_like(points[:, 0])
        # lower_bound_points = np.asarray(self.lower_bound_3d.points)
        # lower_const_dis = points[:, 2] - lower_bound_points[:, 2]
        upper_bound_points = np.asarray(self.upper_bound_3d.points)
        upper_const_dis = points[:, 2] - upper_bound_points[:, 2]

        # prune unreliable points
        reliability[reliability<rely_thresh] = 0
        candidates *= reliability

        # prune points too close to meat
        meat_neighborhoods, _, thread_points = self.KNN_neighborhoods(pcd, thread, 10)
        dis = self.norm_of_thread_to_neighbors(meat_neighborhoods, thread_points)
        dis_rely = dis > dis_thresh
        candidates *= dis_rely

        # prune points with lower thresh too close to meat (reoeat of points too close to meat)
        upper_rely = (dis - upper_const_dis) > dis_thresh
        candidates *= dis_rely

        # prune points too vertical
        z_velocity = points[1:, 2] - points[:-1, 2]
        z_velocity = np.append(z_velocity, velo_thresh) 
        z_velocity = np.abs(z_velocity) + 1e-7

        # multiple by velocity method
        # z_velocity[z_velocity > velo_thresh] = 0
        # candidates *= (1/z_velocity) * np.max(z_velocity)

        # filter by velocity thresh only
        velo_rely = z_velocity < velo_thresh
        candidates *= velo_rely

        # # top three best points
        # top3_idx = np.argsort(candidates)[-3:][::-1]
        # top3 = points[top3_idx]
        top = points[candidates!=0]


        spheres = o3d.geometry.TriangleMesh()
        for i, point in enumerate(top):
            sphere = self.create_spheres_at_points([point], radius=2, color=[0, 0, 1])
            spheres += sphere
        return spheres

    
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
        if self.origin_init == True: self.vis_objects.append(self.origin)
        if self.thread_bound is not None: self.vis_objects.append(self.thread_bound)
        if self.meat_bound is not None: self.vis_objects.append(self.meat_bound)
        if self.needle_bound is not None: self.vis_objects.append(self.needle_bound)

        for object in self.vis_objects:
            ## down sample point cloud ##
            # if o3d.geometry.Geometry.get_geometry_type(object).value == 1: # type point cloud is 1
            #     object = o3d.geometry.PointCloud.random_down_sample(object, 0.3)
            vis.add_geometry(object)
        
        # opt.show_coordinate_frame = True
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