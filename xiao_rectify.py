#!/usr/bin/env python3
import os
import time
import cv2
import yaml
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import pdb
class StereoRectification(Node):
    def __init__(self):
        super().__init__('stereo_rectification')
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.bridge = CvBridge()
        self.cfg, map11, map12, map21, map22, self.P1, self.P2, self.Q = self.load_calibration_config()
        self.rectify_undistort_left = lambda img1 : cv2.remap(img1, map11, map12, cv2.INTER_LINEAR)
        self.rectify_undistort_right = lambda img2 : cv2.remap(img2, map21, map22, cv2.INTER_LINEAR)
        # subscriber and publisher
        self.left_image_sub = self.create_subscription(Image,
                                                       self.cfg["left_image_sub"],
                                                       self.left_image_callback,
                                                       10)
        self.right_image_sub = self.create_subscription(Image,
                                                        self.cfg["right_image_sub"],
                                                        self.right_image_callback,
                                                        10)
        self.left_image_pub = self.create_publisher(Image, self.cfg["left_image_rectified_topic"], 10)
        self.right_image_pub = self.create_publisher(Image, self.cfg["right_image_rectified_topic"], 10)
        self.P1_pub = self.create_publisher(Float32MultiArray, '/stereo/rectified/P1', 10)
        self.P2_pub = self.create_publisher(Float32MultiArray, '/stereo/rectified/P2', 10)
        self.Q_pub = self.create_publisher(Float32MultiArray, '/stereo/rectified/Q', 10)
        self.timer = self.create_timer(1/5, self.pub_camera_info)
    def pub_camera_info(self):
        P1_msg = Float32MultiArray()
        P1_msg.data = self.P1.flatten().tolist()
        self.P1_pub.publish(P1_msg)
        P2_msg = Float32MultiArray()
        P2_msg.data = self.P2.flatten().tolist()
        self.P2_pub.publish(P2_msg)
        Q_msg = Float32MultiArray()
        Q_msg.data = self.Q.flatten().tolist()
        self.Q_pub.publish(Q_msg)
    def left_image_callback(self, msg: Image):
        time_stamp = msg.header.stamp
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
            return
        img = cv2.resize(img,(int(self.cfg["input_img_width"]/self.cfg["scaleing_factor"]),
                              int(self.cfg["input_img_height"]/self.cfg["scaleing_factor"])),
                              interpolation = cv2.INTER_AREA)
        img = self.rectify_undistort_left(img)
        new_msg = self.bridge.cv2_to_imgmsg(img,encoding='rgb8')
        new_msg.header.stamp = time_stamp
        self.left_image_pub.publish(new_msg)
    def right_image_callback(self, msg: Image):
        time_stamp = msg.header.stamp
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
            return
        img = cv2.resize(img,(int(self.cfg["input_img_width"]/self.cfg["scaleing_factor"]),
                              int(self.cfg["input_img_height"]/self.cfg["scaleing_factor"])),
                              interpolation = cv2.INTER_AREA)
        img = self.rectify_undistort_right(img)
        new_msg = self.bridge.cv2_to_imgmsg(img,encoding='rgb8')
        new_msg.header.stamp = time_stamp
        self.right_image_pub.publish(new_msg)
    def load_calibration_config(self):
        '''
        loading config file related to camera calibration
        '''
        with open(os.path.join(self.cwd, 'cfgs', 'dvrk_config.yaml'), 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.camera_file = os.path.join(self.cwd, 'cfgs', cfg["camera_file"])
        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_READ)
        fn = fs.getNode("K1")
        mtx1 = fn.mat()
        mtx1 = mtx1/cfg["scaleing_factor"]/cfg["pre_scaling_factor"]
        mtx1[2,2] = 1
        fn = fs.getNode("K2")
        mtx2 = fn.mat()
        mtx2 = mtx2/cfg["scaleing_factor"]/cfg["pre_scaling_factor"]
        mtx2[2,2] = 1
        fn = fs.getNode("D1")
        dist1 = fn.mat()
        fn = fs.getNode("D2")
        dist2 = fn.mat()
        fn = fs.getNode("R")
        R = fn.mat()
        fn = fs.getNode("T")
        T = fn.mat()
        BINIMG_W = cfg["output_img_width"]
        BINIMG_H = cfg["output_img_height"]
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx1, dist1,
            mtx2, dist2,
            (int(cfg["input_img_width"]/cfg["scaleing_factor"]),
             int(cfg["input_img_height"]/cfg["scaleing_factor"])),
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            newImageSize=(BINIMG_W, BINIMG_H)
        )
        map11,map12 = cv2.initUndistortRectifyMap(mtx1,dist1,R1,P1,(BINIMG_W, BINIMG_H),cv2.CV_16SC2)
        map21,map22 = cv2.initUndistortRectifyMap(mtx2,dist2,R2,P2,(BINIMG_W, BINIMG_H),cv2.CV_16SC2)
        return cfg, map11, map12, map21, map22, P1, P2, Q
if __name__ == '__main__':
    rclpy.init()
    node = StereoRectification()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()56