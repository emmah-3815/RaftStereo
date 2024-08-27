#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.

example: python rope/rosbag_extract.py ./rope/assets/trial_9_single_arm_no_tension.bag ./rope/assets/trial_9_right /stereo/right/rectified_downscaled_image stereo_right
topics: 
/stereo/right/image/compressed
/stereo/left/image/compressed

To write into SSD Path:
/media/emmah/PortableSSD/Arclab_data

"""

import rosbag
topics = bag.get_type_and_topic_info()[1].keys()
types = []
for val in bag.get_type_and_topic_info()[1].values():
     types.append(val[0])


import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory, (existing or new)")
    parser.add_argument("image_topic", help="Image topic.")
    parser.add_argument("output_name", nargs='?', default="frame", help="Image name. (optional)")

    args = parser.parse_args()
    mypath = args.output_dir
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    print ("Extract images from %s on topic %s into %s" % (args.bag_file, \
                                                          args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        try:
            cv_img = bridge.compressed_imgmsg_to_cv2(msg) # tries compressed image convertion to cv2

        except:
            print("something went wrong")

        # try:
            # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # tries regular image convertion to cv2
        # except AttributeError as e:
            # try:
                # cv_img = bridge.compressed_imgmsg_to_cv2(msg) # tries compressed image convertion to cv2
# 
            # except CvBridgeError as e:
                # print(e)
        
        # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) # convert bgr format to rgb
        cv2.imwrite(os.path.join(args.output_dir, args.output_name + "_%06i.png" % count), cv_img)
        print ("Wrote image %i" % count)

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()