#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""print the topics from a rosbag for extraction

example: python rope/rosbag_extract.py ./rope/assets/trial_9_single_arm_no_tension.bag ./rope/assets/trial_9_right /stereo/right/rectified_downscaled_image stereo_right
topics: 
/stereo/right/image/compressed
/stereo/left/image/compressed
_sensor_msgs__CompressedImage
"""




import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")

    args = parser.parse_args()

    bag = rosbag.Bag(args.bag_file, "r")
    # bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages():
        print(topic)
        input()



        # cv_img = bridge.imgmsg_to_cv2(msg) #, desired_encoding="passthrough")
        # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(args.output_dir, args.output_name + "_%06i.png" % count), cv_img)
        # print ("Wrote image %i" % count)

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()