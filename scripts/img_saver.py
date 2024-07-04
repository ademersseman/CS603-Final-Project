#!/usr/bin/python3

#future imports
from __future__ import print_function

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time

rospy.init_node('image_saver', anonymous=True)
bridge = CvBridge()

def save_rgb_image(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'images/rgb_image_{timestamp}.jpg'
        cv2.imwrite(filename, cv_image)
        print(f"Saved {filename}")
    except CvBridgeError as e:
        print(e)

def save_depth_image(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "16UC1")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'images/depth_image_{timestamp}.png'
        cv2.imwrite(filename, cv_image)
        print(f"Saved {filename}")
    except CvBridgeError as e:
        print(e)


rospy.Subscriber("/camera/color/image_raw", Image, save_rgb_image)
rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, save_depth_image)

rospy.spin()