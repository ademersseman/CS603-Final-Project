#!/usr/bin/python3

import replicate
import os
import numpy as np

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
import cv2

def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

    yaw = (euler_from_quaternion([
        p.orientation.x,
        p.orientation.y,
        p.orientation.z,
        p.orientation.w])
    [2])

    return yaw


rospy.init_node('modules', anonymous=True)


class PerceptionModule:

    def __init__(self):

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.odom_sub = rospy.Subscriber("/amcl_pose", PoseStamped, self.odom_callback)
        self.image_data = None
        self.depth_data = None
        self.robot_pose = None
        self.ram_gsam_output = None

        # all callbacks must only update the corresponding data once per loop. So if we update the image data, we must
        # wait for the perception loop to complete before updating the image data again.
        self.open_image_callback = True
        self.open_depth_callback = True
        self.open_odom_callback = True

    def image_callback(self, data):

        # If the image callback is open, we can update the image data. If not, we must wait for the perception-planning
        # loop to finish before updating the image data again.
        if self.open_image_callback:
            self.open_image_callback = False
            self.image_data = self.bridge.imgmsg_to_cv2(data, "bgr8")
            os.remove(os.path.join("camera_imgs", 'camera_image.jpg'))
            cv2.imwrite('camera_imgs/camera_image.jpg', self.image_data)

        # if all callbacks are closed, we can run the perception-planning loop
        if not self.open_image_callback and not self.open_depth_callback and not self.open_odom_callback:
            # run the perception module
            self.run_ramgsam(verbose=True)
            # get the object locations
            object_locations = self.boxes_to_positions() # outputs list of (x,y,object_name) tuples

            # run the planning module [TODO Hongxin]

            # set all callbacks to open so we collect new sensor data
            self.open_image_callback = True
            self.open_depth_callback = True
            self.open_odom_callback = True

    def depth_callback(self, data):
        if self.open_depth_callback:
            self.open_depth_callback = False
            self.depth_data = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def odom_callback(self, data):
        if self.open_odom_callback:
            self.open_odom_callback = False
            self.robot_pose = data.pose.pose

    def run_ramgsam(self, verbose=False):
        if verbose:
            print("Running API")

        self.ram_gsam_output = replicate.run(
            "idea-research/ram-grounded-sam:80a2aede4cf8e3c9f26e96c308d45b23c350dd36f1c381de790715007f1ac0ad",
            input={
                "use_sam_hq": False,
                "input_image": open('camera_imgs/camera_image.jpg', "rb"),
                "show_visualisation": False
            }
        )
        if verbose:
            print("API finished running")
            print("Model output: ", self.ram_gsam_output)

    def boxes_to_positions(self, probs_thresh=0.4):
        """
        Convert the bounding boxes of each object to an objects (x,y) location in 3D space.

        INPUTS:
            self.ram_gsam_output: dict with key to a list of dicts (objects), the output of the RAM-GSAM model.
                Contains the bounding boxes of each object (['box']) stored as [xmin, ymin, xmax, ymax], the class of the
                object (['name']), and the probability of the object detection (['logit']).
            self.depth_data: np.array, the depth data from the camera. Used to determine the distance from
                the robot's location
            self.robot_pose: PoseStamped, the pose of the robot. Used to calculate the 3D location of the
                object.
            probs_thresh: float, the probability threshold for the object detection model. Any object with a logit
                value below this threshold will be discarded from the output.

        OUTPUTS:
            objects_locations: list of tuples, each tuple contains the (x,y) location of an object. Tuples are in
                the form (x, y, object_name)
        """

        objects_locations = []
        for obj in self.ram_gsam_output['mask']:
            if obj['logit'] < probs_thresh:
                continue

            box = obj['box']  # Get the bounding box coordinates [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box

            # Extract depth values within the bounding box region
            depth_values = self.depth_data[ymin:ymax, xmin:xmax]

            # Calculate the average depth value within the bounding box region
            avg_depth_value = np.mean(depth_values)

            # Calculate the 3D coordinates of the object by using the robot's position and orientation
            # and the depth value of the object.

            # Convert quaternion to theta
            theta = get_yaw_from_pose(self.robot_pose.orientation)
            x = self.robot_pose.position.x + avg_depth_value * np.cos(theta)
            y = self.robot_pose.position.y + avg_depth_value * np.sin(theta)

            # Append the 3D coordinates of the object to the list
            objects_locations.append((x, y, obj["name"]))

            return objects_locations


if __name__ == '__main__':
    module = PerceptionModule()
    print("hi there")

    rospy.spin()

    # print(os.environ["REPLICATE_API_TOKEN"])
    # print(os.getcwd())
    #
    # print("Running API")
    # output = replicate.run(
    #    "idea-research/ram-grounded-sam:80a2aede4cf8e3c9f26e96c308d45b23c350dd36f1c381de790715007f1ac0ad",
    #    input={
    #        "use_sam_hq": False,
    #        "input_image": open('camera_imgs/camera_image.jpg', "rb"),
    #        "show_visualisation": False
    #    }
    # )
    # print("API finished running")
    # print(output)

    # example output
    # output = {
    #     'objects': [
    #         {
    #             'class': 'person'
    #         }]
    # }

    # import json
    # # save dictionary output to a file
    # with open('output.json', 'w') as f:
    #    json.dump(output, f)

    # open the json file
    # with open('output.json', 'r') as f:
    #    json_data = json.load(f)