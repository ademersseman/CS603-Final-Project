#!/usr/bin/python3

import replicate
import os
import time
import numpy as np

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
import cv2

from llm_plan import Planer

def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

    yaw = (euler_from_quaternion([
        p.orientation.x,
        p.orientation.y,
        p.orientation.z,
        p.orientation.w])
    [2])

    return yaw


rospy.init_node('triton_assistant', anonymous=True)


class TritonAssistant:

    def __init__(self, user_instruction=None, debug_mode=False, task_id="1"):
        self.finished = 'no'
        self.debug_mode = debug_mode
        self.bridge = CvBridge()
        self.step = 0
        self.task_id = 'task/' + task_id
        os.makedirs(f'{self.task_id}/{str(self.step)}', exist_ok=True)
        self.image_data = None
        self.depth_data = None
        self.robot_pose = None
        self.ram_gsam_output = None

        # all callbacks must only update the corresponding data once per loop. So if we update the image data, we must
        # wait for the perception loop to complete before updating the image data again.
        self.open_image_callback = True
        self.open_depth_callback = True
        self.open_odom_callback = True
        self.localizing = True

        # name: [x, y, z, w, updated_step]
        # self.waypoints = {'unexplored_1': [-0.82, 0.14, 99], 'unexplored_2': [-0.42, -0.25, 99]}
        # 1 is under sink, 2 is by door, 3 is by fridge, 4 is by trash can
        self.waypoints = {'unexplored_1': [-0.15, 0.72, -0.71, 0.7, 99], 'unexplored_2': [-2.2, -1.5, 0.38, 0.92, 99],
                          'unexplored_3': [-2.2, -0.23, 0.14, 1, 99], 'unexplored_4': [0, -1, 0.73, 0.67, 99]}
        self.explored_waypoints = []
        #todo: @Alan, read this info from the map file
        self.planer = Planer(debug_mode=self.debug_mode)
        self.user_instruction = user_instruction

        self.goal = None
        self.goal_semantic = None
        self.nav_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        # self.at_goal = False

        self.star_time = time.time()
        self.api_time = 0

        if self.debug_mode:
            init_pose = PoseWithCovarianceStamped()
            init_pose.pose.pose.position.x = -1.2098531724405264
            init_pose.pose.pose.position.y = -0.5500910593187853
            init_pose.pose.pose.orientation.w = 1.0
            self.robot_pose = init_pose.pose
            self.open_odom_callback = False
            self.waypoints = {'unexplored_1': [-0.10, 0.72, -0.87, 0.49, 99], 'unexplored_2': [-1.88, -1.37, 0.41, 0.91, 99], 'coke_can':[-0.6,-0.08], 'band-aid':[-0.3,-0.5]}

        # self.send_goal(0.70, 0.35, 0.39, 0.92)
        # time.sleep(3)
        self.localizing = False
        self.open_image_callback = True
        self.open_depth_callback = True
        self.open_odom_callback = True

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.odom_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.odom_callback)

    def image_callback(self, data):
        # If the image callback is open, we can update the image data. If not, we must wait for the perception-planning
        # loop to finish before updating the image data again.
        if self.open_image_callback and not self.open_depth_callback:
            print("Image callback received")
            self.open_image_callback = False
            self.image_data = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #print(f"saved image to {self.task_id}/camera_image.jpg")
            # if os.path.exists(os.path.join(self.task_id, 'camera_image.jpg')):
            #     os.remove(os.path.join(self.task_id, 'camera_image.jpg'))
            cv2.imwrite(os.path.join(self.task_id, str(self.step), 'camera_image.jpg'), self.image_data)

        # if all callbacks are closed, we can run the perception-planning loop
        if not self.localizing and not self.open_image_callback and not self.open_depth_callback and not self.open_odom_callback:
            # run the perception module
            self.run_ramgsam(verbose=False)
            # get the object locations
            object_locations = self.boxes_to_positions() # outputs list of (x,y,object_name) tuples
            self.update_waypoints(object_locations)
            if self.debug_mode:
                print(self.waypoints)
            # run the planning module
            if self.at_goal():
                if 'yes' in self.finished.lower():
                    finished_time = time.time()
                    total_time = finished_time - self.star_time
                    os.makedirs(f"{self.task_id}/final", exist_ok=True)
                    with open(f'{self.task_id}/final/statics.txt', 'w') as f:
                        f.write(f"Finished task, found object {self.goal_semantic}, my final pose is at {self.robot_pose}, total time used {total_time}, api time used {self.api_time}, effective time used {total_time - self.api_time}.")
                    print(f"Finished task, found object {self.goal_semantic}, my pose is {self.robot_pose}, total time used {total_time}, api time used {self.api_time}, effective time used {total_time - self.api_time}.")
                    cv2.imwrite(os.path.join(self.task_id, 'final', 'camera_image.jpg'), self.image_data)
                    rospy.signal_shutdown("Finished task")
                else:
                    print("Planning new goal")

                goal, self.finished = self.planer.plan(self.robot_pose, self.user_instruction, self.waypoints, self.explored_waypoints)
                # Strip the parentheses
                x, y, name = goal.strip("()").split(", ")
                x, y = float(x), float(y)
                self.goal_semantic = name

                self.explored_waypoints.append(self.goal_semantic)
                if 'unexplored' in name:
                    # send the goal to the navigation module
                    self.send_goal(self.waypoints[name][0], self.waypoints[name][1], self.waypoints[name][2], self.waypoints[name][3])
                else:
                    self.send_goal(x, y)

            # set all callbacks to open, so we collect new sensor data
            self.open_image_callback = True
            self.open_depth_callback = True
            self.open_odom_callback = True
            self.step += 1
            os.makedirs(f'{self.task_id}/{str(self.step)}', exist_ok=True)

    def depth_callback(self, data):
        if self.open_depth_callback:
            self.open_depth_callback = False
            self.depth_data = self.bridge.imgmsg_to_cv2(data, "16UC1")
            cv2.imwrite(f'{self.task_id}/{str(self.step)}/camera_depth_image.jpg', self.depth_data)
            print("Depth callback received")

    def odom_callback(self, data):
        # print("Odom callback received")
        # if self.open_odom_callback:
            # print(data)
            # save the robot's pose to file
        with open(f'{self.task_id}/{str(self.step)}/robot_pose.txt', 'w') as f:
            f.write(str(data.pose.pose))
        self.open_odom_callback = False
        self.robot_pose = data.pose.pose
        print("Odom callback received: x,y", self.robot_pose.position.x, self.robot_pose.position.y)

    # def status_callback(self, data):
    #     print("Status callback received")
    #     if data.status.status == 3:
    #         self.at_goal = True

    def run_ramgsam(self, verbose=False):
        if self.debug_mode:
            return None
        api_start_time = time.time()

        print("Running API, starting time: ", api_start_time)

        self.ram_gsam_output = replicate.run(
            "idea-research/ram-grounded-sam:80a2aede4cf8e3c9f26e96c308d45b23c350dd36f1c381de790715007f1ac0ad",
            input={
                "use_sam_hq": False,
                "input_image": open(f'{self.task_id}/{str(self.step)}/camera_image.jpg', "rb"),
                "show_visualisation": False
            }
        )
        api_finished_time = time.time()
        self.api_time += api_finished_time - api_start_time
        del self.ram_gsam_output['json_data']['mask'][0]  # remove the background object
        # example output
        # {'json_data': {'mask': [{'label': 'background', 'value': 0},
        #                         {'box': [0.01312255859375, 0.2370758056640625, 640.0107421875, 479.97027587890625],
        #                          'label': 'room', 'logit': 0.45, 'value': 1},
        #                         {'box': [0.10516357421875, 274.33544921875, 640.0984497070312, 479.7895202636719],
        #                          'label': 'floor', 'logit': 0.42, 'value': 2}, {
        #                             'box': [-0.00220489501953125, 201.869384765625, 145.75796508789062,
        #                                     361.15313720703125], 'label': 'pillow', 'logit': 0.37, 'value': 3}],
        #                'tags': 'lamp, chip, dark, floor, light, pillow, room'}, 'masked_img': None,
        #  'rounding_box_img': None, 'tags': 'lamp, chip, dark, floor, light, pillow, room'}
        if verbose:
            print("API finished running")
            print("Model output: ", self.ram_gsam_output)

    def boxes_to_positions(self, probs_thresh=0.3):
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

        for obj in self.ram_gsam_output['json_data']['mask']:
            if obj['logit'] < probs_thresh:
                continue

            box = obj['box']  # Get the bounding box coordinates [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Extract depth values within the bounding box region
            depth_values = self.depth_data[ymin:ymax, xmin:xmax]

            # Calculate the average depth value within the bounding box region
            avg_depth_value = np.mean(depth_values) / 1000  # Convert mm to meters

            # Calculate the 3D coordinates of the object by using the robot's position and orientation
            # and the depth value of the object.

            # Convert quaternion to theta
            theta = get_yaw_from_pose(self.robot_pose)
            if self.debug_mode:
                print(f"Robot pose: {self.robot_pose}")
                print(f"Theta: {theta}")
                print(f"Avg depth value: {avg_depth_value}")
            x = self.robot_pose.position.x + avg_depth_value * np.cos(theta)
            y = self.robot_pose.position.y + avg_depth_value * np.sin(theta)

            # Append the 3D coordinates of the object to the list
            objects_locations.append((x, y, obj["label"]))

        return objects_locations

    def send_goal(self, x, y, z=1.0, w=0.0):
        self.goal = (x, y)
        
        pose_diffs = np.array([x, y]) - np.array([self.robot_pose.position.x, self.robot_pose.position.y])
        
        self.goal = np.array([self.robot_pose.position.x, self.robot_pose.position.y]) + pose_diffs * 0.9
        
        print(f"Motion module started, goal: {self.goal}")
        if self.debug_mode:
            return None
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w
        # self.at_goal = False
        self.nav_pub.publish(pose)

    def update_waypoints(self, object_locations):
        for obj in object_locations:
            if any([x in obj[2] for x in ['apple','coke','can', 'soda', 'pencil', 'beverage']]):
            # if 'door' in obj[2] or 'bag' in obj[2]:
            #     continue
                if obj[2] not in self.waypoints:
                    self.waypoints[obj[2]] = list(obj[:2]) + [self.step]
                    print(f"Object {obj[2]} added to waypoints list at position: {obj[:2]}")
                else:
                    self.waypoints[obj[2]] = list(obj[:2]) + [self.step]
                    print(f"Object {obj[2]} already in waypoints list, original position: {self.waypoints[obj[2]]}, new position: {obj[:2]}")

        if len(self.waypoints) > 15:
            self.waypoints = dict(sorted(self.waypoints.items(), key=lambda x: x[1][-1], reverse=True)[:10])
            print(f"Waypoints length {len(self.waypoints)}, removing oldest waypoints")

        with open(f'{self.task_id}/waypoints.txt', 'w') as f:
            f.write(str(self.waypoints))

    def at_goal(self):
        # todo: @Alan, is there a better way to check if the robot is at the goal using the move_base package?
        if self.goal is None:
            return True
        else:
            return np.linalg.norm(np.array(self.goal) - np.array([self.robot_pose.position.x, self.robot_pose.position.y])) < 0.2


if __name__ == '__main__':
    import argparse
    import sys
    import os
    import time
    parser = argparse.ArgumentParser(description='Triton Assistant')
    parser.add_argument('--user_ins', type=str, default="I'm so hungry.")
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--task_id', type=str, default="2")
    args = parser.parse_args()
    ta = TritonAssistant(user_instruction=args.user_ins, debug_mode=args.debug_mode, task_id=args.task_id)
    print("hi there")

    rospy.spin()
