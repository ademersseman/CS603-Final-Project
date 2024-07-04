#!/usr/bin/python3

# future imports
from __future__ import print_function

# basic imports
import os
import argparse
import time
import numpy as np
import random

# API imports
import replicate
from openai import AzureOpenAI, OpenAI

# ROS imports
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
import cv2

# initialize the node
rospy.init_node('triton_assistant', anonymous=True)


def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw (z-axis rotation) in radians."""

    yaw = (euler_from_quaternion([
        p.orientation.x,
        p.orientation.y,
        p.orientation.z,
        p.orientation.w])
    [2])

    return yaw


def strip_waypoint_astext_for_name(waypoint):
    x, y, name = waypoint.strip("()").split(", ")
    return name


class Waypoint:
    def __init__(self, name, x, y, z=0.0, w=1.0, step=99):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.step = step
        self.explored = False

    def __repr__(self):
        return f"Waypoint(Name={self.name}: Position=(x={self.x:.2f}, y={self.y:.2f}), " \
               f"Orientation=(z={self.z:.2f}, w={self.w:.2f}), Explored={self.explored}, Step={self.step})"


class PerceptionModule:

    def __init__(self, debug_mode=False):
        # static variables
        self.debug_mode = debug_mode

        # dynamic variables
        self.ram_gsam_output = None
        self.object_locations = None
        self.api_time = 0

    def run_ramgsam(self, rgb_image_path):
        """
        This function runs the RAM-Grounded-SAM model on a RGB image stored in the given file path.

        INPUTS:
            rgb_image_path: str, the path to the RGB image file.
            debug_mode: bool, whether to print debug information.

        OUTPUTS:
            ram_gsam_output: dict, the output of the RAM-Grounded-SAM model. Contains the detected objects, their
                bounding boxes, and the probability of the object detection.

        Example Output
        {'json_data':
            {'mask': [{'label': 'background', 'value': 0},
                      {'box': [0.01312255859375, 0.2370758056640625, 640.0107421875, 479.97027587890625],
                            'label': 'room', 'logit': 0.45, 'value': 1},
                      {'box': [0.10516357421875, 274.33544921875, 640.0984497070312, 479.7895202636719],
                            'label': 'floor', 'logit': 0.42, 'value': 2},
                      {'box': [-0.00220489501953125, 201.869384765625, 145.75796508789062, 361.15313720703125],
                            'label': 'pillow', 'logit': 0.37, 'value': 3}],
            'tags': 'lamp, chip, dark, floor, light, pillow, room'}, 'masked_img': None,
            'rounding_box_img': None, 'tags': 'lamp, chip, dark, floor, light, pillow, room'}}
        """
        if self.debug_mode:
            return None
        api_start_time = time.time()

        print("Running API, starting time: ", api_start_time)

        ram_gsam_output = replicate.run(
            "idea-research/ram-grounded-sam:80a2aede4cf8e3c9f26e96c308d45b23c350dd36f1c381de790715007f1ac0ad",
            input={
                "use_sam_hq": False,
                "input_image": open(rgb_image_path, "rb"),
                "show_visualisation": False
            }
        )
        api_finished_time = time.time()
        self.api_time += api_finished_time - api_start_time
        del ram_gsam_output['json_data']['mask'][0]  # remove the background object

        print("API finished running")
        if self.debug_mode:
            print("Model output: ", ram_gsam_output)

        return ram_gsam_output

    def boxes_to_positions(self, ram_gsam_output, depth_data, robot_pose, probs_thresh=0.3):
        """
        Convert the bounding boxes of each object to an objects (x,y) location in 3D space.

        INPUTS:
            - ram_gsam_output: dict with key to a list of dicts (objects), the output of the RAM-Grounded-SAM model. \
              Contains the bounding boxes of each object (['box']) stored as [xmin, ymin, xmax, ymax], the class of \
              the object (['name']), and the probability of the object detection (['logit']).
            - depth_data: np.array, the depth data from the camera. Used to determine the distance from the robot's \
              location
            - robot_pose: PoseStamped, the pose of the robot. Used to calculate the 3D location of the object.
            - probs_thresh: float, the probability threshold for the object detection model. Any object with a logit \
              value below this threshold will be discarded from the output. (default: 0.3)

        OUTPUTS:
            - objects_locations: list of tuples, each tuple contains the (x,y) location of an object. Tuples are in the \
              form (x, y, object_name)
        """

        objects_locations = []

        for obj in ram_gsam_output['json_data']['mask']:
            if obj['logit'] < probs_thresh:
                continue

            box = obj['box']  # Get the bounding box coordinates [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Extract depth values within the bounding box region
            depth_values = depth_data[ymin:ymax, xmin:xmax]

            # Calculate the average depth value within the bounding box region
            avg_depth_value = np.mean(depth_values) / 1000  # Convert mm to meters

            """
            Calculate the 3D coordinates of the object by using the robot's position, orientation
            and the average depth value of the object.
            """

            # Convert quaternion to theta
            theta = get_yaw_from_pose(robot_pose)
            # Calculate the x and y coordinates of the object
            x = robot_pose.position.x + avg_depth_value * np.cos(theta)
            y = robot_pose.position.y + avg_depth_value * np.sin(theta)

            if self.debug_mode:
                print(f"Robot pose: {robot_pose} \t Avg depth value: {avg_depth_value} \t Theta: {theta} \t "
                      f"Position: (x={x:.2f},y={y:.2f}) \t Object name: {obj['label']}")

            # Append the 3D coordinates of the object to the list
            objects_locations.append((x, y, obj["label"]))

        return objects_locations

    def run_perception(self, rgb_image_path, depth_image, robot_pose):
        # run the ram-grounded-sam replicate api
        self.ram_gsam_output = self.run_ramgsam(rgb_image_path)
        # get the object locations
        self.object_locations = self.boxes_to_positions(self.ram_gsam_output, depth_image, robot_pose)

        return self.object_locations


class PlanningModule:
    """
    an LLM planner takes in the map in a scene graph format and the task,
    then selects a waypoint for the robot to go to
    """

    def __init__(self, max_tokens: int = 512, user_instruction = "", debug_mode: bool = True, temperature: float = 0.7, top_p: float = 0.95,
                 lm_id: str = "gpt-4", lm_source: str = "azure"):
        # static variables
        self.user_instruction = user_instruction
        self.debug_mode = debug_mode
        self.lm_id = lm_id
        self.lm_source = lm_source
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = 3

        # dynamic variables
        self.api_time = 0
        self.llm_response_finished = 'no'
        self.nav_goal_name = None

        if self.lm_source == "openai":
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
                max_retries=self.max_retries,
            )
        elif self.lm_source == "azure":
            self.client = AzureOpenAI(
                azure_endpoint="https://chuangsweden.openai.azure.com/",
                api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
                api_version="2024-02-15-preview",
                max_retries=self.max_retries,
            )
        elif self.lm_source == "huggingface":
            # self.client = AutoModelForCausalLM.from_pretrained(self.lm_id)
            pass
        else:
            raise NotImplementedError(f"{self.lm_source} is not supported!")

        def lm_engine(source, lm_id):

            def openai_generate(prompt):
                messages = prompt
                response = self.client.chat.completions.create(
                    model=lm_id,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                with open(f"chat_raw.jsonl", 'a') as f:
                    f.write(response.model_dump_json(indent=4))
                    f.write('\n')
                usage = dict(response.usage)
                response = response.choices[0].message.content
                if self.debug_mode:
                    print('======= prompt ======= \n ', messages)
                    print('======= response ======= \n ', response)
                    print('======= usage ======= \n ', usage)

                return response

            def _generate(prompt):
                if source == 'openai':
                    return openai_generate(prompt)
                elif source == 'azure':
                    return openai_generate(prompt)
                elif source == 'huggingface':
                    raise ValueError("huggingface is not supported!")
                else:
                    raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.lm_source, self.lm_id)

    def termination_check(self, nav_goal_name):
        prompt = [
            {"role": "system",
             "content": "You're a triton robot, who is helpful and can find the best object to meet user needs as fast as possible.\n"
                        "Given the user instruction and the current waypoint your are at, you should determine whether the object that meets the user needs is located now."
             },
            {"role": "user",
             "content": f"User Instruction: {self.user_instruction}\nCurrent waypoint: {nav_goal_name}\nResponse with yes or no only, do not include other information."
             },
        ]
        api_start_time = time.time()
        response = self.generator(prompt)
        api_finished_time = time.time()
        self.api_time += api_finished_time - api_start_time
        finished_words = response.lower().split(' ')
        finished_words = [word.strip().strip('.') for word in finished_words]
        if 'yes' in finished_words and 'no' not in finished_words:
            return True
        return False

    def plan(self, pose, waypoints_dict):
        if self.debug_mode:
            print("waypoint: ", waypoints_dict)

        current_position = (pose.position.x, pose.position.y)
        user_prompt = f"User Instruction: {self.user_instruction}\nCurrent Position: {current_position}\nCurrent Waypoints of Interest:\n"
        waypoints = []
        for waypoint_name, waypoint in waypoints_dict.items():
            if waypoint.explored:
                continue
            waypoints.append(f"({waypoint.x:.2f}, {waypoint.y:.2f}, {waypoint_name})")
        for i, waypoint in enumerate(waypoints):
            user_prompt += f"{chr(65 + i)}. {waypoint}\n"

        user_prompt += ("Please select the best waypoint to explore next to locate the object that meets the user needs as fast as possible. "
                   "Start by typing the letter corresponding to the waypoint you want to explore next.")
        # if self.debug_mode:
        print("Prompt: ", user_prompt)

        prompt = [
                    {"role": "system",
                     "content": "You're a triton robot, who is helpful and can find the best object to meet user needs as fast as possible.\n"
                                "Given the user instruction and the current waypoints of interest, you should select the best waypoint to explore next "
                                "to locate the object that meets the user needs as fast as possible."
                     },
                    {"role": "user",
                     "content": user_prompt
                     },
                ]
        api_start_time = time.time()

        response = self.generator(prompt)

        api_finished_time = time.time()
        self.api_time += api_finished_time - api_start_time
        print("Response: ", response)

        selected_waypoint = None
        first_word = response.split(' ')[0].strip().strip('.').strip('!')
        if len(first_word) == 1 and first_word.isalpha() and first_word.isupper():
            selected_waypoint = waypoints[ord(first_word) - ord('A')]
        else:
            response = response.lower()

            for i, waypoint in enumerate(waypoints):
                option = chr(65 + i + 32) + "."  # "a."
                option_2 = "waypoint " + chr(65 + i + 32)  # "waypoint a"
                if option in response or option_2 in response or waypoint in response :
                    selected_waypoint = waypoint
                    break

        if waypoints is []:
            # send a warning message to the user
            print("No waypoints available to select from!")
            return None

        if selected_waypoint is None:
            print("Warning! No waypoint selected. Randomly select one.")
            selected_waypoint = random.choice(waypoints)

        return strip_waypoint_astext_for_name(selected_waypoint)


class MotionModule:

    def __init__(self, debug_mode=False):
        # static variables
        self.nav_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.debug_mode = debug_mode

        # dynamic variables
        self.nav_goal_pose = None

    def run_motion(self, robot_pose, waypoint):
        """
        Send a goal pose to the move_base package to navigate the robot to a specific location.

        INPUTS:
            - robot_pose: PoseStamped, the current pose of the robot.
            - waypoint: Waypoint, the pose of the target the robot needs to navigate to.
        """
        x, y, z, w = waypoint.x, waypoint.y, waypoint.z, waypoint.w
        # compute the difference between the robot's current pose and the goal pose
        pose_diff = np.array([x, y]) - np.array([robot_pose.position.x, robot_pose.position.y])

        # set the goal pose to be 90% of the way to the goal
        self.nav_goal_pose = np.array([robot_pose.position.x, robot_pose.position.y]) + pose_diff * 1  # TODO: tune

        print(f"Motion module started, goal: {self.nav_goal_pose}")
        if self.debug_mode:
            return None
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = self.nav_goal_pose[0]
        pose.pose.position.y = self.nav_goal_pose[1]
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w

        self.nav_pub.publish(pose)

    def at_goal(self, robot_pose):
        if self.nav_goal_pose is None:
            return True
        else:
            return np.linalg.norm(
                np.array(self.nav_goal_pose) - np.array([robot_pose.position.x, robot_pose.position.y])) < 0.2


class TritonAssistant:

    def __init__(self, user_instruction=None, debug_mode=False, task_id="1"):
        # static variables
        self.debug_mode = debug_mode
        self.bridge = CvBridge()
        self.start_time = time.time()
        self.task_id = 'task/' + task_id
        self.user_instruction = user_instruction

        # modules
        self.perception = PerceptionModule(debug_mode=debug_mode)
        self.planner = PlanningModule(user_instruction=user_instruction, debug_mode=debug_mode)
        self.motion = MotionModule()

        # dynamic sensor variables
        self.image_data = None
        self.depth_data = None
        self.robot_pose = None
        # all callbacks must only update the corresponding data once per loop. So if we update the image data, we must
        # wait for the perception loop to complete before updating the image data again.
        self.open_image_callback = True
        self.open_depth_callback = True
        self.open_odom_callback = True  # TODO: not being used
        self.localizing = False

        # subcribers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.odom_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.odom_callback)

        # dynamic variables
        self.ram_gsam_output = None
        self.nav_goal_name = None
        self.llm_response_finished = 'no'
        self.api_time = 0
        self.step = 0
        os.makedirs(f'{self.task_id}/{str(self.step)}', exist_ok=True)

        # fixed waypoints
        # TODO: read this info from the map file
        # 1 is by door, 2 is by fridge, 3 is under sink, 4 is by trash can
        self.waypoints = {"unexplored_1": Waypoint("unexplored_1", x=-2.2, y=-1.5, z=0.38, w=0.92, step=99),
                          "unexplored_2": Waypoint("unexplored_2", x=-2.2, y=-0.23, z=0.14, w=1, step=99),
                          "unexplored_3": Waypoint("unexplored_3", x=-0.15, y=0.72, z=-0.71, w=0.7, step=99),
                          "unexplored_4": Waypoint("unexplored_4", x=0, y=-1, z=0.73, w=0.67, step=99)}

        if self.debug_mode:
            init_pose = PoseWithCovarianceStamped()
            init_pose.pose.pose.position.x = -1.2098531724405264
            init_pose.pose.pose.position.y = -0.5500910593187853
            init_pose.pose.pose.orientation.w = 1.0
            self.robot_pose = init_pose.pose
            self.open_odom_callback = False
            self.waypoints = {'unexplored_1': Waypoint("unexplored_1", x=-0.1, y=0.72, z=-0.87, w=0.49, step=99),
                              'unexplored_2': Waypoint("unexplored_2", x=-1.88, y=-1.37, z=0.41, w=0.91, step=99),
                              'coke_can': Waypoint("coke_can",x=-0.6, y=-0.08, step=self.step),
                              'band-aid': Waypoint("band-aid", x=-0.3, y=-0.5, step=self.step)}
            # self.send_goal(0.70, 0.35, 0.39, 0.92)

    def image_callback(self, data):
        # If the image callback is open, we can update the image data. If not, we must wait for the perception-planning
        # loop to finish before updating the image data again.
        if self.open_image_callback and not self.open_depth_callback:
            print("Image callback received")
            self.open_image_callback = False
            self.image_data = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # print(f"saved image to {self.task_id}/camera_image.jpg")
            # if os.path.exists(os.path.join(self.task_id, 'camera_image.jpg')):
            #     os.remove(os.path.join(self.task_id, 'camera_image.jpg'))
            cv2.imwrite(f'{self.task_id}/{str(self.step)}/camera_image.jpg', self.image_data)

        # if all callbacks are closed, we can run the perception-planning loop
        if not self.localizing and not self.open_image_callback and not self.open_depth_callback and not self.open_odom_callback:

            if False:
                # run the perception module
                object_locations = self.perception.run_perception(f'{self.task_id}/{str(self.step)}/camera_image.jpg',
                                                                  self.depth_data, self.robot_pose)
                # update the waypoints with the new object locations
                self.update_waypoints(object_locations)

            # run the planning and motion module. If the robot arrived at its last goal pose, we plan a new goal pose
            if self.motion.at_goal(robot_pose=self.robot_pose):
                # check if the user is satisfied with the object found
                if self.planner.termination_check(self.nav_goal_name):
                    self.terminate()
                else:
                    print("Planning new goal")
                # run the planning module
                self.nav_goal_name = self.planner.plan(self.robot_pose, self.waypoints)
                # Set the explored flag to True for the newly selected waypoint, so we don't select it again
                self.waypoints[self.nav_goal_name].explored = True
                # run the motion module
                self.motion.run_motion(self.robot_pose, self.waypoints[self.nav_goal_name])

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
    def update_waypoints(self, object_locations):
        """
        TODO: fill in description
        """
        for obj in object_locations:
            x, y, name = obj
            if any([i in name for i in ['apple', 'coke', 'can', 'soda', 'pencil', 'beverage']]):
                if name not in self.waypoints:
                    self.waypoints[name] = Waypoint(name, x, y, step=self.step)
                    print(f"Object {name} added to waypoints list at position: (x={x:.2f},y={y:.2f})")
                else:
                    print(f"Object {name} already in waypoints list, "
                          f"original position: (x={self.waypoints[name].x:.2f},y={self.waypoints[name].y:.2f})"
                          f"new position: (x={x:.2f},y={y:.2f})")
                    self.waypoints[name].x, self.waypoints[name].y, self.waypoints[name].step = x, y, self.step

        if len(self.waypoints) > 15:
            self.waypoints = dict(sorted(self.waypoints.items(), key=lambda w: w[1].step, reverse=True)[:10])
            print(f"Waypoints length {len(self.waypoints)}, removing oldest waypoints")

        with open(f'{self.task_id}/waypoints.txt', 'w') as f:
            f.write(str(self.waypoints))

        if self.debug_mode:
            print(self.waypoints)

    def terminate(self):
        self.api_time += self.perception.api_time + self.planner.api_time
        finished_time = time.time()
        total_time = finished_time - self.start_time
        os.makedirs(f"{self.task_id}/final", exist_ok=True)
        with open(f'{self.task_id}/final/statics.txt', 'w') as f:
            f.write(f"Finished task, found object {self.nav_goal_name}, my final pose is at "
                    f"{self.robot_pose}, total time used {total_time}, api time used {self.api_time}, "
                    f"effective time used {total_time - self.api_time}.")
        print(f"Finished task, found object {self.nav_goal_name}, my pose is {self.robot_pose}, "
              f"total time used {total_time}, api time used {self.api_time}, "
              f"effective time used {total_time - self.api_time}.")
        cv2.imwrite(os.path.join(self.task_id, 'final', 'camera_image.jpg'), self.image_data)
        rospy.signal_shutdown("Finished task")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Triton Assistant')
    parser.add_argument('--user_ins', type=str, default="I'm so hungry.")
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--task_id', type=str, default="2")
    args = parser.parse_args()
    ta = TritonAssistant(user_instruction=args.user_ins, debug_mode=args.debug_mode, task_id=args.task_id)
    print("Running Triton Assistant...")


    # planner = PlanningModule(debug_mode=True)
    # planner.user_instruction = "I'm so hungry"
    # init_pose = PoseWithCovarianceStamped()
    # init_pose.pose.pose.position.x = -1.2098531724405264
    # init_pose.pose.pose.position.y = -0.5500910593187853
    # init_pose.pose.pose.orientation.w = 1.0
    # waypoints = {'unexplored_1': Waypoint("unexplored_1", x=-0.1, y=0.72, z=-0.87, w=0.49, step=99),
    #                   'unexplored_2': Waypoint("unexplored_2", x=-1.88, y=-1.37, z=0.41, w=0.91, step=99),
    #                   'coke_can': Waypoint("coke_can", x=-0.6, y=-0.08, step=0),
    #                   'band-aid': Waypoint("band-aid", x=-0.3, y=-0.5, step=0)}
    # waypoint = planner.plan(init_pose.pose.pose, waypoints)
    # print(waypoint)
    # print(planner.termination_check(waypoint))

    rospy.spin()
