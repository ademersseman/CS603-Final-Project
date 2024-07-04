#!/usr/bin/python3

# future imports
from __future__ import print_function

import random

import rospy
import time
import json
import os

import numpy as np
import requests
import re
from openai import AzureOpenAI, OpenAI
# from geometry_msgs.msg import PoseStamped
# from move_base_msgs.msg import MoveBaseActionGoal
# rospy.init_node('llm_plan', anonymous=True)


class Planer:
    """
    an LLM planer takes in the map in a scene graph format and the task,
    then selects a waypoint for the robot to go to
    """

    def __init__(
            self,
            max_tokens: int = 512,
            debug_mode: bool = True,
            temperature: float = 0.7,
            top_p: float = 0.95,
            lm_id: str = "gpt-4",
            lm_source: str = "azure",
    ):
        self.robot_pose = None
        self.debug_mode = debug_mode
        self.lm_id = lm_id
        self.lm_source = lm_source
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = 3
        # self.odom_sub = rospy.Subscriber("/amcl_pose", PoseStamped, self.odom_callback)
        # self.nav_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

        if self.lm_source == "openai":
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
                max_retries=self.max_retries,
            )
        elif self.lm_source == "azure":
            self.client = AzureOpenAI(
                azure_endpoint = "https://chuangsweden.openai.azure.com/",
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
                messages = [
                        {"role": "system",
                         "content": "You're a triton robot, who is helpful and can find the best object to meet user needs as fast as possible.\n"
                                "Given the user instruction and the current waypoints of interest, you should select the best waypoint to explore next "
                                "to locate the object that meets the user needs as fast as possible."
                        },
                        {"role": "user",
                         "content": prompt
                        },
                    ]
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

                messages_2 = [
                    {"role": "system",
                     "content": "You're a triton robot, who is helpful and can find the best object to meet user needs as fast as possible.\n"
                                "Given the user instruction and the current waypoints of interest, you should select the best waypoint to explore next "
                                "to locate the object that meets the user needs as fast as possible."
                     },
                    {"role": "user",
                     "content": prompt
                     },
                    {"role": "assistant",
                    "content": response
                    },
                    {"role": "user",
                     "content": "Ok. You've got to the selected waypoint. Is the object that meets the user needs located now? Response with only 'yes' or 'no'"
                     },
                ]
                response_2 = self.client.chat.completions.create(
                    model=lm_id,
                    messages=messages_2,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                with open(f"chat_raw.jsonl", 'a') as f:
                    f.write(response_2.model_dump_json(indent=4))
                    f.write('\n')
                usage_2 = dict(response_2.usage)
                response_2 = response_2.choices[0].message.content
                return response, response_2

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

    def plan(self, pose, user_instruction, waypoints_dict, explored_list):
        # current_position = (self.robot_pose.position.x, self.robot_pose.position.y)
        current_position = (pose.position.x, pose.position.y)
        prompt = f"User Instruction: {user_instruction}\nCurrent Position: {current_position}\nCurrent Waypoints of Interest:\n"
        waypoints = []
        for waypoint, position in waypoints_dict.items():
            print("waypoint: ", waypoint, 'explore_list: ', explored_list)
            if waypoint in explored_list:
                continue
            waypoints.append(f"({position[0]:.2f}, {position[1]:.2f}, {waypoint})")
        for i, waypoint in enumerate(waypoints):
            prompt += f"{chr(65 + i)}. {waypoint}\n"

        prompt += "Please select the best waypoint to explore next to locate the object that meets the user needs as fast as possible."
        # if self.debug_mode:
        print("Prompt: ", prompt)

        response, finished = self.generator(prompt)
        print("Response: ", response)
        print("Finished: ", finished)

        for i, waypoint in enumerate(waypoints):
            option = chr(65 + i) + "."
            if option in response or waypoint in response:
                return waypoint, finished

        print("Warning! No waypoint selected. Randomly select one.")
        random_waypoint = random.choice(waypoints)
        return random_waypoint, finished

    # def odom_callback(self, data):
    #     self.robot_pose = data.pose.pose
    #
    # def send_goal(self, goal):
    #     x, y = goal
    #     pose = PoseStamped()
    #     pose.header.frame_id = "map"
    #     pose.pose.position.x = x
    #     pose.pose.position.y = y
    #     pose.pose.orientation.w = 1.0
    #     self.nav_pub.publish(pose)


if __name__ == "__main__":
    planer = Planer()
    user_instruction = "I'm so thirsty."
    waypoints = ["(-0.6,-0.08,apple)", "(-0.3,-0.5,band-aid)", "(-0.7,0.1,unexplored)", "(-0.3,-0.2,unexplored)"]
    selected_waypoint = planer.plan(user_instruction, waypoints)
    print(f"Selected waypoint: {selected_waypoint}")
    planer.send_goal(selected_waypoint)

