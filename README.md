# Project Description

In this project, we address the challenge of
enabling a physical robot to locate objects within its pre-mapped environment given natural language instructions
from a human user. Using a Triton robot, which comes
equipped with 2D LiDAR and a color-depth camera, we
develop an autonomous control system capable of percep-tion, planning, and navigation. Specifically, we utilize a
state-of-the-art hybrid computer vision model that combines
the Recognize Anything Model (RAM) with the Grounded
Segment Anything Model (Grounded-SAM) to perform automatic dense image annotation and object localization.
Secondly, we leverage Large Language Models for object
reference and high-level exploration planning. Lastly, we use
the move_base package from ROS to navigate to locations
specified by the LLM.

[![Demo Youtube Video](https://i.ytimg.com/an_webp/XO_tAWx22Ko/mqdefault_6s.webp?du=3000&sqp=CNiX_pQG&rs=AOn4CLAQA97Ns1yjOhDexaoklIAAUpg8sg)]((https://youtube.com/shorts/3yT4wYLYD8k))
