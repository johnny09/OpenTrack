# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for AdamSP."""

from pathlib import Path
import numpy as np

ROOT_PATH = Path(__file__).parent.parent.parent.parent / "data" / "xmls" / "pnd_adam_sp"
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_flat_terrain.xml"
FEET_ONLY_ROUGH_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_rough_terrain.xml"

NUM_JOINT = 29


def task_to_xml(task_name: str) -> Path:
    return {
        "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
        "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
    }[task_name]


FEET_SITES = [
    "left_foot",
    "right_foot",
]

FEET_ALL_SITES = [
    "left_foot",
    "right_foot",
    "left_foot_top",
    "right_foot_top",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

LEFT_FEET_GEOMS = [
    # "left_foot1_collision",
    # "left_foot2_collision",
    # "left_foot3_collision",
    # "left_foot4_collision",
    # "left_foot5_collision",
    # "left_foot6_collision",
    "left_foot_collision",
]
RIGHT_FEET_GEOMS = [
    # "right_foot1_collision",
    # "right_foot2_collision",
    # "right_foot3_collision",
    # "right_foot4_collision",
    # "right_foot5_collision",
    # "right_foot6_collision",
    "right_foot_collision",
]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# Joint ranges from adam_sp.xml
RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-2.164, 2.164),  # hipPitch_Left
    (-0.733, 1.605),  # hipRoll_Left
    (-0.785, 0.785),  # hipYaw_Left
    (0.052, 2.391),  # kneePitch_Left
    (-1.0, 0.35),  # anklePitch_Left
    (-0.3491, 0.3491),  # ankleRoll_Left
    # Right leg.
    (-2.164, 2.164),  # hipPitch_Right
    (-1.605, 0.733),  # hipRoll_Right
    (-0.785, 0.785),  # hipYaw_Right
    (0.052, 2.391),  # kneePitch_Right
    (-1.0, 0.35),  # anklePitch_Right
    (-0.3491, 0.3491),  # ankleRoll_Right
    # Waist.
    (-0.279, 0.279),  # waistRoll
    (-0.663, 1.361),  # waistPitch
    (-0.829, 0.829),  # waistYaw
    # Left arm.
    (-3.613, 2.042),  # shoulderPitch_Left
    (-0.628, 2.793),  # shoulderRoll_Left
    (-2.583, 2.583),  # shoulderYaw_Left
    (-2.496, 0.209),  # elbow_Left
    (-2.67, 2.67),  # wristYaw_Left
    (-0.96, 0.96),  # wristPitch_Left
    (-0.96, 0.96),  # wristRoll_Left
    # Right arm.
    (-3.613, 2.042),  # shoulderPitch_Right
    (-2.793, 0.628),  # shoulderRoll_Right
    (-2.583, 2.583),  # shoulderYaw_Right
    (-2.496, 0.209),  # elbow_Right
    (-2.67, 2.67),  # wristYaw_Right
    (-0.96, 0.96),  # wristPitch_Right
    (-0.96, 0.96),  # wristRoll_Right
)

# Velocity limits (29 joints total)
DOF_VEL_LIMITS = [
    # Left leg (6 joints)
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    # Right leg (6 joints)
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    # Waist (3 joints)
    32.0,
    32.0,
    32.0,
    # Left arm (7 joints)
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    # Right arm (7 joints)
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
]

# Joint names from adam_sp.xml (G1 format)
ACTION_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_roll_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]

OBS_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_roll_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]

# Torque limits from adam_sp.xml actuator ranges
TORQUE_LIMIT = np.array(
    [
        # Left leg
        230.0,
        160.0,
        105.0,
        280.0,
        80.0,
        80.0,
        # Right leg
        230.0,
        160.0,
        105.0,
        280.0,
        80.0,
        80.0,
        # Waist
        150.0,
        150.0,
        150.0,
        # Left arm
        65.0,
        65.0,
        65.0,
        30.0,
        6.4,
        6.4,
        6.4,
        # Right arm
        65.0,
        65.0,
        65.0,
        30.0,
        6.4,
        6.4,
        6.4,
    ]
)

DEFAULT_QPOS = np.float32(
    [
        # root pos
        0,
        0,
        0.92,
        # root quat
        1,
        0,
        0,
        0,
        # left leg
        -0.1499,
        -0.00393073,
        -0.0859305,
        0.313304,
        -0.190327,
        -0.0013629,
        # right leg
        -0.1499,
        0.00393073,
        0.0859305,
        0.313304,
        -0.190327,
        0.0013629,
        # waist
        0,
        0,
        0,
        # left arm
        0.0,
        0.1,
        0,
        -0.3,
        0,
        0,
        0,
        # right arm
        0.0,
        -0.1,
        0,
        -0.3,
        0,
        0,
        0,
    ]
)

KPs = np.float32(
    [
        # left leg
        305.0,
        700.0,
        405,
        305.0,
        20.0,
        20.0,
        # right leg
        305.0,
        700.0,
        405,
        305.0,
        20.0,
        20.0,
        # waist
        405.0,
        405.0,
        205.0,
        # left arm
        18.0,
        9.0,
        9.0,
        9.0,
        9.0,
        9.0,
        9.0,
        # right arm
        18.0,
        9.0,
        9.0,
        9.0,
        9.0,
        9.0,
        9.0,
    ]
)

KDs = np.float32(
    [
        # left leg
        6.1,
        30,
        6.1,
        6.1,
        2.5,
        2.5,
        # right leg
        6.1,
        30,
        6.1,
        6.1,
        2.5,
        2.5,
        # waist
        6.1,
        6.1,
        4.1,
        # left arm
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        # right arm
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
    ]
)

# Body links from adam_sp.xml (G1 format)
UPPER_BODY_LINKs = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
]

LOWER_BODY_LINKs = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_roll_link",
    "waist_pitch_link",
    "torso_link",
]

UPPER_BODY_JOINTs = [
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]

FEET_LINKs = ["left_ankle_roll_link", "right_ankle_roll_link"]

SHOULDER_LINKs = ["left_shoulder_pitch_link", "right_shoulder_pitch_link"]


LAFAN1_DATASETS = [
    # "run1_subject2_extended",  # before 15%
    # "run1_subject5_extended",
    # "run2_subject1_extended",
    # "run2_subject4_extended",
    # "walk1_subject1_extended",
    # "walk1_subject2_extended",
    # "walk1_subject5_extended",
    # "walk2_subject1_extended",
    # "walk2_subject3_extended",  # ok
    # "walk2_subject4_extended",
    # "walk3_subject1_extended",
    # "walk3_subject2_extended",
    # "walk3_subject3_extended",
    # "walk3_subject4_extended",
    # "walk3_subject5_extended",
    # "walk4_subject1_extended",
    # "dance1_subject1_extended",
    # "dance1_subject2_extended",
    # "dance1_subject3_extended",  # ok
    # "dance2_subject1_extended",
    # "dance2_subject2_extended",
    # "dance2_subject3_extended",
    # "dance2_subject4_extended",
    # "dance2_subject5_extended",  # ok
]
