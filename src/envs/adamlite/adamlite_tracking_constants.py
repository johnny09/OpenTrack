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
"""Constants for AdamLite."""

from pathlib import Path
import numpy as np

ROOT_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "xmls" / "pnd_adam_lite"
)
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_flat_terrain.xml"
FEET_ONLY_ROUGH_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_rough_terrain.xml"

NUM_JOINT = 25


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

LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "torso"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# Joint ranges from adam_lite.xml
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
    # Left shoulder.
    (-3.613, 2.042),  # shoulderPitch_Left
    (-0.628, 2.793),  # shoulderRoll_Left
    (-2.583, 2.583),  # shoulderYaw_Left
    (-2.496, 0.209),  # elbow_Left
    (
        -2.583,
        2.583,
    ),  # wristYaw_Left (no range specified in XML, using same as shoulderYaw)
    # Right shoulder.
    (-3.613, 2.042),  # shoulderPitch_Right
    (-2.793, 0.628),  # shoulderRoll_Right
    (-2.583, 2.583),  # shoulderYaw_Right
    (-2.496, 0.209),  # elbow_Right
    (
        -2.583,
        2.583,
    ),  # wristYaw_Right (no range specified in XML, using same as shoulderYaw)
)

# Velocity limits (25 joints total)
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
    # Left arm (5 joints)
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
    # Right arm (5 joints)
    32.0,
    32.0,
    32.0,
    32.0,
    32.0,
]

# Joint names from adam_lite.xml
ACTION_JOINT_NAMES = [
    # left leg
    "hipPitch_Left",
    "hipRoll_Left",
    "hipYaw_Left",
    "kneePitch_Left",
    "anklePitch_Left",
    "ankleRoll_Left",
    # right leg
    "hipPitch_Right",
    "hipRoll_Right",
    "hipYaw_Right",
    "kneePitch_Right",
    "anklePitch_Right",
    "ankleRoll_Right",
    # waist
    "waistRoll",
    "waistPitch",
    "waistYaw",
    # left arm
    "shoulderPitch_Left",
    "shoulderRoll_Left",
    "shoulderYaw_Left",
    "elbow_Left",
    "wristYaw_Left",
    # right arm
    "shoulderPitch_Right",
    "shoulderRoll_Right",
    "shoulderYaw_Right",
    "elbow_Right",
    "wristYaw_Right",
]

OBS_JOINT_NAMES = [
    # left leg
    "hipPitch_Left",
    "hipRoll_Left",
    "hipYaw_Left",
    "kneePitch_Left",
    "anklePitch_Left",
    "ankleRoll_Left",
    # right leg
    "hipPitch_Right",
    "hipRoll_Right",
    "hipYaw_Right",
    "kneePitch_Right",
    "anklePitch_Right",
    "ankleRoll_Right",
    # waist
    "waistRoll",
    "waistPitch",
    "waistYaw",
    # left arm
    "shoulderPitch_Left",
    "shoulderRoll_Left",
    "shoulderYaw_Left",
    "elbow_Left",
    "wristYaw_Left",
    # right arm
    "shoulderPitch_Right",
    "shoulderRoll_Right",
    "shoulderYaw_Right",
    "elbow_Right",
    "wristYaw_Right",
]

# Torque limits from adam_lite.xml actuator ranges
TORQUE_LIMIT = np.array(
    [
        # Left leg
        230.0,
        160.0,
        105.0,
        230.0,
        40.0,
        12.0,
        # Right leg
        230.0,
        160.0,
        105.0,
        230.0,
        40.0,
        12.0,
        # Waist
        110.0,
        110.0,
        110.0,
        # Left arm
        65.0,
        65.0,
        65.0,
        30.0,
        6.4,
        # Right arm
        65.0,
        65.0,
        65.0,
        30.0,
        6.4,
    ]
)

DEFAULT_QPOS = np.float32(
    [
        0,
        0,
        0.8,
        1,
        0,
        0,
        0,
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        0,
        0,
        0,
        0.2,
        0.3,
        0,
        1.28,
        0,
        0,
        0,
        0.2,
        -0.3,
        0,
        1.28,
        0,
        0,
        0,
    ]
)

KPs = np.float32(
    [
        100,
        100,
        100,
        200,
        80,
        20,
        100,
        100,
        100,
        200,
        80,
        20,
        300,
        300,
        300,
        90,
        60,
        20,
        60,
        20,
        20,
        20,
        90,
        60,
        20,
        60,
        20,
        20,
        20,
    ]
)

KDs = np.float32(
    [
        2,
        2,
        2,
        4,
        2,
        1,
        2,
        2,
        2,
        4,
        2,
        1,
        10,
        10,
        10,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
    ]
)

# Body links from adam_lite.xml
UPPER_BODY_LINKs = [
    "shoulderPitchLeft",
    "shoulderRollLeft",
    "shoulderYawLeft",
    "elbowLeft",
    "wristYawLeft",
    "shoulderPitchRight",
    "shoulderRollRight",
    "shoulderYawRight",
    "elbowRight",
    "wristYawRight",
]

LOWER_BODY_LINKs = [
    "pelvis",
    "hipPitchLeft",
    "hipRollLeft",
    "thighLeft",
    "shinLeft",
    "anklePitchLeft",
    "toeLeft",
    "hipPitchRight",
    "hipRollRight",
    "thighRight",
    "shinRight",
    "anklePitchRight",
    "toeRight",
    "waistRoll_link",
    "waistPitch_link",
    "torso",
]

UPPER_BODY_JOINTs = [
    # left arm
    "shoulderPitch_Left",
    "shoulderRoll_Left",
    "shoulderYaw_Left",
    "elbow_Left",
    "wristYaw_Left",
    # right arm
    "shoulderPitch_Right",
    "shoulderRoll_Right",
    "shoulderYaw_Right",
    "elbow_Right",
    "wristYaw_Right",
]

FEET_LINKs = ["toeLeft", "toeRight"]

SHOULDER_LINKs = ["shoulderPitchLeft", "shoulderPitchRight"]


LAFAN1_DATASETS = [
    "dance1_subject1",
    "dance1_subject2",
    "dance1_subject3",
    "dance2_subject1",
    "dance2_subject2",
    "dance2_subject3",
    "dance2_subject4",
    "dance2_subject5",
    "fallAndGetUp1_subject1",
    "fallAndGetUp1_subject5",
    "fallAndGetUp2_subject2",
    "fallAndGetUp3_subject1",
    "fight1_subject2",
    "fight1_subject3",
    "fight1_subject5",
    "fightAndSports1_subject1",
    "fightAndSports1_subject4",
    "jumps1_subject1",
    "jumps1_subject2",
    "jumps1_subject5",
    "run1_subject2",
    "run1_subject5",
    "run2_subject1",
    "run2_subject4",
    "sprint1_subject2",
    "sprint1_subject4",
    "walk1_subject1",
    "walk1_subject2",
    "walk1_subject5",
    "walk2_subject1",
    "walk2_subject4",
    "walk3_subject1",
    "walk3_subject2",
    "walk3_subject3",
    "walk3_subject4",
    "walk3_subject5",
    "walk4_subject1",
]
