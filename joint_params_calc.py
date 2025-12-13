import numpy as np

motor_rotor_armature = {
    "PND-130-92-P": 2.74 * 10**-3,
    "PND-130-92-S": 2.74 * 10**-3,
    "PND-80-20-30-S": 2.93 * 10**-4,
    "PND-80-20-50-S": 2.93 * 10**-4,
    "PND-60-17-30-S": 9 * 10**-5,
    "PND-60-17-50-S": 9 * 10**-5,
    "PND-50-52-30-P": 6.1 * 10**-5,
}

actuator_torque_limit = {
    "PND-130-92-P": 113,
    "PND-130-92-S": 150,
    "PND-80-20-30-S": 120,
    "PND-80-20-50-S": 180,
    "PND-60-17-30-S": 55,
    "PND-60-17-50-S": 90,
    "PND-50-52-30-P": 45,
}

joint_actuator_type = {
    "hip_pitch": "PND-130-92-S",
    "hip_roll": "PND-80-20-50-S",
    "hip_yaw": "PND-60-17-30-S",
    "knee": "PND-130-92-S",
    "ankle_pitch": "PND-50-52-30-P",
    "ankle_roll": "PND-50-52-30-P",
    "waist_roll": "PND-60-17-50-S",
    "waist_pitch": "PND-60-17-50-S",
    "waist_yaw": "PND-60-17-50-S",
}

joint_gear_ratio = {
    "hip_pitch": 7,
    "hip_roll": 51,
    "hip_yaw": 31,
    "knee": 7,
    "ankle_pitch": 30,
    "ankle_roll": 30,
    "waist_roll": 51,
    "waist_pitch": 51,
    "waist_yaw": 51,
}

# get joint armature
joint_armature = {}
print("joint_armature:")
for name in joint_actuator_type:
    joint_armature[name] = (
        motor_rotor_armature[joint_actuator_type[name]] * joint_gear_ratio[name] ** 2
    )

    print(f"{name}: {joint_armature[name]}")

# get joint torque limit
joint_torque_limit = {}
print("joint_torque_limit:")
for name in joint_actuator_type:
    joint_torque_limit[name] = actuator_torque_limit[joint_actuator_type[name]]
    print(f"{name}: {joint_torque_limit[name]}")

# get Kp and Kd
omega = 2 * np.pi * 10
damping_ratio = 2.0

for name in joint_actuator_type:
    I = joint_armature[name]
    kp = I * omega**2
    kd = 2 * I * damping_ratio * omega
    print(f"{name}: kp = {kp}, kd = {kd}")
