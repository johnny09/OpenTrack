#!/usr/bin/env python3
"""
测试旋转圆柱体模型的简单脚本
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

# 加载模型
model = mujoco.MjModel.from_xml_path("rotating_cylinder.xml")
data = mujoco.MjData(model)

print("模型加载成功！")
print(f"关节数量: {model.njnt}")
print(f"执行器数量: {model.nu}")
print(f"关节名称: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 0)}")

# 打印动力学属性信息
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_body")
if body_id >= 0:
    print(f"\n旋转体动力学属性:")
    print(f"  质量: {model.body_mass[body_id]:.3f} kg")
    print(f"  惯性矩阵对角线: {model.body_inertia[body_id]}")

control_dt = 0.002
model.opt.timestep = control_dt

# 检查执行器类型
actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 0)
actuator_type = model.actuator_trntype[0]
print(f"\n执行器信息:")
print(f"  名称: {actuator_name}")
print(f"  类型: {actuator_type} (0=motor, 1=position, 2=velocity)")

# 初始化状态为0
data.qpos[0] = 0.0
data.qvel[0] = 0.0
mujoco.mj_forward(model, data)  # 更新动力学状态

# 使用查看器运行仿真
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 设置初始控制信号（目标角度，单位：弧度）
    target_angle = 0.0

    # 仿真循环
    last_print_time = 0.0
    while viewer.is_running():
        step_start = time.time()

        # 设置目标角度（正弦波运动）
        target_angle = np.sin(0.1 * data.time) * 1.5  # 在 -1.5 到 1.5 弧度之间摆动
        # target_angle = 0.0

        joint_kp = 100.0
        joint_kd = 20.0
        pos_error = target_angle - data.qpos[0]
        vel_error = 0.0 - data.qvel[0]
        target_tau = joint_kp * pos_error + joint_kd * vel_error
        data.ctrl[0] = 5

        # 执行一步仿真
        mujoco.mj_step(model, data)

        # 同步查看器（以实时速度运行）
        viewer.sync()

        # 每秒打印一次状态信息
        if data.time - last_print_time >= 1.0:
            joint_pos = data.qpos[0]
            joint_vel = data.qvel[0]
            joint_acc = data.qacc[0]
            actuator_force = data.actuator_force[0]
            print(
                f"时间: {data.time:.2f}s | "
                f"目标角度: {target_angle:.3f}rad | "
                f"实际角度: {joint_pos:.3f}rad | "
                f"角速度: {joint_vel:.3f}rad/s | "
                f"角加速度: {joint_acc:.3f}rad/s² | "
                f"执行器力矩: {actuator_force:.3f}N·m"
            )
            last_print_time = data.time

        # 实时控制（限制仿真速度）
        elapsed = time.time() - step_start
        if elapsed < control_dt:
            time.sleep(control_dt - elapsed)
