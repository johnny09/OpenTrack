#!/usr/bin/env python3
"""
诊断抖动问题的脚本 - 记录状态数据用于分析
"""
import mujoco
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
model = mujoco.MjModel.from_xml_path("rotating_cylinder.xml")
data = mujoco.MjData(model)

# 初始化
data.qpos[0] = 0.0
data.qvel[0] = 0.0
mujoco.mj_forward(model, data)

# 记录数据
times = []
positions = []
velocities = []
accelerations = []
torques = []
targets = []

# 仿真参数
dt = 0.002
target_angle = 0.0
num_steps = 5000  # 10秒的数据

print("开始记录数据...")
for i in range(num_steps):
    # 设置控制
    actuator_type = model.actuator_trntype[0]
    if actuator_type == 1:  # position actuator
        data.ctrl[0] = target_angle
    else:  # motor actuator
        kp, kd = 100.0, 20.0
        pos_error = target_angle - data.qpos[0]
        vel_error = 0.0 - data.qvel[0]
        target_tau = kp * pos_error + kd * vel_error
        data.ctrl[0] = np.clip(target_tau, -50.0, 50.0)
    
    # 记录数据
    times.append(data.time)
    positions.append(data.qpos[0])
    velocities.append(data.qvel[0])
    accelerations.append(data.qacc[0])
    torques.append(data.actuator_force[0])
    targets.append(target_angle)
    
    # 执行一步
    mujoco.mj_step(model, data)

# 转换为numpy数组
times = np.array(times)
positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)
torques = np.array(torques)

# 分析
print("\n=== 诊断结果 ===")
print(f"位置统计:")
print(f"  均值: {np.mean(positions):.6f} rad")
print(f"  标准差: {np.std(positions):.6f} rad")
print(f"  最大值: {np.max(np.abs(positions)):.6f} rad")
print(f"  范围: [{np.min(positions):.6f}, {np.max(positions):.6f}] rad")

print(f"\n角速度统计:")
print(f"  均值: {np.mean(velocities):.6f} rad/s")
print(f"  标准差: {np.std(velocities):.6f} rad/s")
print(f"  最大值: {np.max(np.abs(velocities)):.6f} rad/s")

print(f"\n力矩统计:")
print(f"  均值: {np.mean(torques):.6f} N·m")
print(f"  标准差: {np.std(torques):.6f} N·m")
print(f"  最大值: {np.max(np.abs(torques)):.6f} N·m")

# 计算抖动频率（如果有周期性）
if len(positions) > 100:
    # FFT分析
    fft = np.fft.fft(positions)
    freqs = np.fft.fftfreq(len(positions), dt)
    power = np.abs(fft) ** 2
    
    # 找到主要频率（排除DC分量）
    idx = np.argsort(power[1:len(power)//2])[-5:]
    print(f"\n主要抖动频率:")
    for i in idx:
        freq = freqs[i+1]
        if freq > 0:
            print(f"  {freq:.2f} Hz (功率: {power[i+1]:.2e})")

# 绘制图表
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(times, positions * 1000, 'b-', linewidth=0.5)
axes[0].set_ylabel('位置 (mrad)')
axes[0].set_title('关节位置 vs 时间')
axes[0].grid(True)

axes[1].plot(times, velocities * 1000, 'r-', linewidth=0.5)
axes[1].set_ylabel('角速度 (mrad/s)')
axes[1].set_title('关节角速度 vs 时间')
axes[1].grid(True)

axes[2].plot(times, accelerations, 'g-', linewidth=0.5)
axes[2].set_ylabel('角加速度 (rad/s²)')
axes[2].set_title('关节角加速度 vs 时间')
axes[2].grid(True)

axes[3].plot(times, torques, 'm-', linewidth=0.5)
axes[3].set_ylabel('力矩 (N·m)')
axes[3].set_xlabel('时间 (s)')
axes[3].set_title('执行器力矩 vs 时间')
axes[3].grid(True)

plt.tight_layout()
plt.savefig('shake_diagnosis.png', dpi=150)
print(f"\n图表已保存到 shake_diagnosis.png")

# 判断是否抖动
pos_std = np.std(positions)
if pos_std > 0.001:  # 1 mrad
    print(f"\n⚠️  检测到抖动！位置标准差: {pos_std*1000:.3f} mrad")
    print("建议:")
    print("  1. 增加关节阻尼 (damping)")
    print("  2. 增加PD控制器的kd参数")
    print("  3. 使用RK4积分器")
    print("  4. 检查是否有数值精度问题")
else:
    print(f"\n✓ 系统稳定，位置标准差: {pos_std*1000:.3f} mrad")

