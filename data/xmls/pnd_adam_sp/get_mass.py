import mujoco
import numpy as np

# 加载 MJCF/XML 文件
model = mujoco.MjModel.from_xml_path("scene_mjx_feetonly_flat_terrain.xml")

# 获取每个 body 的质量
body_masses = model.body_mass  # numpy array, 每个 body 的质量 (kg)

# 总质量
total_mass = np.sum(body_masses)

print("Total robot mass =", total_mass, "kg")
