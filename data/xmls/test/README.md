# 旋转圆柱体 MuJoCo 模型

这是一个简单的 MuJoCo 模型示例，包含一个旋转关节和一个圆柱体。

## 文件说明

- `rotating_cylinder.xml`: MuJoCo XML 模型文件
- `test_rotating_cylinder.py`: 测试脚本，演示如何使用模型

## 模型结构

1. **基座 (base)**: 固定的长方体基座
2. **旋转关节 (rotation_joint)**: 绕 Z 轴旋转的铰链关节
3. **圆柱体 (cylinder)**: 由关节驱动的圆柱体几何体
4. **执行器 (rotation_actuator)**: 位置控制执行器，驱动关节旋转

## 使用方法

### 1. 使用 MuJoCo 查看器

```bash
# 在 data/xmls/test 目录下运行
mujoco-viewer rotating_cylinder.xml
```

### 2. 使用 Python 脚本

```bash
cd data/xmls/test
python test_rotating_cylinder.py
```

脚本会：
- 加载模型
- 打开交互式查看器
- 使用正弦波控制信号驱动圆柱体旋转

### 3. 在代码中使用

```python
import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("rotating_cylinder.xml")
data = mujoco.MjData(model)

# 设置控制信号（目标角度，单位：弧度）
data.ctrl[0] = 1.0  # 旋转到 1 弧度

# 运行仿真
for _ in range(1000):
    mujoco.mj_step(model, data)
    print(f"关节角度: {data.qpos[0]:.3f} rad")
```

## 模型参数

- **关节类型**: `hinge` (旋转关节)
- **旋转轴**: Z 轴 (0, 0, 1)
- **角度范围**: -π 到 π 弧度
- **执行器类型**: 位置控制
- **执行器增益**: kp=100, kv=10
- **力范围**: -50 到 50 N·m

## 修改建议

### 改变旋转轴

修改 `joint` 的 `axis` 属性：
```xml
<!-- 绕 Y 轴旋转 -->
<joint name="rotation_joint" type="hinge" axis="0 1 0" />

<!-- 绕 X 轴旋转 -->
<joint name="rotation_joint" type="hinge" axis="1 0 0" />
```

### 改变圆柱体尺寸

修改 `geom` 的 `size` 属性：
```xml
<!-- size="半径 高度" -->
<geom name="cylinder" type="cylinder" size="0.3 0.5" />
```

### 使用速度控制

取消注释速度控制执行器，并注释位置控制：
```xml
<actuator>
  <velocity name="rotation_velocity" joint="rotation_joint" kv="10" forcerange="-50 50" />
</actuator>
```

## 传感器

模型包含以下传感器：
- `joint_position`: 关节位置（角度）
- `joint_velocity`: 关节速度（角速度）

可以通过 `data.sensor()` 访问传感器数据。

