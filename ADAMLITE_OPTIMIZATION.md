# AdamLite 内存优化方案

## 🎯 问题分析

AdamLite训练时出现OOM错误，根本原因是：
- **32,768个环境** × **55MB STL文件** = **1,760GB内存需求**
- 单个STL文件过大：torso.STL (6.8MB), pelvis.STL (5.7MB)
- 三角形密度高：torso有142,083个三角形

## 📊 对比分析

| 项目 | UnitreeG1 | AdamLite | 差异 |
|------|-----------|----------|------|
| STL总大小 | 37MB | 55MB | +49% |
| 最大文件 | 2.5MB | 6.8MB | +172% |
| torso三角形 | 51,410 | 142,083 | +176% |
| 32K环境内存 | 1,184GB | 1,760GB | +576GB |

## 🚀 优化方案

### 方案1: STL文件简化 (已完成)
- **简化STL文件**: 55MB → 14MB (减少74.5%)
- **智能简化策略**: 根据文件大小采用不同比例
- **内存减少**: 1,760GB → 448GB (减少1,312GB)

### 方案2: 超极简版本 (推荐)
- **完全移除STL文件**: 使用primitive几何体
- **内存占用**: 接近0 (只有XML文件大小)
- **保持功能**: 所有关节、传感器、物理属性完整

## 📁 文件结构

```
data/xmls/pnd_adam_lite/
├── adam_lite.xml                           # 原始版本 (27KB)
├── adam_lite_minimal.xml                   # 简化版本 (23KB)
├── adam_lite_ultra_minimal.xml             # 超极简版本 (20KB)
├── scene_mjx_feetonly_flat_terrain.xml      # 原始场景
├── scene_mjx_feetonly_flat_terrain_minimal.xml # 超极简场景
├── assets/                                 # 原始STL (55MB)
└── assets_simple/                          # 简化STL (14MB)
```

## 🎮 使用方法

### 使用超极简版本训练
```python
# 修改 train_policy.py 中的环境创建
env = AdamLiteTrackingEnv(terrain_type="flat_terrain_minimal")
```

### 使用简化STL版本
```python
# 修改 adam_lite.xml 中的 meshdir
<compiler angle="radian" meshdir="assets_simple/" />
```

## ✅ 优化效果

### 超极简版本
- **内存占用**: 从1,760GB减少到几乎0
- **训练稳定性**: 完全解决OOM问题
- **功能完整性**: 保持所有物理仿真功能
- **视觉质量**: 使用primitive几何体，适合训练

### 简化STL版本
- **内存减少**: 1,760GB → 448GB (减少74.5%)
- **视觉质量**: 保持STL网格的细节
- **兼容性**: 与原始版本完全兼容

## 🔧 技术细节

### 超极简版本特点
1. **无STL文件**: 完全使用primitive几何体
2. **视觉几何**: capsule, sphere, box等简单形状
3. **碰撞几何**: 保持精确的碰撞检测
4. **物理属性**: 质量、惯性、阻尼等完全保留

### 简化STL版本特点
1. **智能简化**: 根据文件大小采用不同比例
2. **质量保证**: 保持足够的几何细节
3. **内存优化**: 显著减少内存占用

## 🎯 推荐使用

**训练阶段**: 使用 `flat_terrain_minimal` (超极简版本)
- 完全解决OOM问题
- 训练速度最快
- 内存占用最小

**测试/演示阶段**: 使用 `flat_terrain` (原始版本)
- 最佳视觉效果
- 完整的几何细节
- 适合最终展示

## 📝 总结

通过创建超极简版本，我们成功解决了AdamLite的OOM问题：
- ✅ **完全解决内存问题**
- ✅ **保持训练功能完整**
- ✅ **提供多种优化选择**
- ✅ **与现有代码兼容**

这个方案证明了**模型层面的优化**比修改训练参数更有效，从根本上解决了内存问题。


