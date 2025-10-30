# jax2onnx.py 使用指南

`jax2onnx.py` 是一个将 JAX/Brax 训练的模型转换为 ONNX 格式的工具。

## 功能

这个脚本会：
1. 从实验检查点加载 JAX/Brax 模型
2. 将 JAX 权重转换为 PyTorch 格式
3. 导出为 ONNX 格式
4. 验证 ONNX 模型与原始 JAX 模型的一致性

## 安装依赖

```bash
pip install torch onnx onnxruntime
```

## 基本用法

```bash
python jax2onnx.py --exp_name 10220959_flat_terrain
```

这会：
- 加载实验 `10220959_flat_terrain` 的最新检查点
- 将模型转换为 ONNX 格式
- 保存到 `experiments/10220959_flat_terrain/checkpoints/<latest>/policy.onnx`

## 指定 ONNX Opset 版本

```bash
python jax2onnx.py --exp_name 10220959_flat_terrain --opset-version 13
```

### Opset 版本说明

- **11** (默认)：兼容 TensorRT 7.x 和较老的 ONNX Runtime
- **13-14**：新版本 ONNX 标准
- **17+**：最新版本

选择取决于你的部署环境：
- TensorRT 7.x → opset 11
- TensorRT 8.x+ → opset 13-14
- ONNXRuntime → opset 13 或更高

## 参数

- `exp_name` (必需)：实验名称，对应 `experiments/` 目录下的文件夹名
- `opset_version` (可选，默认 11)：ONNX opset 版本

## 示例

### 转换 G1 机器人模型

```bash
python jax2onnx.py --exp_name 10220959_flat_terrain
```

### 转换 Adam SP 机器人模型

```bash
python jax2onnx.py --exp_name 10241839_adam_sp_flat_terrain --opset-version 13
```

## 输出

转换成功后，你会在终端看到：

```
✓ Successfully exported model to ONNX!
Model exported to ONNX format: experiments/xxx/checkpoints/xxx/policy.onnx
✓ ONNX model validation passed!
```

以及模型输出对比：

```
Predictions:
    JAX : [0.12, 0.45, -0.23, ...]
    ONNX: [0.12, 0.45, -0.23, ...]

Mean absolute error (JAX vs ONNX): 1.23e-06
✓ ONNX model saved to policy.onnx
```

## 转换流程

1. **加载模型**：从检查点恢复 JAX 模型参数
2. **权重转换**：将 JAX 权重转移到 PyTorch MLP
3. **ONNX 导出**：使用 PyTorch 的 ONNX 导出功能
4. **验证**：比较 JAX 和 ONNX 模型的输出

## 代码转换细节

转换过程在以下函数中完成：

- `transfer_weights()`：将 JAX 权重转换为 PyTorch 格式（考虑 JAX/TF 使用 (in, out) 而 PyTorch 使用 (out, in)）
- `convert_torch_to_onnx()`：PyTorch 模型导出为 ONNX
- `validate_onnx_model()`：验证 ONNX 模型输出与 PyTorch 模型匹配
- `convert_jax2onnx()`：主转换流程，包括最终 JAX vs ONNX 对比

## 与 convert_pt_to_onnx.py 的区别

- **jax2onnx.py**：从 JAX/Brax 检查点直接转换到 ONNX（需要重建环境）
- **convert_pt_to_onnx.py**：从已保存的 `.pt` TorchScript 文件转换到 ONNX（更快，但需要先有 `.pt` 文件）

## 故障排查

### 找不到检查点

```
FileNotFoundError: No checkpoint found.
```

确保实验名称正确，且 `experiments/<exp_name>/checkpoints/` 下有检查点文件夹。

### 模型架构不匹配

如果模型有特殊的架构（不是标准的 MLP），可能需要修改 `transfer_weights()` 函数以处理额外的层。

### ONNX Runtime 导入错误

```bash
pip install onnxruntime
```

## 验证转换结果

转换完成后，可以测试加载和使用 ONNX 模型：

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("policy.onnx")
input_name = session.get_inputs()[0].name

# 创建测试输入
test_input = np.random.randn(1, 64).astype(np.float32)

# 运行推理
output = session.run(None, {input_name: test_input})
print(output)
```

## 部署建议

转换后的 ONNX 模型可以用于：

1. **TensorRT**（NVIDIA GPU）
   ```bash
   trtexec --onnx=policy.onnx --saveEngine=policy.trt
   ```

2. **ONNXRuntime**（跨平台）
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession("policy.onnx")
   ```

3. **OpenVINO**（Intel 硬件）
   ```bash
   mo --input_model policy.onnx --output_dir policy_openvino
   ```





