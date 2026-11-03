# 使用说明

[English](../en/usage.md) · [简体中文](usage.md) · [文档](index.md)

## 输入与输出

```python
import numpy as np
from fast_tis_envelope import envelope

e1 = np.array([[1., 0., 0.], [1., 0., 0.]], dtype=np.float32)
e2 = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
result = envelope(e1, e2)
print(result)  # [1.4142135 2.       ]
```

`e1` 和 `e2` 是同形状的有限实数数组 `(N, 3)`，每行是一处体素的三维电场。返回 `(N,)` 的最大包络幅值，单位与输入电场相同。函数假定输入有限，不额外扫描 NaN
和无穷值。计算定义见[算法原理](algorithm.md)。

## 选择后端

`envelope(e1, e2, *, backend="auto", device="auto", parallel=None, compile=False)` 提供以下后端：

| `backend` | 设备 | 返回类型 | 计算精度 |
| --- | --- | --- | --- |
| `numpy` | CPU | NumPy 数组 | float32 / float64 |
| `numba` | CPU | NumPy 数组 | float32 / float64 |
| `jax` | JAX CPU / GPU | JAX 数组 | 遵循 JAX 精度配置 |
| `torch` | CPU / CUDA / MPS | PyTorch 张量 | 至少 float32，取决于设备 |
| `mlx` | MLX CPU / Apple GPU | MLX 数组 | float32 |
| `metal` | Apple GPU | MLX 数组 | float32 |
| `cupy` | NVIDIA CUDA | CuPy 数组 | float32 |

CPU 的 NumPy 和 Numba 保留 float64，将低精度输入至少提升为 float32；MLX、Metal 和 CuPy 将输入转换为 float32。JAX 是否保留 float64
由 `jax_enable_x64` 控制。

默认的 `backend="auto"` 根据输入选择后端：

- NumPy 数组和列表等主机输入在 CPU 上运行，优先使用 Numba，未安装时使用 NumPy。
- JAX、PyTorch 和 CuPy 数组沿用各自的后端；MLX 数组在当前 MLX CPU 或 GPU 环境中分别选择 MLX 或 Metal。
- 主机输入指定 `device="gpu"` 时，优先选择可用的 CuPy/CUDA，否则选择 MLX/Metal。

使用 `resolve_backend` 查看选择结果：

```python
from fast_tis_envelope import resolve_backend

print(resolve_backend(e1, e2))  # numba 或 numpy
```

它返回后端名称；实际计算时还会检查设备是否可用。

## 选择设备

同时指定 `backend` 和 `device`：

```python
cpu = envelope(e1, e2, backend="numba", device="cpu", parallel=True)
apple = envelope(e1, e2, backend="metal", device="mps:0")
cuda = envelope(e1, e2, backend="cupy", device="cuda:0")
torch_cuda = envelope(e1, e2, backend="torch", device="cuda:0")
```

支持 `cpu`、`gpu`、`cuda`、`mps` 和 `metal`，也可添加从 0 开始的编号，如 `cuda:0`。`metal:0` 是 `mps:0` 的别名；MLX 和 MPS
使用编号 0。编号对应所选框架可见的设备顺序。选择不兼容或不可用的设备时会报错。

显式选择设备时，输入按需迁移，返回值位于选定设备；跨设备传输可能产生复制。不同框架的数组应先转换为同一框架。要重复计算，可先把输入放到目标设备，避免每次调用都传输数据。

## 并行与编译

Numba 默认在至少 50,000 行时使用并行内核；`parallel=True` 或 `False` 可覆盖此策略。启动 Python 前通过 `NUMBA_NUM_THREADS` 设置
Numba 的线程数。

```python
result = envelope(e1, e2, backend="torch", device="cuda:0", compile=True)
```

`compile=True` 启用 `torch.compile` 并缓存编译调用。Numba、JAX 和 MLX 使用各自的编译路径，Metal 和 CUDA
内核在首次使用时编译。首次调用和新形状的编译可能需要额外时间。

## GPU 同步

GPU 操作可能异步执行。测量耗时时应先预热，并在每次调用后同步对应设备：

```python
# CuPy：以 cuda:0 为例
import cupy as cp
with cp.cuda.Device(0):
    result = envelope(e1, e2, backend="cupy", device="cuda:0")
    cp.cuda.get_current_stream().synchronize()
```

JAX 可对结果调用 `block_until_ready()`；MLX 可使用 `mx.eval(result)`；PyTorch 使用
`torch.cuda.synchronize(result.device)` 或 `torch.mps.synchronize()`。

Metal 和 CuPy 自定义内核只支持前向计算。张量后端在分段边界和退化点的梯度不作保证。
