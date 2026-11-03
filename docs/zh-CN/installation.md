# 安装说明

[English](../en/installation.md) · [简体中文](installation.md) · [文档](index.md)

## 环境要求

Fast TIS Envelope 需要 Python 3.9+。基础安装依赖 NumPy；Numba、JAX、PyTorch、MLX 和 CuPy 按需安装。Python 导入名为
`fast_tis_envelope`。

## 从源码安装

在项目根目录安装基础版本：

```bash
python -m pip install .
```

通过 extras 安装所需后端，也可以组合多个选项：

```bash
python -m pip install '.[cpu]'          # NumPy + Numba
python -m pip install '.[metal]'        # Apple Silicon：MLX / Metal
python -m pip install '.[jax,torch]'    # JAX + PyTorch
```

## 构建和安装 wheel

在项目根目录构建 wheel，然后选择基础安装或后端 extra：

```bash
python -m pip wheel . --no-deps -w dist
python -m pip install ./dist/fast_tis_envelope-0.2.0-py3-none-any.whl
```

```bash
python -m pip install './dist/fast_tis_envelope-0.2.0-py3-none-any.whl[cpu]'
python -m pip install './dist/fast_tis_envelope-0.2.0-py3-none-any.whl[metal]'
```

## NVIDIA CUDA

CUDA 内核使用 CuPy。已有 CuPy 的环境可直接安装基础包；新环境可选择与 CUDA 版本对应的 extra：

```bash
python -m pip install '.[cuda12]'       # CUDA 12.x
# 或：python -m pip install '.[cuda13]'  # CUDA 13.x
```

同一环境使用一个 CuPy 发行包。CUDA 驱动、运行时和 CuPy 版本的选择见
[CuPy 安装文档](https://docs.cupy.dev/en/stable/install.html)。JAX 和 PyTorch 的 GPU 构建按各自的安装说明选择。

## 检查安装

```bash
python -c "import fast_tis_envelope as f; print(f.__version__)"
```

运行测试可在项目根目录执行 `python -m pip install -e '.[cpu,test]'`，然后运行 `python -m pytest`。GPU 测试需要对应设备和运行时。
