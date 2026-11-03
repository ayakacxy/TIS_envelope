# Installation

[English](installation.md) · [简体中文](../zh-CN/installation.md) · [Documentation](index.md)

## Requirements

Fast TIS Envelope requires Python 3.9+. The base installation uses NumPy; Numba, JAX, PyTorch, MLX
and CuPy are optional. The Python import is `fast_tis_envelope`.

## Install from source

From the project root, install the base package:

```bash
python -m pip install .
```

Select backend extras as needed; extras can be combined:

```bash
python -m pip install '.[cpu]'          # NumPy + Numba
python -m pip install '.[metal]'        # Apple Silicon: MLX / Metal
python -m pip install '.[jax,torch]'    # JAX + PyTorch
```

## Build and install a wheel

Build from the project root, then install the base wheel or an extra:

```bash
python -m pip wheel . --no-deps -w dist
python -m pip install ./dist/fast_tis_envelope-0.2.0-py3-none-any.whl
```

```bash
python -m pip install './dist/fast_tis_envelope-0.2.0-py3-none-any.whl[cpu]'
python -m pip install './dist/fast_tis_envelope-0.2.0-py3-none-any.whl[metal]'
```

## NVIDIA CUDA

The CUDA kernel uses CuPy. If CuPy is already installed, install the base package; for a new
environment, select the extra matching the CUDA version:

```bash
python -m pip install '.[cuda12]'       # CUDA 12.x
# Or: python -m pip install '.[cuda13]'  # CUDA 13.x
```

Use one CuPy distribution per environment. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for compatible CUDA drivers
and runtimes. Select GPU builds of JAX and PyTorch using their installation guides.

## Verify the installation

```bash
python -c "import fast_tis_envelope as f; print(f.__version__)"
```

To run the tests, install `python -m pip install -e '.[cpu,test]'` from the project root and run
`python -m pytest`. GPU tests need the corresponding device and runtime.
