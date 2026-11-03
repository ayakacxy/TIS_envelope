# Usage

[English](usage.md) · [简体中文](../zh-CN/usage.md) · [Documentation](index.md)

## Inputs and output

```python
import numpy as np
from fast_tis_envelope import envelope

e1 = np.array([[1., 0., 0.], [1., 0., 0.]], dtype=np.float32)
e2 = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
result = envelope(e1, e2)
print(result)  # [1.4142135 2.       ]
```

`e1` and `e2` are finite real arrays of the same shape `(N, 3)`, with one three-dimensional field
vector per voxel. The result has shape `(N,)`, with the same units as the input fields. Inputs are
assumed finite; the call does not scan for NaN or infinity. See [Algorithm](algorithm.md) for the
definition.

## Select a backend

`envelope(e1, e2, *, backend="auto", device="auto", parallel=None, compile=False)` supports:

| `backend` | Device | Output | Compute precision |
| --- | --- | --- | --- |
| `numpy` | CPU | NumPy array | float32 / float64 |
| `numba` | CPU | NumPy array | float32 / float64 |
| `jax` | JAX CPU / GPU | JAX array | JAX precision configuration |
| `torch` | CPU / CUDA / MPS | PyTorch tensor | at least float32, device-dependent |
| `mlx` | MLX CPU / Apple GPU | MLX array | float32 |
| `metal` | Apple GPU | MLX array | float32 |
| `cupy` | NVIDIA CUDA | CuPy array | float32 |

NumPy and Numba retain float64 and promote lower-precision inputs to at least float32. MLX, Metal
and CuPy convert inputs to float32. JAX float64 behavior follows `jax_enable_x64`.

With the default `backend="auto"`, dispatch follows the inputs:

- Host inputs such as NumPy arrays and lists run on CPU, using Numba when installed and NumPy
  otherwise.
- JAX, PyTorch and CuPy arrays retain their backend. MLX arrays select MLX or Metal according to the
  active MLX CPU or GPU environment.
- For host inputs, `device="gpu"` selects available CuPy/CUDA first, then MLX/Metal.

Use `resolve_backend` to inspect the choice:

```python
from fast_tis_envelope import resolve_backend

print(resolve_backend(e1, e2))  # numba or numpy
```

It returns the backend name; the calculation also checks device availability.

## Select a device

Specify `backend` and `device` together:

```python
cpu = envelope(e1, e2, backend="numba", device="cpu", parallel=True)
apple = envelope(e1, e2, backend="metal", device="mps:0")
cuda = envelope(e1, e2, backend="cupy", device="cuda:0")
torch_cuda = envelope(e1, e2, backend="torch", device="cuda:0")
```

Devices include `cpu`, `gpu`, `cuda`, `mps` and `metal`, optionally followed by a zero-based index
such as `cuda:0`. `metal:0` is an alias for `mps:0`; MLX and MPS use index 0. Indices follow the
devices visible to the chosen framework. Incompatible or unavailable devices raise an error.

Explicit placement transfers inputs as needed and returns the result on the selected device. Convert
arrays from different frameworks to a common framework before calling. For repeated calculations,
place inputs on the target device first to avoid repeated transfers.

## Parallelism and compilation

Numba uses its parallel kernel for at least 50,000 rows by default; set `parallel=True` or `False`
to override this heuristic. Set `NUMBA_NUM_THREADS` before starting Python to control the thread
pool.

```python
result = envelope(e1, e2, backend="torch", device="cuda:0", compile=True)
```

`compile=True` enables `torch.compile` and caches the compiled callable. Numba, JAX and MLX use
their own compiled paths; Metal and CUDA kernels compile on first use. Initial calls and new shapes
can incur compilation time.

## Synchronize GPU work

GPU operations can execute asynchronously. Warm up first and synchronize the device after each call
when measuring time:

```python
# CuPy example on cuda:0
import cupy as cp
with cp.cuda.Device(0):
    result = envelope(e1, e2, backend="cupy", device="cuda:0")
    cp.cuda.get_current_stream().synchronize()
```

For JAX, call `block_until_ready()` on the result; for MLX, use `mx.eval(result)`; for PyTorch, use
`torch.cuda.synchronize(result.device)` or `torch.mps.synchronize()`.

The custom Metal and CuPy kernels are forward-only. Gradients at piecewise boundaries and degenerate
inputs are not guaranteed for tensor backends.
