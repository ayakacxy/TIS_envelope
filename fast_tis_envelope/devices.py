"""Place inputs for an explicitly requested backend/device pair."""

from __future__ import annotations

from .selection import _parse_device


def _ordinal(index):
    return 0 if index is None else index


def torch_inputs(e1, e2, device):
    import torch
    kind, index = _parse_device(device)
    if kind == "auto":
        return e1, e2
    if kind == "gpu":
        kind = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "unavailable"
        if kind == "unavailable":
            raise RuntimeError("No PyTorch GPU device is available")
        if kind == "mps" and index not in (None, 0):
            raise ValueError("PyTorch MPS only supports device index 0")
    if kind == "cuda":
        idx = _ordinal(index)
        count = torch.cuda.device_count()
        if idx >= count:
            raise ValueError(f"CUDA device {idx} is unavailable to PyTorch (device count: {count})")
        target = torch.device("cuda", idx)
    elif kind == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS device is unavailable")
        target = torch.device("mps")
    else:
        target = torch.device("cpu")
    # Tensor.to preserves autograd links. as_tensor avoids copying host tensors
    # when they already match the requested device.
    return tuple(value.to(target) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=target)
                 for value in (e1, e2))


def jax_inputs(e1, e2, device):
    import jax
    kind, index = _parse_device(device)
    if kind == "auto":
        return e1, e2
    platform = "cpu" if kind == "cpu" else "gpu"
    try:
        devices = jax.local_devices(backend=platform)
    except RuntimeError as exc:
        raise RuntimeError(f"No JAX {platform.upper()} devices are available") from exc
    idx = _ordinal(index)
    if idx >= len(devices):
        raise ValueError(f"JAX {platform.upper()} device {idx} is unavailable (device count: {len(devices)})")
    return jax.device_put(e1, devices[idx]), jax.device_put(e2, devices[idx])


def cupy_inputs(e1, e2, device):
    import cupy as cp
    kind, index = _parse_device(device)
    if kind == "auto":
        return e1, e2
    idx = _ordinal(index)
    count = cp.cuda.runtime.getDeviceCount()
    if idx >= count:
        raise ValueError(f"CUDA device {idx} is unavailable to CuPy (device count: {count})")
    with cp.cuda.Device(idx):
        # Explicit placement may transfer arrays from another visible GPU.
        return tuple(cp.array(value, copy=True) if isinstance(value, cp.ndarray) and value.device.id != idx
                     else cp.asarray(value) for value in (e1, e2))
