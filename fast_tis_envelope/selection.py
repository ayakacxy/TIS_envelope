"""Backend and device selection without importing optional frameworks eagerly."""

from __future__ import annotations

import importlib.util
import re

_BACKENDS = {"auto", "numpy", "numba", "jax", "torch", "mlx", "metal", "cupy"}
_DEVICE_RE = re.compile(r"^(cpu|gpu|cuda|mps|metal)(?::(0|[1-9][0-9]*))?$")


def _parse_device(device):
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    device = device.lower()
    if device == "auto":
        return "auto", None
    match = _DEVICE_RE.fullmatch(device)
    if not match:
        raise ValueError("device must be 'auto', 'cpu', 'gpu', 'cuda', 'mps', or 'metal', optionally followed by :<index>")
    kind, index = match.groups()
    return ("mps" if kind == "metal" else kind), (int(index) if index is not None else None)


def _family(value):
    module = type(value).__module__.split('.')[0]
    if module in ("jax", "jaxlib"):
        return "jax"
    if module in ("torch", "cupy", "mlx"):
        return module
    return "host"


def _gpu_backend(kind="gpu"):
    if kind in ("gpu", "cuda") and importlib.util.find_spec("cupy") is not None:
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > 0:
                return "cupy"
        except Exception:
            pass
    if kind in ("gpu", "mps") and importlib.util.find_spec("mlx") is not None:
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return "metal"
        except Exception:
            pass
    raise RuntimeError(f"No supported {kind.upper()} backend is available; install CuPy with CUDA or MLX on Apple Silicon")


def _validate_backend_device(backend, kind, index):
    if kind == "auto":
        return
    allowed = {
        "numpy": {"cpu"}, "numba": {"cpu"},
        "jax": {"cpu", "gpu", "cuda"},
        "torch": {"cpu", "gpu", "cuda", "mps"},
        "mlx": {"cpu", "gpu", "mps"},
        "metal": {"gpu", "mps"}, "cupy": {"gpu", "cuda"},
    }
    if kind not in allowed[backend]:
        raise ValueError(f"device {kind!r} is not supported by backend {backend!r}")
    if index not in (None, 0) and (backend in {"numpy", "numba", "mlx", "metal"} or kind == "mps" or (backend == "torch" and kind == "cpu")):
        raise ValueError(f"device index {index} is not supported by backend {backend!r} on {kind}")


def resolve_backend(e1, e2, *, backend="auto", device="auto"):
    """Choose a compatible backend; host inputs default to CPU."""
    if not isinstance(backend, str) or backend.lower() not in _BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(sorted(_BACKENDS))}")
    backend = backend.lower()
    kind, index = _parse_device(device)
    f1, f2 = _family(e1), _family(e2)
    if f1 != f2 and f1 != "host" and f2 != "host":
        raise TypeError("inputs from different array frameworks are not supported")
    family = f1 if f1 != "host" else f2
    if backend != "auto":
        compatible = {"numpy": "host", "numba": "host", "jax": "jax", "torch": "torch", "mlx": "mlx", "metal": "mlx", "cupy": "cupy"}[backend]
        if family not in ("host", compatible):
            raise TypeError(f"{family} inputs cannot be passed to backend={backend!r}; convert explicitly")
    elif family == "host":
        backend = _gpu_backend(kind) if kind in ("gpu", "cuda", "mps") else (
            "numba" if importlib.util.find_spec("numba") is not None else "numpy")
    elif family == "mlx":
        if kind == "cpu":
            backend = "mlx"
        elif kind in ("gpu", "mps"):
            backend = "metal"
        else:
            import mlx.core as mx
            backend = "metal" if mx.default_device() == mx.gpu and mx.metal.is_available() else "mlx"
    else:
        backend = family
    _validate_backend_device(backend, kind, index)
    return backend
