"""Fast, portable maximum-envelope solver for temporal interference stimulation."""

from __future__ import annotations

from .selection import resolve_backend

__version__ = "0.2.0"
__all__ = ["envelope", "resolve_backend", "__version__"]


def envelope(e1, e2, *, backend="auto", device="auto", parallel=None, compile=False):
    """Compute the maximum envelope of two finite real fields of shape (N, 3).

    The result has shape (N,) and uses the selected framework's array type.
    Auto selection preserves native framework arrays. Host inputs run on CPU
    unless a GPU device is explicitly requested. For example, device='cuda:0'
    selects the first visible CUDA GPU. GPU kernels use float32.
    ``parallel`` controls the Numba thread pool; by default Numba parallelizes
    inputs with at least 50,000 rows. Set compile=True for torch.compile;
    other compiled backends compile by default. Optional dependencies are lazy.
    """
    chosen = resolve_backend(e1, e2, backend=backend, device=device)
    if not isinstance(compile, bool):
        raise TypeError("compile must be a bool")
    if compile and chosen == "numpy":
        raise ValueError("NumPy does not provide compilation; choose numba or another compiled backend")
    if parallel is not None and chosen != "numba":
        raise ValueError("parallel is only supported by the Numba backend")
    try:
        if chosen == "numpy":
            from .backends.numpy_backend import envelop_numpy
            return envelop_numpy(e1, e2)
        if chosen == "numba":
            from .backends.numba_backend import envelop_numba
            if parallel is None:
                parallel = len(e1) >= 50_000
            if not isinstance(parallel, bool):
                raise TypeError("parallel must be a bool or None")
            return envelop_numba(e1, e2, parallel=parallel)
        if chosen == "jax":
            from .backends.jax_backend import envelop_jax
            from .devices import jax_inputs
            e1, e2 = jax_inputs(e1, e2, device)
            return envelop_jax(e1, e2)
        if chosen == "torch":
            from .backends.torch_backend import envelop_torch
            from .devices import torch_inputs
            e1, e2 = torch_inputs(e1, e2, device)
            if compile:
                return _compiled_torch()(e1, e2)
            return envelop_torch(e1, e2)
        if chosen == "mlx":
            from .backends.mlx_backend import envelop_mlx
            from .selection import _parse_device
            import mlx.core as mx
            kind, _ = _parse_device(device)
            if kind == "auto":
                return envelop_mlx(e1, e2)
            with mx.stream(mx.cpu if kind == "cpu" else mx.gpu):
                return envelop_mlx(e1, e2)
        if chosen == "metal":
            from .backends.mlx_metal_backend import envelop_mlx_metal
            import mlx.core as mx
            with mx.stream(mx.gpu):
                return envelop_mlx_metal(e1, e2)
        if chosen == "cupy":
            from .backends.cupy_backend import envelop_cupy
            from .devices import cupy_inputs
            e1, e2 = cupy_inputs(e1, e2, device)
            return envelop_cupy(e1, e2)
    except ImportError as exc:
        raise ImportError(f"Backend {chosen!r} needs its optional dependencies; see the installation guide") from exc
    raise AssertionError(f"unhandled backend: {chosen}")


from functools import lru_cache


@lru_cache(maxsize=1)
def _compiled_torch():
    import torch
    from .backends.torch_backend import envelop_torch
    return torch.compile(envelop_torch, fullgraph=True)
