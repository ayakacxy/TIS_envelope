"""Resolve historical benchmark source labels to their v2 locations.

The labels remain stable in JSON for compatibility with existing plotters.
The hash always covers the implementation, never a compatibility shim.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PATHS = {
    **{f"env_{name}.py": f"fast_tis_envelope/backends/{name}_backend.py"
       for name in ("numpy", "numba", "jax", "mlx", "mlx_metal", "cupy")},
    "env_torch.py": "benchmarks/torch_baseline.py",
}


def source_path(label):
    return ROOT / _PATHS.get(label, label)
