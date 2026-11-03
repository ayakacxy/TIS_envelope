"""Stress-test all available backends against a float64 CPU reference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.source_paths import source_path
from fast_tis_envelope.backends.numpy_backend import envelop_numpy
from fast_tis_envelope.backends.numba_backend import envelop_numba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--jax", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.samples < 100:
        parser.error("--samples must be at least 100")
    import mlx.core as mx
    from fast_tis_envelope.backends.mlx_backend import envelop_mlx, envelop_mlx_fast
    from fast_tis_envelope.backends.mlx_metal_backend import envelop_mlx_metal

    rng = np.random.default_rng(419)
    n = args.samples
    scale = 10.0 ** rng.uniform(-20, 20, (n, 1))
    x = (rng.normal(size=(n, 3)) * scale).astype(np.float32)
    ratio = 10.0 ** rng.uniform(-6, 6, (n, 1))
    y = (rng.normal(size=(n, 3)) * scale * ratio).astype(np.float32)
    k = n // 10
    y[:k] = (x[:k].astype(np.float64) + rng.normal(size=(k, 3)) * scale[:k] * 1e-7).astype(np.float32)
    y[k:k + k // 2] = x[k:k + k // 2]
    y[k + k // 2:2 * k] = -x[k + k // 2:2 * k]
    y[2 * k:2 * k + k // 2] = 0
    x[2 * k + k // 2:3 * k] = 0
    reference = envelop_numpy(x.astype(np.float64), y.astype(np.float64))
    field_scale = np.maximum(np.max(np.abs(x), axis=1), np.max(np.abs(y), axis=1)).astype(np.float64)
    field_scale[field_scale == 0] = 1
    backends = [
        ("numpy_float32", envelop_numpy),
        ("numba_serial", envelop_numba),
        ("numba_parallel", lambda a, b: envelop_numba(a, b, parallel=True)),
        ("mlx_gpu", envelop_mlx),
        ("metal_gpu", envelop_mlx_metal),
    ]
    if args.jax:
        from fast_tis_envelope.backends.jax_backend import envelop_jax
        backends.append(("jax", envelop_jax))
    root = Path(__file__).resolve().parents[1]
    report = {
        "samples": n,
        "seed": 419,
        "reference": "NumPy float64",
        "source_sha256": {
            name: hashlib.sha256(source_path(name).read_bytes()).hexdigest()
            for name in ("env_numpy.py", "env_numba.py", "env_mlx.py", "env_mlx_metal.py", "env_jax.py")
        },
        "results": {},
    }
    for name, backend in backends:
        result = np.asarray(backend(x, y), dtype=np.float64)
        difference = np.abs(result - reference)
        positive = reference > 0
        normalized = difference / field_scale
        relative = difference[positive] / reference[positive]
        metrics = {
            "nonfinite_count": int((~np.isfinite(result)).sum()),
            "max_error_normalized_by_max_field_component": float(normalized.max()),
            "p99_error_normalized_by_max_field_component": float(np.quantile(normalized, 0.99)),
            "max_relative_error_for_nonzero_reference": float(relative.max()),
            "p99_relative_error_for_nonzero_reference": float(np.quantile(relative, 0.99)),
        }
        if metrics["nonfinite_count"] or metrics["max_relative_error_for_nonzero_reference"] > 2e-4:
            raise AssertionError(f"{name} failed: {metrics}")
        report["results"][name] = metrics
    # The unscaled fast path is tested separately on a bounded input range;
    # the preceding extreme-range test applies only to the stable backends.
    ordinary_scale = 10.0 ** rng.uniform(-3, 2, (n, 1))
    ordinary_x = (rng.normal(size=(n, 3)) * ordinary_scale).astype(np.float32)
    ordinary_y = (rng.normal(size=(n, 3)) * ordinary_scale).astype(np.float32)
    ordinary_y[:k] = (ordinary_x[:k].astype(np.float64) +
                      rng.normal(size=(k, 3)) * ordinary_scale[:k] * 1e-7).astype(np.float32)
    ordinary_y[k:k + k // 2] = ordinary_x[k:k + k // 2]
    ordinary_y[k + k // 2:2 * k] = -ordinary_x[k + k // 2:2 * k]
    ordinary_y[2 * k:2 * k + k // 2] = 0
    ordinary_reference = envelop_numpy(ordinary_x.astype(np.float64), ordinary_y.astype(np.float64))
    ordinary_result = np.asarray(envelop_mlx_fast(ordinary_x, ordinary_y), dtype=np.float64)
    ordinary_error = np.abs(ordinary_result - ordinary_reference)
    ordinary_nonzero = ordinary_reference > 0
    ordinary_relative = ordinary_error[ordinary_nonzero] / ordinary_reference[ordinary_nonzero]
    metrics = {
        "input_scale_range": "1e-3 to 1e2; normal-distributed vector components",
        "nonfinite_count": int((~np.isfinite(ordinary_result)).sum()),
        "max_relative_error_for_nonzero_reference": float(ordinary_relative.max()),
        "p99_relative_error_for_nonzero_reference": float(np.quantile(ordinary_relative, 0.99)),
    }
    if metrics["nonfinite_count"] or metrics["max_relative_error_for_nonzero_reference"] > 2e-4:
        raise AssertionError(f"mlx_gpu_fast failed: {metrics}")
    report["results"]["mlx_gpu_fast_bounded_inputs"] = metrics
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
