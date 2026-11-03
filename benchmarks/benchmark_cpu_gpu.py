"""Measure CPU and GPU envelope solvers on shared inputs for plotting.

Timings are synchronized forward evaluations. Input creation, data preparation
and first compilation are outside the timed region. MLX built-in operators are
measured on CPU and GPU explicitly; Metal and PyTorch MPS run on GPU.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.source_paths import source_path
from fast_tis_envelope.backends.numpy_backend import envelop_numpy


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_dataset(label, e1, e2, args, rng, mx, torch_compiled):
    n = len(e1)
    ref = envelop_numpy(e1.astype(np.float64), e2.astype(np.float64))
    if mx is not None:
        device = mx.cpu if args.device == "cpu" else mx.gpu
        with mx.stream(device):
            inputs = (mx.array(e1), mx.array(e2))
            mx.eval(*inputs)
            mx.synchronize(device)
    input_hash = hashlib.sha256()
    input_hash.update(memoryview(np.ascontiguousarray(e1)).cast("B"))
    input_hash.update(memoryview(np.ascontiguousarray(e2)).cast("B"))

    def cpu_sync(_value):
        return None

    def mlx_call(function, inputs, device):
        with mx.stream(device):
            result = function(*inputs)
            mx.eval(result)
            mx.synchronize(device)
            return result

    if args.device == "cpu":
        from fast_tis_envelope.backends.numba_backend import envelop_numba
        calls = {
            "numpy": (lambda: envelop_numpy(e1, e2), cpu_sync),
            "numba_serial": (lambda: envelop_numba(e1, e2), cpu_sync),
            "numba_parallel": (lambda: envelop_numba(e1, e2, parallel=True), cpu_sync),
        }
        if mx is not None:
            from fast_tis_envelope.backends.mlx_backend import envelop_mlx, envelop_mlx_fast
            calls.update({
                "mlx_cpu": (lambda: mlx_call(envelop_mlx, inputs, device), cpu_sync),
                "mlx_cpu_fast": (lambda: mlx_call(envelop_mlx_fast, inputs, device), cpu_sync),
            })
        if args.jax:
            import jax
            from fast_tis_envelope.backends.jax_backend import envelop_jax
            cpu = jax.devices("cpu")[0]
            jax_inputs = (jax.device_put(e1, cpu), jax.device_put(e2, cpu))
            for value in jax_inputs:
                value.block_until_ready()
            calls["jax_cpu"] = (lambda: envelop_jax(*jax_inputs),
                                lambda value: value.block_until_ready())
    else:
        from fast_tis_envelope.backends.mlx_backend import envelop_mlx, envelop_mlx_fast
        from fast_tis_envelope.backends.mlx_metal_backend import envelop_mlx_metal
        import torch
        from benchmarks.torch_baseline import envelop_torch
        torch_inputs = (torch.from_numpy(e1.copy()).to("mps"), torch.from_numpy(e2.copy()).to("mps"))
        torch.mps.synchronize()
        calls = {
            "mlx_gpu": (lambda: mlx_call(envelop_mlx, inputs, device), cpu_sync),
            "mlx_gpu_fast": (lambda: mlx_call(envelop_mlx_fast, inputs, device), cpu_sync),
            "metal_gpu": (lambda: mlx_call(envelop_mlx_metal, inputs, device), cpu_sync),
            "torch_mps": (lambda: envelop_torch(*torch_inputs), lambda _: torch.mps.synchronize()),
            "torch_mps_compiled": (lambda: torch_compiled(*torch_inputs), lambda _: torch.mps.synchronize()),
        }

    comparison_scale = np.minimum(np.max(np.abs(e1), axis=1), np.max(np.abs(e2), axis=1)).astype(np.float64)
    allowed = 5e-5 * np.abs(ref) + 5e-7 * comparison_scale
    errors = {}
    for name, (function, sync) in calls.items():
        for _ in range(args.warmups):
            result = function()
            sync(result)
        observed = np.asarray(result.cpu().numpy() if name.startswith("torch_mps") else result, dtype=np.float64)
        abs_error = np.abs(observed - ref)
        if not np.isfinite(observed).all() or np.any(abs_error > allowed):
            i = int(np.argmax(abs_error / np.maximum(allowed, np.finfo(np.float64).tiny)))
            raise AssertionError(f"{name} failed at row {i}: actual={observed[i]}, reference={ref[i]}")
        nonzero = ref > 0
        errors[name] = {
            "max_abs_error_vs_float64": float(abs_error.max()),
            "mean_abs_error_vs_float64": float(abs_error.mean()),
            "max_relative_error_vs_float64": float((abs_error[nonzero] / ref[nonzero]).max()) if nonzero.any() else 0,
        }

    samples = {name: [] for name in calls}
    for _ in range(args.repeats):
        order = list(calls)
        rng.shuffle(order)
        for name in order:
            function, sync = calls[name]
            start = time.perf_counter_ns()
            result = function()
            sync(result)
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    records = []
    for name, times in samples.items():
        median = statistics.median(times)
        records.append({
            "dataset": label,
            "n": n,
            "backend": name,
            "input_sha256": input_hash.hexdigest(),
            "median_ms": median,
            "p10_ms": float(np.quantile(times, 0.1)),
            "p90_ms": float(np.quantile(times, 0.9)),
            "million_voxels_per_second": n / (median * 1000),
            "samples_ms": times,
            **errors[name],
        })
    print(f"{label:11s} N={n:>9,}  " + "  ".join(
        f"{item['backend']}={item['median_ms']:.4f} ms" for item in records
    ), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True,
                        help="run CPU and GPU in separate processes to prevent interference")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[100, 1000, 10000, 50000, 100000, 1000000, 1500000, 2000000])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--jax", action="store_true", help="include JAX on CPU")
    parser.add_argument("--no-mlx", action="store_true", help="skip MLX in CPU benchmarks")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--real-inputs", nargs=2, metavar=("E1_NPY", "E2_NPY"),
                        help="optional HCP100610 (N, 3) field pair")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(n < 1 for n in args.sizes) or args.warmups < 1 or args.repeats < 1 or args.threads < 1:
        parser.error("sizes, warmups, repeats and threads must be positive")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be unique")
    if args.jax and args.device != "cpu":
        parser.error("--jax is only available for --device cpu")
    if args.no_mlx and args.device != "cpu":
        parser.error("--no-mlx is only available for --device cpu")
    mx = None
    if not args.no_mlx:
        import mlx.core as mx
    if args.device == "cpu":
        import numba
        numba.set_num_threads(args.threads)
    elif not mx.metal.is_available():
        raise RuntimeError("Metal GPU is not available")
    torch_compiled = None
    if args.device == "gpu":
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS GPU is not available")
        from benchmarks.torch_baseline import envelop_torch
        # One shape-specialized graph per benchmark case, including the real head.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit,
                                                   len(args.sizes) + bool(args.real_inputs) + 1)
        torch_compiled = torch.compile(envelop_torch, backend="inductor", fullgraph=True, dynamic=False)
    if mx is not None:
        mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    files = ("env_numpy.py", "env_numba.py", "env_mlx.py", "env_mlx_metal.py")
    if args.jax:
        import jax
        files += ("env_jax.py",)
    if args.device == "gpu":
        files += ("env_torch.py",)
    mlx_cpu_device = mlx_gpu_device = None
    if mx is not None:
        with mx.stream(mx.cpu):
            mlx_cpu_device = str(mx.default_device())
        with mx.stream(mx.gpu):
            mlx_gpu_device = str(mx.default_device())
    report = {
        "metadata": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
            "numpy": _version("numpy"),
            "numba": _version("numba"),
            "mlx": _version("mlx"),
            "jax": _version("jax") if args.jax else None,
            "jaxlib": _version("jaxlib") if args.jax else None,
            "jax_device": str(jax.devices("cpu")[0]) if args.jax else None,
            "jax_threading": "JAX/XLA default CPU thread pool; not controlled by --threads" if args.jax else None,
            "torch": _version("torch") if args.device == "gpu" else None,
            "torch_device": "mps" if args.device == "gpu" else None,
            "torch_execution": "eager and torch.compile(backend=inductor, fullgraph=True, dynamic=False)" if args.device == "gpu" else None,
            "mlx_cpu_device": mlx_cpu_device,
            "mlx_gpu_device": mlx_gpu_device,
            "benchmark_device": args.device,
            "gpu": {str(k): str(v) for k, v in mx.device_info(mx.gpu).items()} if mx is not None else None,
            "numba_threads": numba.get_num_threads() if args.device == "cpu" else None,
            "dtype": "float32",
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seed": args.seed,
            "synthetic": "independent normal components, mean 0, standard deviation 0.1 V/m; no injected equal rows",
            "timing": "median of randomized-order forward calls, synchronized; inputs and compilation excluded",
            "reference": "NumPy float64 on the same float32 input values",
            "source_sha256": {name: hashlib.sha256(source_path(name).read_bytes()).hexdigest() for name in files},
            "real_input_files": [str(Path(p).resolve()) for p in args.real_inputs] if args.real_inputs else None,
        },
        "results": [],
    }
    random_order = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    for n in sorted(args.sizes):
        e1 = (np_rng.normal(size=(n, 3)) * 0.1).astype(np.float32)
        e2 = (np_rng.normal(size=(n, 3)) * 0.1).astype(np.float32)
        report["results"].extend(_run_dataset("synthetic", e1, e2, args, random_order, mx, torch_compiled))
    if args.real_inputs:
        e1 = np.asarray(np.load(args.real_inputs[0], mmap_mode="r"), dtype=np.float32)
        e2 = np.asarray(np.load(args.real_inputs[1], mmap_mode="r"), dtype=np.float32)
        if e1.shape != e2.shape or e1.ndim != 2 or e1.shape[1] != 3 or not len(e1):
            parser.error("real inputs must have identical, nonempty shape (N, 3)")
        if not np.isfinite(e1).all() or not np.isfinite(e2).all():
            parser.error("real inputs must contain finite values")
        report["results"].extend(_run_dataset("HCP100610", e1, e2, args, random_order, mx, torch_compiled))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
