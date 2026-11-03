"""Paired CPU/CUDA envelope benchmark on shared inputs, in separate processes."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.source_paths import source_path
from fast_tis_envelope.backends.numpy_backend import envelop_numpy


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_case(label, e1, e2, args, random_order, compiled_torch):
    n = len(e1)
    reference = envelop_numpy(e1.astype(np.float64), e2.astype(np.float64))
    scale = np.minimum(np.max(np.abs(e1), axis=1), np.max(np.abs(e2), axis=1)).astype(np.float64)
    allowed = 5e-5 * np.abs(reference) + 5e-7 * scale
    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(e1)).cast('B'))
    digest.update(memoryview(np.ascontiguousarray(e2)).cast('B'))
    noop = lambda: None
    no_sync = lambda result: None
    calls = {}
    if args.device != 'cpu' or not args.numba_only:
        import jax
        from fast_tis_envelope.backends.jax_backend import envelop_jax
        jax_device = jax.devices('cpu' if args.device == 'cpu' else 'gpu')[0]
        jax_inputs = (jax.device_put(e1, jax_device), jax.device_put(e2, jax_device))
        for a in jax_inputs:
            a.block_until_ready()
        calls['jax_cpu' if args.device == 'cpu' else 'jax_cuda'] = (
            lambda: envelop_jax(*jax_inputs), lambda out: out.block_until_ready(), noop)
    if args.device == 'cpu':
        import numba
        from fast_tis_envelope.backends.numba_backend import envelop_numba
        if args.numba_only:
            numba.set_num_threads(32)
            calls['numba_32t'] = (lambda: envelop_numba(e1, e2, parallel=True), no_sync, noop)
        else:
            calls.update({'numpy': (lambda: envelop_numpy(e1, e2), no_sync, noop),
                          'numba_serial': (lambda: envelop_numba(e1, e2), no_sync, noop)})
    else:
        import cupy as cp
        import torch
        from benchmarks.torch_baseline import envelop_torch
        from fast_tis_envelope.backends.cupy_backend import envelop_cupy, envelop_cupy_server
        cupy_inputs = (cp.asarray(e1), cp.asarray(e2))
        torch_inputs = (torch.from_numpy(e1.copy()).to('cuda'), torch.from_numpy(e2.copy()).to('cuda'))
        cp.cuda.Device().synchronize()
        torch.cuda.synchronize()
        cupy_calls = {'cupy_existing': envelop_cupy_server, 'cupy_stable': envelop_cupy}
        for name, fn in cupy_calls.items():
            calls[name] = (lambda fn=fn: fn(*cupy_inputs), lambda out: cp.cuda.Device().synchronize(), noop)
        calls['torch_cuda'] = (lambda: envelop_torch(*torch_inputs), lambda out: torch.cuda.synchronize(), noop)
        calls['torch_cuda_compiled'] = (lambda: compiled_torch(*torch_inputs), lambda out: torch.cuda.synchronize(), noop)
    errors = {}
    for name, (fn, sync, setup) in calls.items():
        setup()
        for _ in range(args.warmups):
            result = fn()
            sync(result)
        if name.startswith('cupy'):
            actual = cp.asnumpy(result).astype(np.float64)
        elif name.startswith('torch'):
            actual = result.cpu().numpy().astype(np.float64)
        else:
            actual = np.asarray(result, dtype=np.float64)
        error = np.abs(actual - reference)
        if not np.isfinite(actual).all() or np.any(error > allowed):
            i = int(np.argmax(error / np.maximum(allowed, np.finfo(np.float64).tiny)))
            raise AssertionError(f'{name} failed {label} N={n} at {i}: actual={actual[i]}, ref={reference[i]}, tolerance={allowed[i]}')
        nonzero = reference > 0
        errors[name] = {'max_abs_error_vs_float64': float(error.max()),
                        'mean_abs_error_vs_float64': float(error.mean()),
                        'max_relative_error_vs_float64': float((error[nonzero] / reference[nonzero]).max()) if nonzero.any() else 0}
    samples = {name: [] for name in calls}
    for _ in range(args.repeats):
        order = list(calls)
        random_order.shuffle(order)
        for name in order:
            fn, sync, setup = calls[name]
            setup()
            start = time.perf_counter_ns()
            result = fn()
            sync(result)
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    records = []
    for name, values in samples.items():
        median = statistics.median(values)
        records.append({'dataset': label, 'n': n, 'backend': name, 'input_sha256': digest.hexdigest(),
                        'median_ms': median, 'p10_ms': float(np.quantile(values, 0.1)),
                        'p90_ms': float(np.quantile(values, 0.9)),
                        'million_voxels_per_second': n / (1000 * median), 'samples_ms': values, **errors[name]})
    print(f'{label:11s} N={n:>9,}  ' + '  '.join(f"{x['backend']}={x['median_ms']:.4f}ms" for x in records), flush=True)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    p.add_argument('--sizes', nargs='+', type=int, default=[100,1000,10000,50000,100000,1000000,1500000,2000000])
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--repeats', type=int, default=31)
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--numba-only', action='store_true', help='CPU: run parallel Numba with 32 threads only')
    p.add_argument('--real-inputs', nargs=2, required=True)
    p.add_argument('--existing-cupy-source', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if min(args.sizes) < 1 or args.warmups < 1 or args.repeats < 1 or len(set(args.sizes)) != len(args.sizes):
        p.error('sizes, warmups, repeats must be positive; sizes must be unique')
    if args.numba_only and args.device != 'cpu':
        p.error('--numba-only requires --device cpu')
    compiled_torch = None
    if args.device == 'gpu':
        import torch
        from benchmarks.torch_baseline import envelop_torch
        from fast_tis_envelope.backends.cupy_backend import _SERVER_SOURCE
        original_source = Path(args.existing_cupy_source).read_text(encoding='utf-8')
        start = original_source.index('__device__ __forceinline__ float envelope_value_n4_strict(')
        end = original_source.index('\n            extern "C" __global__', start)
        if original_source[start:end] not in _SERVER_SOURCE:
            raise ValueError('local CuPy helper does not match the server source')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        if hasattr(torch._dynamo.config, 'recompile_limit'):
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, len(args.sizes)+2)
        else:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, len(args.sizes)+2)
        compiled_torch = torch.compile(envelop_torch, backend='inductor', fullgraph=True, dynamic=False)
    names = ['env_numpy.py', 'env_numba.py', 'env_jax.py', 'env_torch.py', 'env_cupy.py',
             'benchmarks/env_cupy_server.py', 'benchmarks/benchmark_server_cpu_gpu.py']
    cpu_model = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown')
    gpu_info = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'], text=True).strip()
    report = {'metadata': {'benchmark_device': args.device, 'machine': platform.machine(), 'platform': platform.platform(),
                           'cpu_model': cpu_model, 'logical_cpus': os.cpu_count(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                           'gpu': gpu_info, 'python': platform.python_version(),
                           'numpy': version('numpy'), 'numba': version('numba'), 'jax': version('jax'), 'jaxlib': version('jaxlib'),
                           'torch': version('torch'), 'cupy': version('cupy'), 'numba_parallel_threads': 32 if args.numba_only else None,
                           'jax_device': (None if args.device == 'cpu' and args.numba_only else
                                          str(__import__('jax').devices('cpu' if args.device == 'cpu' else 'gpu')[0])),
                           'torch_execution': 'eager and torch.compile(inductor, fullgraph=True, dynamic=False)' if args.device=='gpu' else None,
                           'existing_cupy_source': str(Path(args.existing_cupy_source).resolve()),
                           'existing_cupy_source_sha256': hashlib.sha256(Path(args.existing_cupy_source).read_bytes()).hexdigest(),
                           'dtype': 'float32', 'warmups': args.warmups, 'repeats': args.repeats, 'seed': args.seed,
                           'synthetic': 'independent normal components, mean 0, standard deviation 0.1 V/m; no injected equal rows',
                           'timing': 'median of randomized-order forward calls, synchronized; inputs and compilation excluded',
                           'reference': 'NumPy float64 on the same float32 input values',
                           'source_sha256': {s: hashlib.sha256(source_path(s).read_bytes()).hexdigest() for s in names}},
              'results': []}
    rng = np.random.default_rng(args.seed)
    rand = random.Random(args.seed)
    for n in sorted(args.sizes):
        x = (rng.normal(size=(n,3))*0.1).astype(np.float32)
        y = (rng.normal(size=(n,3))*0.1).astype(np.float32)
        report['results'].extend(run_case('synthetic', x, y, args, rand, compiled_torch))
    e1, e2 = (np.asarray(np.load(p, mmap_mode='r'), dtype=np.float32) for p in args.real_inputs)
    if e1.ndim != 2 or e1.shape[1] != 3 or e1.shape != e2.shape or not len(e1):
        p.error('real inputs must have identical nonempty shape (N, 3)')
    if not np.isfinite(e1).all() or not np.isfinite(e2).all():
        p.error('real inputs must be finite')
    report['results'].extend(run_case('HCP100610', e1, e2, args, rand, compiled_torch))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
