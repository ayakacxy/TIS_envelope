"""Compare all measured local and server backends on the same real-head input."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.plot_cpu_gpu import (CPU_BACKENDS, GPU_BACKENDS, STYLES as LOCAL_STYLES,
                                     FIGURE_WIDTH, _index, _validate, _axis_style, _save, _set_typeface)
from benchmarks.plot_server_cpu_gpu import CPU as SERVER_CPU, GPU as SERVER_GPU, STYLES as SERVER_STYLES, read_reports


def _label(backend):
    labels = {
        'numpy': 'NumPy (CPU)',
        'numba_serial': 'Numba (1T, CPU)',
        'numba_parallel': 'Numba (10T, CPU)',
        'numba_32t': 'Numba (32T, CPU)',
        'jax_cpu': 'JAX (CPU)',
        'mlx_cpu': 'MLX (CPU)',
        'mlx_cpu_fast': 'MLX-fast (CPU)',
        'mlx_gpu': 'MLX (GPU)',
        'mlx_gpu_fast': 'MLX-fast (GPU)',
        'metal_gpu': 'Metal (GPU)',
        'torch_mps': 'PyTorch (eager, GPU)',
        'torch_mps_compiled': 'PyTorch (compiled, GPU)',
        'cupy_existing': 'CuPy (unscaled, GPU)',
        'cupy_stable': 'CuPy (scaled, GPU)',
        'jax_cuda': 'JAX (GPU)',
        'torch_cuda': 'PyTorch (eager, GPU)',
        'torch_cuda_compiled': 'PyTorch (compiled, GPU)',
    }
    return labels[backend]


def load_records(local_cpu, local_gpu, server_cpu, server_cpu32, server_gpu):
    local_cpu = json.loads(Path(local_cpu).read_text())
    local_gpu = json.loads(Path(local_gpu).read_text())
    _validate(local_cpu, local_gpu)
    local_index = {**_index(local_cpu), **_index(local_gpu)}
    server_index, server_metadata = read_reports([server_cpu, server_cpu32, server_gpu])
    local_cases = {(d, n) for d, n, _ in local_index if d != 'synthetic'}
    server_cases = {(d, n) for d, n, _ in server_index if d != 'synthetic'}
    if len(local_cases) != 1 or local_cases != server_cases:
        raise ValueError('expected one identical real-head case on both machines')
    dataset, n = next(iter(local_cases))
    if local_cpu['metadata']['dtype'] != server_metadata['dtype']:
        raise ValueError('different numeric dtype')
    if (local_cpu['metadata']['warmups'], local_cpu['metadata']['repeats']) != (
            server_metadata['warmups'], server_metadata['repeats']):
        raise ValueError('different warmup or repeat counts')
    if local_cpu['metadata']['numba_threads'] != 10:
        raise ValueError('the local Numba label expects 10 threads')
    result = []
    hashes = set()
    for machine, index, backends in (('local', local_index, CPU_BACKENDS + GPU_BACKENDS),
                                     ('server', server_index, SERVER_CPU + SERVER_GPU)):
        for backend in backends:
            r = index[(dataset, n, backend)]
            if r['median_ms'] <= 0 or not np.isfinite(r['median_ms']):
                raise ValueError(f'invalid time for {machine}: {backend}')
            hashes.add(r['input_sha256'])
            color = LOCAL_STYLES[backend]['color'] if machine == 'local' else SERVER_STYLES[backend][1]
            result.append({'machine': machine, 'backend': backend, 'label': _label(backend),
                           'rate': 1000 / r['median_ms'], 'color': color})
    if len(hashes) != 1:
        raise ValueError('different real-head inputs across machines')
    result.sort(key=lambda r: -r['rate'])
    return dataset, n, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-cpu', required=True)
    p.add_argument('--local-gpu', required=True)
    p.add_argument('--server-cpu', required=True)
    p.add_argument('--server-cpu32', required=True)
    p.add_argument('--server-gpu', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    dataset, n, records = load_records(args.local_cpu, args.local_gpu, args.server_cpu, args.server_cpu32, args.server_gpu)
    plt.rcParams.update({'font.size': 8.2, 'axes.labelsize': 8.2,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7.5, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'hatch.linewidth': 0.45})
    _set_typeface()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.4))
    fig.patch.set_facecolor('white')
    fig.text(.5, .975, 'Real head: solver performance', fontsize=11, weight='bold', va='top', ha='center')
    fig.text(.97, .965, f'{n / 1_000_000:.2f}M voxels',
             fontsize=7.8, color='#505050', va='top', ha='right')
    handles = [Patch(facecolor='#DADDE1', edgecolor='#575757', linewidth=.55, label='Local computer (Apple M5)'),
               Patch(facecolor='#DADDE1', edgecolor='#575757', linewidth=.55, hatch='///',
                     label='Server (Xeon / RTX 2080 Ti)')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.10,.928), ncol=2, frameon=False,
               fontsize=7.5, handlelength=1.5, handletextpad=.4, columnspacing=1.5, borderaxespad=0)
    fig.subplots_adjust(left=.10, right=.98, bottom=.28, top=.82)
    x = np.arange(len(records))
    for i, r in enumerate(records):
        server = r['machine'] == 'server'
        bar = ax.bar(i, r['rate'], width=.72, color=r['color'],
                     edgecolor='#555555' if server else 'white', linewidth=.45,
                     hatch='///' if server else None)[0]
        ax.annotate(f"{r['rate']:.0f}", xy=(bar.get_x()+bar.get_width()/2, r['rate']),
                    xytext=(0,3), textcoords='offset points', ha='center', va='bottom',
                    color='#222222', fontsize=5.8, fontweight='bold')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylim(0, max(r['rate'] for r in records) * 4)
    ax.set_ylabel('Calls/s (symlog)', labelpad=3)
    ax.set_xticks(x, [r['label'] for r in records], rotation=55, ha='right', rotation_mode='anchor')
    _axis_style(ax)
    ax.tick_params(axis='x', labelsize=6.5, length=0, pad=5)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save(fig, out, 'real_head_all_methods')


if __name__ == '__main__':
    main()
