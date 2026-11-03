"""Draw print-sized CPU and GPU benchmark figures from paired measurements."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
import numpy as np


# Start with the established layout, then scale the tightly cropped output to
# 180 mm so it fits an A4 page with 15 mm margins on each side.
FIGURE_WIDTH = 170 / 25.4
OUTPUT_WIDTH_INCH = 180 / 25.4
SAVE_PAD_INCH = 0.02
PNG_DPI = 600
# Light, related shades for each implementation family and a mint Metal accent.
# Keep the fill colors consistent across figures; darken thin lines for legibility.
STYLES = {
    "numpy": {"label": "NumPy", "platform": "CPU", "color": "#AAB5C3", "marker": "o", "ls": "-"},
    "numba_serial": {"label": "Numba (1T)", "platform": "CPU", "color": "#85B9D6", "marker": "s", "ls": "-"},
    "numba_parallel": {"label": "Numba", "platform": "CPU", "color": "#5A9DCB", "marker": "^", "ls": "--"},
    "jax_cpu": {"label": "JAX", "platform": "CPU", "color": "#A3B86C", "marker": "D", "ls": "-."},
    "mlx_cpu": {"label": "MLX", "platform": "CPU", "color": "#C7AED4", "marker": "v", "ls": "-"},
    "mlx_cpu_fast": {"label": "MLX-fast", "platform": "CPU", "color": "#A28ABF", "marker": "P", "ls": "--"},
    "mlx_gpu": {"label": "MLX", "platform": "GPU", "color": "#F0C47F", "marker": "v", "ls": "-"},
    "mlx_gpu_fast": {"label": "MLX-fast", "platform": "GPU", "color": "#E99984", "marker": "s", "ls": "--"},
    "metal_gpu": {"label": "Metal", "platform": "GPU", "color": "#72BFB0", "marker": "o", "ls": "-"},
    "torch_mps": {"label": "PyTorch eager", "platform": "MPS", "color": "#DCB2C6", "marker": "D", "ls": "-."},
    "torch_mps_compiled": {"label": "PyTorch compiled", "platform": "MPS", "color": "#AE6E94", "marker": "X", "ls": "--"},
}
CPU_BACKENDS = ("numpy", "numba_serial", "numba_parallel", "jax_cpu", "mlx_cpu", "mlx_cpu_fast")
GPU_BACKENDS = ("mlx_gpu", "mlx_gpu_fast", "metal_gpu", "torch_mps", "torch_mps_compiled")


def _set_typeface():
    """Use Helvetica when installed, with portable alternatives for regeneration."""
    for family in ("Helvetica", "Arial", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            break
        except ValueError:
            continue
    plt.rcParams.update({"font.family": family, "mathtext.fontset": "custom",
                         "mathtext.rm": family, "mathtext.it": f"{family}:italic",
                         "mathtext.bf": f"{family}:bold"})


def _row_legend_order(count, columns):
    """Reorder handles for Matplotlib's column-first legend layout."""
    return [i for col in range(columns) for i in range(col, count, columns)]


def _inline_label(name, platform):
    if name.startswith("Numba (") and name.endswith(")"):
        return f"{name[:-1]}, {platform})"
    return f"{name} ({platform})"


def _index(report):
    return {(r["dataset"], r["n"], r["backend"]): r for r in report["results"]}


def _validate(cpu, gpu):
    if cpu["metadata"]["benchmark_device"] != "cpu" or gpu["metadata"]["benchmark_device"] != "gpu":
        raise ValueError("expected separate CPU and GPU measurement files")
    cpu_sources = cpu["metadata"]["source_sha256"]
    gpu_sources = gpu["metadata"]["source_sha256"]
    core_sources = ("env_numpy.py", "env_numba.py", "env_mlx.py", "env_mlx_metal.py")
    if any(name not in cpu_sources or name not in gpu_sources or cpu_sources[name] != gpu_sources[name]
           for name in core_sources):
        raise ValueError("CPU and GPU measurements used different source versions")
    if "env_jax.py" not in cpu_sources or not cpu["metadata"].get("jax_device"):
        raise ValueError("CPU measurements do not identify the JAX source and device")
    if "env_torch.py" not in gpu_sources or gpu["metadata"].get("torch_device") != "mps":
        raise ValueError("GPU measurements do not identify the PyTorch source and MPS device")
    for key in ("python", "numpy", "numba", "mlx", "machine", "platform"):
        if cpu["metadata"][key] != gpu["metadata"][key]:
            raise ValueError(f"CPU and GPU environments differ for {key}")
    for key in ("dtype", "warmups", "repeats", "seed", "synthetic", "reference"):
        if cpu["metadata"][key] != gpu["metadata"][key]:
            raise ValueError(f"CPU and GPU measurement metadata differs for {key}")
    left, right = _index(cpu), _index(gpu)
    cpu_cases = {(d, n) for d, n, _ in left}
    gpu_cases = {(d, n) for d, n, _ in right}
    if cpu_cases != gpu_cases:
        raise ValueError("CPU and GPU measurement datasets do not match")
    for dataset, n in cpu_cases:
        a = {r["input_sha256"] for (d, size, _), r in left.items() if d == dataset and size == n}
        b = {r["input_sha256"] for (d, size, _), r in right.items() if d == dataset and size == n}
        if len(a) != 1 or a != b:
            raise ValueError(f"different input arrays for {dataset}, N={n}")
        for backend in CPU_BACKENDS:
            if (dataset, n, backend) not in left:
                raise ValueError(f"missing CPU backend {backend} at {dataset}, N={n}")
        for backend in GPU_BACKENDS:
            if (dataset, n, backend) not in right:
                raise ValueError(f"missing GPU backend {backend} at {dataset}, N={n}")


def _size_label(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}k"
    return str(n)


def _axis_style(ax, grid_axis="y"):
    ax.tick_params(axis="both", which="major", labelsize=7.5, length=3, width=0.65,
                   color="#444444", pad=2.5)
    ax.tick_params(axis="both", which="minor", length=1.8, width=0.5, color="#666666")
    ax.grid(True, axis=grid_axis, which="major", linestyle="--",
            linewidth=0.55, color="#B8B8B8", alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#353535")
        spine.set_linewidth(0.7)


def _setup_axis(ax, title, ylabel, sizes):
    ax.set_title(title, loc="left", pad=5, fontsize=9.2, fontweight="bold")
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_xlabel("Voxels / call", labelpad=2)
    ax.set_xticks(np.arange(len(sizes)), [_size_label(n) for n in sizes])
    _axis_style(ax)


def _annotate_fastest_bars(ax, groups):
    """Label the fastest implementation at each input size in calls/s."""
    for bars in zip(*groups):
        winner = max(bars, key=lambda bar: bar.get_height())
        value = winner.get_height()
        ax.annotate(f"{value:,.0f}",
                    xy=(winner.get_x() + winner.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    color="#222222", fontsize=6.4, fontweight="bold")


def _contrast_color(color, minimum):
    """Darken a series color if needed against a white background."""
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    while True:
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        luminance = np.dot(linear, [0.2126, 0.7152, 0.0722])
        if 1.05 / (luminance + 0.05) >= minimum:
            return rgb
        rgb *= 0.95


def _annotate_endpoints(ax, endpoints, last_x):
    """Separate end labels with minimal shifts in log space."""
    ax.figure.canvas.draw()
    low, high = ax.get_ylim()
    axis_height_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    ordered = sorted(endpoints)
    if not ordered:
        return
    levels = np.log10([value for value, _ in ordered])

    def project(gap):
        # Isotonic regression on levels - i*gap prevents overlaps while
        # moving labels both up and down, instead of pushing everything up.
        blocks = []
        for i, level in enumerate(levels):
            blocks.append([float(level - i * gap), 1])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                right, left = blocks.pop(), blocks.pop()
                count = left[1] + right[1]
                blocks.append([(left[0] * left[1] + right[0] * right[1]) / count, count])
        fitted = [mean for mean, count in blocks for _ in range(count)]
        return np.asarray(fitted) + np.arange(len(levels)) * gap

    lo, hi = np.log10(low), np.log10(high)
    for _ in range(15):
        gap = (hi - lo) * (7.2 * 1.35) / axis_height_pt
        placed = project(gap)
        next_lo, next_hi = min(lo, placed[0] - gap / 2), max(hi, placed[-1] + gap / 2)
        if max(abs(next_lo - lo), abs(next_hi - hi)) < 1e-5:
            break
        lo, hi = next_lo, next_hi
    ax.set_ylim(10 ** lo, 10 ** hi)
    for (value, color), level in zip(ordered, placed):
        ax.annotate(f"{value:.2f}", xy=(last_x, value), xytext=(last_x + 0.18, 10 ** level),
                    textcoords="data", color=_contrast_color(color, 4.5), fontsize=7.2, va="center", ha="left",
                    arrowprops={"arrowstyle": "-", "color": _contrast_color(color, 3.0), "linewidth": 0.55,
                                "alpha": 0.8, "shrinkA": 0, "shrinkB": 2},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.2})


def _fit_print_width(fig):
    """Scale the figure and its type to the desired tight-cropped width."""
    for _ in range(4):
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer())
        width = bounds.width + 2 * SAVE_PAD_INCH
        scale = OUTPUT_WIDTH_INCH / width
        if abs(width - OUTPUT_WIDTH_INCH) < 0.05 / 25.4:
            break
        fig.set_size_inches(fig.get_size_inches() * scale)
        for artist in fig.findobj(match=Text):
            artist.set_fontsize(artist.get_fontsize() * scale)
        for artist in fig.findobj(match=Line2D):
            artist.set_linewidth(artist.get_linewidth() * scale)
            artist.set_markersize(artist.get_markersize() * scale)
            artist.set_markeredgewidth(artist.get_markeredgewidth() * scale)
        for artist in fig.findobj(match=Patch):
            artist.set_linewidth(artist.get_linewidth() * scale)


def _save(fig, output_dir, stem):
    pdf, png = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    _fit_print_width(fig)
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": SAVE_PAD_INCH}
    fig.savefig(pdf, **options, metadata={"Creator": "Matplotlib - envelope benchmark"})
    fig.savefig(png, dpi=PNG_DPI, **options)
    plt.close(fig)
    print(f"Saved {pdf} and {png}")


def _make_figure(report, backends, baseline, device_name, output_dir):
    index = _index(report)
    sizes = sorted({n for dataset, n, backend in index if dataset == "synthetic" and backend == baseline})
    positions = np.arange(len(sizes))
    metadata = report["metadata"]
    fig = plt.figure(figsize=(FIGURE_WIDTH, 6.4))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
    ax_rate = fig.add_subplot(grid[0, :])
    ax_time = fig.add_subplot(grid[1, :])
    ax_error = fig.add_subplot(grid[2, 0])
    ax_speed = fig.add_subplot(grid[2, 1])
    fig.patch.set_facecolor("white")
    fig.text(0.10, 0.975, device_name, fontsize=11, weight="bold", va="top")
    fig.text(0.97, 0.975, f"{metadata['gpu']['device_name']}  |  FP32",
             fontsize=8.2, color="#505050", va="top", ha="right")

    def method_label(backend):
        name = (f"Numba ({metadata['numba_threads']}T)" if backend == "numba_parallel"
                else STYLES[backend]["label"])
        return _inline_label(name, STYLES[backend]["platform"])

    handles = [Line2D([0], [0], color=_contrast_color(STYLES[b]["color"], 3.0), marker=STYLES[b]["marker"],
                      markerfacecolor=STYLES[b]["color"],
                      linestyle=STYLES[b]["ls"], linewidth=1.3, markersize=4,
                      label=method_label(b)) for b in backends]
    # Matplotlib fills legends column-first; interleave the handles so each
    # visible row follows the order of the corresponding plotted series.
    order = _row_legend_order(len(backends), 3)
    fig.legend(handles=[handles[i] for i in order], loc="upper left",
               bbox_to_anchor=(0.10, 0.94), ncol=3, frameon=False, fontsize=7.4,
               handlelength=1.2, columnspacing=1.5, handletextpad=0.35, borderaxespad=0)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.085, top=0.85, wspace=0.31, hspace=0.32)

    _setup_axis(ax_rate, "(a) Throughput", "Calls/s", sizes)
    ax_rate.set_xlabel("")
    width = 0.8 / len(backends)
    highest_rate = 0
    rate_bars = []
    for i, backend in enumerate(backends):
        rates = [1000 / index[("synthetic", n, backend)]["median_ms"] for n in sizes]
        highest_rate = max(highest_rate, max(rates))
        rate_bars.append(ax_rate.bar(positions - 0.4 + width * (i + 0.5), rates, width=width,
                                     color=STYLES[backend]["color"], edgecolor="white", linewidth=0.25))
    ax_rate.set_yscale("symlog", linthresh=1)
    ax_rate.set_ylim(0, highest_rate * 12)
    _annotate_fastest_bars(ax_rate, rate_bars)

    _setup_axis(ax_time, "(b) Time per call", "Time (ms)", sizes)
    ax_time.set_xlabel("")
    ax_time.set_yscale("log")
    endpoints = []
    for backend in backends:
        style = STYLES[backend]
        times = [index[("synthetic", n, backend)]["median_ms"] for n in sizes]
        ax_time.plot(positions, times, color=_contrast_color(style["color"], 3.0), marker=style["marker"],
                     markerfacecolor=style["color"],
                     linestyle=style["ls"], linewidth=1.35, markersize=3.6)
        # Label the primary CPU baselines; the other curves remain in the legend.
        if device_name == "GPU" or backend in ("numpy", "numba_parallel", "jax_cpu", "mlx_cpu"):
            endpoints.append((times[-1], style["color"]))
    ax_time.set_xlim(-0.45, len(sizes) + 0.6)
    ax_time.margins(y=0.18)
    _annotate_endpoints(ax_time, endpoints, positions[-1])

    _setup_axis(ax_error, "(c) Error vs NumPy FP64", "Error (V/m)", sizes)
    ax_error.set_yscale("log")
    max_error = [max(index[("synthetic", n, b)]["max_abs_error_vs_float64"] for b in backends) for n in sizes]
    mean_error = [max(index[("synthetic", n, b)]["mean_abs_error_vs_float64"] for b in backends) for n in sizes]
    ax_error.plot(positions, max_error, "o-", color=_contrast_color(STYLES["mlx_cpu_fast"]["color"], 3.0),
                  linewidth=1.4, markersize=3.8,
                  label="Max (worst)")
    ax_error.plot(positions, mean_error, "s-", color=_contrast_color(STYLES["metal_gpu"]["color"], 3.0),
                  linewidth=1.4, markersize=3.8,
                  label="Mean (worst)")
    positive = [x for x in max_error + mean_error if x > 0]
    if positive:
        ax_error.set_ylim(min(positive) / 1.7, max(positive) * 4)
    ax_error.legend(loc="upper left", frameon=False, fontsize=7.4, handlelength=1.6,
                    handletextpad=0.4, borderaxespad=0.35)

    target_title = "(d) Speedup vs NumPy" if device_name == "CPU" else "(d) Speedup vs PyTorch eager"
    _setup_axis(ax_speed, target_title, "Speedup (×)" if device_name == "CPU" else "Speedup (×, log)", sizes)
    targets = (("numba_parallel", "jax_cpu") if device_name == "CPU" else
               ("mlx_gpu", "mlx_gpu_fast", "metal_gpu", "torch_mps_compiled"))
    speed_baseline = baseline if device_name == "CPU" else "torch_mps"
    series = {name: [index[("synthetic", n, speed_baseline)]["median_ms"] /
                     index[("synthetic", n, name)]["median_ms"] for n in sizes]
              for name in targets}
    ax_speed.axhline(1, color="#333333", linewidth=0.75, linestyle="--")
    maximum = max(max(values) for values in series.values())
    if device_name == "CPU":
        for i, name in enumerate(targets):
            width = 0.36
            offset = (i - (len(targets) - 1) / 2) * width
            bars = ax_speed.bar(positions + offset, series[name], width=width * 0.92,
                                color=STYLES[name]["color"],
                                label=(f"Numba ({metadata['numba_threads']}T)" if name == "numba_parallel"
                                       else STYLES[name]["label"]))
            for bar, value in zip(bars, series[name]):
                label = f"{value:.1f}" if value >= 2 else f"{value:.2f}"
                ax_speed.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + maximum * 0.018,
                              label, ha="center", va="bottom", fontsize=6.4, rotation=90)
        ax_speed.set_ylim(0, maximum * 1.55)
        ax_speed.legend(loc="upper left", frameon=False, fontsize=6.5, ncol=2,
                        handlelength=0.9, handletextpad=0.35, columnspacing=0.7)
    else:
        ax_speed.set_yscale("log")
        endpoints = []
        for name in targets:
            style = STYLES[name]
            ax_speed.plot(positions, series[name], color=_contrast_color(style["color"], 3.0),
                          marker=style["marker"], markerfacecolor=style["color"],
                          linestyle=style["ls"], linewidth=1.35, markersize=3.6)
            endpoints.append((series[name][-1], style["color"]))
        ax_speed.set_xlim(-0.45, len(sizes) + 0.65)
        ax_speed.set_ylim(min(0.75, min(min(values) for values in series.values()) * 0.8), maximum * 1.35)
        ticks = [v for v in (1, 2, 5, 10, 20, 50, 100) if v < maximum * 1.35]
        ax_speed.set_yticks(ticks)
        ax_speed.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        _annotate_endpoints(ax_speed, endpoints, positions[-1])

    _save(fig, output_dir, "cpu_performance" if device_name == "CPU" else "gpu_performance")


def _make_comparison(cpu, gpu, output_dir):
    """Compare the same synthetic inputs and, if present, head-model fields."""
    metadata = cpu["metadata"]
    index = {**_index(cpu), **_index(gpu)}
    sizes = sorted({n for dataset, n, backend in index
                    if dataset == "synthetic" and backend == "numba_parallel"})
    real_cases = sorted({(dataset, n) for dataset, n, backend in index
                         if dataset != "synthetic" and backend == "numba_parallel"})
    methods = [
        ("numpy", "NumPy", "CPU"),
        ("numba_serial", "Numba (1T)", "CPU"),
        ("numba_parallel", f"Numba ({metadata['numba_threads']}T)", "CPU"),
        ("jax_cpu", "JAX", "CPU"),
        ("mlx_cpu", "MLX", "CPU"),
        ("mlx_cpu_fast", "MLX-fast", "CPU"),
        ("mlx_gpu", "MLX", "GPU"),
        ("mlx_gpu_fast", "MLX-fast", "GPU"),
        ("metal_gpu", "Metal", "GPU"),
        ("torch_mps", "PyTorch eager", "MPS"),
        ("torch_mps_compiled", "PyTorch compiled", "MPS"),
    ]
    fig, (ax_scale, ax_case) = plt.subplots(
        2, 1, figsize=(FIGURE_WIDTH, 6.2), gridspec_kw={"height_ratios": [1.03, 1]})
    fig.patch.set_facecolor("white")
    fig.text(0.105, 0.975, "CPU vs GPU", fontsize=11, weight="bold", va="top")
    fig.text(0.97, 0.975, f"{metadata['gpu']['device_name']}  |  FP32",
             fontsize=8.2, color="#505050", va="top", ha="right")
    for label, group, x in (("CPU", methods[:6], 0.10), ("GPU", methods[6:], 0.54)):
        fig.text(x, 0.925, label, fontsize=7.6, weight="bold", va="top")
        order = _row_legend_order(len(group), 2)
        handles = [Patch(facecolor=STYLES[group[i][0]]["color"], label=group[i][1]) for i in order]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, 0.902),
                   ncol=2, frameon=False, fontsize=7.0, handlelength=0.9,
                   columnspacing=1.0, handletextpad=0.35, borderaxespad=0)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.79, hspace=0.28)

    positions = np.arange(len(sizes))
    _setup_axis(ax_scale, "(a) Synthetic data", "Calls/s (symlog)", sizes)
    width = 0.88 / len(methods)
    highest = 0
    rate_bars = []
    for i, (backend, _, _) in enumerate(methods):
        rates = [1000 / index[("synthetic", n, backend)]["median_ms"] for n in sizes]
        highest = max(highest, max(rates))
        rate_bars.append(ax_scale.bar(positions - 0.44 + width * (i + 0.5), rates, width=width,
                                      color=STYLES[backend]["color"], edgecolor="white", linewidth=0.25))
    ax_scale.set_yscale("symlog", linthresh=1)
    ax_scale.set_ylim(0, highest * 4)
    _annotate_fastest_bars(ax_scale, rate_bars)

    dataset, n = real_cases[0] if real_cases else ("synthetic", sizes[-1])
    title = "(b) Real head" if real_cases else "(b) Synthetic data"
    size_label = f"{n / 1_000_000:.2f}M" if n >= 1_000_000 else _size_label(n)
    ax_case.set_title(title, loc="left", pad=5, fontsize=9.0, fontweight="bold")
    ax_case.set_title(f"{size_label} voxels", loc="right", pad=5, fontsize=7.8, color="#505050")
    values = [1000 / index[(dataset, n, backend)]["median_ms"] for backend, _, _ in methods]
    bars = ax_case.bar(np.arange(len(methods)), values, width=0.62,
                       color=[STYLES[backend]["color"] for backend, _, _ in methods])
    ax_case.set_ylabel("Calls/s (symlog)", labelpad=3)
    def case_label(name, platform):
        if name.startswith("PyTorch "):
            return f"PyTorch\n({name.removeprefix('PyTorch ')})"
        if name.startswith("Numba (") and name.endswith(")"):
            return f"Numba\n({name[7:-1]}, {platform})"
        return f"{name}\n({platform})"

    ax_case.set_xticks(np.arange(len(methods)),
                       [case_label(name, platform) for _, name, platform in methods])
    ax_case.set_yscale("symlog", linthresh=1)
    ax_case.set_ylim(0, max(values) * 4)
    _axis_style(ax_case, "y")
    ax_case.tick_params(axis="x", labelsize=6.5, length=0, pad=4)
    ax_case.axvline(5.5, color="#B8B8B8", linewidth=0.6, linestyle=":")
    for bar, value in zip(bars, values):
        ax_case.annotate(f"{value:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom",
                         color="#202020", fontsize=7.3, fontweight="bold")

    _save(fig, output_dir, "cpu_gpu_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cpu = json.loads(Path(args.cpu).read_text(encoding="utf-8"))
    gpu = json.loads(Path(args.gpu).read_text(encoding="utf-8"))
    _validate(cpu, gpu)
    plt.rcParams.update({
        "font.size": 8.2, "axes.labelsize": 8.2,
        "axes.titlesize": 9.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.transparent": False,
    })
    _set_typeface()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _make_figure(cpu, CPU_BACKENDS, "numpy", "CPU", output)
    _make_figure(gpu, GPU_BACKENDS, "mlx_gpu", "GPU", output)
    _make_comparison(cpu, gpu, output)


if __name__ == "__main__":
    main()
