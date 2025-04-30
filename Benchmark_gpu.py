import jax
import jax.numpy as jnp
import numpy as np
import torch
import time
from env_jax import envelop_jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Configure GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

# JAX setup
jax.config.update('jax_platform_name', 'gpu')
print(f"JAX devices available: {jax.devices()}")
print(f"JAX is using: {jax.devices()[0]}")

# PyTorch setup
torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {torch_device}")


def envelop_torch(e1, e2):
    """Optimized PyTorch implementation with JIT"""
    # Precompute norms and dot products
    l_x = torch.norm(e1, dim=1)
    l_y = torch.norm(e2, dim=1)
    dot_prod = torch.sum(e1 * e2, dim=1)
    l_x_l_y = l_x * l_y

    # Compute cosine of the angle
    cos_angle = torch.clamp(dot_prod / l_x_l_y, -1.0, 1.0)

    # Flip e1 and cos_angle where cos_angle <= 0
    flip_mask = cos_angle <= 0
    e1_flipped = torch.where(flip_mask.unsqueeze(1), -e1, e1)
    cos_abs = torch.abs(cos_angle)

    # Initialize eam
    eam = torch.zeros_like(l_x)

    # Check for equal vectors
    equal_vectors = torch.all(torch.isclose(e1_flipped, e2), dim=1)
    
    # Compute conditions for all cases
    not_equal = ~equal_vectors
    l_y_lt_l_x = l_y < l_x
    l_x_lt_l_y = ~l_y_lt_l_x & not_equal
    l_y_lt_l_x_cos = l_y < l_x * cos_abs
    l_x_lt_l_y_cos = l_x < l_y * cos_abs
    
    # Define all masks
    mask1 = not_equal & l_y_lt_l_x & l_y_lt_l_x_cos
    mask2 = not_equal & l_y_lt_l_x & ~l_y_lt_l_x_cos
    mask3 = not_equal & l_x_lt_l_y & l_x_lt_l_y_cos
    mask4 = not_equal & l_x_lt_l_y & ~l_x_lt_l_y_cos
    
    # Calculate all possible values
    val_equal = 2 * l_x
    val_mask1 = 2 * l_y
    val_mask3 = 2 * l_x
    
    # For mask2 (condition 2)
    e1_e2_mask2 = torch.where(mask2.unsqueeze(1), e1_flipped - e2, torch.zeros_like(e1))
    cross_prod_mask2 = torch.cross(torch.where(mask2.unsqueeze(1), e2, torch.zeros_like(e2)), e1_e2_mask2, dim=1)
    norm_cross_mask2 = torch.norm(cross_prod_mask2, dim=1)
    norm_e1_e2_mask2 = torch.norm(e1_e2_mask2, dim=1)
    norm_e1_e2_mask2 = torch.where(norm_e1_e2_mask2 == 0, torch.ones_like(norm_e1_e2_mask2), norm_e1_e2_mask2)
    val_mask2 = 2 * norm_cross_mask2 / norm_e1_e2_mask2
    
    # For mask4 (condition 4)
    e1_e2_mask4 = torch.where(mask4.unsqueeze(1), e2 - e1_flipped, torch.zeros_like(e1))
    cross_prod_mask4 = torch.cross(torch.where(mask4.unsqueeze(1), e1_flipped, torch.zeros_like(e1)), e1_e2_mask4, dim=1)
    norm_cross_mask4 = torch.norm(cross_prod_mask4, dim=1)
    norm_e1_e2_mask4 = torch.norm(e1_e2_mask4, dim=1)
    norm_e1_e2_mask4 = torch.where(norm_e1_e2_mask4 == 0, torch.ones_like(norm_e1_e2_mask4), norm_e1_e2_mask4)
    val_mask4 = 2 * norm_cross_mask4 / norm_e1_e2_mask4
    
    # Apply all values according to masks
    eam = torch.where(equal_vectors, val_equal, eam)
    eam = torch.where(mask1, val_mask1, eam)
    eam = torch.where(mask2, val_mask2, eam)
    eam = torch.where(mask3, val_mask3, eam)
    eam = torch.where(mask4, val_mask4, eam)
    
    return eam

def run_benchmark(batch_size, num_runs, dim=3, jit_warmup=500):
    print(f"\n=== Testing with voxel count: {batch_size} ===")
    
    # Use different random seeds for each batch size
    seed = int(batch_size / 1000)  # Create a seed based on batch size
    np.random.seed(seed)
    print(f"Using random seed: {seed}")
    
    # Generate random test data
    e1_np = np.random.randn(batch_size, dim).astype(np.float32)
    e2_np = np.random.randn(batch_size, dim).astype(np.float32)
    
    # Convert to JAX arrays
    e1_jax = jnp.array(e1_np)
    e2_jax = jnp.array(e2_np)
    
    # Convert to PyTorch tensors
    e1_torch = torch.tensor(e1_np, device=torch_device)
    e2_torch = torch.tensor(e2_np, device=torch_device)
    
    # JIT warmup
    if jit_warmup > 0:
        print("Warming up JAX JIT compilation...")
        for _ in range(jit_warmup):
            result = envelop_jax(e1_jax, e2_jax)
            jax.block_until_ready(result)
    
        print("Warming up PyTorch JIT compilation...")
        for _ in range(jit_warmup):
            result = envelop_torch(e1_torch, e2_torch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Test JAX performance
    print(f"JAX test ({num_runs} runs)...")
    start_time_total = time.time()
    for _ in tqdm(range(num_runs)):
        result_jax = envelop_jax(e1_jax, e2_jax)
        jax.block_until_ready(result_jax)
    total_jax_time = time.time() - start_time_total
    avg_jax_time = total_jax_time / num_runs
    
    # Test PyTorch performance
    print(f"PyTorch test ({num_runs} runs)...")
    start_time_total = time.time()
    for _ in tqdm(range(num_runs)):
        result_torch = envelop_torch(e1_torch, e2_torch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    total_torch_time = time.time() - start_time_total
    avg_torch_time = total_torch_time / num_runs
    
    # Analyze result differences
    result_jax_np = np.array(result_jax)
    result_torch_np = result_torch.cpu().numpy()
    result_diff = np.abs(result_jax_np - result_torch_np)
    max_diff = np.max(result_diff)
    mean_diff = np.mean(result_diff)
    
    # Find the index of maximum difference
    max_diff_idx = np.argmax(result_diff)
    
    # Print detailed difference analysis
    print("\n--- Detailed Difference Analysis ---")
    print(f"Max difference: {max_diff:.10f} at index {max_diff_idx}")
    print(f"Mean difference: {mean_diff:.10f}")
    
    # Analyze inputs and outputs at the maximum difference
    print(f"\nValues at maximum difference (index {max_diff_idx}):")
    print(f"  Input e1: {e1_np[max_diff_idx]}")
    print(f"  Input e2: {e2_np[max_diff_idx]}")
    print(f"  JAX result: {result_jax_np[max_diff_idx]:.10f}")
    print(f"  PyTorch result: {result_torch_np[max_diff_idx]:.10f}")
    print(f"  Absolute difference: {result_diff[max_diff_idx]:.10f}")
    
    # Calculate relative error (when result is not zero)
    if result_jax_np[max_diff_idx] != 0:
        rel_error = result_diff[max_diff_idx] / abs(result_jax_np[max_diff_idx])
        print(f"  Relative error: {rel_error:.10f}")
    
    # Calculate error distribution
    percentile_values = [50, 90, 95, 99]
    percentiles = np.percentile(result_diff, percentile_values)
    print("\nError distribution:")
    for p, val in zip(percentile_values, percentiles):
        print(f"  {p}th percentile: {val:.10f}")
    
    # Calculate the proportion of non-zero differences
    epsilon = 1e-7  # Tiny value threshold
    nonzero_diff_ratio = np.mean(result_diff > epsilon)
    print(f"Proportion of values with difference > {epsilon}: {nonzero_diff_ratio:.6f}")
    
    # Calculate speedup ratio
    jax_torch_speedup = avg_torch_time / avg_jax_time
    torch_jax_speedup = avg_jax_time / avg_torch_time
    
    # Print performance results
    print("\n--- Performance Results ---")
    print(f"Batch size: {batch_size}")
    print(f"Average JAX time: {avg_jax_time:.8f}s, Average PyTorch time: {avg_torch_time:.8f}s")
    
    if jax_torch_speedup > 1:
        print(f"JAX is {jax_torch_speedup:.2f}x faster than PyTorch")
    else:
        print(f"PyTorch is {torch_jax_speedup:.2f}x faster than JAX")

    return {
        "batch_size": batch_size,
        "avg_jax_time": avg_jax_time,
        "total_jax_time": total_jax_time,
        "avg_torch_time": avg_torch_time,
        "total_torch_time": total_torch_time,
        "jax_torch_speedup": jax_torch_speedup,
        "max_result_diff": max_diff,
        "mean_result_diff": mean_diff,
        "ops_per_sec_jax": 1.0 / avg_jax_time,
        "ops_per_sec_torch": 1.0 / avg_torch_time,
        "percentile_diffs": dict(zip(percentile_values, percentiles)),
        "nonzero_diff_ratio": nonzero_diff_ratio,
        "max_diff_idx": max_diff_idx,
        "max_diff_inputs": {
            "e1": e1_np[max_diff_idx].tolist(),
            "e2": e2_np[max_diff_idx].tolist()
        },
        "max_diff_outputs": {
            "jax": float(result_jax_np[max_diff_idx]),
            "pytorch": float(result_torch_np[max_diff_idx])
        }
    }

def validate_consistency(num_samples=1000):
    """Validate the consistency between JAX and PyTorch implementations, analyzing errors under different conditions"""
    print("\n=== Validating Implementation Consistency ===")
    
    # Test various special cases
    test_cases = [
        # Random general case
        {"name": "Random Uniform", "gen_func": lambda n: np.random.uniform(-1, 1, (n, 3)).astype(np.float32)},
        # Values close to zero
        {"name": "Near Zero", "gen_func": lambda n: np.random.uniform(-1e-5, 1e-5, (n, 3)).astype(np.float32)},
        # Large values
        {"name": "Large Values", "gen_func": lambda n: np.random.uniform(-1e2, 1e2, (n, 3)).astype(np.float32)},
        # Nearly parallel vectors
        {"name": "Nearly Parallel", "gen_func": lambda n: generate_nearly_parallel(n)},
        # Nearly perpendicular vectors
        {"name": "Nearly Perpendicular", "gen_func": lambda n: generate_nearly_perpendicular(n)},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\nTesting {case['name']} case:")
        # Generate test data
        e1_np = case["gen_func"](num_samples)
        e2_np = case["gen_func"](num_samples)
        
        # Convert to JAX and PyTorch formats
        e1_jax = jnp.array(e1_np)
        e2_jax = jnp.array(e2_np)
        e1_torch = torch.tensor(e1_np, device=torch_device)
        e2_torch = torch.tensor(e2_np, device=torch_device)
        
        # Run the calculations
        result_jax = envelop_jax(e1_jax, e2_jax)
        result_torch = envelop_torch(e1_torch, e2_torch)
        
        # Wait for calculations to finish
        jax.block_until_ready(result_jax)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Analyze result differences
        result_jax_np = np.array(result_jax)
        result_torch_np = result_torch.cpu().numpy()
        result_diff = np.abs(result_jax_np - result_torch_np)
        max_diff = np.max(result_diff)
        mean_diff = np.mean(result_diff)
        max_diff_idx = np.argmax(result_diff)
        
        print(f"  Max difference: {max_diff:.10f}")
        print(f"  Mean difference: {mean_diff:.10f}")
        print(f"  Inputs at max diff: e1={e1_np[max_diff_idx]}, e2={e2_np[max_diff_idx]}")
        print(f"  Outputs at max diff: JAX={result_jax_np[max_diff_idx]:.10f}, PyTorch={result_torch_np[max_diff_idx]:.10f}")
        
        results.append({
            "case": case["name"],
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "inputs": {
                "e1": e1_np[max_diff_idx].tolist(),
                "e2": e2_np[max_diff_idx].tolist()
            },
            "outputs": {
                "jax": float(result_jax_np[max_diff_idx]),
                "pytorch": float(result_torch_np[max_diff_idx])
            }
        })
    
    return results

def generate_nearly_parallel(n):
    """Generate nearly parallel vectors"""
    base = np.random.randn(n, 3).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    base = base / (norms + 1e-10)
    # Add small perturbation
    perturbation = np.random.randn(n, 3).astype(np.float32) * 1e-4
    result = base + perturbation
    # Normalize again
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / (norms + 1e-10)

def generate_nearly_perpendicular(n):
    """Generate nearly perpendicular vector pairs"""
    v1 = np.random.randn(n, 3).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(v1, axis=1, keepdims=True)
    v1 = v1 / (norms + 1e-10)
    
    # Generate perpendicular vectors
    random_perp = np.random.randn(n, 3).astype(np.float32)
    dots = np.sum(v1 * random_perp, axis=1, keepdims=True)
    v2 = random_perp - dots * v1
    
    # Normalize
    norms = np.linalg.norm(v2, axis=1, keepdims=True)
    v2 = v2 / (norms + 1e-10)
    
    # Add small perturbation to make them not perfectly perpendicular
    v2 = v2 + v1 * 1e-4
    norms = np.linalg.norm(v2, axis=1, keepdims=True)
    return v2 / (norms + 1e-10)

# Updated plotting function with expanded axes for the execution time plot
def plot_all_results(all_results):
    """Plot all benchmarking results with expanded axes, improved formatting, and larger fonts"""
    plt.figure(figsize=(10, 6))
    
    # Set larger font sizes for all text elements
    plt.rcParams.update({
        'font.size': 14,          # Base font size
        'axes.titlesize': 16,     # Title font size
        'axes.labelsize': 14,     # Axis label font size
        'xtick.labelsize': 14,    # X-tick label font size
        'ytick.labelsize': 14,    # Y-tick label font size
        'legend.fontsize': 14,    # Legend font size
    })

    # Extract the actual batch sizes from the results
    batch_sizes = [r["batch_size"] for r in all_results]
    
    # Create custom x-axis labels
    x_labels = []
    for size in batch_sizes:
        # Format to scientific notation with one decimal place (e.g., '1.0e+05')
        formatted_size = f'{size:.1e}'
        # Remove the '+0' part from the exponent if present (e.g., '1.0e+05' -> '1.0e5')
        # This handles positive exponents < 10.
        formatted_size = formatted_size.replace('e+0', 'e')
        # If you also had negative exponents with leading zero (e.g., e-04), you might need:
        # formatted_size = formatted_size.replace('e-0', 'e-')
        
        x_labels.append(formatted_size)

    # 1. Operations per second
    plt.subplot(2, 2, 1)
    jax_ops_sec = [r["ops_per_sec_jax"] for r in all_results]
    torch_ops_sec = [r["ops_per_sec_torch"] for r in all_results]
    
    x = np.arange(len(batch_sizes)) if len(batch_sizes) > 1 else [0]
    bar_width = 0.4
    
    plt.bar([i - bar_width / 2 for i in x], jax_ops_sec, bar_width, label='JAX', color='red')
    plt.bar([i + bar_width / 2 for i in x], torch_ops_sec, bar_width, label='PyTorch', color='blue')
    plt.yscale('log')
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max*1.15)  # Increase to 5% extra space for labels
    x_min, x_max = plt.xlim()
    plt.xlim(x_min+0.05, x_max-0.05)  # Add padding to x-axis
    plt.title('Envelope Calculations per Second', fontsize=16, fontweight='bold')
    
    plt.xticks(x, x_labels, fontsize=14) # Use the modified x_labels
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=14,bbox_to_anchor=(0.38, 0.4))
    
    def format_number(num):
        if num >= 1e6:
            return f"{num / 1e6:.1f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.1f}K"
        else:
            return f"{num:.0f}"

    for i, v in enumerate(jax_ops_sec):
        plt.text(i - bar_width / 2, v * 1.05, format_number(v), ha='center', fontsize=11)
    for i, v in enumerate(torch_ops_sec):
        plt.text(i + bar_width / 2, v * 1.05, format_number(v), ha='center', fontsize=11)

    # 2. Time per envelope calculation (MICROSECONDS) - with expanded axes
    plt.subplot(2, 2, 2)
    
    # Convert to microseconds (1000x multiplier compared to milliseconds)
    jax_us = [r["avg_jax_time"] * 1000000 for r in all_results]  # Convert seconds to μs
    torch_us = [r["avg_torch_time"] * 1000000 for r in all_results]  # Convert seconds to μs

    x = np.arange(len(batch_sizes))

    plt.plot(x, jax_us, 'o-', label='JAX', color='red', linewidth=2)
    plt.plot(x, torch_us, 'o-', label='PyTorch', color='blue', linewidth=2)

    plt.yscale('log')
      
    plt.title('Time per Envelope Calculation (μs)', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=14)
    plt.xticks(x, x_labels, fontsize=14) # Use the modified x_labels
    plt.yticks(fontsize=14)

    # EXPANDED AXES: Add padding to x and y axes for the second plot
    # Get current axis limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    
    # Add 15% padding to x-axis on both sides
    x_padding = (x_max - x_min) * 0.07
    plt.xlim(x_min - x_padding, x_max + x_padding)
    
    # Add 20% padding to y-axis (only at the top since this is a log scale)
    plt.ylim(y_min, y_max * 1.25)

    # Add value labels with microsecond units and improved positioning
    for i, (xi, yi) in enumerate(zip(x, jax_us)):
        # Horizontal offset to prevent overlapping with data points
        h_offset = -0.1 if i < len(x)/2 else 0.1  # Move left for first half, right for second half
        plt.text(xi + h_offset, yi * 1.1, f"{yi:.0f} μs", 
                ha='center', va='bottom', fontsize=12, color='red',
                bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
    
    for i, (xi, yi) in enumerate(zip(x, torch_us)):
        h_offset = -0.1 if i < len(x)/2 else 0.1  # Move left for first half, right for second half
        plt.text(xi + h_offset, yi * 1.1, f"{yi:.0f} μs", 
                ha='center', va='bottom', fontsize=12, color='blue',
                bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

    # 3. JAX vs PyTorch result differences
    plt.subplot(2, 2, 3)
    max_diffs = [r["max_result_diff"] for r in all_results]
    mean_diffs = [r["mean_result_diff"] for r in all_results]

    # Add tolerance baseline
    tolerance = 1e-5
    plt.axhline(y=tolerance, color='red', linestyle='--', linewidth=1, label='Tolerance (1e-5)')

    plt.plot(x, max_diffs, 'o-', label='Max Diff', color='purple', linewidth=2)
    plt.plot(x, mean_diffs, 'o-', label='Mean Diff', color='green', linewidth=2)

    # Use the custom x-axis labels
    plt.xticks(x, x_labels, fontsize=14) # Use the modified x_labels
    plt.yticks(fontsize=14)
    plt.yscale('log')
    plt.title('Result Absolute Difference', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=14,bbox_to_anchor=(0.25, 0.55))

    # Mark differences exceeding tolerance
    for i, (y_max, y_mean) in enumerate(zip(max_diffs, mean_diffs)):
        if y_max > tolerance:
            plt.text(i, y_max * 1.2, f'High!', ha='center', color='red', fontsize=10)
        if y_mean > tolerance:
            plt.text(i, y_mean * 0.8, f'High!', ha='center', color='red', fontsize=10)

    # 4. Speedup ratio - with improved label placement
    plt.subplot(2, 2, 4)
    speedups = [r["jax_torch_speedup"] for r in all_results]

    bar_width = 0.6
    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars = plt.bar(x, speedups, bar_width, color=colors)
    
    plt.title('JAX Speedup over PyTorch', fontsize=16, fontweight='bold')
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1)

    # Use the custom x-axis labels
    plt.xticks(x, x_labels, fontsize=14) # Use the modified x_labels
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis range to make room for labels
    max_speedup = max(speedups)
    plt.ylim(0, max_speedup * 1.15)  # Increase to 25% extra space for labels
    
    # Improved label placement to avoid overlap with bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f'{height:.1f}x'
        if height < 1:
            label = f'{1/height:.1f}x (PyTorch)'
            plt.text(bar.get_x() + bar.get_width() / 2., height - 0.5, label, 
                     ha='center', va='bottom', fontsize=12)
        else:
            # Position labels well above the bars with sufficient padding
            y_pos = height + max_speedup * 0.02  # Increase vertical offset to 7% of max height
            plt.text(bar.get_x() + bar.get_width() / 2., y_pos, label, 
                     ha='center', va='bottom', fontsize=12, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'))


    # Add more space between subplots to prevent overlap with larger text
    plt.tight_layout()
    
    plt.savefig('jax_pytorch_comparison.pdf')
    print("Performance comparison charts saved as 'jax_pytorch_comparison.pdf' ")

def print_summary_table(all_results):
    """Print summary table of benchmarking results"""
    print("\n=== JAX vs PyTorch Performance Comparison Summary ===")
    print(f"{'Voxel Count':>10} | {'JAX Total(s)':>15} | {'PyTorch Total(s)':>15} | {'JAX Time/op(ms)':>15} | {'PyTorch Time/op(ms)':>15} | {'JAX/PyTorch Ratio':>15}")
    print("-" * 95)
    for r in all_results:
        jax_faster = r["jax_torch_speedup"] > 1
        ratio_str = f"{r['jax_torch_speedup']:.2f}x (JAX)" if jax_faster else f"{1/r['jax_torch_speedup']:.2f}x (PyTorch)"
        print(f"{r['batch_size']:>10} | {r['total_jax_time']:>15.3f} | {r['total_torch_time']:>15.3f} | {r['avg_jax_time']*1000:>15.3f} | {r['avg_torch_time']*1000:>15.3f} | {ratio_str:>18}")

if __name__ == "__main__":
    # First run consistency validation
    consistency_results = validate_consistency(num_samples=10000)
    
    # Then run performance tests
    test_configs = [
        (10000, 10000),     # 10000 voxels, 1000 iterations
        (100000, 10000),     # 100000 voxels, 100 iterations
        (1000000, 10000),     # 1000000 voxels, 10 iterations
        (1500000,10000),     # 1500000 voxels, 10 iterations
        (2000000, 10000)       # 2000000 voxels, 5 iterations
    ]
    
    all_results = []
    
    # Run all tests
    for batch_size, num_runs in test_configs:
        result = run_benchmark(batch_size, num_runs)
        all_results.append(result)
    
    # Print summary table
    print_summary_table(all_results)
    
    # Plot all results
    plot_all_results(all_results)
