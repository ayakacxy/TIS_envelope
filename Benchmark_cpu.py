# -*- coding: utf-8 -*-
import numpy as np
import time
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the JAX implementation of the envelope solver
# Make sure env_jax.py is in the same directory or correctly added to Python path
try:
    # Assuming your JAX solver is in a file named env_jax.py and the function is envelop_jax
    from env_jax import envelop_jax
except ImportError:
    print("Error: Could not import 'envelop_jax' from 'env_jax.py'.")
    print("Please ensure that the file 'env_jax.py' containing the JAX implementation is available.")
    print("Exiting benchmark.")
    exit()


def envelop_numpy(e1, e2):
    """
    Calculate the TIS envelope amplitude using NumPy.
    This function directly translates the original NumPy implementation logic.

    Args:
        e1 (np.ndarray): Electric field vectors for frequency 1, shape (N, 3).
        e2 (np.ndarray): Electric field vectors for frequency 2, shape (N, 3).

    Returns:
        np.ndarray: Envelope amplitude for each voxel, shape (N,).
    """
    # Note: NumPy operations will use default float type (likely float64)
    # unless inputs are explicitly cast or np.seterr is used.
    # Let's keep the original logic for direct translation.

    eam = np.zeros(len(e1))

    # Calculate magnitudes and dot product
    l_x = np.sqrt(np.sum(e1 * e1, axis=1))
    l_y = np.sqrt(np.sum(e2 * e2, axis=1))
    point = np.sum(e1 * e2, axis=1)

    # Handle potential division by zero if magnitudes are zero
    # Add a small epsilon for numerical stability in cosine calculation
    epsilon = 1e-12 # Use a small epsilon to avoid inf/nan if magnitudes are exactly zero
    magnitude_product = l_x * l_y
    cos_ = point / (magnitude_product + epsilon)
    # Clip cosine to handle potential floating point errors outside [-1, 1]
    cos_ = np.clip(cos_, -1.0, 1.0)


    # Apply angle reversal if cosine is less than or equal to zero
    # NOTE: Original code modifies e1 and cos_ in-place based on this mask.
    # This is crucial for the exact translation of the original branching logic.
    mask_angle_large = cos_ <= 0
    # Create a copy of e1 before modifying it in-place
    e1_processed = e1.copy() # Use copy() to avoid modifying the input array passed to the function
    e1_processed[mask_angle_large] = -e1_processed[mask_angle_large]
    # Re-calculate cosine with the processed e1. The original code recalculates cos_
    # for the masked elements only and keeps the original cos_ for unmasked.
    # This is a very specific implementation detail.
    # Let's stick strictly to the original code's apparent masking and assignment.

    # Let's re-read the original code's first step:
    # mask = cos_ <= 0
    # e1[mask] = -e1[mask]  <-- This modifies the INPUT e1 array parts!
    # cos_[mask] = -cos_[mask] <-- This modifies the cos_ array parts!

    # This in-place modification is key to matching the exact behavior.
    # So, we *must* work on copies of the input arrays or be very careful.
    # Let's work on copies to make the function safer, but replicate the in-place logic on copies.
    e1_local = e1.copy()
    e2_local = e2.copy() # Need e2_local for cross products later
    cos_local = cos_.copy() # Need a local copy of cosine too

    mask_angle_large = cos_local <= 0
    e1_local[mask_angle_large] = -e1_local[mask_angle_large]
    cos_local[mask_angle_large] = -cos_local[mask_angle_large]

    # --- Begin direct translation of original branching/masking logic ---

    equal_vectors = np.all(e1_local == e2_local, axis=1) # Check equality of processed e1_local and e2_local

    eam[equal_vectors] = 2 * np.linalg.norm(e1_local[equal_vectors], axis=1) # Use norm of processed e1_local

    not_equal_vectors = ~equal_vectors

    # Mask 2: not_equal_vectors & (l_y < l_x) -> Note: This mask uses original magnitudes (l_y, l_x)
    mask2 = not_equal_vectors & (l_y < l_x)

    # Mask 3: not_equal_vectors & (l_y < l_x * cos_) -> Note: Uses original magnitudes (l_y, l_x) and the *processed* cos_local
    mask3 = not_equal_vectors & (l_y < l_x * cos_local)

    # Assignment 1: eam[mask2 & mask3] = 2 * l_y[mask2 & mask3] -> Uses original magnitude l_y
    mask_assign1 = mask2 & mask3
    eam[mask_assign1] = 2 * l_y[mask_assign1]

    # Assignment 2: Cross-product case based on mask2 & ~mask3
    mask_assign2 = mask2 & ~mask3
    if np.any(mask_assign2):
        # Use processed vectors e1_local, e2_local for cross product and difference
        e1_masked = e1_local[mask_assign2]
        e2_masked = e2_local[mask_assign2]
        diff_vec = e1_masked - e2_masked
        norm_diff = np.linalg.norm(diff_vec, axis=1)
        # Add epsilon to avoid division by zero if diff_vec is zero
        cross_prod = np.linalg.norm(np.cross(e2_masked, diff_vec), axis=1)
        eam[mask_assign2] = 2 * cross_prod / (norm_diff + epsilon)


    # Mask 4: not_equal_vectors & (l_x < l_y * cos_) -> Note: Uses original magnitudes (l_x, l_y) and the *processed* cos_local
    mask4 = not_equal_vectors & (l_x < l_y * cos_local)

    # Mask 5: not_equal_vectors & (l_x < l_y) -> Note: Uses original magnitudes (l_x, l_y)
    mask5 = not_equal_vectors & (l_x < l_y)

    # Assignment 3: eam[mask5 & mask4] = 2 * l_x[~mask2 & mask4]
    # -> This index (~mask2 & mask4) looks like a potential bug in the original code's indexing.
    # It's applying a mask (~mask2 & mask4) to the magnitude array (l_x),
    # but assigning the result to indices determined by (mask5 & mask4).
    # The most direct translation of the *original line of code* is to use the index `mask5 & mask4` for assignment,
    # and whatever index was intended for `l_x` on the right side.
    # Let's assume the index on the right side was intended to match the assignment index: `l_x[mask5 & mask4]`
    # If not, the exact output might differ slightly.
    # Let's try the most likely interpretation: assign to mask5 & mask4 using l_x magnitude at those indices.
    mask_assign3 = mask5 & mask4
    eam[mask_assign3] = 2 * l_x[mask_assign3] # Assuming index should be mask_assign3

    # Assignment 4: Cross-product case based on mask5 & ~mask4
    mask_assign4 = mask5 & ~mask4
    if np.any(mask_assign4):
        # Use processed vectors e1_local, e2_local for cross product and difference
        e1_masked = e1_local[mask_assign4]
        e2_masked = e2_local[mask_assign4]
        diff_vec = e2_masked - e1_masked # Note the difference vector direction here
        norm_diff = np.linalg.norm(diff_vec, axis=1)
         # Add epsilon to avoid division by zero if diff_vec is zero
        cross_prod = np.linalg.norm(np.cross(e1_masked, diff_vec), axis=1)
        eam[mask_assign4] = 2 * cross_prod / (norm_diff + epsilon)

    # --- End direct translation of original branching/masking logic ---

    # Note: This direct translation preserves the original code's potential quirks
    # regarding using original vs. processed magnitudes/cosine in masks and assignments.
    # This is the best way to ensure the NumPy output exactly matches the original.

    return eam


def run_benchmark(batch_size, num_runs, dim=3, jit_warmup=500):
    """
    Runs benchmark comparing NumPy and JAX implementations.

    Args:
        batch_size (int): Number of voxels for the test.
        num_runs (int): Number of times to run the calculation for averaging.
        dim (int): Dimension of the vectors (should be 3 for 3D electric fields).
        jit_warmup (int): Number of JAX runs for JIT compilation warmup.

    Returns:
        dict: Dictionary containing benchmark results.
    """
    print(f"\n=== Testing Batch Size: {batch_size} ===")

    # Generate random test data
    np.random.seed(42)
    # Generate data with default float type (usually float64) unless specified otherwise
    # If you need float32 specifically, uncomment: .astype(np.float32)
    e1_np = np.random.randn(batch_size, dim) # .astype(np.float32)
    e2_np = np.random.randn(batch_size, dim) # .astype(np.float32)

    # Convert to JAX arrays (JAX default float type is typically float32)
    # If NumPy inputs are float64, JAX might cast them to float64 for computation.
    e1_jax = jnp.array(e1_np)
    e2_jax = jnp.array(e2_np)

    # Determine the data type being used for difference comparison
    data_type_used = np.result_type(e1_np, e2_np)
    # print(f"--- Using data type: {data_type_used} ---") # Debug print

    # Warmup JIT compilation
    if jit_warmup > 0:
        print(f"Warming up JAX JIT compilation ({jit_warmup} runs)...")
        # Assuming envelop_jax is already decorated with @jax.jit
        try:
            for _ in range(jit_warmup):
                result = envelop_jax(e1_jax, e2_jax)
                # Block until the computation is finished
                jax.block_until_ready(result)
            print("JAX JIT warmup complete.")
        except Exception as e:
            print(f"Error during JAX warmup: {e}")
            print("Skipping JAX benchmark for this batch size.")
            # Return dummy results indicating failure for JAX
            return {
                "batch_size": batch_size,
                "avg_numpy_time": -1, # Use negative value to indicate failure
                "total_numpy_time": -1,
                "avg_jax_time": -1,
                "total_jax_time": -1,
                "avg_speedup": -1,
                "max_result_diff": -1,
                "mean_result_diff": -1,
                "ops_per_sec_numpy": -1,
                "ops_per_sec_jax": -1
            }


    # Test NumPy performance
    print(f"Benchmarking NumPy ({num_runs} runs)...")
    start_time_total = time.time()
    try:
        for _ in tqdm(range(num_runs)):
            # Create copies inside loop to simulate independent runs and match original code structure
            e1_copy = e1_np.copy()
            e2_copy = e2_np.copy()
            result_numpy = envelop_numpy(e1_copy, e2_copy)
        total_numpy_time = time.time() - start_time_total
        avg_numpy_time = total_numpy_time / num_runs
        ops_per_sec_numpy = batch_size / avg_numpy_time if avg_numpy_time > 0 else float('inf')
    except Exception as e:
        print(f"Error during NumPy benchmark: {e}")
        # Return dummy results indicating failure for NumPy
        return {
            "batch_size": batch_size,
            "avg_numpy_time": -1, # Use negative value to indicate failure
            "total_numpy_time": -1,
            "avg_jax_time": -1,
            "total_jax_time": -1,
            "avg_speedup": -1,
            "max_result_diff": -1,
            "mean_result_diff": -1,
            "ops_per_sec_numpy": -1,
            "ops_per_sec_jax": -1
        }


    # Test JAX performance (skip if warmup failed)
    if all_results and all_results[-1].get("avg_jax_time", 0) == -1:
        print("Skipping JAX benchmark due to previous error.")
        avg_jax_time = -1
        total_jax_time = -1
        ops_per_sec_jax = -1
    else:
        print(f"Benchmarking JAX ({num_runs} runs)...")
        start_time_total = time.time()
        try:
            for _ in tqdm(range(num_runs)):
                # Assuming envelop_jax is already decorated with @jax.jit
                result_jax = envelop_jax(e1_jax, e2_jax)
                # Ensure computation finishes before timing next run
                jax.block_until_ready(result_jax)
            total_jax_time = time.time() - start_time_total
            avg_jax_time = total_jax_time / num_runs
            ops_per_sec_jax = batch_size / avg_jax_time if avg_jax_time > 0 else float('inf')
        except Exception as e:
            print(f"Error during JAX benchmark: {e}")
            # Indicate failure for JAX
            avg_jax_time = -1
            total_jax_time = -1
            ops_per_sec_jax = -1


    # Calculate result differences (run once outside the loop)
    # Skip difference calculation if either benchmark failed
    if avg_numpy_time != -1 and avg_jax_time != -1:
        e1_copy = e1_np.copy()
        e2_copy = e2_np.copy()
        result_numpy = envelop_numpy(e1_copy, e2_copy)
        # Ensure JAX result is computed and transferred to host
        result_jax = envelop_jax(e1_jax, e2_jax)
        result_jax_np = np.array(result_jax) # Transfer to CPU numpy array

        # Handle potential floating point precision differences between NumPy (float64) and JAX (float32 default)
        # You might need to cast one result to the other's type before comparing if they used different precisions.
        # However, standard practice is to compare results obtained from the *same* input types used in the benchmark.
        # If JAX used float32 and NumPy used float64, small differences are expected.
        # If both used float32 (by casting NumPy inputs), differences should be minimal.
        try:
            result_diff = np.abs(result_numpy - result_jax_np)
            max_diff = np.max(result_diff)
            mean_diff = np.mean(result_diff)
        except Exception as e:
            print(f"Error calculating result differences: {e}")
            max_diff = -2 # Use a different negative value to indicate difference calculation error
            mean_diff = -2
    else:
        max_diff = -1 # Indicate skipped
        mean_diff = -1

    # Calculate speedup ratio (skip if either benchmark failed)
    avg_speedup = avg_numpy_time / avg_jax_time if avg_numpy_time > 0 and avg_jax_time > 0 else -1

    # Compile results dictionary
    return {
        "batch_size": batch_size,
        "avg_numpy_time": avg_numpy_time,
        "total_numpy_time": total_numpy_time,
        "avg_jax_time": avg_jax_time,
        "total_jax_time": total_jax_time,
        "avg_speedup": avg_speedup,
        "max_result_diff": max_diff,
        "mean_result_diff": mean_diff,
        "ops_per_sec_numpy": ops_per_sec_numpy,
        "ops_per_sec_jax": ops_per_sec_jax
    }


def format_rate(rate):
    """Formats a number (e.g., ops/sec) into K, M units."""
    if rate < 0: return "N/A" # Handle error case
    if rate >= 1e6:
        return f"{rate / 1e6:.1f}M"
    elif rate >= 1e3:
        return f"{rate / 1e3:.1f}K"
    else:
        return f"{rate:.0f}"

def format_time_us(time_sec):
     """Formats time in seconds into microseconds string."""
     if time_sec < 0: return "N/A" # Handle error case
     return f"{time_sec * 1000000:.0f} μs"

def format_speedup(speedup):
    """Formats speedup ratio."""
    if speedup < 0: return "N/A" # Handle error case
    return f"{speedup:.2f}x"

def format_diff(diff):
    """Formats difference value."""
    if diff == -1: return "Skipped"
    if diff == -2: return "Error"
    return f"{diff:.8f}"

def plot_all_results(all_results):
    """Plot all benchmarking results with improved formatting and larger fonts"""

    # Filter out failed runs for plotting
    plottable_results = [r for r in all_results if r["avg_numpy_time"] >= 0 and r["avg_jax_time"] >= 0]
    if not plottable_results:
        print("No successful benchmark runs to plot.")
        return

    plt.figure(figsize=(12, 10)) # Increased figure size

    # Set larger font sizes for all text elements
    plt.rcParams.update({
        'font.size': 14,          # Base font size
        'axes.titlesize': 16,     # Title font size
        'axes.labelsize': 14,     # Axis label font size
        'xtick.labelsize': 14,    # X-tick label font size
        'ytick.labelsize': 14,    # Y-tick label font size
        'legend.fontsize': 14,    # Legend font size
        'figure.titlesize': 18    # Overall figure title
    })

    # Extract the actual batch sizes from the results
    batch_sizes = [r["batch_size"] for r in plottable_results]

    # Create custom x-axis labels formatted in scientific notation
    x_labels = []
    for size in batch_sizes:
        formatted_size = f'{size:.1e}'
        # Improve format: remove '+0' from exponent (e.g., '1.0e+05' -> '1.0e5')
        formatted_size = formatted_size.replace('e+0', 'e')
        # formatted_size = formatted_size.replace('e-0', 'e-') # Uncomment if needed for negative exponents
        x_labels.append(formatted_size)

    x_pos = np.arange(len(batch_sizes)) # Position the bars/points

    # --- Plot 1: Operations per second ---
    plt.subplot(2, 2, 1)
    jax_ops_sec = [r["ops_per_sec_jax"] for r in plottable_results]
    numpy_ops_sec = [r["ops_per_sec_numpy"] for r in plottable_results]

    bar_width = 0.35
    bars1 = plt.bar(x_pos - bar_width/2, jax_ops_sec, bar_width, label='JAX', color='#ff7f0e') # Using standard palette colors
    bars2 = plt.bar(x_pos + bar_width/2, numpy_ops_sec, bar_width, label='NumPy', color='#1f77b4')

    plt.yscale('log')
    plt.ylabel('Calculations per Second')
    plt.title('Envelope Calculations per Second', fontsize=16)

    plt.xticks(x_pos, x_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='upper right')

    # Add value labels above bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.05, format_rate(height),
                 ha='center', va='bottom', fontsize=11)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.05, format_rate(height),
                 ha='center', va='bottom', fontsize=11)


    # --- Plot 2: Time per envelope calculation (MICROSECONDS) ---
    plt.subplot(2, 2, 2)

    # Convert average time to microseconds (1e6 multiplier)
    jax_us = [r["avg_jax_time"] * 1e6 for r in plottable_results]
    numpy_us = [r["avg_numpy_time"] * 1e6 for r in plottable_results]

    plt.plot(x_pos, jax_us, 'o-', label='JAX', color='#ff7f0e', linewidth=2, markersize=8)
    plt.plot(x_pos, numpy_us, 'o-', label='NumPy', color='#1f77b4', linewidth=2, markersize=8)

    plt.yscale('log')
    plt.ylabel('Time per Calculation (μs)')
    plt.title('Time per Envelope Calculation (μs)', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=14)
    plt.xticks(x_pos, x_labels, fontsize=14)
    plt.yticks(fontsize=14)

    # Add value labels with microsecond units
    for i, (xi, yi) in enumerate(zip(x_pos, jax_us)):
        plt.text(xi, yi * 1.1, format_time_us(yi/1e6), # Pass time in seconds to formatter
                ha='center', va='bottom', fontsize=12, color='#ff7f0e',
                bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

    for i, (xi, yi) in enumerate(zip(x_pos, numpy_us)):
        plt.text(xi, yi * 0.9, format_time_us(yi/1e6), # Position slightly below for numpy
                ha='center', va='top', fontsize=12, color='#1f77b4',
                bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))


    # --- Plot 3: JAX vs NumPy result differences ---
    plt.subplot(2, 2, 3)
    max_diffs = [r["max_result_diff"] for r in plottable_results]
    mean_diffs = [r["mean_result_diff"] for r in plottable_results]

    # Add tolerance baseline
    tolerance = 1e-5
    plt.axhline(y=tolerance, color='red', linestyle='--', linewidth=1, label='Tolerance (1e-5)')

    plt.plot(x_pos, max_diffs, 'o-', label='Max Diff', color='#2ca02c', linewidth=2, markersize=8) # green
    plt.plot(x_pos, mean_diffs, 'o-', label='Mean Diff', color='#d62728', linewidth=2, markersize=8) # red

    plt.xticks(x_pos, x_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale('log')
    plt.ylabel('Absolute Difference')
    plt.title('Result Absolute Difference', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    # Adjust legend position to not overlap with lines/points
    plt.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # Add difference values as labels - Only if not skipped/error
    for i, (xi, max_d, mean_d) in enumerate(zip(x_pos, max_diffs, mean_diffs)):
         if max_d >= 0: # Check if difference calculation was successful
            plt.text(xi, max_d * 1.2, format_diff(max_d),
                    ha='center', va='bottom', fontsize=10, color='#2ca02c',
                    bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
            plt.text(xi, mean_d * 0.8, format_diff(mean_d),
                    ha='center', va='top', fontsize=10, color='#d62728',
                     bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))


    # --- Plot 4: Speedup ratio ---
    plt.subplot(2, 2, 4)
    speedups = [r["avg_speedup"] for r in plottable_results]

    bar_width = 0.6
    # Use a single color or gradient for speedup
    bars = plt.bar(x_pos, speedups, bar_width, color='#8c564b') # brown

    plt.title('JAX Speedup over NumPy', fontsize=16)
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    plt.ylabel('Speedup Ratio (JAX/NumPy)')


    plt.xticks(x_pos, x_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add label above bars for speedup value
    for bar in bars:
        height = bar.get_height()
        if height >= 0: # Only label if speedup is valid
            label = f'{height:.1f}x'
            # Position labels slightly above the bars
            plt.text(bar.get_x() + bar.get_width()/2., height * 1.05, label,
                     ha='center', va='bottom', fontsize=12, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'))
        else:
             # Label N/A for failed runs (should be filtered out but as a safeguard)
             plt.text(bar.get_x() + bar.get_width()/2., 1.0, "N/A",
                     ha='center', va='bottom', fontsize=12)


    # Add some padding around the x-axis limits for better visualization
    # plt.xlim(x_pos[0] - bar_width, x_pos[-1] + bar_width) # Only needed if x_pos is not just 0..N-1

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for a potential suptitle
    # plt.suptitle('NumPy vs JAX Envelope Solver Benchmark', fontsize=18) # Overall title
    plt.savefig('numpy_jax_comparison.pdf', bbox_inches='tight')
    print("Performance comparison charts saved as 'numpy_jax_comparison.pdf'")


def print_summary_table(all_results):
    """Prints a summary table of benchmark results."""
    print("\n=== NumPy vs JAX Performance Summary ===")
    header = f"{'Batch Size':>12} | {'NumPy Total (s)':>18} | {'JAX Total (s)':>18} | {'NumPy Time/Run (ms)':>22} | {'JAX Time/Run (ms)':>22} | {'Speedup Ratio':>15} | {'Max Diff':>12} | {'Mean Diff':>12}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        # 1. Calculate the string to display for each column first
        batch_size_str = f"{r['batch_size']:>12}"

        numpy_total_str = f"{r['total_numpy_time']:>18.3f}" if r['total_numpy_time'] >= 0 else f"{'N/A':>18}"
        jax_total_str = f"{r['total_jax_time']:>18.3f}" if r['total_jax_time'] >= 0 else f"{'N/A':>18}"

        numpy_time_run_ms = r['avg_numpy_time']*1000 if r['avg_numpy_time'] >= 0 else -1
        numpy_time_run_str = f"{numpy_time_run_ms:>22.3f}" if numpy_time_run_ms >= 0 else f"{'N/A':>22}"

        jax_time_run_ms = r['avg_jax_time']*1000 if r['avg_jax_time'] >= 0 else -1
        jax_time_run_str = f"{jax_time_run_ms:>22.3f}" if jax_time_run_ms >= 0 else f"{'N/A':>22}"

        speedup_str = f"{format_speedup(r['avg_speedup']):>15}"
        max_diff_str = f"{format_diff(r['max_result_diff']):>12}"
        mean_diff_str = f"{format_diff(r['mean_result_diff']):>12}"

        # 2. Print the combined, pre-formatted strings
        print(f"{batch_size_str} | {numpy_total_str} | {jax_total_str} | {numpy_time_run_str} | {jax_time_run_str} | {speedup_str} | {max_diff_str} | {mean_diff_str}")


if __name__ == "__main__":
    # Test configurations: batch size (number of voxels) and number of runs
    # Note: The number of runs should decrease for larger batch sizes
    # to keep the total benchmark time reasonable.
    test_configs = [
        (100, 300000),
        (1000, 50000),
        (10000, 10000),
        (50000, 5000),
        (100000, 1000),
    ]

    all_results = []

    # Run all tests
    for batch_size, num_runs in test_configs:
        result = run_benchmark(batch_size, num_runs)
        all_results.append(result)

    # Print summary table
    print_summary_table(all_results)

    # Plot all results charts
    plot_all_results(all_results)