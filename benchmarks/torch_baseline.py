"""PyTorch envelope implementation extracted from the original GPU benchmark.

Benchmark_gpu.py also initializes JAX/CUDA on import, so the unchanged function
is kept here for use with PyTorch MPS without those import-time side effects.
"""

import torch


def envelop_torch(e1, e2):
    """Original Benchmark_gpu.py tensor-operator implementation (eager)."""
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
