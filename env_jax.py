import jax.numpy as jnp
from jax import jit

@jit
def envelop_jax(e1, e2):
    """Fully vectorized JAX implementation for better GPU utilization"""
    # Precompute norms and dot products
    l_x = jnp.linalg.norm(e1, axis=1)
    l_y = jnp.linalg.norm(e2, axis=1)
    dot_prod = jnp.sum(e1 * e2, axis=1)
    l_x_l_y = l_x * l_y

    # Compute cosine of the angle
    cos_angle = jnp.clip(dot_prod / l_x_l_y, -1.0, 1.0)

    # Flip e1 and cos_angle where cos_angle <= 0
    flip_mask = cos_angle <= 0
    e1_flipped = jnp.where(flip_mask[:, None], -e1, e1)
    cos_abs = jnp.abs(cos_angle)

    # Initialize eam
    eam = jnp.zeros_like(l_x)

    # Check for equal vectors
    equal_vectors = jnp.all(jnp.isclose(e1_flipped, e2), axis=1)
    
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
    e1_e2_mask2 = jnp.where(mask2[:, None], e1_flipped - e2, jnp.zeros_like(e1))
    cross_prod_mask2 = jnp.cross(jnp.where(mask2[:, None], e2, jnp.zeros_like(e2)), e1_e2_mask2)
    norm_cross_mask2 = jnp.linalg.norm(cross_prod_mask2, axis=1)
    norm_e1_e2_mask2 = jnp.linalg.norm(e1_e2_mask2, axis=1)
    norm_e1_e2_mask2 = jnp.where(norm_e1_e2_mask2 == 0, 1.0, norm_e1_e2_mask2)
    val_mask2 = 2 * norm_cross_mask2 / norm_e1_e2_mask2
    
    # For mask4 (condition 4)
    e1_e2_mask4 = jnp.where(mask4[:, None], e2 - e1_flipped, jnp.zeros_like(e1))
    cross_prod_mask4 = jnp.cross(jnp.where(mask4[:, None], e1_flipped, jnp.zeros_like(e1)), e1_e2_mask4)
    norm_cross_mask4 = jnp.linalg.norm(cross_prod_mask4, axis=1)
    norm_e1_e2_mask4 = jnp.linalg.norm(e1_e2_mask4, axis=1)
    norm_e1_e2_mask4 = jnp.where(norm_e1_e2_mask4 == 0, 1.0, norm_e1_e2_mask4)
    val_mask4 = 2 * norm_cross_mask4 / norm_e1_e2_mask4
    
    # Apply all values according to masks
    eam = jnp.where(equal_vectors, val_equal, eam)
    eam = jnp.where(mask1, val_mask1, eam)
    eam = jnp.where(mask2, val_mask2, eam)
    eam = jnp.where(mask3, val_mask3, eam)
    eam = jnp.where(mask4, val_mask4, eam)
    
    return eam

if __name__ == "__main__":
    # Example usage
    e1 = jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    e2 = jnp.array([[0, 1, 0], [1, 0, 0], [1, 1, 0]])
    result = envelop_jax(e1, e2)
    print(result)
    # Expected output:[1.4142135 1.4142135 2.828427 ]