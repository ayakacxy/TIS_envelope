import jax
import jax.numpy as jnp
import numpy as np
from env_jax import envelop_jax


N = 10
key = jax.random.PRNGKey(0)
E1_np = np.random.rand(N, 3) * 0.1  # Example NumPy array
E2_np = np.random.rand(N, 3) * 0.1

# Convert NumPy arrays to JAX arrays to enable JAX acceleration
E1_jax = jnp.array(E1_np)
E2_jax = jnp.array(E2_np)

# Solve for the interference envelope
envelope = envelop_jax(E1_jax, E2_jax)

print("E1 shape:", E1_jax.shape)
print("E2 shape:", E2_jax.shape)
print("Envelope shape:", envelope.shape) # Output shape should be (N,)