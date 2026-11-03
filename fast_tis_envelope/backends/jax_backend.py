"""JAX implementation of the maximum TIS envelope for pairs of 3-D fields."""

import jax
import jax.numpy as jnp


def _norm3(x):
    """Euclidean row norm."""
    return jnp.sqrt(jnp.sum(x * x, axis=1))


@jax.jit
def envelop_jax(e1, e2):
    """Return maximum envelope amplitudes for finite arrays of shape (N, 3).

    This implementation is intended for forward evaluation.
    """
    e1, e2 = jnp.asarray(e1), jnp.asarray(e2)
    if e1.ndim != 2 or e1.shape[1] != 3 or e1.shape != e2.shape:
        raise ValueError("e1 and e2 must have identical shape (N, 3)")
    if jnp.issubdtype(e1.dtype, jnp.complexfloating) or jnp.issubdtype(e2.dtype, jnp.complexfloating):
        raise TypeError("e1 and e2 must be real")
    dtype = jnp.result_type(e1, e2, jnp.float32)
    e1, e2 = e1.astype(dtype), e2.astype(dtype)

    # Normalize each field separately so a weak field is not lost when the
    # other is many orders of magnitude stronger.
    sa = jnp.max(jnp.abs(e1), axis=1)
    sb = jnp.max(jnp.abs(e2), axis=1)
    a = e1 / jnp.where(sa > 0, sa, 1)[:, None]
    b = e2 / jnp.where(sb > 0, sb, 1)[:, None]
    flip = jnp.sum(a * b, axis=1) < 0
    a = jnp.where(flip[:, None], -a, a)
    raw_a = jnp.where(flip[:, None], -e1, e1)

    na, nb = _norm3(a), _norm3(b)
    equal = jnp.all(raw_a == e2, axis=1)

    # Subtract before normalization to preserve near-equal float32 values;
    # halving each input first avoids overflow for large opposite components.
    difference = raw_a * 0.5 - e2 * 0.5
    dscale = jnp.max(jnp.abs(difference), axis=1)
    direction = difference / jnp.where(dscale > 0, dscale, 1)[:, None]
    den = _norm3(direction)
    # E1×(E1-E2) and E2×(E1-E2) have equal magnitudes. Using the weaker
    # field avoids losing the cross product at very large magnitude ratios.
    use_a = sa < sb
    cross_a, cross_b = jnp.cross(a, direction, axis=1), jnp.cross(b, direction, axis=1)
    cross = jnp.where(use_a[:, None], cross_a, cross_b)
    general = jnp.where(use_a, sa, sb) * (2 * _norm3(cross) / jnp.where(den > 0, den, 1))
    # Compare projections on the difference, rather than subtracting nearly
    # equal squared norms. The latter can pick the wrong branch catastrophically.
    b_is_limiting = jnp.sum(b * direction, axis=1) > 0
    a_is_limiting = jnp.sum(a * direction, axis=1) < 0

    # The sum of both field norms is correct for equal and opposite fields.
    return jnp.where(equal, sa * na + sb * nb,
                     jnp.where(b_is_limiting, sb * (2 * nb), jnp.where(a_is_limiting, sa * (2 * na), general)))


if __name__ == "__main__":
    e1 = jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    e2 = jnp.array([[0, 1, 0], [1, 0, 0], [1, 1, 0]])
    print(envelop_jax(e1, e2))
