"""Vectorized CPU reference for the maximum TIS envelope."""

import numpy as np


def _prepare_inputs(e1, e2):
    e1, e2 = np.asarray(e1), np.asarray(e2)
    if e1.ndim != 2 or e1.shape[1] != 3 or e1.shape != e2.shape:
        raise ValueError("e1 and e2 must have identical shape (N, 3)")
    if not (np.issubdtype(e1.dtype, np.number) and np.issubdtype(e2.dtype, np.number)):
        raise TypeError("e1 and e2 must be real numbers")
    if np.iscomplexobj(e1) or np.iscomplexobj(e2):
        raise TypeError("e1 and e2 must be real")
    dtype = np.result_type(e1.dtype, e2.dtype, np.float32)
    dtype = np.float32 if dtype.itemsize <= 4 else np.float64
    return e1.astype(dtype, copy=False), e2.astype(dtype, copy=False)


def envelop_numpy(e1, e2):
    """Compute the envelope on CPU with NumPy, returning float32 or float64.

    Real input arrays must have identical (N, 3) shape and finite values.
    Float16 and smaller are promoted to float32; float64 is retained.
    """
    e1, e2 = _prepare_inputs(e1, e2)
    sa, sb = np.max(np.abs(e1), axis=1), np.max(np.abs(e2), axis=1)
    a = e1 / np.where(sa > 0, sa, 1)[:, None]
    b = e2 / np.where(sb > 0, sb, 1)[:, None]
    flip = np.sum(a * b, axis=1) < 0
    a = np.where(flip[:, None], -a, a)
    raw_a = np.where(flip[:, None], -e1, e1)
    na, nb = np.sqrt(np.sum(a * a, axis=1)), np.sqrt(np.sum(b * b, axis=1))
    equal = np.all(raw_a == e2, axis=1)

    difference = raw_a * 0.5 - e2 * 0.5
    dscale = np.max(np.abs(difference), axis=1)
    direction = difference / np.where(dscale > 0, dscale, 1)[:, None]
    den = np.sqrt(np.sum(direction * direction, axis=1))
    use_a = sa < sb
    cross = np.cross(np.where(use_a[:, None], a, b), direction)
    general = np.where(use_a, sa, sb) * (2 * np.sqrt(np.sum(cross * cross, axis=1)) / np.where(den > 0, den, 1))
    b_is_limiting = np.sum(b * direction, axis=1) > 0
    a_is_limiting = np.sum(a * direction, axis=1) < 0
    return np.where(equal, sa * na + sb * nb,
                    np.where(b_is_limiting, sb * (2 * nb), np.where(a_is_limiting, sa * (2 * na), general)))
