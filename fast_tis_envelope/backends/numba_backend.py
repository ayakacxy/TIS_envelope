"""Numba-compiled serial and parallel CPU maximum-envelope solvers."""

import math

import numpy as np
from numba import njit, prange

from .numpy_backend import _prepare_inputs


@njit(inline="always")
def _envelope_point(x0, x1, x2, y0, y1, y2):
    sa = max(abs(x0), abs(x1), abs(x2))
    sb = max(abs(y0), abs(y1), abs(y2))
    if sa == 0 or sb == 0:
        return 0.0
    a0, a1, a2 = x0 / sa, x1 / sa, x2 / sa
    b0, b1, b2 = y0 / sb, y1 / sb, y2 / sb
    ab = a0 * b0 + a1 * b1 + a2 * b2
    if ab < 0:
        a0, a1, a2 = -a0, -a1, -a2
        x0, x1, x2 = -x0, -x1, -x2
    aa = a0 * a0 + a1 * a1 + a2 * a2
    bb = b0 * b0 + b1 * b1 + b2 * b2
    if x0 == y0 and x1 == y1 and x2 == y2:
        return sa * math.sqrt(aa) + sb * math.sqrt(bb)
    d0, d1, d2 = x0 * 0.5 - y0 * 0.5, x1 * 0.5 - y1 * 0.5, x2 * 0.5 - y2 * 0.5
    ds = max(abs(d0), abs(d1), abs(d2))
    if ds == 0:
        return 0.0
    d0, d1, d2 = d0 / ds, d1 / ds, d2 / ds
    if b0 * d0 + b1 * d1 + b2 * d2 > 0:
        return sb * (2.0 * math.sqrt(bb))
    if a0 * d0 + a1 * d1 + a2 * d2 < 0:
        return sa * (2.0 * math.sqrt(aa))
    if sa < sb:
        v0, v1, v2, s = a0, a1, a2, sa
    else:
        v0, v1, v2, s = b0, b1, b2, sb
    c0 = v1 * d2 - v2 * d1
    c1 = v2 * d0 - v0 * d2
    c2 = v0 * d1 - v1 * d0
    cross_norm = math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
    direction_norm = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    return s * (2.0 * cross_norm / direction_norm)


@njit(cache=True)
def _serial(e1, e2):
    output = np.empty(e1.shape[0], dtype=e1.dtype)
    for i in range(e1.shape[0]):
        output[i] = _envelope_point(e1[i, 0], e1[i, 1], e1[i, 2], e2[i, 0], e2[i, 1], e2[i, 2])
    return output


@njit(parallel=True, cache=True)
def _parallel(e1, e2):
    output = np.empty(e1.shape[0], dtype=e1.dtype)
    for i in prange(e1.shape[0]):
        output[i] = _envelope_point(e1[i, 0], e1[i, 1], e1[i, 2], e2[i, 0], e2[i, 1], e2[i, 2])
    return output


def envelop_numba(e1, e2, *, parallel=False):
    """Compute CPU amplitudes with a Numba JIT kernel.

    With parallel=True, distribute voxels over Numba's CPU thread pool. The
    first call for each dtype/layout compiles a specialization and is slower.
    Input and output requirements match :func:`env_numpy.envelop_numpy`.
    """
    e1, e2 = _prepare_inputs(e1, e2)
    return _parallel(e1, e2) if parallel else _serial(e1, e2)
