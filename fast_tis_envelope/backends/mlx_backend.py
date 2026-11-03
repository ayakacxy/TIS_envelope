"""Maximum TIS envelope using MLX's built-in, differentiable operations."""

import mlx.core as mx


def _prepare_inputs(e1, e2):
    e1 = e1 if isinstance(e1, mx.array) else mx.asarray(e1)
    e2 = e2 if isinstance(e2, mx.array) else mx.asarray(e2)
    if e1.ndim != 2 or e1.shape[1] != 3 or e1.shape != e2.shape:
        raise ValueError("e1 and e2 must have identical shape (N, 3)")
    if mx.issubdtype(e1.dtype, mx.complexfloating) or mx.issubdtype(e2.dtype, mx.complexfloating):
        raise TypeError("e1 and e2 must be real")
    return e1.astype(mx.float32), e2.astype(mx.float32)


def _norm3(x, y, z):
    return mx.sqrt(x * x + y * y + z * z)


def _envelope_ops(e1, e2):
    # Explicit components avoid repeated reductions over the length-3 axis.
    x0, x1, x2 = e1[:, 0], e1[:, 1], e1[:, 2]
    y0, y1, y2 = e2[:, 0], e2[:, 1], e2[:, 2]
    sa = mx.maximum(mx.abs(x0), mx.maximum(mx.abs(x1), mx.abs(x2)))
    sb = mx.maximum(mx.abs(y0), mx.maximum(mx.abs(y1), mx.abs(y2)))
    asafe, bsafe = mx.where(sa > 0, sa, 1), mx.where(sb > 0, sb, 1)
    a0, a1, a2 = x0 / asafe, x1 / asafe, x2 / asafe
    b0, b1, b2 = y0 / bsafe, y1 / bsafe, y2 / bsafe
    flip = a0 * b0 + a1 * b1 + a2 * b2 < 0
    a0, a1, a2 = mx.where(flip, -a0, a0), mx.where(flip, -a1, a1), mx.where(flip, -a2, a2)
    x0, x1, x2 = mx.where(flip, -x0, x0), mx.where(flip, -x1, x1), mx.where(flip, -x2, x2)

    na, nb = _norm3(a0, a1, a2), _norm3(b0, b1, b2)
    equal = (x0 == y0) & (x1 == y1) & (x2 == y2)
    d0, d1, d2 = x0 * 0.5 - y0 * 0.5, x1 * 0.5 - y1 * 0.5, x2 * 0.5 - y2 * 0.5
    dscale = mx.maximum(mx.abs(d0), mx.maximum(mx.abs(d1), mx.abs(d2)))
    dsafe = mx.where(dscale > 0, dscale, 1)
    d0, d1, d2 = d0 / dsafe, d1 / dsafe, d2 / dsafe
    den = _norm3(d0, d1, d2)
    use_a = sa < sb
    v0, v1, v2 = mx.where(use_a, a0, b0), mx.where(use_a, a1, b1), mx.where(use_a, a2, b2)
    c0, c1, c2 = v1 * d2 - v2 * d1, v2 * d0 - v0 * d2, v0 * d1 - v1 * d0
    general = mx.where(use_a, sa, sb) * (2 * _norm3(c0, c1, c2) / mx.where(den > 0, den, 1))
    b_is_limiting = b0 * d0 + b1 * d1 + b2 * d2 > 0
    a_is_limiting = a0 * d0 + a1 * d1 + a2 * d2 < 0
    return mx.where(equal, sa * na + sb * nb,
                    mx.where(b_is_limiting, sb * (2 * nb), mx.where(a_is_limiting, sa * (2 * na), general)))


_compiled_envelope_ops = mx.compile(_envelope_ops)


def envelop_mlx(e1, e2):
    """Return stable float32 MLX amplitudes from real (N, 3) fields.

    Forward-only implementation using compiled built-in MLX operations.
    Inputs must be finite; float64 inputs are converted to float32.
    """
    e1, e2 = _prepare_inputs(e1, e2)
    return _compiled_envelope_ops(e1, e2)


def _envelope_fast_ops(e1, e2):
    x0, x1, x2 = e1[:, 0], e1[:, 1], e1[:, 2]
    y0, y1, y2 = e2[:, 0], e2[:, 1], e2[:, 2]
    flip = x0 * y0 + x1 * y1 + x2 * y2 < 0
    x0, x1, x2 = mx.where(flip, -x0, x0), mx.where(flip, -x1, x1), mx.where(flip, -x2, x2)
    nx = mx.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    ny = mx.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
    equal = (x0 == y0) & (x1 == y1) & (x2 == y2)
    d0, d1, d2 = x0 - y0, x1 - y1, x2 - y2
    d2sum = d0 * d0 + d1 * d1 + d2 * d2
    c0, c1, c2 = y1 * d2 - y2 * d1, y2 * d0 - y0 * d2, y0 * d1 - y1 * d0
    general = 2 * mx.sqrt(c0 * c0 + c1 * c1 + c2 * c2) * mx.rsqrt(mx.where(d2sum > 0, d2sum, 1))
    b_is_limiting = y0 * d0 + y1 * d1 + y2 * d2 > 0
    a_is_limiting = x0 * d0 + x1 * d1 + x2 * d2 < 0
    return mx.where(equal, 2 * nx, mx.where(b_is_limiting, 2 * ny, mx.where(a_is_limiting, 2 * nx, general)))


_compiled_fast_ops = mx.compile(_envelope_fast_ops)


def envelop_mlx_fast(e1, e2):
    """Fast float32 MLX forward path for ordinary field magnitudes.

    The caller must ensure intermediate products and squared differences do
    not overflow or underflow. For very small/large or extremely close fields,
    use envelop_mlx instead; the fast path does not check its numerical range.
    """
    e1, e2 = _prepare_inputs(e1, e2)
    return _compiled_fast_ops(e1, e2)
