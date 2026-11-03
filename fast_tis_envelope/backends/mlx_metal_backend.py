"""Fused, forward-only Metal kernel for the maximum TIS envelope."""

import mlx.core as mx

from .mlx_backend import _prepare_inputs


_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    // MLX passes strides in elements. Each input may be an independent view,
    // including reversed or broadcast axes, without making contiguous copies.
    long j1 = (long)i * e1_strides[0];
    long j2 = (long)i * e2_strides[0];
    long k1 = e1_strides[1];
    long k2 = e2_strides[1];
    float3 x = float3(e1[j1], e1[j1 + k1], e1[j1 + 2 * k1]);
    float3 y = float3(e2[j2], e2[j2 + k2], e2[j2 + 2 * k2]);
    float3 ax = metal::abs(x);
    float3 ay = metal::abs(y);
    float sa = metal::max(ax.x, metal::max(ax.y, ax.z));
    float sb = metal::max(ay.x, metal::max(ay.y, ay.z));
    if (sa == 0.0f || sb == 0.0f) {
        out[i] = 0.0f;
        return;
    }
    float3 a = x / sa;
    float3 b = y / sb;
    float ab = metal::dot(a, b);
    if (ab < 0.0f) {
        a = -a;
        x = -x;
    }
    float aa = metal::dot(a, a);
    float bb = metal::dot(b, b);
    float result;
    if (metal::all(x == y)) {
        result = sa * metal::sqrt(aa) + sb * metal::sqrt(bb);
    } else {
        float3 d = x * 0.5f - y * 0.5f;
        float3 ad = metal::abs(d);
        float ds = metal::max(ad.x, metal::max(ad.y, ad.z));
        if (ds == 0.0f) {
            result = 0.0f;
        } else {
            float3 direction = d / ds;
            if (metal::dot(b, direction) > 0.0f) {
                result = sb * (2.0f * metal::sqrt(bb));
            } else if (metal::dot(a, direction) < 0.0f) {
                result = sa * (2.0f * metal::sqrt(aa));
            } else {
                float3 smaller = sa < sb ? a : b;
                float smaller_scale = sa < sb ? sa : sb;
                result = smaller_scale * (2.0f * metal::length(metal::cross(smaller, direction)) / metal::length(direction));
            }
        }
    }
    out[i] = result;
"""

_kernel = mx.fast.metal_kernel(
    name="fast_tis_envelope_strided_float32",
    input_names=["e1", "e2"],
    output_names=["out"],
    source=_SOURCE,
    ensure_row_contiguous=False,
)


def _envelope_metal(e1, e2):
    n = e1.shape[0]
    if n == 0:
        return mx.zeros((0,), dtype=mx.float32)
    return _kernel(
        inputs=[e1, e2],
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
    )[0]


def envelop_mlx_metal(e1, e2):
    """Return float32 MLX amplitudes, using a custom Metal forward kernel.

    The forward pass runs on a Metal GPU; derivatives are not implemented.
    Noncontiguous MLX input views are read directly using their strides.
    Inputs are expected to be finite; float64 inputs are converted to float32.
    """
    e1, e2 = _prepare_inputs(e1, e2)
    return _envelope_metal(e1, e2)
