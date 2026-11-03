"""CUDA maximum-envelope backends, using CuPy RawKernel.

The existing server formula is extracted from
/home/cxy/MRI2TIS/GNN_TIS/GNN_TIS/losses/cupy_loss.py
(source SHA-256 8adc240a442e2e2da1279e94a8bf97d9b2d1486977c1b803b3c5a18b98de39e1).
The stable kernel follows the same algorithm as env_mlx_metal.py.
"""

import cupy as cp
import numpy as np


_SERVER_SOURCE = r"""
__device__ __forceinline__ float envelope_value_n4_strict(
                float e1x0, float e1y0, float e1z0,
                const float e2x, const float e2y, const float e2z
            ) {
                const float eps = 1.0e-12f;
                const float lx = sqrtf(e1x0 * e1x0 + e1y0 * e1y0 + e1z0 * e1z0);
                const float ly = sqrtf(e2x * e2x + e2y * e2y + e2z * e2z);
                const float point = e1x0 * e2x + e1y0 * e2y + e1z0 * e2z;
                float denom = lx * ly;
                if (denom < eps) denom = eps;
                float cosv = point / denom;
                if (cosv <= 0.0f) {
                    e1x0 = -e1x0; e1y0 = -e1y0; e1z0 = -e1z0;
                    cosv = -cosv;
                }
                float val = 0.0f;
                const bool equal_vectors =
                    (e1x0 == e2x) && (e1y0 == e2y) && (e1z0 == e2z);
                if (equal_vectors) {
                    val = 2.0f * lx;
                } else {
                    if (ly <= lx) {
                        if (ly < lx * cosv) {
                            val = 2.0f * ly;
                        } else {
                            const float dx = e1x0 - e2x;
                            const float dy = e1y0 - e2y;
                            const float dz = e1z0 - e2z;
                            const float cx = e2y * dz - e2z * dy;
                            const float cy = e2z * dx - e2x * dz;
                            const float cz = e2x * dy - e2y * dx;
                            float dn = sqrtf(dx * dx + dy * dy + dz * dz);
                            if (dn < eps) dn = eps;
                            val = 2.0f * sqrtf(cx * cx + cy * cy + cz * cz) / dn;
                        }
                    }
                    if (lx < ly) {
                        if (lx < ly * cosv) {
                            val = 2.0f * lx;
                        } else {
                            const float dx = e2x - e1x0;
                            const float dy = e2y - e1y0;
                            const float dz = e2z - e1z0;
                            const float cx = e1y0 * dz - e1z0 * dy;
                            const float cy = e1z0 * dx - e1x0 * dz;
                            const float cz = e1x0 * dy - e1y0 * dx;
                            float dn = sqrtf(dx * dx + dy * dy + dz * dz);
                            if (dn < eps) dn = eps;
                            val = 2.0f * sqrtf(cx * cx + cy * cy + cz * cz) / dn;
                        }
                    }
                }
                return val;
            }
extern "C" __global__ void envelope_existing(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = envelope_value_n4_strict(a[3*i], a[3*i+1], a[3*i+2], b[3*i], b[3*i+1], b[3*i+2]);
}
"""

_STABLE_SOURCE = r"""
extern "C" __global__ void envelope_stable(const float* e1, const float* e2, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x0=e1[3*i], x1=e1[3*i+1], x2=e1[3*i+2];
    float y0=e2[3*i], y1=e2[3*i+1], y2=e2[3*i+2];
    float sa=fmaxf(fabsf(x0),fmaxf(fabsf(x1),fabsf(x2)));
    float sb=fmaxf(fabsf(y0),fmaxf(fabsf(y1),fabsf(y2)));
    if (sa == 0.0f || sb == 0.0f) { out[i] = 0.0f; return; }
    float a0=x0/sa, a1=x1/sa, a2=x2/sa;
    float b0=y0/sb, b1=y1/sb, b2=y2/sb;
    if (a0*b0+a1*b1+a2*b2 < 0.0f) {
        a0=-a0; a1=-a1; a2=-a2; x0=-x0; x1=-x1; x2=-x2;
    }
    float aa=a0*a0+a1*a1+a2*a2, bb=b0*b0+b1*b1+b2*b2;
    if (x0==y0 && x1==y1 && x2==y2) {
        out[i]=sa*sqrtf(aa)+sb*sqrtf(bb); return;
    }
    float d0=x0*0.5f-y0*0.5f, d1=x1*0.5f-y1*0.5f, d2=x2*0.5f-y2*0.5f;
    float ds=fmaxf(fabsf(d0),fmaxf(fabsf(d1),fabsf(d2)));
    if (ds == 0.0f) { out[i]=0.0f; return; }
    d0/=ds; d1/=ds; d2/=ds;
    if (b0*d0+b1*d1+b2*d2 > 0.0f) { out[i]=sb*(2.0f*sqrtf(bb)); return; }
    if (a0*d0+a1*d1+a2*d2 < 0.0f) { out[i]=sa*(2.0f*sqrtf(aa)); return; }
    bool use_a=sa<sb;
    float v0=use_a?a0:b0, v1=use_a?a1:b1, v2=use_a?a2:b2;
    float c0=v1*d2-v2*d1, c1=v2*d0-v0*d2, c2=v0*d1-v1*d0;
    float normc=sqrtf(c0*c0+c1*c1+c2*c2), normd=sqrtf(d0*d0+d1*d1+d2*d2);
    out[i]=(use_a?sa:sb)*(2.0f*normc/normd);
}
"""

_server_kernel = cp.RawKernel(_SERVER_SOURCE, "envelope_existing")
_stable_kernel = cp.RawKernel(_STABLE_SOURCE, "envelope_stable")


def _prepare_inputs(e1, e2):
    e1, e2 = cp.asarray(e1), cp.asarray(e2)
    if e1.ndim != 2 or e1.shape[1] != 3 or e1.shape != e2.shape:
        raise ValueError("e1 and e2 must have identical shape (N, 3)")
    if cp.issubdtype(e1.dtype, cp.complexfloating) or cp.issubdtype(e2.dtype, cp.complexfloating):
        raise TypeError("e1 and e2 must be real")
    # RawKernel reads row-major inputs; copies, if needed, occur here.
    return cp.ascontiguousarray(e1, dtype=cp.float32), cp.ascontiguousarray(e2, dtype=cp.float32)


def _run(e1, e2, kernel):
    # A caller may use a non-default CUDA device. Keep both inputs and the
    # output on that device rather than silently launching on the current one.
    arrays = [value for value in (e1, e2) if isinstance(value, cp.ndarray)]
    devices = {value.device.id for value in arrays}
    if len(devices) > 1:
        raise ValueError("e1 and e2 must be on the same CUDA device")
    device = next(iter(devices)) if devices else cp.cuda.Device().id
    with cp.cuda.Device(device):
        e1, e2 = _prepare_inputs(e1, e2)
        n = e1.shape[0]
        if n > np.iinfo(np.int32).max:
            raise ValueError("too many rows for the CUDA kernel")
        out = cp.empty((n,), dtype=cp.float32)
        if n:
            kernel(((n + 255) // 256,), (256,), (e1, e2, out, np.int32(n)))
        return out


def envelop_cupy(e1, e2):
    """Stable float32 CUDA envelope for finite real fields of shape (N, 3).

    Inputs may be NumPy or CuPy arrays; output is a CuPy array on the current
    CUDA device. A noncontiguous input is copied before the kernel launch.
    """
    return _run(e1, e2, _stable_kernel)


def envelop_cupy_server(e1, e2):
    """Original server CuPy formula, for ordinary finite field magnitudes.

    This formula uses fixed epsilon guards and unscaled squared components;
    for very small/large or nearly equal fields use envelop_cupy instead.
    """
    return _run(e1, e2, _server_kernel)
