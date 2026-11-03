"""Standalone CUDA envelope kernels for the isolated server benchmark.

The server-existing version extracts the scalar helper from cupy_loss.py;
projection and reward are intentionally excluded from this envelope benchmark.
"""
from pathlib import Path
import cupy as cp
import numpy as np


def _existing_kernel(path):
    text = Path(path).read_text(encoding='utf-8')
    marker = '__device__ __forceinline__ float envelope_value_n4_strict('
    start = text.index(marker)
    end = text.index('\n            extern "C" __global__', start)
    helper = text[start:end]
    source = helper + r'''
    extern "C" __global__ void envelope_existing(const float* a, const float* b, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        out[i] = envelope_value_n4_strict(a[3*i], a[3*i+1], a[3*i+2], b[3*i], b[3*i+1], b[3*i+2]);
    }
    '''
    return cp.RawKernel(source, 'envelope_existing')


_STABLE = r'''
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
'''


def make_calls(existing_path):
    kernels = {'cupy_existing': _existing_kernel(existing_path),
               'cupy_stable': cp.RawKernel(_STABLE, 'envelope_stable')}
    def call(name, a, b):
        n = len(a)
        out = cp.empty((n,), dtype=cp.float32)
        if n:
            kernels[name](((n + 255) // 256,), (256,), (a, b, out, np.int32(n)))
        return out
    return {name: (lambda a, b, name=name: call(name, a, b)) for name in kernels}
