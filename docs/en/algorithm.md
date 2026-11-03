# Algorithm

[English](algorithm.md) · [简体中文](../zh-CN/algorithm.md) · [Documentation](index.md)

## Maximum envelope

For two three-dimensional electric fields $\mathbf E_1$ and $\mathbf E_2$ at the same location, the solver returns the maximum modulation envelope amplitude over unit directions $\mathbf n$:

```math
A(\mathbf E_1,\mathbf E_2)
= 2\max_{\lVert\mathbf n\rVert=1}
\min\!\left(\left|\mathbf E_1\!\cdot\!\mathbf n\right|,
            \left|\mathbf E_2\!\cdot\!\mathbf n\right|\right).
```

Voxels are evaluated independently. Two fields of shape `(N, 3)` produce an array of shape `(N,)`; if the fields are in V/m, the output is also in V/m.

## Piecewise evaluation

Reverse one field when necessary so the angle between the fields is at most 90°. Let the resulting fields be $\mathbf a,\mathbf b$ and $\mathbf d=\mathbf a-\mathbf b$. For unequal fields:

```math
A = \begin{cases}
2\lVert\mathbf b\rVert, & \mathbf b\cdot\mathbf d > 0,\\
2\lVert\mathbf a\rVert, & \mathbf a\cdot\mathbf d < 0,\\
\dfrac{2\lVert\mathbf b\times\mathbf d\rVert}{\lVert\mathbf d\rVert}, & \text{otherwise}.
\end{cases}
```

Equal or opposite fields give $2\lVert\mathbf a\rVert$ after orientation; if either field is zero, the result is zero. The implementation scales each field independently and normalizes the difference before evaluating norms, projections and cross products, reducing intermediate overflow and underflow.

## Implementations

NumPy, JAX, PyTorch and MLX express the cases with array operations. Numba, Metal and CuPy kernels evaluate individual voxels. All backends use the same maximum-envelope definition; see [usage](usage.md) for devices, precision and compilation.

[Optimizations](optimizations.md) · [Performance](performance.md)
