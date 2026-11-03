# 算法原理

[English](../en/algorithm.md) · [简体中文](algorithm.md) · [文档](index.md)

## 最大包络

对同一位置的两组三维电场 $\mathbf E_1$ 和 $\mathbf E_2$，求解器返回所有单位方向 $\mathbf n$ 上的最大调制包络幅值：

```math
A(\mathbf E_1,\mathbf E_2)
= 2\max_{\lVert\mathbf n\rVert=1}
\min\!\left(\left|\mathbf E_1\!\cdot\!\mathbf n\right|,
            \left|\mathbf E_2\!\cdot\!\mathbf n\right|\right).
```

每个体素独立求解。输入为两组形状 `(N, 3)` 的电场，输出为 `(N,)` 的包络幅值；如果电场的单位是 V/m，输出单位也是 V/m。

## 分段计算

将一组电场反向，使两者夹角不超过 90°，记调整后的电场为 $\mathbf a,\mathbf b$，并令 $\mathbf d=\mathbf a-\mathbf b$。当两者不相等时：

```math
A = \begin{cases}
2\lVert\mathbf b\rVert, & \mathbf b\cdot\mathbf d > 0,\\
2\lVert\mathbf a\rVert, & \mathbf a\cdot\mathbf d < 0,\\
\dfrac{2\lVert\mathbf b\times\mathbf d\rVert}{\lVert\mathbf d\rVert}, & \text{其他情况}.
\end{cases}
```

相等或反向的电场经方向调整后得到
$2\lVert\mathbf a\rVert$；任一电场为零时结果为零。实现中分别缩放两组电场，并对差向量归一化，再计算范数、投影和叉积，以减少中间运算的上溢与下溢。

## 实现方式

NumPy、JAX、PyTorch 和 MLX 使用数组操作表达分段条件；Numba、Metal 和 CuPy
内核按体素计算。各后端使用相同的最大包络定义，设备、精度和编译选项见[使用说明](usage.md)。

[优化实现](optimizations.md) · [性能测试](performance.md)
