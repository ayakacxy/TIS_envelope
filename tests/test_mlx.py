import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from fast_tis_envelope.backends.mlx_backend import envelop_mlx, envelop_mlx_fast
from fast_tis_envelope.backends.mlx_metal_backend import envelop_mlx_metal
from tests.common import assert_matches_reference, test_vectors as make_vectors


@pytest.mark.parametrize("backend", [envelop_mlx, envelop_mlx_metal])
def test_mlx_values(backend):
    x, y = make_vectors()
    result = backend(x, y)
    mx.eval(result)
    assert result.dtype == mx.float32
    assert result.shape == (len(x),)
    assert_matches_reference(np.array(result), x, y)


@pytest.mark.parametrize("backend", [envelop_mlx, envelop_mlx_metal])
def test_mlx_views_empty_and_validation(backend):
    x, y = make_vectors()
    a, b = mx.array(x)[::2], mx.array(y)[::-2]
    assert_matches_reference(np.array(backend(a, b)), x[::2], y[::-2])
    assert backend(mx.zeros((0, 3)), mx.zeros((0, 3))).shape == (0,)
    with pytest.raises(ValueError):
        backend(mx.ones((2, 3)), mx.ones((3, 3)))
    with pytest.raises(ValueError):
        backend(mx.ones((2, 2)), mx.ones((2, 2)))
    with pytest.raises(TypeError):
        backend(mx.array([[1j, 0j, 0j]]), mx.ones((1, 3)))


@pytest.mark.parametrize("backend", [envelop_mlx, envelop_mlx_metal])
def test_mlx_very_different_field_magnitudes(backend):
    x = np.array([[1e-20, 0, 0]] * 3, np.float32)
    y = np.array([[1e20, 0, 0], [0, 1e20, 0], [1e20, 1e20, 0]], np.float32)
    expected = np.full(3, 2e-20, np.float32)
    np.testing.assert_allclose(np.array(backend(x, y)), expected, rtol=2e-6, atol=0)
    np.testing.assert_allclose(np.array(backend(y, x)), expected, rtol=2e-6, atol=0)


def test_metal_compilation():
    x = mx.array([[1.1, 0.3, -0.4], [0.5, -0.7, 1.2]])
    y = mx.array([[0.4, 1.2, 0.2], [-0.1, 0.3, 0.7]])
    result = mx.compile(envelop_mlx_metal)(x, y)
    np.testing.assert_allclose(np.array(result), np.array(envelop_mlx(x, y)), rtol=2e-5, atol=1e-6)


def test_mlx_builtin_vmap():
    x = mx.reshape(mx.arange(24, dtype=mx.float32) / 9 + 1, (2, 4, 3))
    y = x * 0.37
    actual = mx.vmap(envelop_mlx)(x, y)
    expected = mx.reshape(envelop_mlx(mx.reshape(x, (-1, 3)), mx.reshape(y, (-1, 3))), (2, 4))
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-6, atol=1e-6)


def test_metal_accepts_strided_input_views():
    rng = np.random.default_rng(1228)
    x = rng.normal(size=(257, 3)).astype(np.float32)
    y = rng.normal(size=(257, 3)).astype(np.float32)
    mx_x, mx_y = mx.array(x), mx.array(y)
    pair = mx.stack([mx_x, mx_y], axis=-1)
    mx.eval(pair)
    cases = [
        (pair[:, :, 0], pair[:, :, 1], x, y),
        (mx_x[::-1], mx_y[::-1], x[::-1], y[::-1]),
        (mx_x[:, ::-1], mx_y[:, ::-1], x[:, ::-1], y[:, ::-1]),
        (mx_x[1::2], mx_y[-2::-2], x[1::2], y[-2::-2]),
        (mx.broadcast_to(mx_x[:1], (len(x), 3)), mx_y, np.broadcast_to(x[:1], x.shape), y),
        (mx.transpose(mx.array(x.T)), mx.transpose(mx.array(y.T)), x, y),
        (pair[:, :, 0], mx_y, x, y),
    ]
    for a, b, expected_a, expected_b in cases:
        actual = np.array(envelop_mlx_metal(a, b))
        assert_matches_reference(actual, expected_a, expected_b)
    compiled = mx.compile(envelop_mlx_metal)(pair[:, :, 0], pair[:, :, 1])
    assert_matches_reference(np.array(compiled), x, y)


@pytest.mark.parametrize("count", [1, 2, 31, 32, 255, 256, 257])
def test_metal_strided_dispatch_boundaries(count):
    rng = np.random.default_rng(count)
    x = rng.normal(size=(count, 3)).astype(np.float32)
    y = rng.normal(size=(count, 3)).astype(np.float32)
    pair = mx.stack([mx.array(x), mx.array(y)], axis=-1)
    actual = np.array(envelop_mlx_metal(pair[:, :, 0], pair[:, :, 1]))
    assert_matches_reference(actual, x, y)


def test_mlx_fast_forward_values():
    rng = np.random.default_rng(811)
    x = rng.normal(size=(1000, 3)).astype(np.float32)
    y = rng.normal(size=(1000, 3)).astype(np.float32)
    scales = (10 ** rng.uniform(-3, 2, size=(1000, 1))).astype(np.float32)
    x, y = x * scales, y * scales
    y[:10] = x[:10]
    y[10:20] = -x[10:20]
    y[20:30] = 0
    y[30:40] = x[30:40] + rng.normal(size=(10, 3)).astype(np.float32) * scales[30:40] * 1e-6
    x[40] = [0.0025003631599247456, -0.0027910673525184393, 0.0027729261200875044]
    y[40] = [0.002500363392755389, -0.002791067585349083, 0.002772926352918148]
    assert_matches_reference(np.array(envelop_mlx_fast(x, y)), x, y)
    assert_matches_reference(np.array(envelop_mlx_fast(mx.array(x)[::2], mx.array(y)[::2])), x[::2], y[::2])
    assert envelop_mlx_fast(mx.zeros((0, 3)), mx.zeros((0, 3))).shape == (0,)
