"""CUDA tests for both standalone CuPy envelope backends."""
import numpy as np
import pytest

cp = pytest.importorskip('cupy')
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CUDA GPU unavailable', allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip('CUDA GPU unavailable', allow_module_level=True)

from fast_tis_envelope.backends.cupy_backend import envelop_cupy, envelop_cupy_server
from fast_tis_envelope.backends.numpy_backend import envelop_numpy
from tests.common import assert_matches_reference, test_vectors as make_vectors


def test_cupy_stable_reference_and_extreme_scales():
    x, y = make_vectors()
    result = envelop_cupy(x, y)
    assert result.dtype == cp.float32 and result.shape == (len(x),)
    assert_matches_reference(cp.asnumpy(result), x, y)
    x = np.array([[1e-20, 0, 0]] * 3, np.float32)
    y = np.array([[1e20, 0, 0], [0, 1e20, 0], [1e20, 1e20, 0]], np.float32)
    expected = np.full(3, 2e-20, np.float32)
    np.testing.assert_allclose(cp.asnumpy(envelop_cupy(x, y)), expected, rtol=2e-6, atol=0)
    np.testing.assert_allclose(cp.asnumpy(envelop_cupy(y, x)), expected, rtol=2e-6, atol=0)


def test_cupy_stable_wide_range_and_near_equal_fields():
    rng = np.random.default_rng(419)
    n = 200_000
    scale = 10.0 ** rng.uniform(-20, 20, (n, 1))
    x = (rng.normal(size=(n, 3)) * scale).astype(np.float32)
    ratio = 10.0 ** rng.uniform(-6, 6, (n, 1))
    y = (rng.normal(size=(n, 3)) * scale * ratio).astype(np.float32)
    k = n // 10
    y[:k] = (x[:k].astype(np.float64) + rng.normal(size=(k, 3)) * scale[:k] * 1e-7).astype(np.float32)
    y[k:k + k // 2] = x[k:k + k // 2]
    y[k + k // 2:2 * k] = -x[k + k // 2:2 * k]
    y[2 * k:2 * k + k // 2] = 0
    x[2 * k + k // 2:3 * k] = 0
    reference = envelop_numpy(x.astype(np.float64), y.astype(np.float64))
    actual = cp.asnumpy(envelop_cupy(x, y)).astype(np.float64)
    assert np.isfinite(actual).all()
    positive = reference > 0
    assert np.all(actual[~positive] == 0)
    assert np.max(np.abs(actual[positive] - reference[positive]) / reference[positive]) < 2e-4


@pytest.mark.parametrize('fn', [envelop_cupy, envelop_cupy_server])
def test_cupy_bounded_fields_and_strided_views(fn):
    rng = np.random.default_rng(182)
    x = rng.normal(size=(257, 3)).astype(np.float32) * 0.1
    y = rng.normal(size=(257, 3)).astype(np.float32) * 0.1
    xa, ya = cp.asarray(x), cp.asarray(y)
    cases = [(xa, ya, x, y), (xa[::-2], ya[::2], x[::-2], y[::2]),
             (xa[:, ::-1], ya[:, ::-1], x[:, ::-1], y[:, ::-1]),
             (cp.broadcast_to(xa[:1], xa.shape), ya, np.broadcast_to(x[:1], x.shape), y)]
    for a, b, ref_a, ref_b in cases:
        result = fn(a, b)
        assert_matches_reference(cp.asnumpy(result), ref_a, ref_b)


@pytest.mark.parametrize('fn', [envelop_cupy, envelop_cupy_server])
def test_cupy_empty_and_validation(fn):
    assert fn(cp.zeros((0, 3)), cp.zeros((0, 3))).shape == (0,)
    with pytest.raises(ValueError):
        fn(cp.ones((2, 3)), cp.ones((3, 3)))
    with pytest.raises(ValueError):
        fn(cp.ones((2, 2)), cp.ones((2, 2)))
    with pytest.raises(TypeError):
        fn(cp.asarray([[1j, 0j, 0j]]), cp.ones((1, 3)))
