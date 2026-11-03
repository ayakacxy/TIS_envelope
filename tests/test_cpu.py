import numpy as np
import pytest

from fast_tis_envelope.backends.numpy_backend import envelop_numpy
pytest.importorskip("numba")
from fast_tis_envelope.backends.numba_backend import envelop_numba
from tests.common import assert_matches_reference, test_vectors as make_vectors


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", [envelop_numpy, envelop_numba, lambda a, b: envelop_numba(a, b, parallel=True)])
def test_values_against_geometry(dtype, backend):
    x, y = make_vectors(dtype)
    result = backend(x, y)
    assert result.dtype == dtype
    assert_matches_reference(result, x, y, rtol=3e-5 if dtype == np.float32 else 1e-12,
                             atol=2e-6 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("backend", [envelop_numpy, envelop_numba, lambda a, b: envelop_numba(a, b, parallel=True)])
def test_noncontiguous_empty_and_validation(backend):
    x, y = make_vectors()
    assert_matches_reference(backend(x[::2], y[::-2]), x[::2], y[::-2])
    assert backend(np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)).shape == (0,)
    with pytest.raises(ValueError):
        backend(np.ones((2, 3)), np.ones((3, 3)))
    with pytest.raises(ValueError):
        backend(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(TypeError):
        backend(np.ones((2, 3), complex), np.ones((2, 3)))


def test_float64_dynamic_range():
    x = np.array([[1e-150, 0, 0], [1e150, 0, 0], [1e-150, 0, 0]])
    y = np.array([[0, 1e-150, 0], [0, 1e150, 0], [0, 0, 0]])
    expected = np.array([np.sqrt(2) * 1e-150, np.sqrt(2) * 1e150, 0])
    for backend in (envelop_numpy, envelop_numba):
        np.testing.assert_allclose(backend(x, y), expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("backend", [envelop_numpy, envelop_numba, lambda a, b: envelop_numba(a, b, parallel=True)])
def test_very_different_field_magnitudes(backend):
    x = np.array([[1e-20, 0, 0]] * 3, np.float32)
    y = np.array([[1e20, 0, 0], [0, 1e20, 0], [1e20, 1e20, 0]], np.float32)
    expected = np.full(3, 2e-20, np.float32)
    np.testing.assert_allclose(backend(x, y), expected, rtol=2e-6, atol=0)
    np.testing.assert_allclose(backend(y, x), expected, rtol=2e-6, atol=0)
