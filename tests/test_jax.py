import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fast_tis_envelope.backends.jax_backend import envelop_jax
from tests.common import assert_matches_reference, test_vectors as make_vectors


def test_jax_values_equal_and_zero():
    x, y = make_vectors()
    assert_matches_reference(np.array(envelop_jax(x, y)), x, y)
    zero = jnp.zeros((1, 3), dtype=jnp.float32)
    one = jnp.array([[1.0, 0, 0]])
    for a, b, expected in ((zero, zero, 0), (one, zero, 0), (one, one, 2), (one, -one, 2)):
        np.testing.assert_allclose(np.array(envelop_jax(a, b)), [expected], atol=1e-6)


def test_jax_very_different_field_magnitudes():
    x = np.array([[1e-20, 0, 0]] * 3, np.float32)
    y = np.array([[1e20, 0, 0], [0, 1e20, 0], [1e20, 1e20, 0]], np.float32)
    expected = np.full(3, 2e-20, np.float32)
    np.testing.assert_allclose(np.array(envelop_jax(x, y)), expected, rtol=2e-6, atol=0)
    np.testing.assert_allclose(np.array(envelop_jax(y, x)), expected, rtol=2e-6, atol=0)
