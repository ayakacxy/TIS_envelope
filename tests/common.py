"""Independent geometric oracle and common test vectors."""

import numpy as np


def geometric_reference(e1, e2):
    """Maximize 2 min(|E1·n|, |E2·n|) over unit directions in float64.

    In the fields' plane an optimum occurs at a field direction or where the
    two absolute projections are equal. This is deliberately independent of
    the cross-product piecewise formula used by the implementations.
    """
    e1, e2 = np.asarray(e1, dtype=np.float64), np.asarray(e2, dtype=np.float64)
    values = np.zeros(len(e1), dtype=np.float64)
    for i, (x, y) in enumerate(zip(e1, e2)):
        scale = max(np.max(np.abs(x)), np.max(np.abs(y)))
        if scale == 0:
            continue
        x, y = x / scale, y / scale
        directions = [x, y]
        for difference in (x - y, x + y):
            norm = np.linalg.norm(difference)
            if norm > 0:
                unit = difference / norm
                directions.append(x - np.dot(x, unit) * unit)
        for direction in directions:
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                value = 2 * min(abs(np.dot(x, direction)), abs(np.dot(y, direction)))
                values[i] = max(values[i], scale * value)
    return values


def test_vectors(dtype=np.float32):
    rng = np.random.default_rng(1590)
    x = rng.normal(size=(300, 3))
    y = rng.normal(size=(300, 3))
    scale = 10.0 ** rng.uniform(-18, 18, size=(300, 1))
    x, y = x * scale, y * scale
    x[:8] = [[1, 0, 0], [1, 0, 0], [1e-9, 0, 0], [1e-9, 0, 0],
             [1, 2, 3], [1, 2, 3], [0, 0, 0], [1, 0, 0]]
    y[:8] = [[1, 0, 0], [-1, 0, 0], [0, 1e-9, 0], [0, 0, 0],
             [1, 2, 3], [-1, -2, -3], [0, 0, 0], [0, 1, 0]]
    y[8:18] = x[8:18]
    y[18:28] = -x[18:28]
    y[28:38] = 0
    x[38:48] = 0
    # Close fields include differences smaller than the old isclose tolerance.
    y[48:58] = x[48:58] * (1 + 2e-6)
    y[58:88] = x[58:88] + rng.normal(size=(30, 3)) * scale[58:88] * 1e-7
    # Regression: comparing dot(E1,E2) to |E1|² in float32 selected the
    # cross-product branch and underestimated this result by about 95%.
    x[88] = [0.0025003631599247456, -0.0027910673525184393, 0.0027729261200875044]
    y[88] = [0.002500363392755389, -0.002791067585349083, 0.002772926352918148]
    return x.astype(dtype), y.astype(dtype)


def assert_matches_reference(actual, e1, e2, *, rtol=3e-5, atol=2e-6):
    expected = geometric_reference(e1, e2)
    scale = np.maximum(np.max(np.abs(e1), axis=1), np.max(np.abs(e2), axis=1)).astype(np.float64)
    scale[scale == 0] = 1
    np.testing.assert_allclose(np.asarray(actual) / scale, expected / scale, rtol=rtol, atol=atol)
