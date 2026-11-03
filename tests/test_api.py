"""Public API contracts and backend dispatch."""
import importlib.util
import numpy as np
import pytest

from fast_tis_envelope import envelope, resolve_backend
from tests.common import assert_matches_reference, test_vectors as make_vectors


def test_cpu_auto_and_explicit():
    x, y = make_vectors()
    assert resolve_backend(x, y) == ("numba" if importlib.util.find_spec("numba") else "numpy")
    for name in ("auto", "numpy", "numba"):
        if name == "numba" and importlib.util.find_spec("numba") is None:
            continue
        result = envelope(x, y, backend=name)
        assert isinstance(result, np.ndarray)
        assert_matches_reference(result, x, y)


def test_cpu_dtype_views_and_empty():
    x, y = make_vectors(np.float64)
    result = envelope(x[::-2], y[::2], backend="numpy")
    assert result.dtype == np.float64
    assert_matches_reference(result, x[::-2], y[::2], rtol=1e-12, atol=1e-12)
    assert envelope(np.empty((0, 3)), np.empty((0, 3)), backend="numpy").shape == (0,)


def test_bad_options_and_shapes():
    x = np.ones((2, 3), np.float32)
    with pytest.raises(ValueError, match="backend"):
        envelope(x, x, backend="bogus")
    with pytest.raises(ValueError, match="device"):
        envelope(x, x, device="invalid")
    with pytest.raises(ValueError, match="parallel"):
        envelope(x, x, backend="numpy", parallel=True)
    with pytest.raises(ValueError, match="shape"):
        envelope(x, x[:1], backend="numpy")
    with pytest.raises(TypeError, match="real"):
        envelope(x.astype(complex), x, backend="numpy")


def test_numba_parallel_option():
    pytest.importorskip("numba")
    x, y = make_vectors()
    assert_matches_reference(envelope(x, y, backend="numba", parallel=True), x, y)
    with pytest.raises(TypeError, match="parallel"):
        envelope(x, y, backend="numba", parallel="yes")


def test_torch_native_and_reference():
    torch = pytest.importorskip("torch")
    x, y = make_vectors()
    a, b = torch.tensor(x), torch.tensor(y)
    assert resolve_backend(a, b) == "torch"
    result = envelope(a, b)
    assert result.device == a.device and result.dtype == torch.float32
    assert_matches_reference(result.numpy(), x, y)
    empty = torch.empty((0, 3))
    assert envelope(empty, empty).shape == (0,)
    with pytest.raises(TypeError, match="convert explicitly"):
        envelope(a, b, backend="numpy")


def test_jax_native_and_reference():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    x, y = make_vectors()
    a, b = jnp.asarray(x), jnp.asarray(y)
    assert resolve_backend(a, b) == "jax"
    assert_matches_reference(np.asarray(envelope(a, b)), x, y)


def test_mlx_native_and_reference():
    mx = pytest.importorskip("mlx.core")
    x, y = make_vectors()
    a, b = mx.array(x), mx.array(y)
    assert resolve_backend(a, b) in ("mlx", "metal")
    assert_matches_reference(np.array(envelope(a, b)), x, y)


def test_torch_compile_matches_eager():
    torch = pytest.importorskip("torch")
    x, y = make_vectors()
    a, b = torch.tensor(x), torch.tensor(y)
    result = envelope(a, b, compile=True)
    torch.testing.assert_close(result, envelope(a, b), rtol=3e-5, atol=1e-6)


def test_compile_option_validation():
    x = np.ones((2, 3), np.float32)
    with pytest.raises(ValueError, match="NumPy"):
        envelope(x, x, backend="numpy", compile=True)
    with pytest.raises(TypeError, match="compile"):
        envelope(x, x, compile="yes")


def test_cupy_native_and_host_gpu():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA GPU unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA GPU unavailable")
    x, y = make_vectors()
    a, b = cp.asarray(x), cp.asarray(y)
    assert resolve_backend(a, b) == "cupy"
    assert resolve_backend(x, y, device="gpu") == "cupy"
    assert_matches_reference(cp.asnumpy(envelope(a, b)), x, y)
    assert_matches_reference(cp.asnumpy(envelope(x, y, device="gpu")), x, y)


def test_auto_falls_back_to_numpy_when_numba_not_installed(monkeypatch):
    import fast_tis_envelope.selection as selection
    original_find_spec = selection.importlib.util.find_spec
    monkeypatch.setattr(selection.importlib.util, "find_spec",
                        lambda name: None if name == "numba" else original_find_spec(name))
    x = np.array([[1., 0., 0.]], dtype=np.float32)
    assert resolve_backend(x, x) == "numpy"
    np.testing.assert_array_equal(envelope(x, x), [2.])


@pytest.mark.parametrize("device", ["cpu", "cpu:0"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_explicit_cpu_device(backend, device):
    if backend == "numba":
        pytest.importorskip("numba")
    x, y = make_vectors()
    assert resolve_backend(x, y, backend=backend, device=device) == backend
    assert_matches_reference(envelope(x, y, backend=backend, device=device), x, y)


@pytest.mark.parametrize("backend,device", [
    ("numpy", "cuda:0"), ("numba", "gpu"), ("metal", "cpu"),
    ("cupy", "cpu"), ("mlx", "cuda:0"), ("torch", "cpu:1"),
    ("metal", "mps:1"), ("numpy", "cpu:2"), ("jax", "mps"),
])
def test_incompatible_backend_or_index(backend, device):
    x = np.ones((1, 3), np.float32)
    with pytest.raises(ValueError, match="device"):
        envelope(x, x, backend=backend, device=device)


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:x", "cuda:01", "cuda:", "cpu:0:1"])
def test_invalid_device_syntax(device):
    with pytest.raises(ValueError, match="device"):
        resolve_backend([[1, 0, 0]], [[1, 0, 0]], device=device)


def test_torch_select_and_move_to_cpu():
    torch = pytest.importorskip("torch")
    x, y = make_vectors()
    result = envelope(x, y, backend="torch", device="cpu:0")
    assert result.device.type == "cpu"
    assert_matches_reference(result.numpy(), x, y)
    assert resolve_backend(result, result, device="cpu") == "torch"
    with pytest.raises(ValueError, match="CUDA device"):
        envelope(x, y, backend="torch", device=f"cuda:{torch.cuda.device_count()}")


def test_jax_select_cpu():
    jax = pytest.importorskip("jax")
    x, y = make_vectors()
    result = envelope(x, y, backend="jax", device="cpu:0")
    assert result.device.platform == "cpu"
    assert_matches_reference(np.asarray(result), x, y)
    with pytest.raises(ValueError, match="device"):
        envelope(x, y, backend="jax", device=f"cpu:{len(jax.local_devices(backend='cpu'))}")


def test_mlx_select_device():
    mx = pytest.importorskip("mlx.core")
    x, y = make_vectors()
    for backend, device in (("mlx", "cpu:0"), ("mlx", "gpu:0"), ("metal", "mps:0")):
        actual = np.array(envelope(x, y, backend=backend, device=device))
        assert_matches_reference(actual, x, y)


def test_cupy_select_cuda_device():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
        if not count:
            pytest.skip("CUDA GPU unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA GPU unavailable")
    x, y = make_vectors()
    a = cp.asarray(x)
    result = envelope(a, y, backend="cupy", device="cuda:0")
    assert result.device.id == 0
    assert_matches_reference(cp.asnumpy(result), x, y)
    with pytest.raises(ValueError, match="CUDA device"):
        envelope(x, y, backend="cupy", device=f"cuda:{count}")
    if count >= 2:
        with cp.cuda.Device(1):
            a, b = cp.asarray(x), cp.asarray(y)
        result = envelope(a, b, backend="cupy", device="cuda:0")
        assert result.device.id == 0
        assert_matches_reference(cp.asnumpy(result), x, y)


def test_torch_select_mps_and_transfer():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS GPU unavailable")
    x, y = make_vectors()
    gpu = envelope(x, y, backend="torch", device="mps:0", compile=True)
    torch.mps.synchronize()
    assert gpu.device.type == "mps"
    assert_matches_reference(gpu.cpu().numpy(), x, y)
    a = torch.as_tensor(x, device="mps")
    b = torch.as_tensor(y, device="mps")
    cpu = envelope(a, b, device="cpu")
    assert cpu.device.type == "cpu"
    assert_matches_reference(cpu.numpy(), x, y)
