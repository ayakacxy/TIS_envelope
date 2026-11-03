"""Compatibility alias for the Fast TIS Envelope public API."""
from fast_tis_envelope import __version__, envelope, resolve_backend

__all__ = ["envelope", "resolve_backend", "__version__"]
