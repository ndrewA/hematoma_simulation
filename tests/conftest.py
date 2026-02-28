"""Shared fixtures for the test suite."""

import numpy as np
import pytest
from pathlib import Path

from preprocessing.validation import CheckContext


_NOFILE = Path("/tmp/_validation_test_nonexistent")


def _make_ctx(mat, sdf=None, dx=1.0):
    """Build a real CheckContext with arrays injected (bypasses lazy NIfTI loading)."""
    N = mat.shape[0]
    paths = {
        "mat": _NOFILE,
        "sdf": _NOFILE,
        "brain": _NOFILE,
        "fs": _NOFILE,
        "meta": _NOFILE,
        "fiber": _NOFILE,
        "t1w": _NOFILE,
        "val_dir": _NOFILE,
    }
    ctx = CheckContext.__new__(CheckContext)
    ctx.paths = paths
    ctx.N = N
    ctx.dx = dx
    ctx.subject = "test"
    ctx.profile = "debug"
    ctx.verbose = False
    ctx._mat = mat.astype(np.uint8)
    ctx._sdf = sdf if sdf is not None else np.ones((N, N, N), dtype=np.float32)
    ctx._brain = None
    ctx._meta = None
    ctx._headers = None
    ctx._fiber_img = None
    ctx._fiber_data = None
    ctx._cache = {}
    ctx.results = []
    ctx.census = {}
    ctx.metrics = {}
    ctx.has_simnibs = False
    return ctx


@pytest.fixture
def make_ctx():
    """Factory fixture that returns the _make_ctx helper."""
    return _make_ctx
