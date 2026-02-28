"""Shared fixtures for the test suite."""

import numpy as np
import pytest
from pathlib import Path

import nibabel as nib
from preprocessing.validation import CheckContext


_NOFILE = Path("/tmp/_validation_test_nonexistent")


class _MockNiftiImg:
    """Minimal NIfTI-like object for header checks (avoids large allocations)."""

    def __init__(self, shape, dtype=np.float32, affine=None):
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self.affine = affine if affine is not None else np.eye(4)

    @property
    def shape(self):
        return self._shape

    def get_data_dtype(self):
        return self._dtype


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


def _make_fiber_ctx(mat, fiber_data, fiber_affine, sdf=None, dx=1.0,
                    mat_affine=None):
    """Build a CheckContext with fiber data and headers injected."""
    ctx = _make_ctx(mat, sdf=sdf, dx=dx)
    N = mat.shape[0]
    if mat_affine is None:
        mat_affine = np.diag([float(dx), float(dx), float(dx), 1.0])
        mat_affine[:3, 3] = -N * dx / 2.0
    ctx._fiber_data = fiber_data.astype(np.float32)
    ctx._fiber_img = nib.Nifti1Image(fiber_data.astype(np.float32), fiber_affine)
    ctx._headers = {
        "mat": nib.Nifti1Image(ctx._mat, mat_affine),
        "sdf": nib.Nifti1Image(ctx._sdf, mat_affine),
        "brain": nib.Nifti1Image(
            np.zeros((N, N, N), dtype=np.uint8), mat_affine),
    }
    return ctx


def _make_header_ctx(N=10, dx=1.0, affine=None, meta=None,
                     mat_dtype=np.uint8, sdf_dtype=np.float32,
                     fiber_img=None):
    """Build a CheckContext with NIfTI headers and metadata injected."""
    if affine is None:
        affine = np.diag([float(dx), float(dx), float(dx), 1.0])
        affine[:3, 3] = -N * dx / 2.0
    headers = {
        "mat": nib.Nifti1Image(
            np.zeros((N, N, N), dtype=mat_dtype), affine),
        "sdf": nib.Nifti1Image(
            np.zeros((N, N, N), dtype=sdf_dtype), affine),
        "brain": nib.Nifti1Image(
            np.zeros((N, N, N), dtype=np.uint8), affine),
    }
    if meta is None:
        meta = {
            "grid_size": N,
            "dx_mm": float(dx),
            "affine_grid_to_phys": affine.tolist(),
            "affine_phys_to_grid": np.linalg.inv(affine).tolist(),
        }
    ctx = CheckContext.__new__(CheckContext)
    ctx.paths = {k: _NOFILE for k in
                 ("mat", "sdf", "brain", "fs", "meta", "fiber", "t1w", "val_dir")}
    ctx.N = N
    ctx.dx = float(dx)
    ctx.subject = "test"
    ctx.profile = "debug"
    ctx.verbose = False
    ctx._mat = np.zeros((N, N, N), dtype=np.uint8)
    ctx._sdf = np.ones((N, N, N), dtype=np.float32)
    ctx._brain = None
    ctx._meta = meta
    ctx._headers = headers
    ctx._fiber_img = fiber_img
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


@pytest.fixture
def make_fiber_ctx():
    """Factory fixture that returns the _make_fiber_ctx helper."""
    return _make_fiber_ctx


@pytest.fixture
def make_header_ctx():
    """Factory fixture that returns the _make_header_ctx helper."""
    return _make_header_ctx
