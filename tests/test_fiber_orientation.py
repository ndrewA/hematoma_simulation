"""Tests for preprocessing/fiber_orientation.py — structure tensor, WM mask."""

import numpy as np
import pytest

from preprocessing.fiber_orientation import (
    _build_aniso_lut,
    _ANISO_LABELS,
    compute_structure_tensor,
    apply_wm_mask,
    build_wm_mask,
)
from preprocessing.utils import FS_LUT_SIZE


# ---------------------------------------------------------------------------
# _build_aniso_lut
# ---------------------------------------------------------------------------
class TestBuildAnisoLut:
    def test_shape(self):
        lut = _build_aniso_lut()
        assert lut.shape == (FS_LUT_SIZE,)
        assert lut.dtype == bool

    def test_aniso_labels_true(self):
        lut = _build_aniso_lut()
        for lab in _ANISO_LABELS:
            assert lut[lab], f"Label {lab} should be anisotropic"

    def test_non_aniso_false(self):
        lut = _build_aniso_lut()
        # Label 3 (left cortical GM) should not be anisotropic
        assert not lut[3]
        assert not lut[42]  # right cortical GM
        assert not lut[10]  # thalamus


# ---------------------------------------------------------------------------
# compute_structure_tensor
# ---------------------------------------------------------------------------
class TestComputeStructureTensor:
    def test_output_shape(self):
        shape = (4, 5, 6)
        dyads = [np.random.randn(*shape, 3).astype(np.float32) for _ in range(3)]
        fracs = [np.random.rand(*shape).astype(np.float32) for _ in range(3)]

        M0 = compute_structure_tensor(dyads, fracs)
        assert M0.shape == shape + (6,)
        assert M0.dtype == np.float32

    def test_single_fiber_x_direction(self):
        shape = (2, 2, 2)
        # Single fiber along X with fraction 1.0
        v = np.zeros(shape + (3,), dtype=np.float32)
        v[..., 0] = 1.0  # unit vector along X
        dyads = [v, np.zeros_like(v), np.zeros_like(v)]
        fracs = [np.ones(shape, dtype=np.float32),
                 np.zeros(shape, dtype=np.float32),
                 np.zeros(shape, dtype=np.float32)]

        M0 = compute_structure_tensor(dyads, fracs)

        # M_00 should be 1.0, all others 0
        np.testing.assert_allclose(M0[..., 0], 1.0)  # M_00
        np.testing.assert_allclose(M0[..., 1], 0.0)  # M_11
        np.testing.assert_allclose(M0[..., 2], 0.0)  # M_22
        np.testing.assert_allclose(M0[..., 3], 0.0)  # M_01
        np.testing.assert_allclose(M0[..., 4], 0.0)  # M_02
        np.testing.assert_allclose(M0[..., 5], 0.0)  # M_12

    def test_isotropic_fractions(self):
        shape = (3, 3, 3)
        # Three orthogonal fibers with equal fractions
        vx = np.zeros(shape + (3,), dtype=np.float32)
        vx[..., 0] = 1.0
        vy = np.zeros(shape + (3,), dtype=np.float32)
        vy[..., 1] = 1.0
        vz = np.zeros(shape + (3,), dtype=np.float32)
        vz[..., 2] = 1.0

        f = np.full(shape, 1 / 3, dtype=np.float32)
        dyads = [vx, vy, vz]
        fracs = [f, f.copy(), f.copy()]

        M0 = compute_structure_tensor(dyads, fracs)

        # Diagonal should be ~1/3 each, off-diagonal ~0
        np.testing.assert_allclose(M0[..., 0], 1 / 3, atol=1e-6)
        np.testing.assert_allclose(M0[..., 1], 1 / 3, atol=1e-6)
        np.testing.assert_allclose(M0[..., 2], 1 / 3, atol=1e-6)
        np.testing.assert_allclose(M0[..., 3], 0.0, atol=1e-6)

    def test_fraction_weighting(self):
        shape = (2, 2, 2)
        v = np.zeros(shape + (3,), dtype=np.float32)
        v[..., 0] = 1.0
        dyads = [v, np.zeros_like(v), np.zeros_like(v)]

        f_half = np.full(shape, 0.5, dtype=np.float32)
        fracs = [f_half, np.zeros(shape, dtype=np.float32),
                 np.zeros(shape, dtype=np.float32)]

        M0 = compute_structure_tensor(dyads, fracs)

        # With fraction 0.5, M_00 should be 0.5
        np.testing.assert_allclose(M0[..., 0], 0.5)

    def test_frees_dyad_memory(self):
        shape = (2, 2, 2)
        dyads = [np.zeros(shape + (3,), dtype=np.float32) for _ in range(3)]
        fracs = [np.zeros(shape, dtype=np.float32) for _ in range(3)]

        compute_structure_tensor(dyads, fracs)

        for d in dyads:
            assert d is None

    def test_symmetry_via_eigenvalues(self):
        # Random dyads → M0 should be PSD at every voxel
        rng = np.random.default_rng(42)
        shape = (3, 3, 3)
        dyads = [rng.standard_normal(shape + (3,)).astype(np.float32) for _ in range(3)]
        fracs = [np.abs(rng.standard_normal(shape)).astype(np.float32) for _ in range(3)]

        M0 = compute_structure_tensor(dyads, fracs)

        # Check PSD at every voxel
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    m = M0[i, j, k]
                    mat = np.array([
                        [m[0], m[3], m[4]],
                        [m[3], m[1], m[5]],
                        [m[4], m[5], m[2]],
                    ])
                    eigvals = np.linalg.eigvalsh(mat)
                    assert np.all(eigvals >= -1e-6), \
                        f"Non-PSD at ({i},{j},{k}): eigvals={eigvals}"


# ---------------------------------------------------------------------------
# apply_wm_mask
# ---------------------------------------------------------------------------
class TestApplyWmMask:
    def test_zeros_non_aniso(self):
        M0 = np.ones((5, 5, 5, 6), dtype=np.float32)
        is_aniso = np.zeros((5, 5, 5), dtype=bool)
        is_aniso[2, 2, 2] = True

        apply_wm_mask(M0, is_aniso)

        # Only aniso voxel should remain
        np.testing.assert_allclose(M0[2, 2, 2], 1.0)
        assert M0[0, 0, 0, 0] == 0.0

    def test_preserves_aniso(self):
        M0 = np.ones((3, 3, 3, 6), dtype=np.float32) * 0.42
        is_aniso = np.ones((3, 3, 3), dtype=bool)

        apply_wm_mask(M0, is_aniso)

        np.testing.assert_allclose(M0, 0.42)


# ---------------------------------------------------------------------------
# build_wm_mask
# ---------------------------------------------------------------------------
class TestBuildWmMask:
    def test_identity_transform(self):
        # Same affine for both spaces → direct lookup
        affine = np.eye(4)
        fs_data = np.zeros((10, 10, 10), dtype=np.int16)
        fs_data[5, 5, 5] = 2  # cerebral WM → anisotropic
        aniso_lut = _build_aniso_lut()

        is_aniso, fs_labels = build_wm_mask(affine, fs_data, affine, (10, 10, 10), aniso_lut)

        assert is_aniso[5, 5, 5]
        assert fs_labels[5, 5, 5] == 2
        assert not is_aniso[0, 0, 0]  # label 0 → not anisotropic
