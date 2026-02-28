"""Tests for preprocessing/skull_sdf.py — bone threshold, growth, closing, signed EDT."""

import numpy as np
import pytest

from preprocessing.skull_sdf import (
    compute_bone_threshold,
    compute_signed_edt,
    extract_voxel_size,
    grow_skull_interior,
    morphological_close,
)


# ---------------------------------------------------------------------------
# extract_voxel_size
# ---------------------------------------------------------------------------
class TestExtractVoxelSize:
    def test_isotropic(self):
        affine = np.diag([-0.7, 0.7, 0.7, 1.0])
        assert extract_voxel_size(affine) == pytest.approx(0.7)

    def test_positive_diagonal(self):
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        assert extract_voxel_size(affine) == pytest.approx(1.0)

    def test_returns_float(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        result = extract_voxel_size(affine)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_bone_threshold
# ---------------------------------------------------------------------------
class TestComputeBoneThreshold:
    def test_basic(self):
        rng = np.random.default_rng(42)
        # Simulate brain T2w values: mean ~1000, spread ~100
        t2w = np.zeros((20, 20, 20), dtype=np.float32)
        brain_mask = np.zeros((20, 20, 20), dtype=bool)
        brain_mask[5:15, 5:15, 5:15] = True
        t2w[brain_mask] = rng.normal(1000, 100, size=int(brain_mask.sum())).astype(np.float32)

        threshold, stats = compute_bone_threshold(t2w, brain_mask, bone_z=-0.5)

        assert threshold is not None
        assert stats is not None
        assert stats["bone_z"] == -0.5
        assert stats["brain_median"] == pytest.approx(np.median(t2w[brain_mask]), rel=1e-5)
        # Threshold should be below median (bone_z is negative)
        assert threshold < stats["brain_median"]

    def test_too_few_brain_voxels(self):
        t2w = np.ones((5, 5, 5), dtype=np.float32) * 500
        brain_mask = np.zeros((5, 5, 5), dtype=bool)
        brain_mask[0, 0, 0] = True  # only 1 voxel

        threshold, stats = compute_bone_threshold(t2w, brain_mask, bone_z=-0.5)
        assert threshold is None
        assert stats is None

    def test_uniform_values_scale_zero(self):
        # All brain voxels have the same value → MAD = 0 → scale < 1
        t2w = np.ones((20, 20, 20), dtype=np.float32) * 500
        brain_mask = np.zeros((20, 20, 20), dtype=bool)
        brain_mask[2:18, 2:18, 2:18] = True

        threshold, stats = compute_bone_threshold(t2w, brain_mask, bone_z=-0.5)
        assert threshold is None
        assert stats is None

    def test_z_score_formula(self):
        # Verify threshold = median + z * scale exactly
        rng = np.random.default_rng(123)
        t2w = np.zeros((20, 20, 20), dtype=np.float32)
        brain_mask = np.zeros((20, 20, 20), dtype=bool)
        brain_mask[2:18, 2:18, 2:18] = True
        t2w[brain_mask] = rng.normal(800, 150, size=int(brain_mask.sum())).astype(np.float32)

        bone_z = -1.0
        threshold, stats = compute_bone_threshold(t2w, brain_mask, bone_z)

        expected = stats["brain_median"] + bone_z * stats["brain_scale"]
        assert threshold == pytest.approx(expected, rel=1e-5)

    def test_positive_z(self):
        # Positive z-score → threshold above median
        rng = np.random.default_rng(7)
        t2w = np.zeros((20, 20, 20), dtype=np.float32)
        brain_mask = np.zeros((20, 20, 20), dtype=bool)
        brain_mask[2:18, 2:18, 2:18] = True
        t2w[brain_mask] = rng.normal(1000, 100, size=int(brain_mask.sum())).astype(np.float32)

        threshold, stats = compute_bone_threshold(t2w, brain_mask, bone_z=1.0)
        assert threshold > stats["brain_median"]


# ---------------------------------------------------------------------------
# grow_skull_interior
# ---------------------------------------------------------------------------
class TestGrowSkullInterior:
    def _make_inputs(self, size=30):
        """Create a small sphere brain inside a head mask."""
        brain = np.zeros((size, size, size), dtype=bool)
        head = np.zeros((size, size, size), dtype=bool)
        c = size // 2
        coords = np.mgrid[:size, :size, :size] - c
        dist = np.sqrt((coords ** 2).sum(axis=0))
        brain[dist <= 5] = True
        head[dist <= 12] = True
        # T2w: high inside head (tissue), low outside (air)
        t2w = np.zeros((size, size, size), dtype=np.float32)
        t2w[head] = 1000.0
        return brain, head, t2w

    def test_brain_preserved(self):
        brain, head, t2w = self._make_inputs()
        result, stats = grow_skull_interior(
            brain, head, t2w, voxel_size=1.0, bone_threshold=500.0,
            max_growth_mm=5.0, min_neighbors=1,
        )
        # Every brain voxel must still be in the result
        assert np.all(result[brain])

    def test_grows_beyond_brain(self):
        brain, head, t2w = self._make_inputs()
        result, stats = grow_skull_interior(
            brain, head, t2w, voxel_size=1.0, bone_threshold=500.0,
            max_growth_mm=5.0, min_neighbors=1,
        )
        assert stats["grown_voxels"] > 0
        assert result.sum() > brain.sum()

    def test_respects_head_mask(self):
        brain, head, t2w = self._make_inputs()
        result, stats = grow_skull_interior(
            brain, head, t2w, voxel_size=1.0, bone_threshold=500.0,
            max_growth_mm=20.0, min_neighbors=1,
        )
        assert not np.any(result & ~head)

    def test_stops_at_bone(self):
        brain, head, t2w = self._make_inputs()
        # Set T2w to 0 outside brain → everything is "bone"
        t2w[:] = 0.0
        t2w[brain] = 1000.0
        result, stats = grow_skull_interior(
            brain, head, t2w, voxel_size=1.0, bone_threshold=500.0,
            max_growth_mm=10.0, min_neighbors=1,
        )
        # Growth should be minimal — only brain + maybe 1 layer of
        # voxels that are partially in the bright zone
        assert stats["grown_voxels"] == 0

    def test_curvature_gating(self):
        brain, head, t2w = self._make_inputs()
        # High min_neighbors restricts growth
        result_strict, _ = grow_skull_interior(
            brain.copy(), head, t2w, voxel_size=1.0, bone_threshold=500.0,
            max_growth_mm=5.0, min_neighbors=20,
        )
        result_loose, _ = grow_skull_interior(
            brain.copy(), head, t2w, voxel_size=1.0, bone_threshold=500.0,
            max_growth_mm=5.0, min_neighbors=1,
        )
        assert result_strict.sum() <= result_loose.sum()

    def test_stats_keys(self):
        brain, head, t2w = self._make_inputs()
        _, stats = grow_skull_interior(
            brain, head, t2w, voxel_size=1.0, bone_threshold=500.0,
            max_growth_mm=3.0, min_neighbors=1,
        )
        assert "iterations" in stats
        assert "brain_voxels" in stats
        assert "grown_voxels" in stats
        assert "total_voxels" in stats
        assert stats["total_voxels"] == stats["brain_voxels"] + stats["grown_voxels"]

    def test_no_atlas(self):
        brain, head, t2w = self._make_inputs()
        result, stats = grow_skull_interior(
            brain, head, t2w, voxel_size=1.0, bone_threshold=500.0,
            max_growth_mm=3.0, p_brain=None,
        )
        assert stats["atlas_lambda"] is None


# ---------------------------------------------------------------------------
# morphological_close
# ---------------------------------------------------------------------------
class TestMorphologicalClose:
    def test_zero_radius_returns_input(self):
        interior = np.ones((10, 10, 10), dtype=bool)
        brain = interior.copy()
        head = interior.copy()
        result = morphological_close(interior, brain, head, 1.0, close_radius_mm=0.0)
        np.testing.assert_array_equal(result, interior)

    def test_brain_preserved(self):
        interior = np.zeros((20, 20, 20), dtype=bool)
        brain = np.zeros((20, 20, 20), dtype=bool)
        head = np.ones((20, 20, 20), dtype=bool)
        brain[8:12, 8:12, 8:12] = True
        interior[8:12, 8:12, 8:12] = True
        result = morphological_close(interior, brain, head, 1.0, close_radius_mm=2.0)
        assert np.all(result[brain])

    def test_fills_concavity(self):
        # Create a shell with a 1-voxel dent
        interior = np.zeros((20, 20, 20), dtype=bool)
        brain = np.zeros((20, 20, 20), dtype=bool)
        head = np.ones((20, 20, 20), dtype=bool)
        interior[5:15, 5:15, 5:15] = True
        brain[7:13, 7:13, 7:13] = True
        # Carve a small concavity
        interior[5, 8:12, 8:12] = False
        result = morphological_close(interior, brain, head, 1.0, close_radius_mm=2.0)
        # Closing should fill the 1-voxel concavity
        assert result.sum() >= interior.sum()

    def test_stays_within_head(self):
        interior = np.zeros((20, 20, 20), dtype=bool)
        brain = np.zeros((20, 20, 20), dtype=bool)
        head = np.zeros((20, 20, 20), dtype=bool)
        head[3:17, 3:17, 3:17] = True
        brain[8:12, 8:12, 8:12] = True
        interior[6:14, 6:14, 6:14] = True
        result = morphological_close(interior, brain, head, 1.0, close_radius_mm=5.0)
        assert not np.any(result & ~head)


# ---------------------------------------------------------------------------
# compute_signed_edt
# ---------------------------------------------------------------------------
class TestComputeSignedEdt:
    def test_sign_convention(self):
        # Negative inside, positive outside
        interior = np.zeros((20, 20, 20), dtype=bool)
        interior[5:15, 5:15, 5:15] = True
        sdf = compute_signed_edt(interior, voxel_size=1.0, sigma_mm=0.0)

        assert sdf[10, 10, 10] < 0  # center = deep inside
        assert sdf[0, 0, 0] > 0     # corner = far outside

    def test_dtype_float32(self):
        interior = np.zeros((10, 10, 10), dtype=bool)
        interior[3:7, 3:7, 3:7] = True
        sdf = compute_signed_edt(interior, voxel_size=1.0, sigma_mm=0.0)
        assert sdf.dtype == np.float32

    def test_distance_magnitude(self):
        # For a cube, center voxel should be ~5mm from the nearest surface
        interior = np.zeros((20, 20, 20), dtype=bool)
        interior[5:15, 5:15, 5:15] = True
        sdf = compute_signed_edt(interior, voxel_size=1.0, sigma_mm=0.0)

        # Center is at (10,10,10), nearest boundary at 5 voxels away
        # SDF should be approximately -5.0 (negative inside)
        assert sdf[10, 10, 10] == pytest.approx(-5.0, abs=0.5)

    def test_voxel_size_scaling(self):
        interior = np.zeros((20, 20, 20), dtype=bool)
        interior[5:15, 5:15, 5:15] = True

        sdf_1mm = compute_signed_edt(interior, voxel_size=1.0, sigma_mm=0.0)
        sdf_2mm = compute_signed_edt(interior, voxel_size=2.0, sigma_mm=0.0)

        # Same voxel, but 2mm spacing → distances should be ~2x larger
        ratio = sdf_2mm[10, 10, 10] / sdf_1mm[10, 10, 10]
        assert ratio == pytest.approx(2.0, abs=0.1)

    def test_smoothing_preserves_sign(self):
        interior = np.zeros((20, 20, 20), dtype=bool)
        interior[5:15, 5:15, 5:15] = True
        sdf = compute_signed_edt(interior, voxel_size=1.0, sigma_mm=2.0)

        # Deep interior should still be negative after smoothing
        assert sdf[10, 10, 10] < 0
        # Far exterior should still be positive
        assert sdf[0, 0, 0] > 0

    def test_no_smoothing(self):
        interior = np.zeros((10, 10, 10), dtype=bool)
        interior[3:7, 3:7, 3:7] = True
        sdf = compute_signed_edt(interior, voxel_size=1.0, sigma_mm=0.0)
        # Without smoothing, should have exact staircase distances
        # Voxel at (3,5,5) is 1 voxel inside the boundary → SDF = -1.0
        assert abs(sdf[3, 5, 5]) <= 1.0
