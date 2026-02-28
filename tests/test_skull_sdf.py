"""Tests for preprocessing/skull_sdf.py — bone threshold, growth, closing, signed EDT."""

import numpy as np
import nibabel as nib
import pytest

from preprocessing.skull_sdf import (
    compute_bone_threshold,
    compute_signed_edt,
    extract_voxel_size,
    grow_skull_interior,
    load_atlas_brain_prob,
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


# ---------------------------------------------------------------------------
# load_atlas_brain_prob
# ---------------------------------------------------------------------------
class TestLoadAtlasBrainProb:
    def test_3d_atlas(self, tmp_path):
        # 3D pre-computed P(brain) volume
        data = np.full((10, 10, 10), 0.5, dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "p_brain_3d.nii"
        nib.save(img, str(path))

        result = load_atlas_brain_prob(
            str(path), target_affine=np.eye(4), target_shape=(10, 10, 10))

        assert result is not None
        assert result.shape == (10, 10, 10)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 0.5, atol=0.05)

    def test_4d_atlas_sums_channels(self, tmp_path):
        # SPM-style 4D TPM: P_brain = ch0 + ch1 + ch2
        data = np.zeros((10, 10, 10, 6), dtype=np.float32)
        data[:, :, :, 0] = 0.3  # GM
        data[:, :, :, 1] = 0.3  # WM
        data[:, :, :, 2] = 0.2  # CSF
        data[:, :, :, 3] = 0.1  # bone
        data[:, :, :, 4] = 0.05  # soft tissue
        data[:, :, :, 5] = 0.05  # air
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "tpm_4d.nii"
        nib.save(img, str(path))

        result = load_atlas_brain_prob(
            str(path), target_affine=np.eye(4), target_shape=(10, 10, 10))

        assert result is not None
        # GM + WM + CSF = 0.8
        np.testing.assert_allclose(result, 0.8, atol=0.05)

    def test_4d_atlas_clamps_above_one(self, tmp_path):
        # Channels sum > 1.0 → should be clamped
        data = np.zeros((10, 10, 10, 6), dtype=np.float32)
        data[:, :, :, 0] = 0.5
        data[:, :, :, 1] = 0.5
        data[:, :, :, 2] = 0.5  # sum = 1.5
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "tpm_over.nii"
        nib.save(img, str(path))

        result = load_atlas_brain_prob(
            str(path), target_affine=np.eye(4), target_shape=(10, 10, 10))

        assert result is not None
        assert np.all(result <= 1.0)

    def test_missing_file_returns_none(self, tmp_path):
        path = tmp_path / "nonexistent.nii"

        result = load_atlas_brain_prob(
            str(path), target_affine=np.eye(4), target_shape=(5, 5, 5))

        assert result is None

    def test_identity_affine_preserves(self, tmp_path):
        # Identity affine, same shape → output ≈ input
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, (8, 8, 8)).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "identity.nii"
        nib.save(img, str(path))

        result = load_atlas_brain_prob(
            str(path), target_affine=np.eye(4), target_shape=(8, 8, 8))

        assert result is not None
        np.testing.assert_allclose(result, data, atol=0.01)

    def test_scaled_affine_resamples(self, tmp_path):
        # Atlas at 2mm voxels, target at 1mm → output shape is target
        data = np.full((5, 5, 5), 0.7, dtype=np.float32)
        atlas_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img = nib.Nifti1Image(data, atlas_affine)
        path = tmp_path / "atlas_2mm.nii"
        nib.save(img, str(path))

        target_shape = (10, 10, 10)
        result = load_atlas_brain_prob(
            str(path), target_affine=np.eye(4), target_shape=target_shape)

        assert result is not None
        assert result.shape == target_shape
        # Interior voxels should have values near 0.7
        assert result[5, 5, 5] == pytest.approx(0.7, abs=0.1)

    def test_output_clipped_0_1(self, tmp_path):
        # Atlas with values outside [0,1] → output clipped
        data = np.full((10, 10, 10), -0.5, dtype=np.float32)
        data[5, 5, 5] = 2.0
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "out_of_range.nii"
        nib.save(img, str(path))

        result = load_atlas_brain_prob(
            str(path), target_affine=np.eye(4), target_shape=(10, 10, 10))

        assert result is not None
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
