"""Tests for preprocessing/resample_sources.py — bbox, grid meta, affine checks."""

import numpy as np
import pytest

from preprocessing.resample_sources import (
    build_grid_meta,
    compute_brain_bbox,
    verify_source_affine,
)


# ---------------------------------------------------------------------------
# compute_brain_bbox
# ---------------------------------------------------------------------------
class TestComputeBrainBbox:
    def test_basic(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 3:17, 8:12] = 1

        bbox_min, bbox_max, centroid, n_voxels = compute_brain_bbox(mask)

        np.testing.assert_array_equal(bbox_min, [5, 3, 8])
        np.testing.assert_array_equal(bbox_max, [14, 16, 11])
        assert n_voxels == 10 * 14 * 4

    def test_centroid(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[4:6, 4:6, 4:6] = 1  # 2x2x2 cube centered at (4.5, 4.5, 4.5)

        _, _, centroid, _ = compute_brain_bbox(mask)
        np.testing.assert_allclose(centroid, [4.5, 4.5, 4.5])

    def test_single_voxel(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[3, 7, 2] = 1

        bbox_min, bbox_max, centroid, n_voxels = compute_brain_bbox(mask)
        np.testing.assert_array_equal(bbox_min, [3, 7, 2])
        np.testing.assert_array_equal(bbox_max, [3, 7, 2])
        assert n_voxels == 1

    def test_empty_mask_raises(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_brain_bbox(mask)


# ---------------------------------------------------------------------------
# verify_source_affine
# ---------------------------------------------------------------------------
class TestVerifySourceAffine:
    def test_expected_hcp_affine(self):
        affine = np.diag([-0.7, 0.7, 0.7, 1.0])
        assert verify_source_affine(affine) is True

    def test_unexpected_affine_warns(self):
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        assert verify_source_affine(affine) is False


# ---------------------------------------------------------------------------
# build_grid_meta
# ---------------------------------------------------------------------------
class TestBuildGridMeta:
    def _make_args(self, N=10, dx=1.0):
        from types import SimpleNamespace
        return SimpleNamespace(subject="test", profile="debug", N=N, dx=dx)

    def test_basic_fields(self):
        N, dx = 10, 1.0
        args = self._make_args(N, dx)
        grid_affine = np.diag([dx, dx, dx, 1.0])
        grid_affine[:3, 3] = -N * dx / 2.0
        source_affine = np.diag([-0.7, 0.7, 0.7, 1.0])
        meta = build_grid_meta(
            args, grid_affine, source_affine, (260, 311, 260),
            np.array([3, 3, 3]), np.array([7, 7, 7]),
            np.array([5.0, 5.0, 5.0]), 125,
        )
        assert meta["grid_size"] == 10
        assert meta["dx_mm"] == 1.0
        assert meta["subject_id"] == "test"
        assert meta["profile"] == "debug"
        assert meta["domain_extent_mm"] == 10.0

    def test_phys_to_grid_is_inverse(self):
        N, dx = 10, 2.0
        args = self._make_args(N, dx)
        grid_affine = np.diag([dx, dx, dx, 1.0])
        grid_affine[:3, 3] = -N * dx / 2.0
        meta = build_grid_meta(
            args, grid_affine, np.diag([-0.7, 0.7, 0.7, 1.0]),
            (260, 311, 260), np.array([3, 3, 3]), np.array([7, 7, 7]),
            np.array([5.0, 5.0, 5.0]), 125,
        )
        g2p = np.array(meta["affine_grid_to_phys"])
        p2g = np.array(meta["affine_phys_to_grid"])
        np.testing.assert_allclose(g2p @ p2g, np.eye(4), atol=1e-10)

    def test_brain_volume_ml(self):
        N, dx = 10, 2.0
        args = self._make_args(N, dx)
        grid_affine = np.diag([dx, dx, dx, 1.0])
        meta = build_grid_meta(
            args, grid_affine, np.eye(4), (1, 1, 1),
            np.array([0, 0, 0]), np.array([9, 9, 9]),
            np.array([5.0, 5.0, 5.0]), 100,
        )
        # brain_volume_ml = 100 * 2.0³ / 1000 = 0.8
        assert meta["brain_volume_ml"] == 0.8
