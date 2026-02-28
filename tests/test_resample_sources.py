"""Tests for preprocessing/resample_sources.py — bbox, grid meta, affine checks."""

import numpy as np
import pytest

from preprocessing.resample_sources import compute_brain_bbox, verify_source_affine


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
    def test_expected_hcp_affine(self, capsys):
        affine = np.diag([-0.7, 0.7, 0.7, 1.0])
        verify_source_affine(affine)
        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_unexpected_affine_warns(self, capsys):
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        verify_source_affine(affine)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
