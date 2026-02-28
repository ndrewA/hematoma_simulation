"""Tests for preprocessing/dural_membrane.py — geometry helpers."""

import numpy as np
import pytest

from preprocessing.dural_membrane import (
    _build_fs_luts,
    _compute_crop_bbox,
    _extract_membrane,
    _get_edges,
    _shoelace_area,
    LEFT_CEREBRAL_LABELS,
    LEFT_CEREBRAL_RANGE,
    RIGHT_CEREBRAL_LABELS,
    RIGHT_CEREBRAL_RANGE,
    CC_LABELS,
)
from preprocessing.utils import FS_LUT_SIZE


# ---------------------------------------------------------------------------
# _build_fs_luts
# ---------------------------------------------------------------------------
class TestBuildFsLuts:
    def test_lut_sizes(self):
        left, right, cc = _build_fs_luts()
        assert left.shape == (FS_LUT_SIZE,)
        assert right.shape == (FS_LUT_SIZE,)
        assert cc.shape == (FS_LUT_SIZE,)

    def test_left_labels(self):
        left, _, _ = _build_fs_luts()
        for lab in LEFT_CEREBRAL_LABELS:
            assert left[lab]
        for lab in range(LEFT_CEREBRAL_RANGE[0], LEFT_CEREBRAL_RANGE[1] + 1):
            assert left[lab]

    def test_right_labels(self):
        _, right, _ = _build_fs_luts()
        for lab in RIGHT_CEREBRAL_LABELS:
            assert right[lab]
        for lab in range(RIGHT_CEREBRAL_RANGE[0], RIGHT_CEREBRAL_RANGE[1] + 1):
            assert right[lab]

    def test_cc_labels(self):
        _, _, cc = _build_fs_luts()
        for lab in CC_LABELS:
            assert cc[lab]

    def test_no_overlap_left_right(self):
        left, right, _ = _build_fs_luts()
        overlap = left & right
        assert not np.any(overlap)


# ---------------------------------------------------------------------------
# _compute_crop_bbox
# ---------------------------------------------------------------------------
class TestComputeCropBbox:
    def test_basic(self):
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        mat[5:15, 5:15, 5:15] = 1
        slices = _compute_crop_bbox(mat, pad_vox=2)

        assert slices[0].start == 3   # 5 - 2
        assert slices[0].stop == 17   # 14 + 2 + 1
        assert slices[1].start == 3
        assert slices[2].start == 3

    def test_padding_clipped_to_bounds(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[0:3, 0:3, 0:3] = 1
        slices = _compute_crop_bbox(mat, pad_vox=5)

        assert slices[0].start == 0   # clipped, not -5
        assert slices[0].stop <= 10

    def test_encompasses_all_nonzero(self):
        mat = np.zeros((15, 15, 15), dtype=np.uint8)
        mat[3, 7, 11] = 1
        mat[10, 2, 5] = 1
        slices = _compute_crop_bbox(mat, pad_vox=0)

        cropped = mat[slices]
        assert cropped.sum() == mat.sum()


# ---------------------------------------------------------------------------
# _extract_membrane
# ---------------------------------------------------------------------------
class TestExtractMembrane:
    def test_sign_change_detected(self):
        # Simple linear phi: negative on one side, positive on other
        phi = np.zeros((10, 10, 10), dtype=np.float32)
        phi[:5, :, :] = -5.0
        phi[5:, :, :] = 5.0
        eligible = np.ones((10, 10, 10), dtype=bool)

        surface = _extract_membrane(phi, eligible, t_target_mm=0.5)

        # Surface should be concentrated around the sign change (index 4-5)
        assert surface[4, 5, 5] or surface[5, 5, 5]
        # Far from boundary should be empty
        assert not surface[0, 5, 5]
        assert not surface[9, 5, 5]

    def test_respects_eligible(self):
        phi = np.zeros((10, 10, 10), dtype=np.float32)
        phi[:5, :, :] = -5.0
        phi[5:, :, :] = 5.0
        eligible = np.zeros((10, 10, 10), dtype=bool)  # nothing eligible

        surface = _extract_membrane(phi, eligible, t_target_mm=1.0)
        assert not np.any(surface)

    def test_uniform_phi_no_surface(self):
        phi = np.ones((10, 10, 10), dtype=np.float32) * 5.0
        eligible = np.ones((10, 10, 10), dtype=bool)

        surface = _extract_membrane(phi, eligible, t_target_mm=1.0)
        # No sign changes → no barrier surface (thickness might add some)
        # but phi is uniformly 5.0, so |phi| < 1.0 is false everywhere
        assert not np.any(surface)

    def test_thickening(self):
        phi = np.linspace(-5, 5, 20).reshape(20, 1, 1) * np.ones((1, 5, 5))
        phi = phi.astype(np.float32)
        eligible = np.ones(phi.shape, dtype=bool)

        thin = _extract_membrane(phi, eligible, t_target_mm=0.1)
        thick = _extract_membrane(phi, eligible, t_target_mm=3.0)

        assert thick.sum() >= thin.sum()


# ---------------------------------------------------------------------------
# _get_edges
# ---------------------------------------------------------------------------
class TestGetEdges:
    def test_basic(self):
        proj = np.zeros((5, 10), dtype=bool)
        proj[2, 3:7] = True  # y=2, z=3..6

        top, bot, exists = _get_edges(proj, 5, 10)

        assert exists[2]
        assert top[2] == 6   # max z
        assert bot[2] == 3   # min z
        assert not exists[0]
        assert not exists[4]

    def test_empty_projection(self):
        proj = np.zeros((5, 10), dtype=bool)
        top, bot, exists = _get_edges(proj, 5, 10)
        assert not np.any(exists)

    def test_single_pixel(self):
        proj = np.zeros((5, 10), dtype=bool)
        proj[1, 4] = True
        top, bot, exists = _get_edges(proj, 5, 10)

        assert exists[1]
        assert top[1] == 4
        assert bot[1] == 4


# ---------------------------------------------------------------------------
# _shoelace_area
# ---------------------------------------------------------------------------
class TestShoelaceArea:
    def test_unit_square(self):
        ys = [0, 1, 1, 0]
        zs = [0, 0, 1, 1]
        assert _shoelace_area(ys, zs) == pytest.approx(1.0)

    def test_triangle(self):
        ys = [0, 4, 0]
        zs = [0, 0, 3]
        assert _shoelace_area(ys, zs) == pytest.approx(6.0)

    def test_reversed_winding(self):
        # Shoelace should give same magnitude regardless of winding
        ys = [0, 0, 1, 1]
        zs = [0, 1, 1, 0]
        assert _shoelace_area(ys, zs) == pytest.approx(1.0)

    def test_degenerate_line(self):
        ys = [0, 1, 2]
        zs = [0, 0, 0]
        assert _shoelace_area(ys, zs) == pytest.approx(0.0)
