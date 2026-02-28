"""Tests for preprocessing/dural_membrane.py — geometry helpers + reconstruction."""

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

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
    classify_hemispheres,
    _detect_crista_galli,
    _draw_line_safe,
    merge_dural,
    check_idempotency,
    _collect_falx_polygon,
    _collect_boundary_polylines,
    _build_free_edge_controls,
    _solve_bezier_shape,
    _rasterize_falx_cookie,
    _detect_falx_geometry,
    _compute_midplane_membrane,
    _measure_notch_ellipse,
    print_membrane_continuity,
    print_csf_components,
    print_thickness_estimate,
    print_junction_thickness,
    reconstruct_falx,
    reconstruct_tentorium,
    _FalxGeometry,
    _MOF_LEFT,
    _MOF_RIGHT,
)
from preprocessing.utils import FS_LUT_SIZE


# ---------------------------------------------------------------------------
# Shared helper for geometry-dependent tests
# ---------------------------------------------------------------------------
def _make_falx_geometry(Y=60, Z=80, anchor_y=10, anchor_z=50.0,
                        crista_y=40, crista_z=20.0, genu_y=35, mid_x=15,
                        mem_y_min=5, mem_y_max=55):
    """Build a minimal _FalxGeometry with plausible edge profiles."""
    mem_top_z = np.full(Y, -1, dtype=int)
    mem_bot_z = np.full(Y, Z, dtype=int)
    mem_exists = np.zeros(Y, dtype=bool)
    for y in range(mem_y_min, mem_y_max + 1):
        mem_exists[y] = True
        mem_top_z[y] = Z - 10
        mem_bot_z[y] = 10

    tent_top_z = np.full(Y, -1, dtype=int)
    tent_exists = np.zeros(Y, dtype=bool)
    for y in range(mem_y_min, anchor_y + 1):
        tent_exists[y] = True
        tent_top_z[y] = int(anchor_z)

    cc_top_z = np.full(Y, -1, dtype=int)
    cc_exists = np.zeros(Y, dtype=bool)
    for y in range(15, genu_y + 1):
        cc_exists[y] = True
        cc_top_z[y] = int(anchor_z) + 5

    cc_landmarks = {
        "splenium": (15, 45.6),
        "body": (25, 25.7),
        "genu": (genu_y, 21.3),
    }

    return _FalxGeometry(
        mem_top_z=mem_top_z, mem_bot_z=mem_bot_z, mem_exists=mem_exists,
        mem_y_min=mem_y_min, mem_y_max=mem_y_max,
        tent_top_z=tent_top_z, tent_exists=tent_exists,
        cc_landmarks=cc_landmarks, cc_top_z=cc_top_z, cc_exists=cc_exists,
        anchor_y=anchor_y, anchor_z=anchor_z,
        crista_y=crista_y, crista_z=crista_z,
        genu_y=genu_y, mid_x=mid_x,
    )


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


# ---------------------------------------------------------------------------
# classify_hemispheres
# ---------------------------------------------------------------------------
class TestClassifyHemispheres:
    def test_left_labels_detected(self):
        left_lut, right_lut, _ = _build_fs_luts()
        fs = np.zeros((3, 3, 3), dtype=np.int16)
        fs[1, 1, 1] = 2  # left cerebral WM

        left, right = classify_hemispheres(fs, left_lut, right_lut)

        assert left[1, 1, 1]
        assert not right[1, 1, 1]

    def test_right_labels_detected(self):
        left_lut, right_lut, _ = _build_fs_luts()
        fs = np.zeros((3, 3, 3), dtype=np.int16)
        fs[1, 1, 1] = 41  # right cerebral WM

        left, right = classify_hemispheres(fs, left_lut, right_lut)

        assert not left[1, 1, 1]
        assert right[1, 1, 1]

    def test_negative_labels_safe(self):
        left_lut, right_lut, _ = _build_fs_luts()
        fs = np.array([[[-5]]], dtype=np.int16)

        left, right = classify_hemispheres(fs, left_lut, right_lut)

        assert not left[0, 0, 0]
        assert not right[0, 0, 0]

    def test_no_overlap(self):
        left_lut, right_lut, _ = _build_fs_luts()
        # Spread of labels including both hemispheres + Desikan ranges
        labels = [2, 3, 41, 42, 1001, 2001, 0, 10, 49]
        fs = np.array(labels, dtype=np.int16).reshape(3, 3, 1)

        left, right = classify_hemispheres(fs, left_lut, right_lut)

        assert not np.any(left & right)


# ---------------------------------------------------------------------------
# _detect_crista_galli
# ---------------------------------------------------------------------------
class TestDetectCristaGalli:
    def test_basic_detection(self):
        fs_crop = np.zeros((11, 40, 40), dtype=np.int16)
        mid_x = 5
        genu_y = 15
        # Place MOF at (mid_x, y=20, z=5)
        fs_crop[mid_x, 20, 5] = _MOF_LEFT

        crista_y, crista_z = _detect_crista_galli(fs_crop, mid_x, genu_y)

        assert crista_y == 20
        assert crista_z == 5

    def test_raises_when_no_mof_anterior(self):
        fs_crop = np.zeros((11, 40, 40), dtype=np.int16)
        mid_x = 5
        genu_y = 15
        # Place MOF at y=10, which is <= genu_y
        fs_crop[mid_x, 10, 5] = _MOF_LEFT

        with pytest.raises(ValueError, match="medial orbitofrontal"):
            _detect_crista_galli(fs_crop, mid_x, genu_y)

    def test_uses_right_mof_label(self):
        fs_crop = np.zeros((11, 40, 40), dtype=np.int16)
        mid_x = 5
        genu_y = 10
        # Only right MOF label present
        fs_crop[mid_x, 20, 8] = _MOF_RIGHT

        crista_y, crista_z = _detect_crista_galli(fs_crop, mid_x, genu_y)

        assert crista_y == 20
        assert crista_z == 8

    def test_picks_most_inferior_z(self):
        fs_crop = np.zeros((11, 40, 40), dtype=np.int16)
        mid_x = 5
        genu_y = 10
        # Multiple MOF voxels at different z
        fs_crop[mid_x, 20, 15] = _MOF_LEFT
        fs_crop[mid_x, 25, 7] = _MOF_LEFT
        fs_crop[mid_x, 30, 3] = _MOF_LEFT

        crista_y, crista_z = _detect_crista_galli(fs_crop, mid_x, genu_y)

        assert crista_z == 3  # minimum z


# ---------------------------------------------------------------------------
# _draw_line_safe
# ---------------------------------------------------------------------------
class TestDrawLineSafe:
    def test_horizontal_line(self):
        target = np.zeros((10, 10), dtype=bool)
        _draw_line_safe(2, 1, 2, 8, target, 10, 10)

        # All pixels on row y=2 from z=1 to z=8 should be set
        assert np.all(target[2, 1:9])
        assert not target[0, 5]

    def test_vertical_line(self):
        target = np.zeros((10, 10), dtype=bool)
        _draw_line_safe(1, 3, 7, 3, target, 10, 10)

        assert np.all(target[1:8, 3])
        assert not target[5, 0]

    def test_out_of_bounds_clamped(self):
        target = np.zeros((10, 10), dtype=bool)
        # Endpoints far outside grid — should not crash
        _draw_line_safe(-5, -5, 15, 15, target, 10, 10)

        # Clamped to (0,0)→(9,9), should mark some diagonal pixels
        assert target[0, 0]
        assert target[9, 9]

    def test_single_point(self):
        target = np.zeros((10, 10), dtype=bool)
        _draw_line_safe(4, 4, 4, 4, target, 10, 10)

        assert target[4, 4]
        assert target.sum() == 1


# ---------------------------------------------------------------------------
# merge_dural
# ---------------------------------------------------------------------------
class TestMergeDural:
    def test_basic_merge(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8) * 2
        falx = np.zeros((10, 10, 10), dtype=bool)
        tent = np.zeros((10, 10, 10), dtype=bool)
        falx[0:5, :, :] = True
        tent[5:10, :, :] = True

        n_falx, n_tent, n_overlap, n_total = merge_dural(mat, falx, tent)

        assert n_falx == 500
        assert n_tent == 500
        assert n_overlap == 0
        assert n_total == 1000
        assert np.all(mat == 10)

    def test_overlap_counted(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8) * 2
        falx = np.zeros((10, 10, 10), dtype=bool)
        tent = np.zeros((10, 10, 10), dtype=bool)
        falx[3:7, :, :] = True
        tent[5:9, :, :] = True

        n_falx, n_tent, n_overlap, n_total = merge_dural(mat, falx, tent)

        assert n_overlap == 200  # indices 5,6 overlap
        assert n_total == n_falx + n_tent - n_overlap

    def test_empty_masks(self):
        mat = np.ones((5, 5, 5), dtype=np.uint8) * 3
        falx = np.zeros((5, 5, 5), dtype=bool)
        tent = np.zeros((5, 5, 5), dtype=bool)

        n_falx, n_tent, n_overlap, n_total = merge_dural(mat, falx, tent)

        assert (n_falx, n_tent, n_overlap, n_total) == (0, 0, 0, 0)
        assert np.all(mat == 3)  # unchanged

    def test_modifies_mat_in_place(self):
        mat = np.ones((5, 5, 5), dtype=np.uint8)
        falx = np.zeros((5, 5, 5), dtype=bool)
        falx[2, 2, 2] = True
        tent = np.zeros((5, 5, 5), dtype=bool)
        orig_id = id(mat)

        merge_dural(mat, falx, tent)

        assert id(mat) == orig_id
        assert mat[2, 2, 2] == 10


# ---------------------------------------------------------------------------
# check_idempotency
# ---------------------------------------------------------------------------
class TestCheckIdempotency:
    def test_no_existing_dural(self):
        mat = np.ones((5, 5, 5), dtype=np.uint8) * 2
        fs = np.full((5, 5, 5), 3, dtype=np.int16)
        mat_before = mat.copy()

        result = check_idempotency(mat, fs)

        assert result == 0
        np.testing.assert_array_equal(mat, mat_before)

    def test_resets_to_tissue(self):
        mat = np.zeros((5, 5, 5), dtype=np.uint8)
        fs = np.zeros((5, 5, 5), dtype=np.int16)
        # Set one voxel to dural (u8=10) where FS label is 2 (WM → u8=1)
        mat[2, 2, 2] = 10
        fs[2, 2, 2] = 2

        result = check_idempotency(mat, fs)

        assert result == 1
        assert mat[2, 2, 2] == 1  # restored to cerebral WM

    def test_resets_vacuum_to_csf(self):
        mat = np.zeros((5, 5, 5), dtype=np.uint8)
        fs = np.zeros((5, 5, 5), dtype=np.int16)
        # u8=10 where FS label is 0 → maps to vacuum (u8=0) → should become CSF (u8=8)
        mat[2, 2, 2] = 10
        fs[2, 2, 2] = 0

        check_idempotency(mat, fs)

        assert mat[2, 2, 2] == 8


# ---------------------------------------------------------------------------
# _measure_notch_ellipse
# ---------------------------------------------------------------------------
class TestMeasureNotchEllipse:
    def _make_notch_scene(self, X=40, Y=50, Z=40, dx_mm=1.0):
        """Build a mat_crop with brainstem, cerebellum, and a cerebellar gap."""
        mat = np.zeros((X, Y, Z), dtype=np.uint8)
        mid_x = X // 2
        # Brainstem at center spanning a range of z
        mat[mid_x - 2:mid_x + 2, Y // 2 - 3:Y // 2 + 3, 15:30] = 6
        # Left cerebellar cortex
        mat[3:mid_x - 3, Y // 2 - 8:Y // 2 + 8, 10:28] = 4
        # Right cerebellar cortex
        mat[mid_x + 3:X - 3, Y // 2 - 8:Y // 2 + 8, 10:28] = 5
        return mat

    def test_returns_ellipse(self):
        mat = self._make_notch_scene()
        result = _measure_notch_ellipse(mat, dx_mm=1.0)

        assert result is not None
        assert result.ndim == 2
        assert result.dtype == bool

    def test_returns_none_no_brainstem(self):
        mat = np.zeros((20, 30, 20), dtype=np.uint8)
        # Only cerebellar tissue, no brainstem (label 6)
        mat[2:8, :, :] = 4
        mat[12:18, :, :] = 5

        result = _measure_notch_ellipse(mat, dx_mm=1.0)
        assert result is None

    def test_returns_none_no_cerebellar_gap(self):
        mat = np.zeros((20, 30, 20), dtype=np.uint8)
        # Brainstem only, no cerebellar tissue → no gap to measure
        mat[8:12, 10:20, 5:15] = 6

        result = _measure_notch_ellipse(mat, dx_mm=1.0)
        assert result is None

    def test_ellipse_shape(self):
        mat = self._make_notch_scene(X=30, Y=40, Z=30)
        result = _measure_notch_ellipse(mat, dx_mm=1.0)

        if result is not None:
            assert result.shape == (30, 40)


# ---------------------------------------------------------------------------
# _collect_falx_polygon
# ---------------------------------------------------------------------------
class TestCollectFalxPolygon:
    def test_returns_arrays(self):
        geo = _make_falx_geometry()
        inner_ys = np.linspace(geo.anchor_y, geo.crista_y, 50)
        inner_zs = np.full(50, 50.0)

        poly_y, poly_z = _collect_falx_polygon(
            inner_ys, inner_zs, geo, tent_post_y=geo.mem_y_min)

        assert isinstance(poly_y, np.ndarray)
        assert isinstance(poly_z, np.ndarray)
        assert len(poly_y) == len(poly_z)
        assert len(poly_y) > 0

    def test_starts_at_inner_ys(self):
        geo = _make_falx_geometry()
        inner_ys = np.linspace(geo.anchor_y, geo.crista_y, 50)
        inner_zs = np.full(50, 50.0)

        poly_y, poly_z = _collect_falx_polygon(
            inner_ys, inner_zs, geo, tent_post_y=geo.mem_y_min)

        assert poly_y[0] == pytest.approx(inner_ys[0])
        assert poly_z[0] == pytest.approx(inner_zs[0])

    def test_polygon_has_area(self):
        geo = _make_falx_geometry()
        inner_ys = np.linspace(geo.anchor_y, geo.crista_y, 50)
        inner_zs = np.full(50, 50.0)

        poly_y, poly_z = _collect_falx_polygon(
            inner_ys, inner_zs, geo, tent_post_y=geo.mem_y_min)

        area = _shoelace_area(poly_y, poly_z)
        assert area > 0


# ---------------------------------------------------------------------------
# _collect_boundary_polylines
# ---------------------------------------------------------------------------
class TestCollectBoundaryPolylines:
    def test_returns_polyline_and_area(self):
        geo = _make_falx_geometry()

        outer_y, outer_z, post_area = _collect_boundary_polylines(geo)

        assert len(outer_y) > 0
        assert len(outer_y) == len(outer_z)
        assert post_area >= 0

    def test_posterior_area_positive(self):
        # With tent in posterior region and dome ceiling, area > 0
        geo = _make_falx_geometry()

        _, _, post_area = _collect_boundary_polylines(geo)

        assert post_area > 0

    def test_no_posterior_range_zero_area(self):
        # Set mem_y_min = anchor_y so posterior loop range is empty
        geo = _make_falx_geometry(mem_y_min=10, anchor_y=10)

        _, _, post_area = _collect_boundary_polylines(geo)

        assert post_area == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _build_free_edge_controls
# ---------------------------------------------------------------------------
class TestBuildFreeEdgeControls:
    def test_output_shapes(self):
        geo = _make_falx_geometry()

        pchip_ys, pchip_zs, genu_ctrl_z = _build_free_edge_controls(geo)

        expected_len = geo.genu_y - geo.anchor_y
        assert len(pchip_ys) == expected_len
        assert len(pchip_zs) == expected_len
        assert isinstance(genu_ctrl_z, float)

    def test_pchip_ys_range(self):
        geo = _make_falx_geometry()

        pchip_ys, _, _ = _build_free_edge_controls(geo)

        assert pchip_ys[0] == pytest.approx(geo.anchor_y)
        assert pchip_ys[-1] == pytest.approx(geo.genu_y - 1)

    def test_single_cc_landmark(self):
        # Only genu landmark → 2 control points (anchor + genu), PCHIP still works
        geo = _make_falx_geometry()
        geo.cc_landmarks = {"genu": (geo.genu_y, 21.3)}

        pchip_ys, pchip_zs, genu_ctrl_z = _build_free_edge_controls(geo)

        assert len(pchip_ys) == geo.genu_y - geo.anchor_y
        assert not np.any(np.isnan(pchip_zs))


# ---------------------------------------------------------------------------
# _solve_bezier_shape
# ---------------------------------------------------------------------------
class TestSolveBezierShape:
    def _get_bezier_inputs(self):
        """Build inputs for _solve_bezier_shape from the shared geometry."""
        geo = _make_falx_geometry()
        pchip_ys, pchip_zs, genu_ctrl_z = _build_free_edge_controls(geo)
        outer_y, outer_z, post_area = _collect_boundary_polylines(geo)
        return pchip_ys, pchip_zs, genu_ctrl_z, outer_y, outer_z, post_area, geo

    def test_returns_arrays(self):
        pchip_ys, pchip_zs, genu_ctrl_z, outer_y, outer_z, post_area, geo = \
            self._get_bezier_inputs()

        result = _solve_bezier_shape(
            pchip_ys, pchip_zs, genu_ctrl_z,
            outer_y, outer_z, post_area, geo, dx_mm=1.0)

        assert result is not None
        inner_ys, inner_zs = result
        assert len(inner_ys) == len(inner_zs)
        assert len(inner_ys) > len(pchip_ys)  # PCHIP + Bezier

    def test_degenerate_returns_none(self):
        # Make genu and crista at the same y with mem_top == mem_bot
        # → skull contour start == end → skull_len == 0 → returns None
        geo = _make_falx_geometry(crista_y=35, crista_z=40.0, genu_y=35,
                                  mem_y_max=35)
        geo.mem_exists[:] = False
        geo.mem_exists[35] = True
        geo.mem_top_z[35] = 40
        geo.mem_bot_z[35] = 40
        geo.cc_landmarks = {"genu": (35, 40.0)}

        pchip_ys, pchip_zs, genu_ctrl_z = _build_free_edge_controls(geo)
        outer_y, outer_z, post_area = _collect_boundary_polylines(geo)

        result = _solve_bezier_shape(
            pchip_ys, pchip_zs, genu_ctrl_z,
            outer_y, outer_z, post_area, geo, dx_mm=1.0)

        assert result is None

    def test_inner_curve_spans_anchor_to_crista(self):
        pchip_ys, pchip_zs, genu_ctrl_z, outer_y, outer_z, post_area, geo = \
            self._get_bezier_inputs()

        result = _solve_bezier_shape(
            pchip_ys, pchip_zs, genu_ctrl_z,
            outer_y, outer_z, post_area, geo, dx_mm=1.0)

        assert result is not None
        inner_ys, inner_zs = result
        # Starts near anchor, ends near crista
        assert inner_ys[0] == pytest.approx(geo.anchor_y, abs=1)
        assert inner_ys[-1] == pytest.approx(geo.crista_y, abs=2)


# ---------------------------------------------------------------------------
# _detect_falx_geometry
# ---------------------------------------------------------------------------
class TestDetectFalxGeometry:
    def _make_scene(self, with_tent=True):
        """Build a simple scene for geometry detection."""
        X, Y, Z = 20, 50, 40
        mid_x = X // 2
        # Vertical membrane plane at midline
        membrane = np.zeros((X, Y, Z), dtype=bool)
        membrane[mid_x, 5:45, 5:35] = True
        # Skull SDF: negative everywhere (all intracranial)
        skull_crop = np.full((X, Y, Z), -5.0, dtype=np.float32)
        # FS labels: left and right + MOF for crista + CC for landmarks
        fs_crop = np.zeros((X, Y, Z), dtype=np.int16)
        fs_crop[:mid_x, :, :] = 2     # left WM
        fs_crop[mid_x:, :, :] = 41    # right WM
        # CC labels at midline
        fs_crop[mid_x, 12:14, 20:25] = 251  # splenium
        fs_crop[mid_x, 22:24, 20:25] = 253  # body
        fs_crop[mid_x, 32:34, 20:25] = 255  # genu
        # MOF labels anterior to genu (y > 32)
        fs_crop[mid_x, 38, 8] = _MOF_LEFT

        tent_crop = None
        if with_tent:
            tent_crop = np.zeros((X, Y, Z), dtype=bool)
            tent_crop[mid_x - 3:mid_x + 3, 5:15, 18:22] = True

        cc_landmarks = {"splenium": (13, 45.6), "body": (23, 25.7),
                        "genu": (33, 21.3)}
        cc_top_z = np.full(Y, -1, dtype=int)
        cc_exists = np.zeros(Y, dtype=bool)
        for y in range(12, 35):
            cc_exists[y] = True
            cc_top_z[y] = 25

        return membrane, skull_crop, tent_crop, cc_landmarks, cc_top_z, cc_exists, fs_crop

    def test_returns_dataclass(self):
        membrane, skull, tent, cc_lm, cc_top, cc_ex, fs = self._make_scene()

        geo = _detect_falx_geometry(
            membrane, skull, tent, cc_lm, cc_top, cc_ex, 1.0, fs)

        assert isinstance(geo, _FalxGeometry)
        assert hasattr(geo, "mem_top_z")
        assert hasattr(geo, "anchor_y")
        assert hasattr(geo, "crista_y")

    def test_anchor_from_tent(self):
        membrane, skull, tent, cc_lm, cc_top, cc_ex, fs = self._make_scene(
            with_tent=True)

        geo = _detect_falx_geometry(
            membrane, skull, tent, cc_lm, cc_top, cc_ex, 1.0, fs)

        # Anchor should come from the tentorium midline
        assert geo.anchor_y > 0

    def test_no_tent_fallback(self):
        membrane, skull, _, cc_lm, cc_top, cc_ex, fs = self._make_scene(
            with_tent=False)

        geo = _detect_falx_geometry(
            membrane, skull, None, cc_lm, cc_top, cc_ex, 1.0, fs)

        # Without tent, anchor falls back to membrane bottom at junction
        assert isinstance(geo.anchor_y, int)
        assert isinstance(geo.anchor_z, float)


# ---------------------------------------------------------------------------
# _rasterize_falx_cookie
# ---------------------------------------------------------------------------
class TestRasterizeFalxCookie:
    def _make_cookie_inputs(self):
        """Build membrane, inner curve, and geometry for rasterization."""
        geo = _make_falx_geometry(Y=60, Z=80)
        X = 30
        # Vertical plane membrane
        membrane = np.zeros((X, 60, 80), dtype=bool)
        membrane[X // 2, 5:55, 10:70] = True
        # Simple inner curve
        inner_ys = np.linspace(geo.anchor_y, geo.crista_y, 100)
        inner_zs = np.full(100, 50.0)
        n_membrane = int(membrane.sum())
        return membrane, inner_ys, inner_zs, geo, n_membrane

    def test_output_shape_matches_full(self):
        membrane, inner_ys, inner_zs, geo, n_mem = self._make_cookie_inputs()
        full_shape = (40, 70, 90)
        crop_slices = (slice(5, 35), slice(0, 60), slice(0, 80))

        falx = _rasterize_falx_cookie(
            membrane, inner_ys, inner_zs, geo, None,
            crop_slices, full_shape, n_mem)

        assert falx.shape == full_shape
        assert falx.dtype == bool

    def test_subset_of_membrane(self):
        membrane, inner_ys, inner_zs, geo, n_mem = self._make_cookie_inputs()
        full_shape = (30, 60, 80)
        crop_slices = (slice(0, 30), slice(0, 60), slice(0, 80))

        falx = _rasterize_falx_cookie(
            membrane, inner_ys, inner_zs, geo, None,
            crop_slices, full_shape, n_mem)

        # Every falx voxel should have been True in the membrane
        assert np.all(membrane[falx[crop_slices]])

    def test_no_tent_cut_when_tent_none(self):
        membrane, inner_ys, inner_zs, geo, n_mem = self._make_cookie_inputs()
        full_shape = (30, 60, 80)
        crop_slices = (slice(0, 30), slice(0, 60), slice(0, 80))

        # Should not crash with tent_crop=None
        falx = _rasterize_falx_cookie(
            membrane, inner_ys, inner_zs, geo, None,
            crop_slices, full_shape, n_mem)

        assert falx.sum() > 0


# ---------------------------------------------------------------------------
# _compute_midplane_membrane
# ---------------------------------------------------------------------------
class TestComputeMidplaneMembrane:
    def _make_bilateral_scene(self, with_cc=True):
        """Build FS labels with left/right hemispheres and optional CC."""
        X, Y, Z = 20, 30, 20
        fs = np.zeros((X, Y, Z), dtype=np.int16)
        # Left hemisphere on one side of midline
        fs[:X // 2, :, :] = 2    # left cerebral WM
        fs[X // 2:, :, :] = 41   # right cerebral WM

        if with_cc:
            # CC sub-regions at midline at different y positions
            fs[X // 2, 8:10, 8:12] = 251   # splenium
            fs[X // 2, 14:16, 8:12] = 253  # body
            fs[X // 2, 20:22, 8:12] = 255  # genu

        skull = np.full((X, Y, Z), -5.0, dtype=np.float32)
        return fs, skull

    def test_basic_membrane_detected(self):
        fs, skull = self._make_bilateral_scene()

        membrane, n_mem, cc_lm, cc_top, cc_ex = \
            _compute_midplane_membrane(fs, skull, dx_mm=1.0, thickness_mm=1.0)

        assert n_mem > 0
        assert membrane.shape == fs.shape

    def test_cc_landmarks_found(self):
        fs, skull = self._make_bilateral_scene(with_cc=True)

        _, _, cc_lm, _, _ = \
            _compute_midplane_membrane(fs, skull, dx_mm=1.0, thickness_mm=1.0)

        assert "splenium" in cc_lm
        assert "body" in cc_lm
        assert "genu" in cc_lm

    def test_no_cc_empty_landmarks(self):
        fs, skull = self._make_bilateral_scene(with_cc=False)

        _, _, cc_lm, _, _ = \
            _compute_midplane_membrane(fs, skull, dx_mm=1.0, thickness_mm=1.0)

        assert cc_lm == {}


# ---------------------------------------------------------------------------
# reconstruct_tentorium
# ---------------------------------------------------------------------------
class TestReconstructTentorium:
    def test_returns_bool_array(self):
        shape = (30, 40, 30)
        mat = np.zeros(shape, dtype=np.uint8)
        # Cerebral tissue in upper half (z > 15)
        mat[5:25, 5:35, 16:28] = 1
        # Cerebellar tissue in lower half (z < 15)
        mat[8:22, 10:30, 3:14] = 4
        crop_slices = (slice(0, 30), slice(0, 40), slice(0, 30))

        tent = reconstruct_tentorium(mat, dx_mm=1.0, notch_radius=5.0,
                                     crop_slices=crop_slices)

        assert tent.shape == shape
        assert tent.dtype == bool

    def test_tentorium_between_tissues(self):
        shape = (30, 40, 30)
        mat = np.zeros(shape, dtype=np.uint8)
        # Cerebral above z=15 (z=15..27)
        mat[5:25, 5:35, 15:28] = 1
        # Cerebellar below z=15 (z=3..14), adjacent to cerebral
        mat[8:22, 10:30, 3:15] = 4
        crop_slices = (slice(0, 30), slice(0, 40), slice(0, 30))

        tent = reconstruct_tentorium(mat, dx_mm=1.0, notch_radius=5.0,
                                     crop_slices=crop_slices)

        assert tent.any(), "tentorium should be non-empty with separated cerebral/cerebellar tissue"
        tent_z = np.where(tent.any(axis=(0, 1)))[0]
        median_z = np.median(tent_z)
        assert 10 < median_z < 20

    def test_empty_mat_no_crash(self):
        shape = (10, 10, 10)
        mat = np.zeros(shape, dtype=np.uint8)
        crop_slices = (slice(0, 10), slice(0, 10), slice(0, 10))

        # All-zero mat: no tissue to watershed between
        tent = reconstruct_tentorium(mat, dx_mm=1.0, notch_radius=5.0,
                                     crop_slices=crop_slices)

        assert tent.shape == shape
        assert not tent.any()


# ---------------------------------------------------------------------------
# reconstruct_falx — regression
# ---------------------------------------------------------------------------
class TestReconstructFalxRegression:
    def test_degenerate_geometry_returns_none(self):
        """Regression (f5041ca): reconstruct_falx should return None, not crash,
        when geometry is degenerate (e.g. no crista galli landmarks)."""
        X, Y, Z = 10, 10, 10
        mat = np.zeros((X, Y, Z), dtype=np.uint8)
        mat[:X // 2, :, :] = 1
        mat[X // 2:, :, :] = 1
        fs = np.zeros((X, Y, Z), dtype=np.int16)
        fs[4, 5, 5] = 2   # left WM
        fs[5, 5, 5] = 41  # right WM
        skull_sdf = np.full((X, Y, Z), -5.0, dtype=np.float32)
        crop_slices = (slice(0, X), slice(0, Y), slice(0, Z))

        result = reconstruct_falx(mat, fs, skull_sdf, dx_mm=1.0,
                                  crop_slices=crop_slices)
        assert result is None


# ---------------------------------------------------------------------------
# print_membrane_continuity
# ---------------------------------------------------------------------------
class TestPrintMembraneContinuity:
    def test_single_component(self, capsys):
        falx = np.zeros((10, 10, 10), dtype=bool)
        falx[5, 3:7, 3:7] = True  # one connected blob
        tent = np.zeros((10, 10, 10), dtype=bool)  # empty

        print_membrane_continuity(falx, tent, dx_mm=1.0)
        out = capsys.readouterr().out
        assert "1 components" in out
        assert "100.0%" in out
        assert "skipped" in out  # tent should be skipped

    def test_multiple_components(self, capsys):
        falx = np.zeros((20, 20, 20), dtype=bool)
        falx[10, 2:5, 2:5] = True   # blob 1
        falx[10, 15:18, 15:18] = True  # blob 2 (disconnected)
        tent = np.zeros((20, 20, 20), dtype=bool)

        print_membrane_continuity(falx, tent, dx_mm=1.0)
        out = capsys.readouterr().out
        assert "2 components" in out
        assert "second:" in out

    def test_empty_masks(self, capsys):
        falx = np.zeros((5, 5, 5), dtype=bool)
        tent = np.zeros((5, 5, 5), dtype=bool)

        print_membrane_continuity(falx, tent, dx_mm=1.0)
        out = capsys.readouterr().out
        assert out.count("skipped") >= 2


# ---------------------------------------------------------------------------
# print_csf_components
# ---------------------------------------------------------------------------
class TestPrintCsfComponents:
    def test_two_blobs(self, capsys):
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        mat[2:5, 2:5, 2:5] = 8    # CSF blob 1
        mat[15:18, 15:18, 15:18] = 8  # CSF blob 2

        print_csf_components(mat, dx_mm=1.0)
        out = capsys.readouterr().out
        assert "#1:" in out
        assert "#2:" in out
        assert "Total: 2 components" in out

    def test_no_csf(self, capsys):
        mat = np.zeros((5, 5, 5), dtype=np.uint8)

        print_csf_components(mat, dx_mm=1.0)
        out = capsys.readouterr().out
        assert "skipped (0 voxels)" in out


# ---------------------------------------------------------------------------
# print_thickness_estimate
# ---------------------------------------------------------------------------
class TestPrintThicknessEstimate:
    def test_thick_slab(self, capsys):
        falx = np.zeros((20, 20, 20), dtype=bool)
        falx[8:12, 3:17, 3:17] = True  # 4-voxel thick slab
        tent = np.zeros((20, 20, 20), dtype=bool)

        print_thickness_estimate(falx, tent, dx_mm=1.0)
        out = capsys.readouterr().out
        assert "Falx: ~" in out
        assert "skipped" in out  # tent

    def test_thin_plane(self, capsys):
        falx = np.zeros((20, 20, 20), dtype=bool)
        falx[10, 3:17, 3:17] = True  # 1-voxel thick plane
        tent = np.zeros((20, 20, 20), dtype=bool)

        print_thickness_estimate(falx, tent, dx_mm=1.0)
        out = capsys.readouterr().out
        # 1-voxel thick: no interior after erosion → n_surface = n → thickness formula
        n = int(falx.sum())
        eroded = binary_erosion(falx)
        n_interior = int(eroded.sum())
        if n_interior == 0:
            # All surface: thickness = n / (n/2) * dx = 2*dx
            assert "Falx: ~2.00 mm" in out
        else:
            assert "Falx: ~" in out

    def test_empty(self, capsys):
        falx = np.zeros((5, 5, 5), dtype=bool)
        tent = np.zeros((5, 5, 5), dtype=bool)

        print_thickness_estimate(falx, tent, dx_mm=1.0)
        out = capsys.readouterr().out
        assert out.count("skipped") >= 2


# ---------------------------------------------------------------------------
# print_junction_thickness
# ---------------------------------------------------------------------------
class TestPrintJunctionThickness:
    def test_overlap_with_warning(self, capsys):
        falx = np.zeros((20, 20, 20), dtype=bool)
        tent = np.zeros((20, 20, 20), dtype=bool)
        mat = np.ones((20, 20, 20), dtype=np.uint8) * 10
        # 4 contiguous z-voxels overlap in one column
        falx[10, 10, 5:9] = True
        tent[10, 10, 5:9] = True

        print_junction_thickness(falx, tent, dx_mm=1.0, mat=mat)
        out = capsys.readouterr().out
        assert "Overlap: 4" in out
        assert "Max z-run: 4" in out
        assert "WARNING" in out

    def test_no_overlap(self, capsys):
        falx = np.zeros((10, 10, 10), dtype=bool)
        tent = np.zeros((10, 10, 10), dtype=bool)
        mat = np.ones((10, 10, 10), dtype=np.uint8)
        falx[5, 5, 2:5] = True
        tent[5, 5, 7:9] = True  # no overlap

        print_junction_thickness(falx, tent, dx_mm=1.0, mat=mat)
        out = capsys.readouterr().out
        assert "No overlap" in out

    def test_short_run_no_warning(self, capsys):
        falx = np.zeros((10, 10, 10), dtype=bool)
        tent = np.zeros((10, 10, 10), dtype=bool)
        mat = np.ones((10, 10, 10), dtype=np.uint8) * 10
        # 2 contiguous z-voxels overlap
        falx[5, 5, 4:6] = True
        tent[5, 5, 4:6] = True

        print_junction_thickness(falx, tent, dx_mm=1.0, mat=mat)
        out = capsys.readouterr().out
        assert "Max z-run: 2" in out
        assert "WARNING" not in out
