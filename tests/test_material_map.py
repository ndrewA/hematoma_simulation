"""Tests for preprocessing/material_map.py — LUT construction and label mapping."""

import numpy as np
import pytest

from preprocessing.material_map import (
    DIRECT_MAP,
    FALLBACK_MAP,
    build_lut,
    apply_mapping,
    collect_warnings,
    _SENTINEL,
)
from preprocessing.utils import FS_LUT_SIZE


# ---------------------------------------------------------------------------
# build_lut
# ---------------------------------------------------------------------------
class TestBuildLut:
    def test_lut_shape(self):
        lut, _ = build_lut()
        assert lut.shape == (FS_LUT_SIZE,)
        assert lut.dtype == np.uint8

    def test_direct_labels_returned(self):
        lut, direct_labels = build_lut()
        assert direct_labels == set(DIRECT_MAP.keys())

    def test_vacuum_maps_to_zero(self):
        lut, _ = build_lut()
        assert lut[0] == 0

    def test_cerebral_wm(self):
        lut, _ = build_lut()
        assert lut[2] == 1    # Left cerebral WM
        assert lut[41] == 1   # Right cerebral WM

    def test_cortical_gm(self):
        lut, _ = build_lut()
        assert lut[3] == 2    # Left cortical GM
        assert lut[42] == 2   # Right cortical GM

    def test_desikan_parcellations(self):
        lut, _ = build_lut()
        # ctx-lh range: 1001–1035
        for fs in range(1001, 1036):
            assert lut[fs] == 2
        # ctx-rh range: 2001–2035
        for fs in range(2001, 2036):
            assert lut[fs] == 2

    def test_deep_gm(self):
        lut, _ = build_lut()
        assert lut[10] == 3   # Thalamus L
        assert lut[49] == 3   # Thalamus R
        assert lut[12] == 3   # Putamen L
        assert lut[17] == 3   # Hippocampus L

    def test_ventricles(self):
        lut, _ = build_lut()
        assert lut[4] == 7    # Lateral ventricle L
        assert lut[43] == 7   # Lateral ventricle R
        assert lut[14] == 7   # 3rd ventricle
        assert lut[15] == 7   # 4th ventricle

    def test_brainstem(self):
        lut, _ = build_lut()
        assert lut[16] == 6

    def test_choroid_plexus(self):
        lut, _ = build_lut()
        assert lut[31] == 9
        assert lut[63] == 9

    def test_fallback_labels_present(self):
        lut, _ = build_lut()
        for fs_label, u8_class in FALLBACK_MAP.items():
            if fs_label < FS_LUT_SIZE:
                assert lut[fs_label] == u8_class

    def test_unmapped_slots_get_sentinel(self):
        lut, _ = build_lut()
        all_mapped = set(DIRECT_MAP.keys()) | set(FALLBACK_MAP.keys())
        # Find a slot that is unmapped
        for i in range(FS_LUT_SIZE):
            if i not in all_mapped:
                assert lut[i] == _SENTINEL, f"Slot {i} should be sentinel"
                break


# ---------------------------------------------------------------------------
# apply_mapping
# ---------------------------------------------------------------------------
class TestApplyMapping:
    @pytest.fixture
    def lut(self):
        lut, _ = build_lut()
        return lut

    def test_direct_mapping(self, lut):
        fs = np.array([[[2, 41], [3, 42]]], dtype=np.int16)
        mat = apply_mapping(fs, lut)
        assert mat.dtype == np.uint8
        np.testing.assert_array_equal(mat, [[[1, 1], [2, 2]]])

    def test_negative_labels_clamped(self, lut):
        fs = np.array([-5, 0, 2], dtype=np.int16)
        mat = apply_mapping(fs, lut)
        assert mat[0] == lut[0]  # clamped to index 0

    def test_out_of_range_falls_to_gm(self, lut):
        fs = np.array([9999], dtype=np.int16)
        mat = apply_mapping(fs, lut)
        assert mat[0] == 2  # cortical GM fallback

    def test_sentinel_falls_to_gm(self, lut):
        # Find an unmapped slot
        all_mapped = set(DIRECT_MAP.keys()) | set(FALLBACK_MAP.keys())
        unmapped = None
        for i in range(FS_LUT_SIZE):
            if i not in all_mapped:
                unmapped = i
                break
        assert unmapped is not None
        fs = np.array([unmapped], dtype=np.int16)
        mat = apply_mapping(fs, lut)
        assert mat[0] == 2  # cortical GM fallback

    def test_shape_preserved(self, lut):
        fs = np.zeros((5, 6, 7), dtype=np.int16)
        mat = apply_mapping(fs, lut)
        assert mat.shape == (5, 6, 7)

    def test_all_valid_classes(self, lut):
        # Every output should be in {0..11}
        fs = np.arange(FS_LUT_SIZE, dtype=np.int16)
        mat = apply_mapping(fs, lut)
        assert set(np.unique(mat)).issubset(set(range(12)))


# ---------------------------------------------------------------------------
# collect_warnings
# ---------------------------------------------------------------------------
class TestCollectWarnings:
    def test_no_warnings_for_direct_labels(self, capsys):
        _, direct_labels = build_lut()
        # Only direct labels present
        fs = np.array(sorted(direct_labels)[:10], dtype=np.int16)
        collect_warnings(fs, direct_labels)
        captured = capsys.readouterr()
        assert "Fallback" not in captured.out
        assert "Unknown" not in captured.out

    def test_fallback_detected(self, capsys):
        _, direct_labels = build_lut()
        # Label 1 (Cerebral Exterior) is in FALLBACK_MAP
        fs = np.array([0, 2, 1], dtype=np.int16)
        collect_warnings(fs, direct_labels)
        captured = capsys.readouterr()
        assert "Fallback" in captured.out

    def test_unknown_detected(self, capsys):
        _, direct_labels = build_lut()
        # Label 999 is in neither DIRECT_MAP nor FALLBACK_MAP
        fs = np.array([0, 999], dtype=np.int16)
        collect_warnings(fs, direct_labels)
        captured = capsys.readouterr()
        assert "Unknown" in captured.out
