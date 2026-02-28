"""Tests for preprocessing/validation/checks.py — helpers + complex check functions."""

import numpy as np
import nibabel as nib
import pytest
from scipy.ndimage import distance_transform_edt

from preprocessing.validation.checks import (
    _adjacent_labels,
    _classify_dural,
    _compute_falx_cc_heights,
    _compute_surface_distances,
    _compute_volume_census,
    _find_landmark,
    _principal_direction,
    check_c1,
    check_c2,
    check_c3,
    check_c4,
    check_c5,
    check_c6,
    check_c7,
    check_c8,
    check_c9,
    check_c10,
    check_c11,
    check_d1,
    check_d2,
    check_d3,
    check_d4,
    check_d5,
    check_f1,
    check_f2,
    check_f3,
    check_f4,
    check_f5,
    check_f6,
    check_g1,
    check_g2,
    check_g3,
    check_h1,
    check_h2,
    check_h3,
    check_h4,
    check_h5,
    check_h6,
    check_h7,
    check_h8,
    check_h9,
    check_h10,
    check_m1,
    check_m2,
    check_m3,
    check_m4,
    check_v1,
    check_v2,
    check_v3,
    check_v4,
    check_v5,
    check_v6,
)
from tests.conftest import (
    _make_ctx,
    _make_fiber_ctx,
    _make_header_ctx,
    _MockNiftiImg,
)


# ═══════════════════════════════════════════════════════════════════════════
# _principal_direction
# ═══════════════════════════════════════════════════════════════════════════

class TestPrincipalDirection:
    def test_x_dominant(self):
        # m6 = [m00, m11, m22, m01, m02, m12]
        # Identity-like with large XX → principal direction ≈ [1, 0, 0]
        m6 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v = _principal_direction(m6)
        assert v is not None
        np.testing.assert_allclose(np.abs(v), [1.0, 0.0, 0.0], atol=1e-10)

    def test_isotropic(self):
        m6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        v = _principal_direction(m6)
        assert v is not None
        # Degenerate eigenvalues — any unit vector is valid
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-10)

    def test_all_zero_returns_none(self):
        m6 = np.zeros(6)
        assert _principal_direction(m6) is None


# ═══════════════════════════════════════════════════════════════════════════
# _find_landmark
# ═══════════════════════════════════════════════════════════════════════════

class TestFindLandmark:
    def _make_fiber(self, shape=(10, 12, 10)):
        """Return zeroed fiber_data with shape (*shape, 6)."""
        return np.zeros((*shape, 6), dtype=np.float32)

    def test_finds_dominant_axis(self):
        fd = self._make_fiber()
        # Voxel (5, 6, 4) has strong X-dominant tensor: m00=0.9, trace=1.0
        fd[5, 6, 4, 0] = 0.9   # m00
        fd[5, 6, 4, 1] = 0.05  # m11
        fd[5, 6, 4, 2] = 0.05  # m22
        result = _find_landmark(fd, dominant_axis=0, x_range=(0, 10))
        assert result == (5, 6, 4)

    def test_returns_none_no_candidates(self):
        fd = self._make_fiber()
        result = _find_landmark(fd, dominant_axis=0, x_range=(0, 10))
        assert result is None

    def test_respects_x_range(self):
        fd = self._make_fiber()
        fd[2, 6, 4, 0] = 0.9
        fd[2, 6, 4, 1] = 0.05
        fd[2, 6, 4, 2] = 0.05
        # Candidate at x=2, but x_range excludes it
        result = _find_landmark(fd, dominant_axis=0, x_range=(5, 10))
        assert result is None

    def test_fraction_threshold(self):
        fd = self._make_fiber()
        # frac = 0.4/1.0 = 0.4 < default 0.5
        fd[5, 6, 4, 0] = 0.4
        fd[5, 6, 4, 1] = 0.3
        fd[5, 6, 4, 2] = 0.3
        result = _find_landmark(fd, dominant_axis=0, x_range=(0, 10),
                                frac_thresh=0.5)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# _adjacent_labels
# ═══════════════════════════════════════════════════════════════════════════

class TestAdjacentLabels:
    def test_basic_adjacency(self):
        # 5x5x5 volume: label 1 at x=0, tissue at x=1, label 2 at x=2
        labeled = np.zeros((5, 5, 5), dtype=np.int32)
        labeled[0, :, :] = 1
        labeled[2, :, :] = 2
        tissue = np.zeros((5, 5, 5), dtype=bool)
        tissue[1, :, :] = True
        adj = _adjacent_labels(labeled, tissue)
        assert adj == {1, 2}

    def test_no_adjacency(self):
        # Tissue at x=1, labels only at x=4 — gap
        labeled = np.zeros((5, 5, 5), dtype=np.int32)
        labeled[4, :, :] = 1
        tissue = np.zeros((5, 5, 5), dtype=bool)
        tissue[1, :, :] = True
        adj = _adjacent_labels(labeled, tissue)
        assert adj == set()

    def test_six_connectivity_not_diagonal(self):
        # Label at (0,0,0), tissue at (1,1,1) — diagonal only, no face adj
        labeled = np.zeros((3, 3, 3), dtype=np.int32)
        labeled[0, 0, 0] = 1
        tissue = np.zeros((3, 3, 3), dtype=bool)
        tissue[1, 1, 1] = True
        adj = _adjacent_labels(labeled, tissue)
        assert adj == set()

    def test_empty_tissue_mask(self):
        labeled = np.zeros((3, 3, 3), dtype=np.int32)
        labeled[1, 1, 1] = 1
        tissue = np.zeros((3, 3, 3), dtype=bool)
        adj = _adjacent_labels(labeled, tissue)
        assert adj == set()


# ═══════════════════════════════════════════════════════════════════════════
# _classify_dural
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyDural:
    def test_basic_classification(self):
        # 20³ volume: cerebellar at z=0-4, dural at z=5 (near) and z=15 (far)
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        mat[:, :, 0:5] = 4   # cerebellar GM
        mat[10, 10, 5] = 10  # dural near cerebellum → tentorium
        mat[10, 10, 15] = 10  # dural far from cerebellum → falx
        ctx = _make_ctx(mat)
        falx, tent = _classify_dural(ctx)
        assert tent[10, 10, 5]
        assert falx[10, 10, 15]

    def test_no_dural_returns_empty(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)  # all GM, no dural
        ctx = _make_ctx(mat)
        falx, tent = _classify_dural(ctx)
        assert falx.sum() == 0
        assert tent.sum() == 0

    def test_all_dural_no_cerebellum(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[5, 5, 5] = 10  # dural
        ctx = _make_ctx(mat)
        falx, tent = _classify_dural(ctx)
        assert falx[5, 5, 5]      # classified as falx (no cerebellum)
        assert tent.sum() == 0

    def test_dilation_distance(self):
        # Dural 4 voxels from cerebellum — beyond 3-iteration dilation → falx
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        mat[10, 10, 0] = 5   # cerebellar WM
        mat[10, 10, 5] = 10  # dural: 5 voxels away (> 3 dilation iterations)
        ctx = _make_ctx(mat)
        falx, tent = _classify_dural(ctx)
        assert falx[10, 10, 5]
        assert not tent[10, 10, 5]


# ═══════════════════════════════════════════════════════════════════════════
# _compute_volume_census
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeVolumeCensus:
    def test_known_volumes(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[:, :, :] = 1  # 1000 voxels of cerebral WM
        sdf = -np.ones((10, 10, 10), dtype=np.float32)  # all inside skull
        ctx = _make_ctx(mat, sdf=sdf, dx=2.0)
        vc = _compute_volume_census(ctx)
        # vol_voxel_ml = 2.0³ / 1000 = 0.008
        # 1000 voxels × 0.008 = 8.0 mL
        assert abs(vc["parenchyma_ml"] - 8.0) < 0.01

    def test_parenchyma_sum(self):
        # Parenchyma ids: [1,2,3,4,5,6,9]
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        # 10 voxels each at z-slices for parenchyma classes
        for i, cls in enumerate([1, 2, 3, 4, 5, 6, 9]):
            mat[i, 0, 0] = cls
        sdf = np.ones((20, 20, 20), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=1.0)
        vc = _compute_volume_census(ctx)
        # 7 voxels × 1.0³/1000 = 0.007 mL
        assert abs(vc["parenchyma_ml"] - 0.007) < 1e-6

    def test_icv_from_sdf(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = np.ones((10, 10, 10), dtype=np.float32)
        # 50 voxels with sdf < 0
        sdf[:5, :, :] = -1.0  # 500 voxels
        ctx = _make_ctx(mat, sdf=sdf, dx=1.0)
        vc = _compute_volume_census(ctx)
        # 500 × 1.0/1000 = 0.5 mL
        assert abs(vc["icv_ml"] - 0.5) < 1e-6
        assert vc["n_sdf_neg"] == 500

    def test_empty_volume(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=1.0)
        vc = _compute_volume_census(ctx)
        assert vc["parenchyma_ml"] == 0.0
        assert vc["icv_ml"] == 0.0
        assert vc["n_nonzero_mat"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# check_d4 — isolated vacuum islands inside skull
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckD4:
    def test_no_islands_passes(self):
        # Tissue fills the interior, vacuum only on boundary (exterior)
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[2:8, 2:8, 2:8] = 1  # tissue block in center
        sdf = np.ones((10, 10, 10), dtype=np.float32)
        sdf[2:8, 2:8, 2:8] = -1.0  # inside skull
        ctx = _make_ctx(mat, sdf=sdf)
        check_d4(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_island_inside_skull_fails(self):
        # Tissue fills interior except a vacuum pocket inside skull
        mat = np.zeros((12, 12, 12), dtype=np.uint8)
        mat[2:10, 2:10, 2:10] = 1  # tissue
        mat[5, 5, 5] = 0            # vacuum pocket inside
        sdf = np.ones((12, 12, 12), dtype=np.float32)
        sdf[2:10, 2:10, 2:10] = -1.0
        ctx = _make_ctx(mat, sdf=sdf)
        check_d4(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_island_outside_skull_ok(self):
        # Small vacuum cluster but sdf>0 at those voxels (outside skull)
        mat = np.zeros((12, 12, 12), dtype=np.uint8)
        mat[2:10, 2:10, 2:10] = 1
        mat[5, 5, 5] = 0  # vacuum pocket
        sdf = np.ones((12, 12, 12), dtype=np.float32)
        sdf[2:10, 2:10, 2:10] = -1.0
        sdf[5, 5, 5] = 1.0  # but sdf > 0 means outside skull
        ctx = _make_ctx(mat, sdf=sdf)
        check_d4(ctx)
        assert ctx.results[-1]["status"] == "PASS"


# ═══════════════════════════════════════════════════════════════════════════
# check_d5 — SDF cut-cell gradient quality
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckD5:
    def test_perfect_sdf_passes(self):
        # Build SDF from EDT of a sphere: |∇| ≈ 1.0 everywhere
        N = 30
        dx = 1.0
        coords = np.mgrid[:N, :N, :N].astype(np.float32)
        center = N / 2
        r = N / 3
        dist = np.sqrt((coords[0] - center)**2 +
                       (coords[1] - center)**2 +
                       (coords[2] - center)**2)
        sdf = (dist - r).astype(np.float32)
        mat = np.zeros((N, N, N), dtype=np.uint8)
        ctx = _make_ctx(mat, sdf=sdf, dx=dx)
        check_d5(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_no_cut_cells_passes(self):
        # SDF with all values >> dx (no zero-crossing)
        N = 10
        sdf = np.full((N, N, N), 100.0, dtype=np.float32)
        mat = np.zeros((N, N, N), dtype=np.uint8)
        ctx = _make_ctx(mat, sdf=sdf, dx=1.0)
        check_d5(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "no cut-cell" in ctx.results[-1]["value"]

    def test_noisy_gradient_fails(self):
        # SDF with random noise at boundary → gradient is wild
        N = 20
        dx = 1.0
        rng = np.random.default_rng(123)
        sdf = rng.uniform(-0.4, 0.4, (N, N, N)).astype(np.float32)
        mat = np.zeros((N, N, N), dtype=np.uint8)
        ctx = _make_ctx(mat, sdf=sdf, dx=dx)
        check_d5(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_boundary_cut_cells_filtered(self):
        """Regression (f5041ca): boundary voxels were using clamped one-sided differences."""
        N = 10
        sdf = np.full((N, N, N), 5.0, dtype=np.float32)
        # Place zero-crossing only at array boundary faces (x=0, x=N-1)
        # These are the only cut cells, and should all be excluded
        sdf[0, :, :] = -0.3
        sdf[-1, :, :] = -0.3
        mat = np.zeros((N, N, N), dtype=np.uint8)
        ctx = _make_ctx(mat, sdf=sdf, dx=1.0)
        check_d5(ctx)
        # Should pass — boundary cut-cells are excluded, not biased
        assert ctx.results[-1]["status"] == "PASS"


# ═══════════════════════════════════════════════════════════════════════════
# check_c4 — falx barrier separates left/right supratentorial CSF
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckC4:
    def _make_c4_ctx(self, mat, cc_z_sup=5):
        """Build ctx with synthetic fs_data cache for C4."""
        from preprocessing.utils import FS_LUT_SIZE
        ctx = _make_ctx(mat)
        N = mat.shape[0]

        # Build minimal LUTs: left hemisphere x < N//2, right x > N//2
        left_lut = np.zeros(FS_LUT_SIZE, dtype=bool)
        right_lut = np.zeros(FS_LUT_SIZE, dtype=bool)
        left_lut[1] = True   # label 1 = left tissue
        right_lut[2] = True  # label 2 = right tissue

        # Build fs_safe: left hemisphere has label 1, right has label 2
        fs_safe = np.zeros((N, N, N), dtype=np.int16)
        fs_safe[:N // 2, :, :] = 1   # left
        fs_safe[N // 2:, :, :] = 2   # right

        ctx._cache["fs_data"] = {
            "cc_z_sup": cc_z_sup,
            "fs_safe": fs_safe,
            "left_lut": left_lut,
            "right_lut": right_lut,
        }
        # C4 needs fs path to "exist" — override check by pre-populating cache
        return ctx

    def test_separated_csf_passes(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        # Left CSF in left hemisphere, right CSF in right hemisphere, above cc_z_sup=5
        mat[3:8, 5:15, 8:18] = 8    # left CSF
        mat[12:17, 5:15, 8:18] = 8  # right CSF (separate component)
        # Dural at midline separating them
        mat[9:11, 5:15, 8:18] = 10
        ctx = self._make_c4_ctx(mat, cc_z_sup=5)
        check_c4(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_bridging_csf_fails(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        # Single CSF blob spanning both hemispheres above cc_z_sup=5
        mat[3:17, 5:15, 8:18] = 8
        ctx = self._make_c4_ctx(mat, cc_z_sup=5)
        check_c4(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_supratentorial_csf_skips(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        # CSF only below cc_z_sup=15 → none above
        mat[3:17, 5:15, 0:10] = 8
        ctx = self._make_c4_ctx(mat, cc_z_sup=15)
        check_c4(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


# ═══════════════════════════════════════════════════════════════════════════
# Domain Closure (D1–D3)
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckD1:
    def test_no_vacuum_passes(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf)
        check_d1(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_vacuum_inside_fails(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)
        mat[5, 5, 5] = 0  # vacuum inside skull
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf)
        check_d1(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckD2:
    def test_no_tissue_outside_passes(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[3:7, 3:7, 3:7] = 1
        sdf = np.ones((10, 10, 10), dtype=np.float32)
        sdf[3:7, 3:7, 3:7] = -1.0
        ctx = _make_ctx(mat, sdf=sdf)
        check_d2(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_tissue_outside_fails(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[5, 5, 5] = 1  # tissue outside skull
        sdf = np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf)
        check_d2(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckD3:
    def test_brain_inside_passes(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf)
        ctx._brain = np.ones((10, 10, 10), dtype=np.uint8)
        check_d3(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_brain_outside_fails(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf)
        ctx._brain = np.ones((10, 10, 10), dtype=np.uint8)
        check_d3(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


# ═══════════════════════════════════════════════════════════════════════════
# Material Integrity (M1–M4)
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckM1:
    def test_valid_passes(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[5, 5, 5] = 11
        ctx = _make_ctx(mat)
        check_m1(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_out_of_range_fails(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[5, 5, 5] = 12
        ctx = _make_ctx(mat)
        check_m1(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckM2:
    def test_no_255_passes(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        check_m2(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_has_255_fails(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[5, 5, 5] = 255
        ctx = _make_ctx(mat)
        check_m2(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckM3:
    def test_all_present_passes(self):
        mat = np.zeros((12, 12, 12), dtype=np.uint8)
        for i in range(1, 12):
            mat[i, 0, 0] = i
        ctx = _make_ctx(mat)
        check_m3(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_missing_warns(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)  # only class 1
        ctx = _make_ctx(mat)
        check_m3(ctx)
        assert ctx.results[-1]["status"] == "WARN"


class TestCheckM4:
    def test_consistent_passes(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        ctx._brain = np.ones((10, 10, 10), dtype=np.uint8)
        check_m4(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_brain_vacuum_fails(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        ctx._brain = np.ones((10, 10, 10), dtype=np.uint8)
        check_m4(ctx)
        assert ctx.results[-1]["status"] == "WARN"


# ═══════════════════════════════════════════════════════════════════════════
# Volume Sanity (V1–V6)
# dx=10mm → vol_voxel_ml = 1.0 mL, 10³ = 1000 voxels = 1000 mL
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckV1:
    def test_normal_passes(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)  # 1000 mL parenchyma
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v1(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_low_fails(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[0, 0, 0] = 1  # 1 mL, below 850
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v1(ctx)
        assert ctx.results[-1]["status"] == "WARN"


class TestCheckV2:
    def test_normal_passes(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[0, :2, :] = 7  # 20 voxels = 20 mL ventricular
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v2(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_excessive_fails(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[0, :, :] = 7  # 100 mL, above 80
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v2(ctx)
        assert ctx.results[-1]["status"] == "WARN"


class TestCheckV3:
    def test_normal_passes(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)  # parenchyma
        mat[:, :, 0] = 8  # 100 subarachnoid voxels = 10% ICV
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v3(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_low_fails(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)
        mat[0, 0, 0] = 8  # 1 voxel = 0.1% ICV
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v3(ctx)
        assert ctx.results[-1]["status"] == "WARN"


class TestCheckV4:
    def test_normal_passes(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[0, :, 0] = 10  # 10 voxels = 10 mL dural
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v4(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_zero_fails(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)  # no dural
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v4(ctx)
        assert ctx.results[-1]["status"] == "WARN"


class TestCheckV5:
    def test_consistent_passes(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v5(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_inconsistent_fails(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[:5, :, :] = 1  # 500 tissue but 1000 inside skull
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v5(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_sdf_negative_fails(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)
        sdf = np.ones((10, 10, 10), dtype=np.float32)  # all positive
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v5(ctx)
        assert ctx.results[-1]["status"] == "WARN"
        assert "no SDF<0" in ctx.results[-1]["value"]


class TestCheckV6:
    def test_reports_census(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8)
        sdf = -np.ones((10, 10, 10), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf, dx=10.0)
        check_v6(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert len(ctx.census) > 0
        assert "brain_parenchyma_mL" in ctx.metrics


# ═══════════════════════════════════════════════════════════════════════════
# Compartmentalization (C1)
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckC1:
    def test_single_component(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[2:8, 2:8, 2:8] = 1
        ctx = _make_ctx(mat)
        check_c1(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert ctx.metrics["active_domain_components"] == 1

    def test_multiple_components(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[1, 1, 1] = 1
        mat[8, 8, 8] = 1
        ctx = _make_ctx(mat)
        check_c1(ctx)
        assert ctx.results[-1]["status"] == "PASS"  # INFO always passes
        assert ctx.metrics["active_domain_components"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# Dural Membrane (C2, C3, C5–C11)
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckC2:
    def test_single_blob_passes(self):
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        mat[10, 3:17, 10:18] = 10  # one connected falx region
        ctx = _make_ctx(mat)
        check_c2(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_fragmented_fails(self):
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        for i in range(10):
            mat[10, 2 * i, 10] = 10  # 10 isolated dural voxels
        ctx = _make_ctx(mat)
        check_c2(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_falx_skips(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        check_c2(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


class TestCheckC3:
    def test_csf_adjacent_passes(self):
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        mat[10, 10, 5:15] = 6  # brainstem
        z_min, z_max = 5, 14
        z_upper = z_min + 2 * (z_max - z_min) // 3  # = 11
        mat[11, 10, z_upper] = 8  # CSF adjacent to brainstem
        ctx = _make_ctx(mat)
        check_c3(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_no_csf_fails(self):
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        mat[10, 10, 5:15] = 6  # brainstem only
        ctx = _make_ctx(mat)
        check_c3(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_brainstem_skips(self):
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        ctx = _make_ctx(mat)
        check_c3(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


class TestCheckC5:
    def test_high_coverage_passes(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        mat[N // 2, 5:15, 5:15] = 10  # dural at midline fissure
        sdf = -np.ones((N, N, N), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf)
        check_c5(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_low_coverage_fails(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        mat[N // 2, 5:15, 5:15] = 8  # CSF at midline, no dural
        sdf = -np.ones((N, N, N), dtype=np.float32)
        ctx = _make_ctx(mat, sdf=sdf)
        check_c5(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_fissure_skips(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        sdf = np.ones((N, N, N), dtype=np.float32)  # all outside skull
        ctx = _make_ctx(mat, sdf=sdf)
        check_c5(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


class TestCheckC6:
    def test_covered_passes(self):
        N = 30
        mat = np.zeros((N, N, N), dtype=np.uint8)
        mat[14:16, 14:16, 7:15] = 6   # brainstem
        mat[5:25, 5:25, 0:7] = 4      # cerebellar
        mat[5:25, 5:25, 17:28] = 1    # cerebral
        mat[5:25, 5:25, 10:13] = 10   # dural between
        ctx = _make_ctx(mat)
        check_c6(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_uncovered_fails(self):
        N = 30
        mat = np.zeros((N, N, N), dtype=np.uint8)
        mat[14:16, 14:16, 7:15] = 6
        mat[5:25, 5:25, 0:7] = 4
        mat[5:25, 5:25, 17:28] = 1
        # No dural
        ctx = _make_ctx(mat)
        check_c6(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_brainstem_skips(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        check_c6(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


class TestCheckC7:
    def test_separated_passes(self):
        N = 30
        mat = np.zeros((N, N, N), dtype=np.uint8)
        mat[14:16, 14:16, 7:15] = 6  # brainstem
        mat[5:10, 5:10, 0:5] = 8     # infra CSF
        mat[5:10, 5:10, 25:29] = 8   # supra CSF
        ctx = _make_ctx(mat)
        check_c7(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_no_csf_skips(self):
        N = 30
        mat = np.zeros((N, N, N), dtype=np.uint8)
        mat[14:16, 14:16, 7:15] = 6  # brainstem only
        ctx = _make_ctx(mat)
        check_c7(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]

    def test_no_brainstem_skips(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        mat[2:8, 2:8, 2:8] = 8  # CSF but no brainstem (label 6)
        ctx = _make_ctx(mat)
        check_c7(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped (no brainstem)" in ctx.results[-1]["value"]


class TestCheckC8:
    def test_anatomical_area_passes(self):
        N = 30
        dx = 3.0
        mat = np.zeros((N, N, N), dtype=np.uint8)
        mat[:, :, 0:5] = 4             # cerebellar (for tent classification)
        mat[1:29, 1:23, 6] = 10        # tent (near cerebellum)
        mat[15, 1:29, 9:28] = 10       # falx (far from cerebellum)
        ctx = _make_ctx(mat, dx=dx)
        check_c8(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_small_area_fails(self):
        mat = np.zeros((20, 20, 20), dtype=np.uint8)
        mat[10, 10, 10] = 10  # single dural voxel
        ctx = _make_ctx(mat, dx=1.0)
        check_c8(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_dural_skips(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        check_c8(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


class TestCheckC9:
    def _make_c9c10_ctx(self, cc_label_id, cc_y, height_vox, N=40, dx=1.0):
        """Build ctx with fs_data and dural_classification caches."""
        from preprocessing.utils import FS_LUT_SIZE
        mat = np.zeros((N, N, N), dtype=np.uint8)
        ctx = _make_ctx(mat, dx=dx)
        fs_safe = np.zeros((N, N, N), dtype=np.int16)
        if cc_label_id is not None:
            fs_safe[:, cc_y, :] = cc_label_id
        ctx._cache["fs_data"] = {
            "cc_z_sup": 25,
            "fs_safe": fs_safe,
            "left_lut": np.zeros(FS_LUT_SIZE, dtype=bool),
            "right_lut": np.zeros(FS_LUT_SIZE, dtype=bool),
        }
        falx = np.zeros((N, N, N), dtype=bool)
        if height_vox is not None and height_vox > 0:
            z_start = 20 - height_vox // 2
            falx[:, cc_y, z_start:z_start + height_vox] = True
        tent = np.zeros((N, N, N), dtype=bool)
        ctx._cache["dural_classification"] = (falx, tent)
        return ctx

    def test_normal_passes(self):
        ctx = self._make_c9c10_ctx(253, 20, 25)  # 25mm in [14, 37]
        check_c9(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_short_fails(self):
        ctx = self._make_c9c10_ctx(253, 20, 5)  # 5mm < 14
        check_c9(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_cc_skips(self):
        ctx = self._make_c9c10_ctx(None, 20, 0)  # no CC body
        check_c9(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


class TestCheckC10:
    def test_normal_passes(self):
        ctx = TestCheckC9()._make_c9c10_ctx(255, 15, 20)  # 20mm in [10, 33]
        check_c10(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_short_fails(self):
        ctx = TestCheckC9()._make_c9c10_ctx(255, 15, 5)  # 5mm < 10
        check_c10(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_cc_skips(self):
        ctx = TestCheckC9()._make_c9c10_ctx(None, 15, 0)
        check_c10(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


class TestCheckC11:
    def test_adequate_passes(self):
        N = 40
        mat = np.zeros((N, N, N), dtype=np.uint8)
        falx = np.zeros((N, N, N), dtype=bool)
        tent = np.zeros((N, N, N), dtype=bool)
        falx[20, 5:38, 20:30] = True    # sagittal curtain
        tent[10:30, 5:38, 18] = True     # horizontal sheet
        ctx = _make_ctx(mat)
        ctx._cache["dural_classification"] = (falx, tent)
        check_c11(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_short_fails(self):
        N = 40
        mat = np.zeros((N, N, N), dtype=np.uint8)
        falx = np.zeros((N, N, N), dtype=bool)
        tent = np.zeros((N, N, N), dtype=bool)
        falx[20, 18:22, 20:25] = True   # narrow y-range
        tent[18:22, 18:22, 18] = True
        ctx = _make_ctx(mat)
        ctx._cache["dural_classification"] = (falx, tent)
        check_c11(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_overlap_skips(self):
        N = 40
        mat = np.zeros((N, N, N), dtype=np.uint8)
        falx = np.zeros((N, N, N), dtype=bool)
        tent = np.zeros((N, N, N), dtype=bool)
        tent[10:30, 5:35, 5] = True  # tent exists, no falx
        ctx = _make_ctx(mat)
        ctx._cache["dural_classification"] = (falx, tent)
        check_c11(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


# ═══════════════════════════════════════════════════════════════════════════
# Fiber Texture (F1–F6)
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckF1:
    def test_high_coverage_passes(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        mat[5:15, 5:15, 5:15] = 1  # WM
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        fiber_data[5:15, 5:15, 5:15, 0] = 0.3  # nonzero trace
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4), dx=1.0,
                              mat_affine=np.eye(4))
        check_f1(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_low_coverage_fails(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        mat[5:15, 5:15, 5:15] = 1
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4), dx=1.0,
                              mat_affine=np.eye(4))
        check_f1(ctx)
        assert ctx.results[-1]["status"] == "WARN"

    def test_no_wm_fails(self):
        N = 10
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4), dx=1.0,
                              mat_affine=np.eye(4))
        check_f1(ctx)
        assert ctx.results[-1]["status"] == "WARN"


class TestCheckF2:
    def test_psd_passes(self):
        N = 10
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        fiber_data[3:7, 3:7, 3:7, 0] = 0.3
        fiber_data[3:7, 3:7, 3:7, 1] = 0.3
        fiber_data[3:7, 3:7, 3:7, 2] = 0.3
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4))
        check_f2(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_negative_diag_fails(self):
        N = 10
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        fiber_data[5, 5, 5, 0] = -0.5
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4))
        check_f2(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckF3:
    def test_bound_passes(self):
        N = 10
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        fiber_data[5, 5, 5, :3] = 0.3  # trace = 0.9
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4))
        check_f3(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_exceeds_fails(self):
        N = 10
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        fiber_data[5, 5, 5, :3] = 0.5  # trace = 1.5
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4))
        check_f3(ctx)
        assert ctx.results[-1]["status"] == "WARN"


class TestCheckF4:
    def test_x_dominant_passes(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        fiber_data[10, 10, 10, 0] = 0.9   # m00 (XX)
        fiber_data[10, 10, 10, 1] = 0.05
        fiber_data[10, 10, 10, 2] = 0.05
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4))
        check_f4(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_no_cc_skips(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4))
        check_f4(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


class TestCheckF5:
    def test_z_dominant_passes(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        x_pos = 7  # 0.35 * 20 = 7, within x_range (6, 9)
        fiber_data[x_pos, 10, 10, 0] = 0.05
        fiber_data[x_pos, 10, 10, 1] = 0.05
        fiber_data[x_pos, 10, 10, 2] = 0.9  # m22 (ZZ)
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4))
        check_f5(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_no_ic_skips(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4))
        check_f5(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "skipped" in ctx.results[-1]["value"]


class TestCheckF6:
    def test_round_trip_passes(self):
        N = 20
        dx = 1.0
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        mat_affine = np.diag([dx, dx, dx, 1.0])
        mat_affine[:3, 3] = -N * dx / 2.0
        fiber_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        fiber_affine[:3, 3] = -N / 2.0
        ctx = _make_fiber_ctx(mat, fiber_data, fiber_affine, dx=dx,
                              mat_affine=mat_affine)
        check_f6(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_wrong_center_fails(self):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        fiber_data = np.zeros((N, N, N, 6), dtype=np.float32)
        mat_affine = np.diag([1.0, 1.0, 1.0, 1.0])  # no translation
        ctx = _make_fiber_ctx(mat, fiber_data, np.eye(4), dx=1.0,
                              mat_affine=mat_affine)
        check_f6(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: _compute_falx_cc_heights, _compute_surface_distances
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeFalxCcHeights:
    def test_returns_heights(self):
        from preprocessing.utils import FS_LUT_SIZE
        N = 40
        mat = np.zeros((N, N, N), dtype=np.uint8)
        ctx = _make_ctx(mat)
        fs_safe = np.zeros((N, N, N), dtype=np.int16)
        fs_safe[:, 20, :] = 253  # CC body at y=20
        fs_safe[:, 30, :] = 255  # CC genu at y=30
        ctx._cache["fs_data"] = {
            "cc_z_sup": 25, "fs_safe": fs_safe,
            "left_lut": np.zeros(FS_LUT_SIZE, dtype=bool),
            "right_lut": np.zeros(FS_LUT_SIZE, dtype=bool),
        }
        falx = np.zeros((N, N, N), dtype=bool)
        falx[:, 20, 10:30] = True   # 20 voxels at body y
        falx[:, 30, 15:25] = True   # 10 voxels at genu y
        ctx._cache["dural_classification"] = (falx, np.zeros_like(falx))
        heights = _compute_falx_cc_heights(ctx)
        assert heights["body_mm"] == 20.0
        assert heights["genu_mm"] == 10.0

    def test_no_cc_returns_none(self):
        from preprocessing.utils import FS_LUT_SIZE
        N = 40
        ctx = _make_ctx(np.zeros((N, N, N), dtype=np.uint8))
        ctx._cache["fs_data"] = {
            "cc_z_sup": 25,
            "fs_safe": np.zeros((N, N, N), dtype=np.int16),
            "left_lut": np.zeros(FS_LUT_SIZE, dtype=bool),
            "right_lut": np.zeros(FS_LUT_SIZE, dtype=bool),
        }
        falx = np.zeros((N, N, N), dtype=bool)
        ctx._cache["dural_classification"] = (falx, falx.copy())
        heights = _compute_falx_cc_heights(ctx)
        assert heights["body_mm"] is None
        assert heights["genu_mm"] is None

    def test_no_falx_returns_none(self):
        from preprocessing.utils import FS_LUT_SIZE
        N = 40
        ctx = _make_ctx(np.zeros((N, N, N), dtype=np.uint8))
        fs_safe = np.zeros((N, N, N), dtype=np.int16)
        fs_safe[:, 20, :] = 253
        ctx._cache["fs_data"] = {
            "cc_z_sup": 25, "fs_safe": fs_safe,
            "left_lut": np.zeros(FS_LUT_SIZE, dtype=bool),
            "right_lut": np.zeros(FS_LUT_SIZE, dtype=bool),
        }
        falx = np.zeros((N, N, N), dtype=bool)
        ctx._cache["dural_classification"] = (falx, falx.copy())
        heights = _compute_falx_cc_heights(ctx)
        assert heights["body_mm"] is None


class TestComputeSurfaceDistances:
    def _make_sphere_sdf(self, N, center, r):
        coords = np.mgrid[:N, :N, :N].astype(np.float32)
        dist = np.sqrt(sum((coords[i] - center[i])**2 for i in range(3)))
        return (dist - r).astype(np.float32)

    def test_identical_sdfs_zero(self):
        N = 30
        sdf = self._make_sphere_sdf(N, [N / 2] * 3, N / 3)
        affine = np.eye(4)
        ctx = _make_ctx(np.zeros((N, N, N), dtype=np.uint8), sdf=sdf)
        ctx._headers = {
            "mat": nib.Nifti1Image(ctx._mat, affine),
            "sdf": nib.Nifti1Image(ctx._sdf, affine),
            "brain": nib.Nifti1Image(np.zeros((N, N, N), dtype=np.uint8),
                                     affine),
        }
        ctx._cache["simnibs_resampled"] = {
            "sdf": sdf.copy(),
            "labels": np.zeros((N, N, N), dtype=np.int16),
            "inner_boundary": np.zeros((N, N, N), dtype=bool),
        }
        sd = _compute_surface_distances(ctx)
        assert sd["d_o2s"].mean() < 0.5
        assert sd["d_s2o"].mean() < 0.5

    def test_offset_sdfs_nonzero(self):
        N = 30
        r = N / 3
        sdf_ours = self._make_sphere_sdf(N, [N / 2] * 3, r)
        sdf_sim = self._make_sphere_sdf(N, [N / 2 + 3, N / 2, N / 2], r)
        affine = np.eye(4)
        ctx = _make_ctx(np.zeros((N, N, N), dtype=np.uint8), sdf=sdf_ours)
        ctx._headers = {
            "mat": nib.Nifti1Image(ctx._mat, affine),
            "sdf": nib.Nifti1Image(ctx._sdf, affine),
            "brain": nib.Nifti1Image(np.zeros((N, N, N), dtype=np.uint8),
                                     affine),
        }
        ctx._cache["simnibs_resampled"] = {
            "sdf": sdf_sim,
            "labels": np.zeros((N, N, N), dtype=np.int16),
            "inner_boundary": np.zeros((N, N, N), dtype=bool),
        }
        sd = _compute_surface_distances(ctx)
        assert sd["d_o2s"].mean() > 0.5
        assert sd["d_s2o"].mean() > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Ground Truth (G1–G3)
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckG1:
    def test_small_assd_passes(self):
        ctx = _make_ctx(np.zeros((10, 10, 10), dtype=np.uint8))
        ctx._cache["surface_distances"] = {
            "vox_ours": np.array([[5, 5, 5]]),
            "phys_ours": np.array([[5.0, 5.0, 5.0]]),
            "vox_sim": np.array([[5, 5, 5]]),
            "phys_sim": np.array([[5.0, 5.0, 5.0]]),
            "d_o2s": np.array([0.5, 0.3, 0.2]),
            "d_s2o": np.array([0.4, 0.6]),
        }
        check_g1(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_large_assd_fails(self):
        ctx = _make_ctx(np.zeros((10, 10, 10), dtype=np.uint8))
        ctx._cache["surface_distances"] = {
            "vox_ours": np.array([[5, 5, 5]]),
            "phys_ours": np.array([[5.0, 5.0, 5.0]]),
            "vox_sim": np.array([[5, 5, 5]]),
            "phys_sim": np.array([[5.0, 5.0, 5.0]]),
            "d_o2s": np.array([3.0, 4.0, 5.0]),
            "d_s2o": np.array([2.5, 3.5]),
        }
        check_g1(ctx)
        assert ctx.results[-1]["status"] == "WARN"


class TestCheckG2:
    def test_regional_reported(self):
        ctx = _make_ctx(np.zeros((10, 10, 10), dtype=np.uint8))
        ctx._cache["surface_distances"] = {
            "vox_ours": np.array([[5, 5, 2], [5, 5, 5], [5, 5, 8]]),
            "phys_ours": np.array([[5., 5., 2.], [5., 5., 5.], [5., 5., 8.]]),
            "vox_sim": np.array([[5, 5, 5]]),
            "phys_sim": np.array([[5.0, 5.0, 5.0]]),
            "d_o2s": np.array([1.0, 0.5, 1.5]),
            "d_s2o": np.array([0.8]),
        }
        check_g2(ctx)
        assert ctx.results[-1]["status"] == "PASS"


class TestCheckG3:
    def test_per_direction_reported(self):
        ctx = _make_ctx(np.zeros((10, 10, 10), dtype=np.uint8))
        ctx._cache["surface_distances"] = {
            "vox_ours": np.array([[5, 5, 5]]),
            "phys_ours": np.array([[5.0, 5.0, 5.0]]),
            "vox_sim": np.array([[5, 5, 5]]),
            "phys_sim": np.array([[5.0, 5.0, 5.0]]),
            "d_o2s": np.array([1.0, 0.5]),
            "d_s2o": np.array([0.8, 1.2]),
        }
        check_g3(ctx)
        assert ctx.results[-1]["status"] == "PASS"
        assert "gt_surface_hausdorff_mm" in ctx.metrics


# ═══════════════════════════════════════════════════════════════════════════
# Header Consistency (H1–H10)
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckH1:
    def test_matching_passes(self):
        ctx = _make_header_ctx(N=10, dx=1.0)
        check_h1(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_mismatched_fails(self):
        ctx = _make_header_ctx(N=10, dx=1.0)
        bad_affine = ctx._headers["sdf"].affine.copy()
        bad_affine[0, 3] += 1.0
        ctx._headers["sdf"] = nib.Nifti1Image(
            np.zeros((10, 10, 10), dtype=np.float32), bad_affine)
        check_h1(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckH2:
    def test_correct_passes(self):
        ctx = _make_header_ctx(N=10, dx=1.0)
        check_h2(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_wrong_fails(self):
        ctx = _make_header_ctx(N=10, dx=1.0)
        affine = ctx._headers["mat"].affine
        ctx._headers["mat"] = nib.Nifti1Image(
            np.zeros((8, 8, 8), dtype=np.uint8), affine)
        check_h2(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckH3:
    def test_uint8_passes(self):
        ctx = _make_header_ctx(N=10, dx=1.0, mat_dtype=np.uint8)
        check_h3(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_wrong_fails(self):
        ctx = _make_header_ctx(N=10, dx=1.0, mat_dtype=np.int16)
        check_h3(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckH4:
    def test_float32_passes(self):
        ctx = _make_header_ctx(N=10, dx=1.0, sdf_dtype=np.float32)
        check_h4(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_wrong_fails(self):
        ctx = _make_header_ctx(N=10, dx=1.0, sdf_dtype=np.float64)
        check_h4(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckH5:
    def test_float32_passes(self):
        fiber_img = _MockNiftiImg((145, 174, 145, 6), dtype=np.float32)
        ctx = _make_header_ctx(N=10, dx=1.0, fiber_img=fiber_img)
        check_h5(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_wrong_fails(self):
        fiber_img = _MockNiftiImg((145, 174, 145, 6), dtype=np.float64)
        ctx = _make_header_ctx(N=10, dx=1.0, fiber_img=fiber_img)
        check_h5(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckH6:
    def test_correct_passes(self):
        fiber_img = _MockNiftiImg((145, 174, 145, 6))
        ctx = _make_header_ctx(N=10, dx=1.0, fiber_img=fiber_img)
        check_h6(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_wrong_fails(self):
        fiber_img = _MockNiftiImg((10, 10, 10, 6))
        ctx = _make_header_ctx(N=10, dx=1.0, fiber_img=fiber_img)
        check_h6(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckH7:
    def test_matching_passes(self):
        ctx = _make_header_ctx(N=10, dx=2.0)
        check_h7(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_mismatched_fails(self):
        ctx = _make_header_ctx(N=10, dx=2.0)
        ctx._meta["dx_mm"] = 1.0
        check_h7(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckH8:
    def test_matching_passes(self):
        ctx = _make_header_ctx(N=10, dx=1.0)
        check_h8(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_mismatched_fails(self):
        ctx = _make_header_ctx(N=10, dx=1.0)
        ctx._meta["grid_size"] = 20
        check_h8(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckH9:
    def test_matching_passes(self):
        ctx = _make_header_ctx(N=10, dx=1.0)
        check_h9(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_mismatched_fails(self):
        ctx = _make_header_ctx(N=10, dx=1.0)
        ctx._meta["affine_grid_to_phys"] = np.eye(4).tolist()
        check_h9(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"


class TestCheckH10:
    def test_centered_passes(self):
        ctx = _make_header_ctx(N=10, dx=1.0)
        check_h10(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_off_center_fails(self):
        affine = np.diag([1.0, 1.0, 1.0, 1.0])  # no translation
        ctx = _make_header_ctx(N=10, dx=1.0, affine=affine)
        check_h10(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"

    def test_fiber_inside_passes(self):
        N = 10
        dx = 1.0
        mat_affine = np.diag([dx, dx, dx, 1.0])
        mat_affine[:3, 3] = -N * dx / 2.0
        # Fiber affine: ACPC (0,0,0) maps inside fiber volume
        fiber_data = np.ones((20, 20, 20, 6), dtype=np.float32)
        fiber_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        fiber_affine[:3, 3] = -10.0  # center at origin
        fiber_img = nib.Nifti1Image(fiber_data, fiber_affine)
        ctx = _make_header_ctx(N=N, dx=dx, fiber_img=fiber_img)
        check_h10(ctx)
        assert ctx.results[-1]["status"] == "PASS"

    def test_fiber_outside_fails(self):
        N = 10
        dx = 1.0
        # Fiber affine: large offset so ACPC (0,0,0) maps outside
        fiber_data = np.ones((5, 5, 5, 6), dtype=np.float32)
        fiber_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        fiber_affine[:3, 3] = 100.0  # far away
        fiber_img = nib.Nifti1Image(fiber_data, fiber_affine)
        ctx = _make_header_ctx(N=N, dx=dx, fiber_img=fiber_img)
        check_h10(ctx)
        assert ctx.results[-1]["status"] == "CRITICAL"
        assert "ACPC outside fiber" in ctx.results[-1]["value"]


# ═══════════════════════════════════════════════════════════════════════════
# C5 — CC boundary masking
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckC5BoundaryMasking:
    def test_cc_boundary_masking(self, tmp_path):
        N = 20
        mat = np.zeros((N, N, N), dtype=np.uint8)
        sdf = -np.ones((N, N, N), dtype=np.float32)
        mid_x = N // 2
        cc_z_sup = 10  # CC extends up to z=10

        # Fissure voxels: CSF at midline above and below cc_z_sup
        mat[mid_x, 5:15, 3:8] = 8    # below CC — should be excluded
        mat[mid_x, 5:15, 12:18] = 8  # above CC — should be counted
        # Dural covering above CC
        mat[mid_x, 5:15, 12:18] = 10

        ctx = _make_ctx(mat, sdf=sdf)
        # Simulate fs path existence
        ctx.paths["fs"] = tmp_path / "fs.nii.gz"
        ctx.paths["fs"].touch()
        # Pre-populate cache with cc_z_sup
        ctx._cache["fs_data"] = {"cc_z_sup": cc_z_sup}

        check_c5(ctx)
        # With dural covering above CC, should pass
        assert ctx.results[-1]["status"] == "PASS"
