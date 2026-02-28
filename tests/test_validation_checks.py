"""Tests for preprocessing/validation/checks.py — helpers + complex check functions."""

import numpy as np
import pytest
from pathlib import Path
from scipy.ndimage import distance_transform_edt

from preprocessing.validation.checks import (
    _adjacent_labels,
    _classify_dural,
    _compute_volume_census,
    _find_landmark,
    _principal_direction,
    check_c4,
    check_d4,
    check_d5,
)
from preprocessing.validation import CheckContext


# ---------------------------------------------------------------------------
# Shared helper — build a CheckContext with pre-loaded arrays
# ---------------------------------------------------------------------------

_NOFILE = Path("/tmp/_validation_test_nonexistent")


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
