"""Tests for preprocessing/subarachnoid_csf.py — fringe recovery, CSF fill, domain closure."""

import numpy as np
import pytest

from preprocessing.subarachnoid_csf import (
    fill_subarachnoid_csf,
    check_domain_closure,
    recover_fringe_tissue,
)


# ---------------------------------------------------------------------------
# fill_subarachnoid_csf
# ---------------------------------------------------------------------------
class TestFillSubarachnoidCsf:
    def test_sulcal_csf(self):
        # Vacuum inside brain → sulcal CSF
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = np.full((10, 10, 10), 10.0, dtype=np.float32)  # outside skull
        brain = np.zeros((10, 10, 10), dtype=np.uint8)

        # Small brain region with vacuum
        brain[4:6, 4:6, 4:6] = 1
        sdf[3:7, 3:7, 3:7] = -5.0  # inside skull

        n_sulcal, n_shell, sulcal, shell = fill_subarachnoid_csf(mat, sdf, brain)

        assert n_sulcal == 8  # 2x2x2 brain region
        assert np.all(mat[brain == 1] == 8)

    def test_shell_csf(self):
        # Inside skull, outside brain, vacuum → shell CSF
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = np.full((10, 10, 10), 10.0, dtype=np.float32)
        brain = np.zeros((10, 10, 10), dtype=np.uint8)

        # Skull interior larger than brain
        sdf[2:8, 2:8, 2:8] = -5.0
        brain[4:6, 4:6, 4:6] = 1
        # Give brain voxels some tissue label so they don't count as vacuum
        mat[brain == 1] = 2

        n_sulcal, n_shell, sulcal, shell = fill_subarachnoid_csf(mat, sdf, brain)

        # Shell = inside skull & outside brain & vacuum
        # Shell = 6^3 skull interior minus 2^3 brain = 208
        assert n_shell == 208
        assert n_sulcal == 0  # brain voxels already labeled

    def test_does_not_overwrite_tissue(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = np.full((10, 10, 10), -5.0, dtype=np.float32)
        brain = np.zeros((10, 10, 10), dtype=np.uint8)

        # Label some voxels as WM
        mat[4:6, 4:6, 4:6] = 1
        brain[3:7, 3:7, 3:7] = 1

        fill_subarachnoid_csf(mat, sdf, brain)

        # WM voxels should still be WM
        assert np.all(mat[4:6, 4:6, 4:6] == 1)

    def test_modifies_in_place(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = np.full((10, 10, 10), -5.0, dtype=np.float32)
        brain = np.ones((10, 10, 10), dtype=np.uint8)

        mat_id = id(mat)
        fill_subarachnoid_csf(mat, sdf, brain)
        assert id(mat) == mat_id  # same object
        assert np.any(mat == 8)   # data was actually modified


# ---------------------------------------------------------------------------
# check_domain_closure
# ---------------------------------------------------------------------------
class TestCheckDomainClosure:
    def test_passes_when_no_vacuum_inside(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8) * 8  # all CSF
        sdf = np.full((10, 10, 10), -5.0, dtype=np.float32)
        # Should not raise or exit
        check_domain_closure(mat, sdf)

    def test_passes_vacuum_outside_skull(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = np.full((10, 10, 10), 5.0, dtype=np.float32)  # all outside skull
        check_domain_closure(mat, sdf)

    def test_fails_on_vacuum_inside(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        sdf = np.full((10, 10, 10), -5.0, dtype=np.float32)  # inside skull
        with pytest.raises(ValueError):
            check_domain_closure(mat, sdf)


# ---------------------------------------------------------------------------
# recover_fringe_tissue
# ---------------------------------------------------------------------------
class TestRecoverFringeTissue:
    def test_recovers_near_boundary_voxels(self):
        size = 20
        mat = np.zeros((size, size, size), dtype=np.uint8)
        brain = np.zeros((size, size, size), dtype=np.uint8)

        # Brain region
        brain[5:15, 5:15, 5:15] = 1
        # Tissue in brain interior
        mat[6:14, 6:14, 6:14] = 2  # cortical GM

        # Fringe: brain=1, mat=0, near tissue AND near brain boundary
        # The layer at brain[5,:,:] etc. satisfies both criteria

        n_recovered = recover_fringe_tissue(mat, brain, dx_mm=1.0, max_dist_mm=2.0)
        assert n_recovered > 0
        # Recovered voxels should be assigned cortical GM (class 2)
        fringe = (brain == 1) & (mat == 2)
        assert np.count_nonzero(fringe) >= n_recovered

    def test_no_recovery_when_all_labeled(self):
        mat = np.ones((10, 10, 10), dtype=np.uint8) * 2
        brain = np.ones((10, 10, 10), dtype=np.uint8)
        n_recovered = recover_fringe_tissue(mat, brain, dx_mm=1.0)
        assert n_recovered == 0

    def test_interior_sulcal_not_recovered(self):
        # Vacuum deep inside the brain (far from boundary) should NOT be recovered
        size = 30
        mat = np.zeros((size, size, size), dtype=np.uint8)
        brain = np.zeros((size, size, size), dtype=np.uint8)

        brain[3:27, 3:27, 3:27] = 1
        mat[4:26, 4:26, 4:26] = 2  # tissue

        # Create a vacuum pocket deep inside (far from brain boundary)
        mat[13:17, 13:17, 13:17] = 0  # sulcal vacuum

        n_recovered = recover_fringe_tissue(mat, brain, dx_mm=1.0, max_dist_mm=1.0)

        # The deep vacuum should remain unlabeled
        assert np.all(mat[14:16, 14:16, 14:16] == 0)

    def test_effective_dist_at_least_dx(self):
        # At coarse grid (dx=5mm), max_dist_mm=1.0 should be elevated to dx
        size = 10
        mat = np.zeros((size, size, size), dtype=np.uint8)
        brain = np.zeros((size, size, size), dtype=np.uint8)

        brain[2:8, 2:8, 2:8] = 1
        mat[3:7, 3:7, 3:7] = 2

        n_recovered = recover_fringe_tissue(mat, brain, dx_mm=5.0, max_dist_mm=1.0)
        # With dx=5mm, effective_dist=5mm, so fringe layer should be recovered
        assert n_recovered > 0
