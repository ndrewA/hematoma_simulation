"""Tests for preprocessing/validation/ — decorator, resolve, report, record, dispatch."""

import nibabel as nib
import numpy as np
import pytest
from pathlib import Path

from preprocessing.validation import (
    CheckContext,
    build_report,
    resolve_checks,
    run_checks,
)
from preprocessing.validation.checks import (
    CheckDef,
    _REGISTRY,
    _REGISTRY_BY_ID,
)


# ---------------------------------------------------------------------------
# Shared helper — same pattern as test_validation_checks.py
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
# Check decorator & registry
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckDecorator:
    def test_registry_populated(self):
        assert len(_REGISTRY) > 0

    def test_registry_by_id_matches(self):
        for defn in _REGISTRY:
            assert defn.check_id in _REGISTRY_BY_ID
            assert _REGISTRY_BY_ID[defn.check_id] is defn

    def test_check_ids_unique(self):
        ids = [d.check_id for d in _REGISTRY]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"


# ═══════════════════════════════════════════════════════════════════════════
# resolve_checks
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveChecks:
    def test_none_returns_none(self):
        assert resolve_checks(None) is None

    def test_single_check_id(self):
        result = resolve_checks("D4")
        assert result == {"D4"}

    def test_phase_expands(self):
        result = resolve_checks("header")
        # Should contain all H* checks
        h_checks = {d.check_id for d in _REGISTRY if d.phase == "header"}
        assert result == h_checks
        assert len(result) > 0

    def test_unknown_token_returns_none(self):
        result = resolve_checks("nonexistent_xyz")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# build_report
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildReport:
    def _result(self, check_id, status, severity="WARN"):
        return {
            "id": check_id,
            "severity": severity,
            "description": f"test {check_id}",
            "status": status,
            "value": None,
        }

    def test_overall_pass(self):
        results = [
            self._result("D1", "PASS"),
            self._result("D2", "PASS"),
        ]
        report = build_report("test", "debug", 128, 2.0, results, {}, {})
        assert report["overall_status"] == "PASS"

    def test_overall_warn(self):
        results = [
            self._result("D1", "PASS"),
            self._result("D2", "WARN"),
        ]
        report = build_report("test", "debug", 128, 2.0, results, {}, {})
        assert report["overall_status"] == "WARN"

    def test_overall_fail(self):
        results = [
            self._result("D1", "PASS"),
            self._result("H1", "CRITICAL", severity="CRITICAL"),
        ]
        report = build_report("test", "debug", 128, 2.0, results, {}, {})
        assert report["overall_status"] == "FAIL"


# ═══════════════════════════════════════════════════════════════════════════
# CheckContext.record
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckContextRecord:
    def test_pass_recorded(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        ctx.record("H1", True)
        assert len(ctx.results) == 1
        assert ctx.results[0]["status"] == "PASS"
        assert ctx.results[0]["id"] == "H1"

    def test_fail_critical(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        ctx.record("H1", False)
        # H1 severity is CRITICAL
        assert ctx.results[0]["status"] == "CRITICAL"

    def test_fail_info_stays_info(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        ctx.record("V6", False)
        # V6 severity is INFO — failed INFO stays INFO
        assert ctx.results[0]["status"] == "INFO"


# ═══════════════════════════════════════════════════════════════════════════
# run_checks
# ═══════════════════════════════════════════════════════════════════════════

class TestRunChecks:
    def _make_valid_ctx(self):
        """Build a ctx that passes header + domain checks with in-memory data."""
        N = 10
        dx = 1.0
        affine = np.eye(4) * dx
        affine[3, 3] = 1.0
        affine[:3, 3] = -N / 2 * dx  # center at origin

        mat_data = np.zeros((N, N, N), dtype=np.uint8)
        mat_data[2:8, 2:8, 2:8] = 1
        sdf_data = np.ones((N, N, N), dtype=np.float32)
        sdf_data[2:8, 2:8, 2:8] = -1.0
        brain_data = np.zeros((N, N, N), dtype=np.uint8)
        brain_data[2:8, 2:8, 2:8] = 1

        ctx = _make_ctx(mat_data, sdf=sdf_data, dx=dx)
        ctx._brain = brain_data

        # Pre-populate headers with in-memory NIfTI images so header checks work
        ctx._headers = {
            "mat": nib.Nifti1Image(mat_data, affine),
            "sdf": nib.Nifti1Image(sdf_data, affine),
            "brain": nib.Nifti1Image(brain_data, affine),
        }
        ctx._meta = {
            "dx_mm": dx,
            "grid_size": N,
            "affine_grid_to_phys": affine.tolist(),
        }
        return ctx

    def test_selected_filters(self):
        ctx = self._make_valid_ctx()
        run_checks(ctx, selected={"D1"}, no_dural=True, no_fiber=True,
                   no_ground_truth=True)
        result_ids = {r["id"] for r in ctx.results}
        assert "D1" in result_ids
        # Should not contain unselected checks
        assert "D2" not in result_ids
        assert "M1" not in result_ids

    def test_skip_no_dural(self):
        ctx = self._make_valid_ctx()
        run_checks(ctx, selected=None, no_dural=True, no_fiber=True,
                   no_ground_truth=True)
        result_ids = {r["id"] for r in ctx.results}
        # Dural-phase checks should not be present
        dural_ids = {d.check_id for d in _REGISTRY if d.phase == "dural"}
        assert result_ids.isdisjoint(dural_ids)
        # But non-dural phases should be present
        assert any(r["id"].startswith("D") for r in ctx.results)

    def test_skip_no_simnibs(self):
        ctx = self._make_valid_ctx()
        ctx.has_simnibs = False
        run_checks(ctx, selected=None, no_dural=True, no_fiber=True,
                   no_ground_truth=False)
        # G* checks should be auto-skipped with "skipped (no SimNIBS)"
        gt_results = [r for r in ctx.results
                      if r["id"].startswith("G")]
        assert len(gt_results) > 0
        for r in gt_results:
            assert r["value"] == "skipped (no SimNIBS)"
