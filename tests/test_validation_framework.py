"""Tests for preprocessing/validation/ — decorator, resolve, report, record, dispatch."""

import nibabel as nib
import numpy as np
import pytest

from preprocessing.validation import (
    CheckContext,
    _parse_figure_set,
    build_report,
    resolve_checks,
    run_checks,
)
from preprocessing.validation.checks import (
    CheckDef,
    _REGISTRY,
    _REGISTRY_BY_ID,
)
from tests.conftest import _make_ctx


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


# ═══════════════════════════════════════════════════════════════════════════
# resolve_checks — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveChecksEdgeCases:
    def test_mixed_valid_and_invalid(self, capsys):
        result = resolve_checks("D4,nonexistent,header")
        assert "D4" in result
        h_checks = {d.check_id for d in _REGISTRY if d.phase == "header"}
        assert h_checks.issubset(result)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_multiple_check_ids(self):
        result = resolve_checks("D1,D2,M1")
        assert result == {"D1", "D2", "M1"}

    def test_whitespace_stripped(self):
        result = resolve_checks("  D4 , header ")
        assert "D4" in result
        h_checks = {d.check_id for d in _REGISTRY if d.phase == "header"}
        assert h_checks.issubset(result)


# ═══════════════════════════════════════════════════════════════════════════
# build_report — structure
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildReportStructure:
    def _result(self, check_id, status, severity="WARN"):
        return {
            "id": check_id,
            "severity": severity,
            "description": f"test {check_id}",
            "status": status,
            "value": None,
        }

    def test_empty_results(self):
        report = build_report("s", "p", 128, 2.0, [], {}, {})
        assert report["overall_status"] == "PASS"
        assert report["checks"] == {}
        assert len(report["figures"]) == 6

    def test_checks_dict_structure(self):
        results = [
            self._result("D1", "PASS"),
            self._result("D2", "WARN"),
        ]
        report = build_report("s", "p", 128, 2.0, results, {}, {})
        assert "D1" in report["checks"]
        assert "D2" in report["checks"]
        assert report["checks"]["D1"]["status"] == "PASS"
        assert report["checks"]["D2"]["status"] == "WARN"

    def test_census_metrics_preserved(self):
        census = {"0": {"name": "vacuum", "voxels": 100}}
        metrics = {"brain_mL": 1200.0}
        report = build_report("s", "p", 128, 2.0, [], census, metrics)
        assert report["volume_census"] == census
        assert report["key_metrics"] == metrics

    def test_has_timestamp(self):
        report = build_report("s", "p", 128, 2.0, [], {}, {})
        assert "timestamp" in report
        # ISO 8601 format contains 'T' separator
        assert "T" in report["timestamp"]

    def test_metadata_fields(self):
        report = build_report("sub1", "prod", 512, 0.5, [], {}, {})
        assert report["subject_id"] == "sub1"
        assert report["profile"] == "prod"
        assert report["grid_size"] == 512
        assert report["dx_mm"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# _parse_figure_set
# ═══════════════════════════════════════════════════════════════════════════

class TestParseFigureSet:
    def test_none_returns_none(self):
        assert _parse_figure_set(None) is None

    def test_all_returns_none(self):
        assert _parse_figure_set("all") is None

    def test_single(self):
        assert _parse_figure_set("3") == {3}

    def test_comma_separated(self):
        assert _parse_figure_set("1,3") == {1, 3}

    def test_whitespace(self):
        assert _parse_figure_set(" 1 , 3 ") == {1, 3}


# ═══════════════════════════════════════════════════════════════════════════
# run_checks — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestRunChecksEdgeCases:
    def _make_valid_ctx(self, fiber_img=None):
        N = 10
        dx = 1.0
        affine = np.eye(4) * dx
        affine[3, 3] = 1.0
        affine[:3, 3] = -N / 2 * dx

        mat_data = np.zeros((N, N, N), dtype=np.uint8)
        mat_data[2:8, 2:8, 2:8] = 1
        sdf_data = np.ones((N, N, N), dtype=np.float32)
        sdf_data[2:8, 2:8, 2:8] = -1.0
        brain_data = np.zeros((N, N, N), dtype=np.uint8)
        brain_data[2:8, 2:8, 2:8] = 1

        ctx = _make_ctx(mat_data, sdf=sdf_data, dx=dx)
        ctx._brain = brain_data
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
        ctx._fiber_img = fiber_img
        return ctx

    def test_header_critical_aborts_early(self):
        N = 10
        dx = 1.0
        bad_affine = np.diag([dx, dx, dx, 1.0])  # translation [0,0,0] — off center
        mat_data = np.zeros((N, N, N), dtype=np.uint8)
        mat_data[2:8, 2:8, 2:8] = 1
        sdf_data = np.ones((N, N, N), dtype=np.float32)
        sdf_data[2:8, 2:8, 2:8] = -1.0
        brain_data = np.zeros((N, N, N), dtype=np.uint8)
        brain_data[2:8, 2:8, 2:8] = 1

        ctx = _make_ctx(mat_data, sdf=sdf_data, dx=dx)
        ctx._brain = brain_data
        ctx._headers = {
            "mat": nib.Nifti1Image(mat_data, bad_affine),
            "sdf": nib.Nifti1Image(sdf_data, bad_affine),
            "brain": nib.Nifti1Image(brain_data, bad_affine),
        }
        ctx._meta = {
            "dx_mm": dx,
            "grid_size": N,
            "affine_grid_to_phys": bad_affine.tolist(),
        }
        run_checks(ctx, selected=None, no_dural=True, no_fiber=True,
                   no_ground_truth=True)
        result_ids = {r["id"] for r in ctx.results}
        # H10 should fail as CRITICAL
        assert any(r["id"] == "H10" and r["status"] == "CRITICAL"
                   for r in ctx.results)
        # No non-header checks should have run
        non_header = {r["id"] for r in ctx.results
                      if not r["id"].startswith("H")}
        assert len(non_header) == 0

    def test_fiber_skip_when_no_fiber_img(self):
        ctx = self._make_valid_ctx(fiber_img=None)
        run_checks(ctx, selected=None, no_dural=True, no_fiber=False,
                   no_ground_truth=True)
        fiber_header_results = [r for r in ctx.results
                                if r["id"] in ("H5", "H6")]
        for r in fiber_header_results:
            assert r["value"] == "skipped (no fiber)"

    def test_no_fiber_flag_skips_fiber_phase(self):
        ctx = self._make_valid_ctx()
        run_checks(ctx, selected=None, no_dural=True, no_fiber=True,
                   no_ground_truth=True)
        result_ids = {r["id"] for r in ctx.results}
        fiber_ids = {d.check_id for d in _REGISTRY if d.phase == "fiber"}
        assert result_ids.isdisjoint(fiber_ids)

    def test_no_ground_truth_flag(self):
        ctx = self._make_valid_ctx()
        ctx.has_simnibs = True
        run_checks(ctx, selected=None, no_dural=True, no_fiber=True,
                   no_ground_truth=True)
        result_ids = {r["id"] for r in ctx.results}
        gt_ids = {d.check_id for d in _REGISTRY if d.phase == "ground_truth"}
        assert result_ids.isdisjoint(gt_ids)


# ═══════════════════════════════════════════════════════════════════════════
# CheckContext.record — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckContextRecordEdgeCases:
    def test_verbose_prints(self, capsys):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        ctx.verbose = True
        ctx.record("H1", True, value="ok")
        captured = capsys.readouterr()
        assert "H1" in captured.out
        assert "PASS" in captured.out
        assert "ok" in captured.out

    def test_value_stored(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        ctx.record("H1", True, value="42 violations")
        assert ctx.results[-1]["value"] == "42 violations"

    def test_warn_severity(self):
        mat = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = _make_ctx(mat)
        ctx.record("D4", False)
        assert ctx.results[-1]["status"] == "WARN"
