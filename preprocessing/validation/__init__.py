"""Cross-cutting validation and diagnostic visualization for all preprocessing outputs.

Final preprocessing step — quality gate before the solver runs.  Validates
header consistency, domain closure, material integrity, volume sanity,
dural compartmentalization, and fiber coverage across all outputs.

Produces:
  - validation_report.json  (machine-readable)
  - fig1-fig4 PNGs          (diagnostic figures)
  - launch_fsleyes.sh       (convenience viewer script)

Usage:
    python -m preprocessing.validation --subject 157336 --profile debug --verbose
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from preprocessing.utils import PROFILES, processed_dir, raw_dir


# ---------------------------------------------------------------------------
# Phase display names (for console summary)
# ---------------------------------------------------------------------------
PHASE_DISPLAY = {
    "header":      "Header Consistency",
    "domain":      "Domain Closure",
    "material":    "Material Integrity",
    "volume":      "Volume Sanity",
    "compartment": "Compartmentalization",
    "dural":       "Dural Membrane",
    "fiber":       "Fiber Coverage",
}


# ---------------------------------------------------------------------------
# CheckContext — holds lazily-loaded data and cached intermediates
# ---------------------------------------------------------------------------
class CheckContext:
    def __init__(self, paths, N, dx, subject, profile, verbose):
        self.paths = paths
        self.N = N
        self.dx = dx
        self.subject = subject
        self.profile = profile
        self.verbose = verbose
        self._mat = self._sdf = self._brain = self._meta = None
        self._headers = self._fiber_img = self._fiber_data = None
        self._cache = {}
        self.results = []
        self.census = {}
        self.metrics = {}

    # -- Lazy properties --

    @property
    def mat(self):
        if self._mat is None:
            self._mat = np.asarray(
                nib.load(str(self.paths["mat"])).dataobj, dtype=np.uint8)
        return self._mat

    @property
    def sdf(self):
        if self._sdf is None:
            self._sdf = np.asarray(
                nib.load(str(self.paths["sdf"])).dataobj, dtype=np.float32)
        return self._sdf

    @property
    def brain(self):
        if self._brain is None:
            self._brain = np.asarray(
                nib.load(str(self.paths["brain"])).dataobj, dtype=np.uint8)
        return self._brain

    @property
    def meta(self):
        if self._meta is None:
            with open(self.paths["meta"]) as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def headers(self):
        if self._headers is None:
            self._headers = {}
            for name in ("mat", "sdf", "brain", "fs"):
                self._headers[name] = nib.load(str(self.paths[name]))
        return self._headers

    @property
    def mat_affine(self):
        return self.headers["mat"].affine

    @property
    def fiber_img(self):
        if self._fiber_img is None:
            if self.paths["fiber"].exists():
                self._fiber_img = nib.load(str(self.paths["fiber"]))
            else:
                return None
        return self._fiber_img

    @property
    def fiber_data(self):
        if self._fiber_data is None:
            img = self.fiber_img
            if img is not None:
                self._fiber_data = img.get_fdata(dtype=np.float32)
            else:
                return None
        return self._fiber_data

    def get_cached(self, key, compute_fn):
        """Return cached intermediate, computing on first access."""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def record(self, check_id, passed, value=None):
        """Record check result.  Severity/description from registry."""
        from preprocessing.validation.checks import _REGISTRY_BY_ID

        defn = _REGISTRY_BY_ID[check_id]
        severity = defn.severity
        description = defn.description
        status = "PASS" if passed else severity
        if not passed and severity == "INFO":
            status = "INFO"

        entry = {
            "id": check_id,
            "severity": severity,
            "description": description,
            "status": status,
            "value": value,
        }
        self.results.append(entry)
        if self.verbose:
            val_str = f"  ({value})" if value is not None else ""
            print(f"  {check_id:4s} [{severity:8s}] "
                  f"{description:50s} {status}{val_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse CLI arguments for validation."""
    parser = argparse.ArgumentParser(
        description="Cross-cutting validation of all preprocessing outputs."
    )
    parser.add_argument("--subject", required=True, help="HCP subject ID")
    parser.add_argument(
        "--profile", required=True, choices=list(PROFILES.keys()),
        help="Simulation profile",
    )
    parser.add_argument("--no-images", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--no-fiber", action="store_true",
                        help="Skip fiber checks (Phase 4)")
    parser.add_argument("--no-dural", action="store_true",
                        help="Skip dural checks (Phase 3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-check detail")
    parser.add_argument(
        "--only", type=str, default=None,
        help="Run only specified checks/phases (comma-separated). "
             "Accepts check IDs (C5,C6) or phase names "
             "(header,domain,material,volume,compartment,dural,fiber). "
             "Implies --no-images unless overridden.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Resolve --only
# ---------------------------------------------------------------------------
def resolve_checks(only_arg):
    """Parse ``--only`` value into a set of check IDs, or *None* for all."""
    from preprocessing.validation.checks import _REGISTRY

    if only_arg is None:
        return None

    # Build phase -> IDs from registry
    phase_checks: dict[str, set[str]] = {}
    for defn in _REGISTRY:
        phase_checks.setdefault(defn.phase, set()).add(defn.check_id)

    selected = set()
    for token in only_arg.split(","):
        token = token.strip()
        lower = token.lower()
        upper = token.upper()
        if lower in phase_checks:
            selected.update(phase_checks[lower])
        elif upper in {d.check_id for d in _REGISTRY}:
            selected.add(upper)
        else:
            print(f"WARNING: unknown check or phase '{token}'")
    return selected or None


# ---------------------------------------------------------------------------
# Path builder
# ---------------------------------------------------------------------------
def _build_paths(subject, profile):
    """Return dict of all input/output paths."""
    out = processed_dir(subject, profile)
    raw = raw_dir(subject)
    fiber_dir = out.parent

    val_dir = out / "validation"

    return {
        "mat":     out / "material_map.nii.gz",
        "sdf":     out / "skull_sdf.nii.gz",
        "brain":   out / "brain_mask.nii.gz",
        "fs":      out / "fs_labels_resampled.nii.gz",
        "meta":    out / "grid_meta.json",
        "fiber":   fiber_dir / "fiber_M0.nii.gz",
        "t1w":     raw / "T1w_acpc_dc_restore_brain.nii.gz",
        "val_dir": val_dir,
        "report":  val_dir / "validation_report.json",
        "fsleyes": val_dir / "launch_fsleyes.sh",
        "fig1":    val_dir / "fig1_material_map.png",
        "fig2":    val_dir / "fig2_dural_detail.png",
        "fig3":    val_dir / "fig3_skull_sdf.png",
        "fig4":    val_dir / "fig4_fiber_dec.png",
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_checks(ctx, selected, no_dural, no_fiber):
    """Execute selected checks via the registry."""
    from preprocessing.validation.checks import _REGISTRY

    # Header checks first (cheap, gate everything else)
    for defn in _REGISTRY:
        if defn.phase != "header":
            continue
        if selected is not None and defn.check_id not in selected:
            continue
        if "fiber_img" in defn.needs and ctx.fiber_img is None:
            ctx.record(defn.check_id, True, value="skipped (no fiber)")
            continue
        defn.func(ctx)

    # Abort on critical header failures (before loading volumes)
    if any(r["severity"] == "CRITICAL" and r["status"] != "PASS"
           for r in ctx.results):
        return

    # All other checks
    for defn in _REGISTRY:
        if defn.phase == "header":
            continue
        if selected is not None and defn.check_id not in selected:
            continue
        if no_dural and defn.phase == "dural":
            continue
        if no_fiber and defn.phase == "fiber":
            continue
        if "fiber_img" in defn.needs and ctx.fiber_img is None:
            ctx.record(defn.check_id, True, value="skipped (no fiber)")
            continue
        if "fs" in defn.needs and not ctx.paths["fs"].exists():
            ctx.record(defn.check_id, True, value="skipped (no fs_labels)")
            continue
        defn.func(ctx)


# ---------------------------------------------------------------------------
# Report + FSLeyes script + Console summary
# ---------------------------------------------------------------------------
def build_report(subject, profile, N, dx, all_results, census, metrics):
    """Build the validation_report.json dict."""
    has_critical = any(r["status"] == "CRITICAL" for r in all_results)
    has_warn = any(r["status"] == "WARN" for r in all_results)
    if has_critical:
        overall = "FAIL"
    elif has_warn:
        overall = "WARN"
    else:
        overall = "PASS"

    checks = {}
    for r in all_results:
        checks[r["id"]] = {
            "severity": r["severity"],
            "description": r["description"],
            "status": r["status"],
            "value": r["value"],
        }

    figures = [
        "fig1_material_map.png",
        "fig2_dural_detail.png",
        "fig3_skull_sdf.png",
        "fig4_fiber_dec.png",
    ]

    return {
        "subject_id": subject,
        "profile": profile,
        "grid_size": N,
        "dx_mm": dx,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "checks": checks,
        "volume_census": census,
        "key_metrics": metrics,
        "figures": figures,
    }


def write_fsleyes_script(path, val_dir):
    """Write launch_fsleyes.sh convenience script."""
    script = '''#!/bin/bash
# Launch FSLeyes with all preprocessing outputs overlaid
DIR="$(dirname "$0")/.."
fsleyes \\
  "$DIR/skull_sdf.nii.gz" -cm blue-lightblue -dr -30 30 \\
  "$DIR/material_map.nii.gz" -cm random -ot label \\
  "$DIR/brain_mask.nii.gz" -cm green -a 30 \\
  "$DIR/fs_labels_resampled.nii.gz" -cm random -ot label -a 0 &
'''
    path.write_text(script)
    os.chmod(str(path), 0o755)
    print(f"  Saved {path}")


def print_console_summary(all_results, census, metrics, subject, profile, N, dx):
    """Print human-readable summary to console."""
    from preprocessing.validation.checks import _REGISTRY

    print()
    print("=" * 63)
    print(f"  Validation Report: {subject} / {profile} ({N}\u00b3, {dx} mm)")
    print("=" * 63)

    # Build phase -> ordered check IDs from registry
    phase_order = []
    phase_ids: dict[str, list[str]] = {}
    for defn in _REGISTRY:
        if defn.phase not in phase_ids:
            phase_order.append(defn.phase)
            phase_ids[defn.phase] = []
        phase_ids[defn.phase].append(defn.check_id)

    results_by_id = {r["id"]: r for r in all_results}

    for phase in phase_order:
        check_ids = phase_ids[phase]
        phase_results = [results_by_id[cid] for cid in check_ids
                         if cid in results_by_id]
        if not phase_results:
            continue
        phase_name = PHASE_DISPLAY.get(phase, phase)
        print(f"\n  {phase_name}")
        print(f"  {'─' * 59}")
        for r in phase_results:
            desc = r["description"][:42]
            status = r["status"]
            val = ""
            if r["value"] is not None:
                val = f"  ({str(r['value'])[:30]})"
            dots = "." * max(1, 47 - len(desc))
            print(f"  {r['id']:4s} {desc} {dots} {status}{val}")

    # Census
    if census:
        print(f"\n  Volume Census")
        print(f"  {'─' * 59}")
        for uid in range(12):
            entry = census.get(str(uid), {})
            name = entry.get("name", "???")
            voxels = entry.get("voxels", 0)
            vol = entry.get("volume_mL", 0)
            print(f"  u8={uid:<3d} {name:<24s} {voxels:>12,} vox  {vol:>10.1f} mL")

    # Overall
    n_pass = sum(1 for r in all_results if r["status"] == "PASS")
    n_warn = sum(1 for r in all_results if r["status"] == "WARN")
    n_fail = sum(1 for r in all_results if r["status"] in ("CRITICAL", "FAIL"))
    n_total = len(all_results)

    if n_fail > 0:
        overall = "FAIL"
    elif n_warn > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    print()
    print("=" * 63)
    print(f"  OVERALL: {overall}  ({n_pass}/{n_total} passed, "
          f"{n_warn} warnings, {n_fail} failures)")
    print("=" * 63)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    """Orchestrate cross-cutting validation."""
    t_total = time.monotonic()
    args = parse_args(argv)
    subject = args.subject
    profile = args.profile
    N, dx = PROFILES[profile]
    no_fiber = args.no_fiber
    no_dural = args.no_dural
    verbose = args.verbose

    selected = resolve_checks(args.only)
    no_images = args.no_images or (selected is not None)

    print(f"Subject: {subject}")
    print(f"Profile: {profile}  (N={N}, dx={dx} mm)")
    if selected is not None:
        print(f"Checks: {', '.join(sorted(selected))}")
    print(f"Flags: no_images={no_images}, no_fiber={no_fiber}, no_dural={no_dural}")
    print()

    paths = _build_paths(subject, profile)

    if not paths["mat"].exists():
        print(f"FATAL: material_map not found: {paths['mat']}")
        sys.exit(1)

    if not no_fiber and not paths["fiber"].exists():
        print(f"WARNING: fiber_M0 not found, skipping fiber checks")
        no_fiber = True

    paths["val_dir"].mkdir(parents=True, exist_ok=True)

    ctx = CheckContext(paths, N, dx, subject, profile, verbose)
    run_checks(ctx, selected, no_dural, no_fiber)

    # Abort early on critical header failures
    critical_fails = [r for r in ctx.results
                      if r["severity"] == "CRITICAL" and r["status"] != "PASS"]
    if critical_fails and all(r["id"].startswith("H") for r in critical_fails):
        # Only header criticals — abort was already handled in run_checks
        if not any(not r["id"].startswith("H") for r in ctx.results):
            print(f"\nCRITICAL: {len(critical_fails)} header checks failed — aborting")
            report = build_report(subject, profile, N, dx,
                                  ctx.results, ctx.census, ctx.metrics)
            with open(paths["report"], "w") as f:
                json.dump(report, f, indent=2)
            print_console_summary(ctx.results, ctx.census, ctx.metrics,
                                  subject, profile, N, dx)
            sys.exit(1)

    # Ensure domain closure metric
    ctx.metrics.setdefault("domain_closure_violations", 0)
    for r in ctx.results:
        if r["id"] == "D1" and r["status"] != "PASS":
            ctx.metrics["domain_closure_violations"] = r["value"]

    # Figures
    if not no_images:
        from preprocessing.validation.figures import generate_all_figures
        generate_all_figures(ctx)

    # Write JSON report
    report = build_report(subject, profile, N, dx,
                          ctx.results, ctx.census, ctx.metrics)
    with open(paths["report"], "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved {paths['report']}")

    # Write fsleyes script
    write_fsleyes_script(paths["fsleyes"], paths["val_dir"])

    # Console summary
    print_console_summary(ctx.results, ctx.census, ctx.metrics,
                          subject, profile, N, dx)

    print(f"\n  Total wall time: {time.monotonic() - t_total:.1f}s")
