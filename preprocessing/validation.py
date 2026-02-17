"""Cross-cutting validation and diagnostic visualization for all preprocessing outputs.

Final preprocessing step — quality gate before the solver runs.  Validates
header consistency, domain closure, material integrity, volume sanity,
dural compartmentalization, and fiber coverage across all outputs.

Produces:
  - validation_report.json  (machine-readable)
  - fig1–fig5 PNGs          (diagnostic figures)
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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.table import Table
import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    label as cc_label,
    map_coordinates,
)

from preprocessing.utils import (
    PROFILES,
    build_grid_affine,
    processed_dir,
    raw_dir,
    resample_to_grid,
)
from preprocessing.material_map import CLASS_NAMES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECK_DEFS = {
    # Phase 1: Header Consistency
    "H1":  ("CRITICAL", "Bitwise-identical affines across grid NIfTIs"),
    "H2":  ("CRITICAL", "Shape matches grid_meta grid_size"),
    "H3":  ("CRITICAL", "material_map dtype is uint8"),
    "H4":  ("CRITICAL", "skull_sdf dtype is float32"),
    "H5":  ("CRITICAL", "fiber_M0 dtype is float32"),
    "H6":  ("CRITICAL", "fiber_M0 shape matches expected dimensions"),
    "H7":  ("CRITICAL", "grid_meta dx_mm matches NIfTI affine diagonal"),
    "H8":  ("CRITICAL", "grid_meta grid_size matches NIfTI shape"),
    "H9":  ("CRITICAL", "grid_meta affine matches NIfTI sform"),
    "H10": ("CRITICAL", "ACPC origin round-trip through grid and fiber affines"),
    # Phase 2: Domain Closure
    "D1":  ("CRITICAL", "Zero vacuum inside skull"),
    "D2":  ("CRITICAL", "Zero tissue outside skull"),
    "D3":  ("CRITICAL", "Brain containment in skull"),
    "D4":  ("WARN",     "No isolated vacuum islands inside skull"),
    "D5":  ("WARN",     "SDF Eikonal property: gradient magnitude near 1.0"),
    # Phase 2: Material Integrity
    "M1":  ("CRITICAL", "All values in {0..11}"),
    "M2":  ("CRITICAL", "Does not contain 255"),
    "M3":  ("WARN",     "All expected classes {1..11} present"),
    "M4":  ("WARN",     "Brain mask consistency"),
    # Phase 2: Volume Sanity
    "V1":  ("WARN",     "Brain parenchyma volume"),
    "V2":  ("WARN",     "Ventricular CSF volume"),
    "V3":  ("WARN",     "Subarachnoid CSF volume"),
    "V4":  ("WARN",     "Dural membrane volume"),
    "V5":  ("WARN",     "ICV vs nonzero-material consistency"),
    "V6":  ("INFO",     "Complete volume census"),
    # Phase 2: Compartmentalization
    "C1":  ("INFO",     "Active domain connected components"),
    # Phase 3: Dural Membrane
    "C2":  ("WARN",     "Falx largest component > 90%"),
    "C3":  ("WARN",     "Tentorium largest component > 90%"),
    "C4":  ("WARN",     "Tentorial notch patency"),
    # Phase 4: Fiber Texture
    "F1":  ("WARN",     "WM forward-transform coverage >= 90%"),
    "F2":  ("CRITICAL", "M_0 diagonal elements >= 0 (PSD)"),
    "F3":  ("WARN",     "Trace <= 1.0 bound"),
    "F4":  ("WARN",     "CC landmark X-dominant"),
    "F5":  ("WARN",     "IC landmark Z-dominant"),
    "F6":  ("CRITICAL", "Fiber affine round-trip (copy of H10)"),
}

# Material colormap: u8 index -> RGBA
MATERIAL_COLORS = [
    (0.0, 0.0, 0.0, 0.0),    # 0: Vacuum (transparent)
    (1.0, 1.0, 1.0, 1.0),    # 1: Cerebral WM (white)
    (0.3, 0.3, 0.3, 1.0),    # 2: Cortical GM (dark gray)
    (1.0, 1.0, 0.0, 1.0),    # 3: Deep GM (yellow)
    (1.0, 1.0, 0.94, 1.0),   # 4: Cerebellar WM (ivory)
    (0.5, 0.5, 0.0, 1.0),    # 5: Cerebellar Cortex (olive)
    (1.0, 0.65, 0.0, 1.0),   # 6: Brainstem (orange)
    (0.0, 0.0, 1.0, 1.0),    # 7: Ventricular CSF (blue)
    (0.0, 1.0, 1.0, 1.0),    # 8: Subarachnoid CSF (cyan)
    (1.0, 0.0, 1.0, 1.0),    # 9: Choroid Plexus (magenta)
    (1.0, 0.0, 0.0, 1.0),    # 10: Dural Membrane (red)
    (0.0, 1.0, 0.0, 1.0),    # 11: Vessel / Venous Sinus (green)
]
_MAT_CMAP = ListedColormap(MATERIAL_COLORS)
_MAT_NORM = BoundaryNorm(np.arange(-0.5, 12.5, 1.0), _MAT_CMAP.N)


# ---------------------------------------------------------------------------
# Helper: record check result
# ---------------------------------------------------------------------------
def record_check(results, check_id, passed, value=None, verbose=False):
    """Append a check result to the results list."""
    severity, description = CHECK_DEFS[check_id]
    status = "PASS" if passed else severity  # CRITICAL, WARN, or INFO on fail
    if not passed and severity == "INFO":
        status = "INFO"  # INFO checks don't fail
    entry = {
        "id": check_id,
        "severity": severity,
        "description": description,
        "status": status,
        "value": value,
    }
    results.append(entry)
    if verbose:
        val_str = f"  ({value})" if value is not None else ""
        print(f"  {check_id:4s} [{severity:8s}] {description:50s} {status}{val_str}")


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
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Path builder
# ---------------------------------------------------------------------------
def _build_paths(subject, profile):
    """Return dict of all input/output paths."""
    out = processed_dir(subject, profile)
    raw = raw_dir(subject)
    fiber_dir = out.parent  # profile-independent: data/processed/{subject}/

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
        "fig5":    val_dir / "fig5_summary.png",
    }


# ---------------------------------------------------------------------------
# Phase 1: Header Consistency (H1-H10)
# ---------------------------------------------------------------------------
def run_header_checks(paths, N, dx, has_fiber, verbose):
    """Run header consistency checks H1-H10.

    Returns (results, header_info) where header_info contains data needed
    by later phases (affines, shapes, etc.).
    """
    results = []

    # Load headers only (no data)
    grid_files = {
        "mat":   paths["mat"],
        "sdf":   paths["sdf"],
        "brain": paths["brain"],
        "fs":    paths["fs"],
    }
    headers = {}
    for name, path in grid_files.items():
        img = nib.load(str(path))
        headers[name] = img

    # Load grid_meta
    with open(paths["meta"]) as f:
        meta = json.load(f)

    # Fiber header (optional)
    fiber_img = None
    if has_fiber and paths["fiber"].exists():
        fiber_img = nib.load(str(paths["fiber"]))

    # H1: Bitwise-identical affines
    mat_affine = headers["mat"].affine
    all_match = all(
        np.array_equal(mat_affine, headers[k].affine)
        for k in ("sdf", "brain", "fs")
    )
    record_check(results, "H1", all_match, verbose=verbose)

    # H2: Shape == (N, N, N)
    shapes_ok = all(headers[k].shape == (N, N, N) for k in grid_files)
    record_check(results, "H2", shapes_ok,
                 value=f"{N}x{N}x{N}" if shapes_ok else str({k: headers[k].shape for k in grid_files}),
                 verbose=verbose)

    # H3: mat dtype uint8
    record_check(results, "H3",
                 headers["mat"].get_data_dtype() == np.uint8,
                 verbose=verbose)

    # H4: sdf dtype float32
    record_check(results, "H4",
                 headers["sdf"].get_data_dtype() == np.float32,
                 verbose=verbose)

    # H5/H6: fiber checks
    if fiber_img is not None:
        record_check(results, "H5",
                     fiber_img.get_data_dtype() == np.float32,
                     verbose=verbose)
        record_check(results, "H6",
                     fiber_img.shape == (145, 174, 145, 6),
                     value=str(fiber_img.shape),
                     verbose=verbose)
    else:
        record_check(results, "H5", True, value="skipped (no fiber)", verbose=verbose)
        record_check(results, "H6", True, value="skipped (no fiber)", verbose=verbose)

    # H7: meta dx_mm matches affine diagonal
    meta_dx = float(meta["dx_mm"])
    record_check(results, "H7",
                 abs(meta_dx - float(mat_affine[0, 0])) < 1e-6,
                 value=f"meta={meta_dx}, affine={float(mat_affine[0,0])}",
                 verbose=verbose)

    # H8: meta grid_size matches shape
    record_check(results, "H8",
                 int(meta["grid_size"]) == headers["mat"].shape[0],
                 verbose=verbose)

    # H9: meta affine matches NIfTI sform
    meta_affine = np.array(meta["affine_grid_to_phys"])
    record_check(results, "H9",
                 np.allclose(meta_affine, mat_affine, atol=1e-6),
                 verbose=verbose)

    # H10: ACPC origin round-trip
    h10_pass = True
    h10_value = None
    phys_origin = np.array([0.0, 0.0, 0.0, 1.0])
    grid_vox = np.linalg.inv(mat_affine) @ phys_origin
    if not np.allclose(grid_vox[:3], N / 2, atol=0.5):
        h10_pass = False
        h10_value = f"grid_vox={grid_vox[:3]}, expected={N/2}"
    if fiber_img is not None and h10_pass:
        diff_vox = np.linalg.inv(fiber_img.affine) @ phys_origin
        for d in range(3):
            if not (0 <= diff_vox[d] < fiber_img.shape[d]):
                h10_pass = False
                h10_value = f"ACPC outside fiber on axis {d}: {diff_vox[d]:.1f}"
                break
    record_check(results, "H10", h10_pass, value=h10_value, verbose=verbose)

    header_info = {
        "mat_affine": mat_affine,
        "meta": meta,
        "fiber_img": fiber_img,
        "h10_pass": h10_pass,
    }
    return results, header_info


# ---------------------------------------------------------------------------
# Phase 2: Domain checks (D1-D5)
# ---------------------------------------------------------------------------
def run_domain_checks(mat, sdf, brain, meta, verbose):
    """Run domain closure checks D1-D5."""
    results = []
    N = mat.shape[0]
    dx = float(meta["dx_mm"])

    # D1: Zero vacuum inside skull
    n_vac_inside = int(np.count_nonzero((sdf < 0) & (mat == 0)))
    record_check(results, "D1", n_vac_inside == 0,
                 value=n_vac_inside, verbose=verbose)

    # D2: Zero tissue outside skull
    tissue_lut = np.zeros(256, dtype=bool)
    tissue_lut[1:12] = True
    n_tissue_outside = int(np.count_nonzero((sdf > 0) & tissue_lut[mat]))
    record_check(results, "D2", n_tissue_outside == 0,
                 value=n_tissue_outside, verbose=verbose)

    # D3: Brain containment
    brain_outside = int(np.count_nonzero((brain == 1) & (sdf >= 0)))
    record_check(results, "D3", brain_outside == 0,
                 value=brain_outside, verbose=verbose)

    # D4: Vacuum islands inside skull
    vacuum = (mat == 0)
    vacuum_labels, n_comp = cc_label(vacuum)
    sizes = np.bincount(vacuum_labels.ravel())
    main_label = int(sizes[1:].argmax()) + 1 if n_comp > 0 else 0
    n_islands = 0
    island_voxels = 0
    if n_comp > 1:
        for label_id in range(1, n_comp + 1):
            if label_id == main_label:
                continue
            component = (vacuum_labels == label_id)
            if np.all(sdf[component] < 0):
                n_islands += 1
                island_voxels += int(component.sum())
    del vacuum_labels
    record_check(results, "D4", n_islands == 0,
                 value=f"{n_islands} islands, {island_voxels} voxels",
                 verbose=verbose)

    # D5: SDF Eikonal property (sampled gradient magnitude)
    rng = np.random.default_rng(42)
    interior = np.argwhere(sdf < -1.0)
    n_sample = min(100_000, len(interior))
    if n_sample > 0:
        sample = interior[rng.choice(len(interior), size=n_sample, replace=False)]
        grad_mag_sq = np.zeros(n_sample)
        for axis in range(3):
            fwd = sample.copy()
            fwd[:, axis] = np.clip(fwd[:, axis] + 1, 0, N - 1)
            bwd = sample.copy()
            bwd[:, axis] = np.clip(bwd[:, axis] - 1, 0, N - 1)
            diff = (sdf[fwd[:, 0], fwd[:, 1], fwd[:, 2]]
                    - sdf[bwd[:, 0], bwd[:, 1], bwd[:, 2]])
            grad_mag_sq += (diff / (2 * dx)) ** 2
        grad_mag = np.sqrt(grad_mag_sq)
        p5, p95 = float(np.percentile(grad_mag, 5)), float(np.percentile(grad_mag, 95))
        d5_pass = (p5 >= 0.8) and (p95 <= 1.2)
        record_check(results, "D5", d5_pass,
                     value=f"p5={p5:.3f}, p95={p95:.3f}",
                     verbose=verbose)
    else:
        record_check(results, "D5", True,
                     value="skipped (no interior voxels)", verbose=verbose)

    return results


# ---------------------------------------------------------------------------
# Phase 2: Material checks (M1-M4)
# ---------------------------------------------------------------------------
def run_material_checks(mat, brain, verbose):
    """Run material integrity checks M1-M4."""
    results = []
    unique_vals = set(int(v) for v in np.unique(mat))

    # M1: All values in {0..11}
    valid_range = set(range(12))
    out_of_range = unique_vals - valid_range
    record_check(results, "M1", len(out_of_range) == 0,
                 value=sorted(out_of_range) if out_of_range else None,
                 verbose=verbose)

    # M2: No 255
    record_check(results, "M2", 255 not in unique_vals, verbose=verbose)

    # M3: All classes 1-11 present
    missing = set(range(1, 12)) - unique_vals
    record_check(results, "M3", len(missing) == 0,
                 value=f"missing: {sorted(missing)}" if missing else "all present",
                 verbose=verbose)

    # M4: Brain mask consistency — every brain voxel has tissue/fluid
    brain_vacuum = int(np.count_nonzero((brain == 1) & (mat == 0)))
    record_check(results, "M4", brain_vacuum == 0,
                 value=brain_vacuum, verbose=verbose)

    return results


# ---------------------------------------------------------------------------
# Phase 2: Volume checks (V1-V6)
# ---------------------------------------------------------------------------
def run_volume_checks(mat, sdf, meta, verbose):
    """Run volume sanity checks V1-V6. Returns (results, census, metrics)."""
    results = []
    dx = float(meta["dx_mm"])
    vol_voxel_ml = dx ** 3 / 1000.0

    # Build census
    census = {}
    counts = np.bincount(mat.ravel(), minlength=12)
    for uid in range(12):
        n = int(counts[uid])
        vol = round(n * vol_voxel_ml, 2)
        census[str(uid)] = {
            "name": CLASS_NAMES.get(uid, "???"),
            "voxels": n,
            "volume_mL": vol,
        }

    # Compute volumes
    parenchyma_ids = [1, 2, 3, 4, 5, 6, 9]
    parenchyma_ml = sum(int(counts[i]) for i in parenchyma_ids) * vol_voxel_ml
    ventricular_ml = int(counts[7]) * vol_voxel_ml
    subarachnoid_ml = int(counts[8]) * vol_voxel_ml
    dural_ml = int(counts[10]) * vol_voxel_ml
    n_sdf_neg = int(np.count_nonzero(sdf < 0))
    icv_ml = n_sdf_neg * vol_voxel_ml
    n_nonzero_mat = int(np.count_nonzero(mat > 0))

    # V1: Brain parenchyma [800, 2000]
    record_check(results, "V1", 800 <= parenchyma_ml <= 2000,
                 value=f"{parenchyma_ml:.1f} mL", verbose=verbose)

    # V2: Ventricular CSF [10, 60]
    record_check(results, "V2", 10 <= ventricular_ml <= 60,
                 value=f"{ventricular_ml:.1f} mL", verbose=verbose)

    # V3: Subarachnoid CSF [100, 500]
    record_check(results, "V3", 100 <= subarachnoid_ml <= 500,
                 value=f"{subarachnoid_ml:.1f} mL", verbose=verbose)

    # V4: Dural membrane [2, 50]
    record_check(results, "V4", 2 <= dural_ml <= 50,
                 value=f"{dural_ml:.1f} mL", verbose=verbose)

    # V5: ICV vs nonzero-material consistency
    if n_sdf_neg > 0:
        frac_diff = abs(n_sdf_neg - n_nonzero_mat) / n_sdf_neg
        record_check(results, "V5", frac_diff < 0.02,
                     value=f"{frac_diff:.4f} ({n_sdf_neg} vs {n_nonzero_mat})",
                     verbose=verbose)
    else:
        record_check(results, "V5", False,
                     value="no SDF<0 voxels", verbose=verbose)

    # V6: Census (INFO only)
    record_check(results, "V6", True, value="see census", verbose=verbose)

    metrics = {
        "brain_parenchyma_mL": round(parenchyma_ml, 1),
        "intracranial_volume_mL": round(icv_ml, 1),
        "ventricular_csf_mL": round(ventricular_ml, 1),
        "subarachnoid_csf_mL": round(subarachnoid_ml, 1),
        "dural_membrane_mL": round(dural_ml, 1),
    }

    return results, census, metrics


# ---------------------------------------------------------------------------
# Phase 2: Active domain (C1)
# ---------------------------------------------------------------------------
def run_active_domain_check(mat, verbose):
    """Run active domain connected components check C1. Returns (results, n_components)."""
    results = []

    active = (mat > 0)
    labels, n = cc_label(active)
    sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1] if n > 0 else []
    del labels

    top_str = ", ".join(str(int(s)) for s in sizes[:3]) if len(sizes) > 0 else "none"
    record_check(results, "C1", True,
                 value=f"{n} components (top: {top_str})",
                 verbose=verbose)

    return results, n


# ---------------------------------------------------------------------------
# Phase 3: Dural checks (C2-C4)
# ---------------------------------------------------------------------------
def run_dural_checks(mat, verbose):
    """Run dural membrane checks C2-C4.

    Returns (results, falx_n_comp, tent_n_comp).
    """
    results = []
    N = mat.shape[0]
    dural = (mat == 10)
    n_dural = int(np.count_nonzero(dural))

    if n_dural == 0:
        record_check(results, "C2", True,
                     value="skipped (0 dural voxels)", verbose=verbose)
        record_check(results, "C3", True,
                     value="skipped (0 dural voxels)", verbose=verbose)
        record_check(results, "C4", True,
                     value="skipped (0 dural voxels)", verbose=verbose)
        return results, 0, 0

    mid_x = N // 2

    # C2: Falx — dural voxels near midline (±5 voxels of x = N/2)
    falx_region = dural.copy()
    falx_region[:mid_x - 5, :, :] = False
    falx_region[mid_x + 5:, :, :] = False
    n_falx = int(np.count_nonzero(falx_region))
    falx_n_comp = 0
    if n_falx > 0:
        falx_labels, falx_n_comp = cc_label(falx_region)
        falx_sizes = np.sort(np.bincount(falx_labels.ravel())[1:])[::-1]
        frac_largest = float(falx_sizes[0]) / float(falx_sizes.sum())
        record_check(results, "C2", frac_largest > 0.90,
                     value=f"{falx_n_comp} comp, largest={frac_largest:.1%}",
                     verbose=verbose)
        del falx_labels
    else:
        record_check(results, "C2", True,
                     value="skipped (0 falx voxels)", verbose=verbose)

    # C3: Tentorium — dural voxels away from midline
    tent_region = dural & ~falx_region
    n_tent = int(np.count_nonzero(tent_region))
    tent_n_comp = 0
    if n_tent > 0:
        tent_labels, tent_n_comp = cc_label(tent_region)
        tent_sizes = np.sort(np.bincount(tent_labels.ravel())[1:])[::-1]
        frac_largest = float(tent_sizes[0]) / float(tent_sizes.sum())
        record_check(results, "C3", frac_largest > 0.90,
                     value=f"{tent_n_comp} comp, largest={frac_largest:.1%}",
                     verbose=verbose)
        del tent_labels
    else:
        record_check(results, "C3", True,
                     value="skipped (0 tentorial voxels)", verbose=verbose)
    del falx_region, tent_region

    # C4: Tentorial notch patency
    brainstem = (mat == 6)
    if not brainstem.any():
        record_check(results, "C4", True,
                     value="skipped (no brainstem)", verbose=verbose)
    else:
        z_indices = np.where(brainstem.any(axis=(0, 1)))[0]
        z_min, z_max = int(z_indices.min()), int(z_indices.max())
        z_upper = z_min + 2 * (z_max - z_min) // 3

        bs_slice = brainstem[:, :, z_upper]
        csf_slice = (mat[:, :, z_upper] == 8)
        bs_dilated = binary_dilation(bs_slice, iterations=1)
        csf_adjacent = csf_slice & bs_dilated & ~bs_slice
        n_adjacent = int(np.count_nonzero(csf_adjacent))
        record_check(results, "C4", n_adjacent > 0,
                     value=f"{n_adjacent} adjacent CSF voxels at z={z_upper}",
                     verbose=verbose)

    return results, falx_n_comp, tent_n_comp


# ---------------------------------------------------------------------------
# Phase 4: Fiber checks (F1-F6)
# ---------------------------------------------------------------------------
def run_fiber_checks(mat, fiber_img, mat_affine, h10_pass, verbose):
    """Run fiber texture checks F1-F6.

    Returns (results, metrics_update).
    """
    results = []
    metrics = {}

    fiber_data = fiber_img.get_fdata(dtype=np.float32)
    fiber_affine = fiber_img.affine

    # F1: Forward-transform coverage
    wm_lut = np.zeros(256, dtype=bool)
    wm_lut[[1, 4, 6]] = True
    wm_mask = wm_lut[mat]
    wm_indices = np.argwhere(wm_mask)

    rng = np.random.default_rng(42)
    n_sample = min(50_000, len(wm_indices))
    coverage_pct = 0.0
    if n_sample > 0:
        sample = wm_indices[rng.choice(len(wm_indices), size=n_sample, replace=False)]

        # Composite transform: grid voxel -> physical -> diffusion voxel
        phys_to_diff = np.linalg.inv(fiber_affine)
        grid_to_diff = phys_to_diff @ mat_affine

        grid_homo = np.column_stack([sample, np.ones(n_sample)])
        diff_coords = (grid_to_diff @ grid_homo.T).T[:, :3]

        trace_at_wm = np.zeros(n_sample)
        for ch in range(3):  # diagonal: M_00, M_11, M_22
            vals = map_coordinates(
                fiber_data[..., ch], diff_coords.T,
                order=1, mode="constant", cval=0.0,
            )
            trace_at_wm += vals

        coverage_pct = float((trace_at_wm > 0).sum()) / n_sample * 100
        record_check(results, "F1", coverage_pct >= 90.0,
                     value=f"{coverage_pct:.1f}%", verbose=verbose)

        # Compute trace stats for metrics
        nonzero_trace = trace_at_wm[trace_at_wm > 0]
        if len(nonzero_trace) > 0:
            metrics["fiber_trace_mean"] = round(float(np.mean(nonzero_trace)), 4)
            metrics["fiber_trace_p95"] = round(float(np.percentile(nonzero_trace, 95)), 4)
    else:
        record_check(results, "F1", False,
                     value="no WM voxels", verbose=verbose)

    metrics["fiber_wm_coverage_pct"] = round(coverage_pct, 1)

    # F2: PSD — diagonal elements >= 0
    f2_pass = True
    f2_detail = []
    for ch in range(3):
        diag = fiber_data[..., ch]
        n_negative = int(np.count_nonzero(diag < -1e-7))
        if n_negative > 0:
            f2_pass = False
            f2_detail.append(f"ch{ch}: {n_negative} neg")

    # Full eigenvalue check on 10k sample
    nonzero_mask = np.any(fiber_data != 0, axis=-1)
    nonzero_indices = np.argwhere(nonzero_mask)
    n_eig_sample = min(10_000, len(nonzero_indices))
    n_eig_fail = 0
    if n_eig_sample > 0:
        eig_sample = nonzero_indices[rng.choice(len(nonzero_indices), size=n_eig_sample, replace=False)]
        for idx in eig_sample:
            i, j, k = idx
            m = fiber_data[i, j, k]
            mat3 = np.array([
                [m[0], m[3], m[4]],
                [m[3], m[1], m[5]],
                [m[4], m[5], m[2]],
            ], dtype=np.float64)
            eigvals = np.linalg.eigvalsh(mat3)
            if np.any(eigvals < -1e-7):
                n_eig_fail += 1
                f2_pass = False

    if n_eig_fail > 0:
        f2_detail.append(f"eigen: {n_eig_fail}/{n_eig_sample} fail")

    record_check(results, "F2", f2_pass,
                 value="; ".join(f2_detail) if f2_detail else None,
                 verbose=verbose)

    # F3: Trace <= 1.0
    trace_vol = fiber_data[..., 0] + fiber_data[..., 1] + fiber_data[..., 2]
    n_over = int(np.count_nonzero(trace_vol > 1.0 + 1e-5))
    record_check(results, "F3", n_over == 0,
                 value=f"{n_over} voxels over", verbose=verbose)

    # F4: CC landmark X-dominant
    # Derive CC landmark from fiber data: find voxels where M_00/trace > 0.5
    # near the midline (x ~ shape/2), then take centroid
    cc_vox = _find_landmark(fiber_data, dominant_axis=0,
                            x_range=(fiber_data.shape[0] // 2 - 5,
                                     fiber_data.shape[0] // 2 + 5))
    if cc_vox is not None:
        m6 = fiber_data[cc_vox[0], cc_vox[1], cc_vox[2]]
        cc_dir = _principal_direction(m6)
        f4_pass = cc_dir is not None and abs(cc_dir[0]) > 0.7
        record_check(results, "F4", f4_pass,
                     value=f"vox={cc_vox}, dir={cc_dir}",
                     verbose=verbose)
    else:
        record_check(results, "F4", True,
                     value="skipped (no CC-like voxels found)",
                     verbose=verbose)

    # F5: IC landmark Z-dominant
    # Derive IC landmark from fiber data: find voxels where M_22/trace > 0.5
    # lateral to midline (x in ~40-65% of range)
    shape = fiber_data.shape[:3]
    ic_vox = _find_landmark(fiber_data, dominant_axis=2,
                            x_range=(int(shape[0] * 0.3),
                                     int(shape[0] * 0.45)))
    if ic_vox is not None:
        m6 = fiber_data[ic_vox[0], ic_vox[1], ic_vox[2]]
        ic_dir = _principal_direction(m6)
        f5_pass = ic_dir is not None and abs(ic_dir[2]) > 0.7
        record_check(results, "F5", f5_pass,
                     value=f"vox={ic_vox}, dir={ic_dir}",
                     verbose=verbose)
    else:
        record_check(results, "F5", True,
                     value="skipped (no IC-like voxels found)",
                     verbose=verbose)

    # F6: Copy of H10
    record_check(results, "F6", h10_pass,
                 value="copied from H10", verbose=verbose)

    return results, metrics, fiber_data


def _find_landmark(fiber_data, dominant_axis, x_range, frac_thresh=0.5,
                   trace_thresh=0.1):
    """Find a landmark voxel where the given axis dominates the tensor.

    Searches for voxels where M_diag[dominant_axis]/trace > frac_thresh,
    restricted to x_range=(lo, hi). Returns the voxel (i,j,k) with the
    highest dominance fraction, or None if none found.
    """
    trace = fiber_data[..., 0] + fiber_data[..., 1] + fiber_data[..., 2]
    nonzero = trace > trace_thresh
    diag = fiber_data[..., dominant_axis]
    frac = np.zeros_like(trace)
    frac[nonzero] = diag[nonzero] / trace[nonzero]
    candidates = (frac > frac_thresh) & nonzero
    # Restrict to x_range
    x_lo, x_hi = x_range
    mask = np.zeros_like(candidates)
    mask[x_lo:x_hi, :, :] = candidates[x_lo:x_hi, :, :]

    if not mask.any():
        return None
    # Pick the voxel with the highest dominance fraction
    frac_masked = np.where(mask, frac, 0.0)
    best = np.unravel_index(np.argmax(frac_masked), frac_masked.shape)
    return (int(best[0]), int(best[1]), int(best[2]))


def _principal_direction(m6):
    """Extract principal eigenvector from 6-component upper triangle.

    Returns 3-vector or None if tensor is all zeros.
    """
    if np.allclose(m6, 0):
        return None
    mat3 = np.array([
        [m6[0], m6[3], m6[4]],
        [m6[3], m6[1], m6[5]],
        [m6[4], m6[5], m6[2]],
    ], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(mat3)
    return eigvecs[:, -1]


# ---------------------------------------------------------------------------
# Figure 1: Material Map Triplanar
# ---------------------------------------------------------------------------
def generate_fig1(mat, t1w, N, subject, profile, dx, path):
    """Material map triplanar: 2 rows x 3 columns."""
    mid = N // 2

    slices_mat = [
        mat[mid, :, :],    # axial
        mat[:, mid, :],    # coronal
        mat[:, :, mid],    # sagittal
    ]
    slices_t1w = [
        t1w[mid, :, :],
        t1w[:, mid, :],
        t1w[:, :, mid],
    ] if t1w is not None else [None, None, None]

    titles = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Material Map — {subject} / {profile} ({N}³, {dx} mm)",
                 fontsize=14, fontweight="bold")

    for col in range(3):
        # Row 1: material map alone
        ax = axes[0, col]
        ax.imshow(slices_mat[col].T, origin="lower", cmap=_MAT_CMAP,
                  norm=_MAT_NORM, interpolation="nearest")
        ax.set_title(f"{titles[col]} — material")
        ax.axis("off")

        # Row 2: material map at 40% alpha over T1w
        ax = axes[1, col]
        if slices_t1w[col] is not None:
            ax.imshow(slices_t1w[col].T, origin="lower", cmap="gray",
                      interpolation="nearest")
        ax.imshow(slices_mat[col].T, origin="lower", cmap=_MAT_CMAP,
                  norm=_MAT_NORM, interpolation="nearest", alpha=0.4)
        ax.set_title(f"{titles[col]} — overlay on T1w")
        ax.axis("off")

    # Add legend
    _add_material_legend(fig)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _add_material_legend(fig):
    """Add material class legend to the right side of the figure."""
    handles = []
    for uid in range(12):
        color = MATERIAL_COLORS[uid]
        if uid == 0:
            continue  # skip vacuum
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color[:3],
                                     ec="black", linewidth=0.5))
    labels = [f"{uid}: {CLASS_NAMES.get(uid, '???')}" for uid in range(1, 12)]
    fig.legend(handles, labels, loc="center right", fontsize=8,
               title="Material Classes", title_fontsize=9)


# ---------------------------------------------------------------------------
# Figure 2: Dural Membrane Detail
# ---------------------------------------------------------------------------
def generate_fig2(mat, t1w, N, subject, profile, dx, path):
    """Dural membrane detail: 3x3 panels."""
    mid_x = N // 2
    dural = (mat == 10)

    if not dural.any():
        print(f"  Skipping fig2 (no dural voxels)")
        return

    # Find brainstem z-range for tentorial level
    brainstem = (mat == 6)
    if brainstem.any():
        z_indices = np.where(brainstem.any(axis=(0, 1)))[0]
        z_min, z_max = int(z_indices.min()), int(z_indices.max())
        z_tent = z_min + 2 * (z_max - z_min) // 3
    else:
        z_tent = N // 2

    # Find CC y-range for coronal slices
    # Use material u8=1 (WM) near midline as proxy
    wm_mid = (mat[mid_x - 2:mid_x + 2, :, :] == 1)
    if wm_mid.any():
        y_indices = np.where(wm_mid.any(axis=(0, 2)))[0]
        y_ant = int(y_indices.min()) + (int(y_indices.max()) - int(y_indices.min())) // 4
        y_mid = (int(y_indices.min()) + int(y_indices.max())) // 2
        y_post = int(y_indices.max()) - (int(y_indices.max()) - int(y_indices.min())) // 4
    else:
        y_ant, y_mid, y_post = N // 3, N // 2, 2 * N // 3

    # Find cerebellum center for posterior fossa
    cerebellar = (mat == 5)
    if cerebellar.any():
        cb_y = int(np.mean(np.where(cerebellar.any(axis=(0, 2)))[0]))
    else:
        cb_y = y_post

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle(f"Dural Membrane Detail — {subject} / {profile}",
                 fontsize=14, fontweight="bold")

    def _show_dural_panel(ax, mat_slice, t1w_slice, title):
        """Show a panel with dural highlighted in magenta, others at 20% on T1w."""
        if t1w_slice is not None:
            ax.imshow(t1w_slice.T, origin="lower", cmap="gray",
                      interpolation="nearest")

        # Non-dural material at 20% opacity
        mat_display = mat_slice.copy().astype(float)
        mat_display[mat_slice == 10] = np.nan  # mask dural for separate layer
        ax.imshow(mat_display.T, origin="lower", cmap=_MAT_CMAP,
                  norm=_MAT_NORM, interpolation="nearest", alpha=0.2)

        # Dural voxels in magenta at full opacity
        dural_mask = (mat_slice == 10)
        if dural_mask.any():
            dural_overlay = np.zeros(mat_slice.shape + (4,))
            dural_overlay[dural_mask, :] = [1.0, 0.0, 1.0, 1.0]
            ax.imshow(np.transpose(dural_overlay, (1, 0, 2)),
                      origin="lower", interpolation="nearest")

        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Row 1
    _show_dural_panel(axes[0, 0], mat[mid_x, :, :],
                      t1w[mid_x, :, :] if t1w is not None else None,
                      f"Midsagittal x={mid_x}")

    # Binary falx mask
    falx_region = dural.copy()
    falx_region[:mid_x - 5, :, :] = False
    falx_region[mid_x + 5:, :, :] = False
    ax = axes[0, 1]
    ax.imshow(falx_region[mid_x, :, :].T, origin="lower", cmap="Reds",
              interpolation="nearest")
    ax.set_title(f"Falx mask, midsag x={mid_x}", fontsize=9)
    ax.axis("off")
    del falx_region

    _show_dural_panel(axes[0, 2], mat[:, y_mid, :],
                      t1w[:, y_mid, :] if t1w is not None else None,
                      f"Coronal y={y_mid} (mid CC)")

    # Row 2
    _show_dural_panel(axes[1, 0], mat[:, :, z_tent],
                      t1w[:, :, z_tent] if t1w is not None else None,
                      f"Axial z={z_tent} (tentorial)")

    # Zoomed notch region
    if brainstem.any():
        bs_x = int(np.mean(np.where(brainstem[:, :, z_tent].any(axis=1))[0]))
        bs_y = int(np.mean(np.where(brainstem[:, :, z_tent].any(axis=0))[0]))
        pad = 30
        x_lo, x_hi = max(0, bs_x - pad), min(N, bs_x + pad)
        y_lo, y_hi = max(0, bs_y - pad), min(N, bs_y + pad)
        _show_dural_panel(axes[1, 1], mat[x_lo:x_hi, y_lo:y_hi, z_tent],
                          t1w[x_lo:x_hi, y_lo:y_hi, z_tent] if t1w is not None else None,
                          f"Notch zoom z={z_tent}")
    else:
        axes[1, 1].set_title("No brainstem", fontsize=9)
        axes[1, 1].axis("off")

    _show_dural_panel(axes[1, 2], mat[:, cb_y, :],
                      t1w[:, cb_y, :] if t1w is not None else None,
                      f"Coronal y={cb_y} (post. fossa)")

    # Row 3: CC coronal slices
    for col_idx, (y_slice, label) in enumerate([
        (y_ant, "anterior CC"),
        (y_mid, "mid CC"),
        (y_post, "posterior CC"),
    ]):
        _show_dural_panel(axes[2, col_idx], mat[:, y_slice, :],
                          t1w[:, y_slice, :] if t1w is not None else None,
                          f"Coronal y={y_slice} ({label})")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Skull SDF + Domain Boundary
# ---------------------------------------------------------------------------
def generate_fig3(mat, sdf, t1w, brain, N, subject, profile, dx, path):
    """Skull SDF + domain boundary: 2x3 panels."""
    mid = N // 2

    # Row 1: SDF contours on T1w (axial, coronal, sagittal)
    contour_levels = [-20, -10, -5, 0, 5, 10]
    contour_colors = ["#0000FF", "#4444FF", "#8888FF", "#000000", "#FF8888", "#FF0000"]
    contour_lw = [0.5, 0.5, 0.5, 2.0, 0.5, 0.5]

    sdf_slices = [sdf[mid, :, :], sdf[:, mid, :], sdf[:, :, mid]]
    t1w_slices = ([t1w[mid, :, :], t1w[:, mid, :], t1w[:, :, mid]]
                  if t1w is not None else [None, None, None])
    brain_slices = [brain[mid, :, :], brain[:, mid, :], brain[:, :, mid]]
    titles = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Skull SDF + Domain Boundary — {subject} / {profile}",
                 fontsize=14, fontweight="bold")

    for col in range(3):
        ax = axes[0, col]
        if t1w_slices[col] is not None:
            ax.imshow(t1w_slices[col].T, origin="lower", cmap="gray",
                      interpolation="nearest")
        ax.contour(sdf_slices[col].T, levels=contour_levels,
                   colors=contour_colors, linewidths=contour_lw)
        ax.set_title(f"{titles[col]} — SDF contours", fontsize=9)
        ax.axis("off")

    # Row 2, panel 1: Filled SDF (RdBu_r, ±30)
    ax = axes[1, 0]
    im = ax.imshow(sdf[mid, :, :].T, origin="lower", cmap="RdBu_r",
                   vmin=-30, vmax=30, interpolation="nearest")
    ax.set_title("Axial — SDF colormap", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.7, label="SDF (mm)")

    # Row 2, panel 2: Brain mask boundary on SDF
    ax = axes[1, 1]
    ax.imshow(sdf[:, mid, :].T, origin="lower", cmap="RdBu_r",
              vmin=-30, vmax=30, interpolation="nearest")
    brain_dilated = binary_dilation(brain_slices[1])
    brain_boundary = brain_dilated & ~brain_slices[1].astype(bool)
    boundary_overlay = np.zeros(brain_boundary.shape + (4,))
    boundary_overlay[brain_boundary, :] = [0.0, 1.0, 0.0, 0.8]
    ax.imshow(np.transpose(boundary_overlay, (1, 0, 2)),
              origin="lower", interpolation="nearest")
    ax.set_title("Coronal — brain boundary on SDF", fontsize=9)
    ax.axis("off")

    # Row 2, panel 3: Sylvian fissure zoom
    ax = axes[1, 2]
    # Sylvian fissure: coronal near mid-y, lateral region
    pad = min(40, N // 4)
    x_lo = max(0, mid - pad)
    x_hi = min(N, mid + pad)
    z_lo = max(0, mid - pad)
    z_hi = min(N, mid + pad)
    sdf_zoom = sdf[x_lo:x_hi, mid, z_lo:z_hi]
    ax.imshow(sdf_zoom.T, origin="lower", cmap="RdBu_r",
              vmin=-30, vmax=30, interpolation="nearest")
    ax.contour(sdf_zoom.T, levels=[0], colors=["black"], linewidths=[2.0])
    ax.set_title("Sylvian zoom (coronal)", fontsize=9)
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Fiber DEC Map
# ---------------------------------------------------------------------------
def generate_fig4(fiber_data, subject, profile, path):
    """Direction-Encoded Color map at native fiber resolution: 2x2 + legend."""
    shape = fiber_data.shape[:3]  # (145, 174, 145)

    # Slice indices (approximate for subject 157336)
    slices_info = [
        ("Axial z=72 (CC level)", 2, 72),
        ("Coronal y=100", 1, 100),
        ("Sagittal x=72", 0, 72),
        ("Axial z=82 (corona radiata)", 2, 82),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                             gridspec_kw={"width_ratios": [1, 1, 0.3]})
    fig.suptitle(f"Fiber Orientation DEC Map — {subject} / {profile}",
                 fontsize=14, fontweight="bold")

    panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, ((title, axis, slice_idx), (row, col)) in enumerate(
            zip(slices_info, panel_positions)):
        ax = axes[row, col]

        if slice_idx >= shape[axis]:
            ax.set_title(f"{title} — OOB", fontsize=9)
            ax.axis("off")
            continue

        # Extract the 2D slice of 6-component tensor
        if axis == 0:
            tensor_slice = fiber_data[slice_idx, :, :, :]
        elif axis == 1:
            tensor_slice = fiber_data[:, slice_idx, :, :]
        else:
            tensor_slice = fiber_data[:, :, slice_idx, :]

        dec_rgb = _compute_dec_slice(tensor_slice)
        ax.imshow(np.transpose(dec_rgb, (1, 0, 2)), origin="lower",
                  interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Color sphere legend (right column)
    for row in range(2):
        axes[row, 2].axis("off")

    # Simple legend text
    ax_leg = axes[0, 2]
    ax_leg.text(0.5, 0.8, "DEC Convention", ha="center", va="top",
                fontsize=11, fontweight="bold", transform=ax_leg.transAxes)
    ax_leg.text(0.5, 0.6, "R = Left-Right (X)", ha="center", va="top",
                fontsize=10, color="red", transform=ax_leg.transAxes)
    ax_leg.text(0.5, 0.45, "G = Ant-Post (Y)", ha="center", va="top",
                fontsize=10, color="green", transform=ax_leg.transAxes)
    ax_leg.text(0.5, 0.3, "B = Sup-Inf (Z)", ha="center", va="top",
                fontsize=10, color="blue", transform=ax_leg.transAxes)
    ax_leg.text(0.5, 0.1, "Brightness = trace", ha="center", va="top",
                fontsize=9, transform=ax_leg.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _compute_dec_slice(tensor_slice):
    """Compute DEC RGB image from a 2D slice of 6-component tensors.

    tensor_slice: (W, H, 6)
    Returns: (W, H, 3) float32 RGB in [0, 1]
    """
    W, H = tensor_slice.shape[:2]
    trace = tensor_slice[..., 0] + tensor_slice[..., 1] + tensor_slice[..., 2]
    trace_max = trace.max()
    if trace_max == 0:
        return np.zeros((W, H, 3), dtype=np.float32)

    rgb = np.zeros((W, H, 3), dtype=np.float32)
    nonzero = trace > 0

    # Batch eigendecomposition per row for memory efficiency
    for i in range(W):
        nz_cols = np.where(nonzero[i])[0]
        if len(nz_cols) == 0:
            continue

        m = tensor_slice[i, nz_cols]  # (K, 6)
        # Build batch of 3x3 symmetric matrices
        mats = np.zeros((len(nz_cols), 3, 3), dtype=np.float64)
        mats[:, 0, 0] = m[:, 0]
        mats[:, 1, 1] = m[:, 1]
        mats[:, 2, 2] = m[:, 2]
        mats[:, 0, 1] = mats[:, 1, 0] = m[:, 3]
        mats[:, 0, 2] = mats[:, 2, 0] = m[:, 4]
        mats[:, 1, 2] = mats[:, 2, 1] = m[:, 5]

        eigvals, eigvecs = np.linalg.eigh(mats)
        principal = eigvecs[:, :, -1]  # (K, 3) — largest eigenvalue
        abs_principal = np.abs(principal)

        brightness = trace[i, nz_cols] / trace_max
        rgb[i, nz_cols] = abs_principal * brightness[:, np.newaxis]

    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Figure 5: Validation Summary
# ---------------------------------------------------------------------------
def generate_fig5(all_results, census, metrics, subject, profile, N, dx,
                  fig_paths, path):
    """Summary figure: check table + census + metrics + thumbnails."""
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1.5, 0.5, 1.5], hspace=0.3)

    # Section 1: Check results table
    ax1 = fig.add_subplot(gs[0])
    ax1.axis("off")
    ax1.set_title(f"Validation Summary — {subject} / {profile} ({N}³, {dx} mm)",
                  fontsize=14, fontweight="bold", pad=20)

    # Build table data
    col_labels = ["ID", "Severity", "Description", "Status", "Value"]
    cell_colors = []
    cell_text = []
    for r in all_results:
        status = r["status"]
        if status == "PASS":
            row_color = ["#d4edda"] * 5  # green
        elif status == "WARN":
            row_color = ["#fff3cd"] * 5  # yellow
        elif status in ("CRITICAL", "FAIL"):
            row_color = ["#f8d7da"] * 5  # red
        else:
            row_color = ["#e2e3e5"] * 5  # gray (INFO)
        cell_colors.append(row_color)
        val_str = str(r["value"])[:50] if r["value"] is not None else ""
        cell_text.append([r["id"], r["severity"], r["description"], status, val_str])

    if cell_text:
        table = ax1.table(cellText=cell_text, colLabels=col_labels,
                          cellColours=cell_colors, loc="center",
                          cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.2)

    # Section 2: Census table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title("Material Census", fontsize=12, fontweight="bold")

    census_labels = ["u8", "Class", "Voxels", "Volume (mL)"]
    census_text = []
    for uid in range(12):
        entry = census.get(str(uid), {})
        census_text.append([
            str(uid),
            entry.get("name", "???"),
            f"{entry.get('voxels', 0):,}",
            f"{entry.get('volume_mL', 0):.1f}",
        ])

    if census_text:
        table2 = ax2.table(cellText=census_text, colLabels=census_labels,
                           loc="center", cellLoc="left")
        table2.auto_set_font_size(False)
        table2.set_fontsize(8)
        table2.scale(1, 1.3)

    # Section 3: Key metrics
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    metric_str = "  |  ".join(f"{k}: {v}" for k, v in metrics.items())
    ax3.text(0.5, 0.5, metric_str, ha="center", va="center", fontsize=9,
             transform=ax3.transAxes, family="monospace")

    # Section 4: Thumbnails
    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")
    thumb_names = ["fig1", "fig2", "fig3", "fig4"]
    n_thumbs = 0
    for i, name in enumerate(thumb_names):
        fpath = fig_paths.get(name)
        if fpath is not None and Path(fpath).exists():
            try:
                thumb_img = plt.imread(str(fpath))
                inset = ax4.inset_axes([i * 0.25 + 0.02, 0.05, 0.22, 0.9])
                inset.imshow(thumb_img)
                inset.axis("off")
                inset.set_title(name, fontsize=8)
                n_thumbs += 1
            except Exception:
                pass

    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Report + FSLeyes script + Console summary
# ---------------------------------------------------------------------------
def build_report(subject, profile, N, dx, all_results, census, metrics):
    """Build the validation_report.json dict."""
    # Determine overall status
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
        "fig5_summary.png",
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
    print()
    print("=" * 63)
    print(f"  Validation Report: {subject} / {profile} ({N}\u00b3, {dx} mm)")
    print("=" * 63)

    # Group checks by phase
    phases = [
        ("Header Consistency", ["H1", "H2", "H3", "H4", "H5", "H6",
                                 "H7", "H8", "H9", "H10"]),
        ("Domain Closure",     ["D1", "D2", "D3", "D4", "D5"]),
        ("Material Integrity", ["M1", "M2", "M3", "M4"]),
        ("Volume Sanity",      ["V1", "V2", "V3", "V4", "V5", "V6"]),
        ("Compartmentalization", ["C1", "C2", "C3", "C4"]),
        ("Fiber Coverage",     ["F1", "F2", "F3", "F4", "F5", "F6"]),
    ]

    results_by_id = {r["id"]: r for r in all_results}

    for phase_name, check_ids in phases:
        print(f"\n  {phase_name}")
        print(f"  {'─' * 59}")
        for cid in check_ids:
            r = results_by_id.get(cid)
            if r is None:
                continue
            desc = r["description"][:42]
            status = r["status"]
            val = ""
            if r["value"] is not None:
                val = f"  ({str(r['value'])[:30]})"
            dots = "." * max(1, 47 - len(desc))
            print(f"  {cid:4s} {desc} {dots} {status}{val}")

    # Census
    dx_val = float(dx)
    vol_voxel_ml = dx_val ** 3 / 1000.0
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
    n_info = sum(1 for r in all_results if r["status"] == "INFO")
    n_total = len(all_results)

    has_critical = n_fail > 0
    has_warn = n_warn > 0
    if has_critical:
        overall = "FAIL"
    elif has_warn:
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
    no_images = args.no_images
    no_fiber = args.no_fiber
    no_dural = args.no_dural
    verbose = args.verbose

    print(f"Subject: {subject}")
    print(f"Profile: {profile}  (N={N}, dx={dx} mm)")
    print(f"Flags: no_images={no_images}, no_fiber={no_fiber}, no_dural={no_dural}")
    print()

    paths = _build_paths(subject, profile)

    # Check critical inputs exist
    if not paths["mat"].exists():
        print(f"FATAL: material_map not found: {paths['mat']}")
        sys.exit(1)

    # Auto-detect fiber availability
    has_fiber = (not no_fiber) and paths["fiber"].exists()
    if not no_fiber and not paths["fiber"].exists():
        print(f"WARNING: fiber_M0 not found, skipping fiber checks")
        has_fiber = False

    # Create output directory
    paths["val_dir"].mkdir(parents=True, exist_ok=True)

    all_results = []
    census = {}
    metrics = {}

    # ── Phase 1: Header Checks ──
    print("─── Phase 1: Header Consistency ───")
    t0 = time.monotonic()
    header_results, header_info = run_header_checks(
        paths, N, dx, has_fiber, verbose)
    all_results.extend(header_results)
    print(f"  Phase 1: {time.monotonic() - t0:.1f}s")

    # Abort on critical header failures
    critical_fails = [r for r in header_results
                      if r["severity"] == "CRITICAL" and r["status"] != "PASS"]
    if critical_fails:
        print(f"\nCRITICAL: {len(critical_fails)} header checks failed — aborting")
        # Write partial report
        report = build_report(subject, profile, N, dx, all_results, census, metrics)
        paths["report"].parent.mkdir(parents=True, exist_ok=True)
        with open(paths["report"], "w") as f:
            json.dump(report, f, indent=2)
        print_console_summary(all_results, census, metrics, subject, profile, N, dx)
        sys.exit(1)

    # ── Phase 2: Grid Volumes ──
    print("\n─── Phase 2: Grid Volumes ───")
    t0 = time.monotonic()

    print("  Loading grid volumes...")
    mat_img = nib.load(str(paths["mat"]))
    mat = np.asarray(mat_img.dataobj, dtype=np.uint8)
    sdf_img = nib.load(str(paths["sdf"]))
    sdf = np.asarray(sdf_img.dataobj, dtype=np.float32)
    brain_img = nib.load(str(paths["brain"]))
    brain = np.asarray(brain_img.dataobj, dtype=np.uint8)

    with open(paths["meta"]) as f:
        meta = json.load(f)

    # Domain checks (D1-D5)
    print("  Running domain checks...")
    domain_results = run_domain_checks(mat, sdf, brain, meta, verbose)
    all_results.extend(domain_results)

    # Material checks (M1-M4)
    print("  Running material checks...")
    material_results = run_material_checks(mat, brain, verbose)
    all_results.extend(material_results)

    # Volume checks (V1-V6)
    print("  Running volume checks...")
    volume_results, census, vol_metrics = run_volume_checks(mat, sdf, meta, verbose)
    all_results.extend(volume_results)
    metrics.update(vol_metrics)

    # Active domain (C1)
    print("  Running active domain check...")
    c1_results, n_active_comp = run_active_domain_check(mat, verbose)
    all_results.extend(c1_results)
    metrics["active_domain_components"] = n_active_comp

    # Figures 1, 3 (need T1w)
    t1w = None
    if not no_images:
        print("  Resampling T1w to grid...")
        t1w_path = paths["t1w"]
        if t1w_path.exists():
            grid_affine = header_info["mat_affine"]
            t1w = resample_to_grid(
                str(t1w_path), grid_affine, (N, N, N),
                order=1, cval=0.0, dtype=np.float32,
            )
        else:
            print(f"  WARNING: T1w not found, figures will lack underlay")

        print("  Generating Figure 1...")
        try:
            generate_fig1(mat, t1w, N, subject, profile, dx, paths["fig1"])
        except Exception as e:
            print(f"  WARNING: Figure 1 failed: {e}")

        print("  Generating Figure 3...")
        try:
            generate_fig3(mat, sdf, t1w, brain, N, subject, profile, dx, paths["fig3"])
        except Exception as e:
            print(f"  WARNING: Figure 3 failed: {e}")

    # Free SDF, T1w for Phase 3
    del sdf
    del t1w
    t1w = None

    print(f"  Phase 2: {time.monotonic() - t0:.1f}s")

    # ── Phase 3: Dural Membrane ──
    if not no_dural and int(np.count_nonzero(mat == 10)) > 0:
        print("\n─── Phase 3: Dural Membrane ───")
        t0 = time.monotonic()

        print("  Running dural checks...")
        dural_results, falx_n, tent_n = run_dural_checks(mat, verbose)
        all_results.extend(dural_results)
        metrics["falx_components"] = falx_n
        metrics["tentorium_components"] = tent_n

        # Figure 2 (needs T1w re-resample)
        if not no_images:
            print("  Re-resampling T1w for Figure 2...")
            t1w_path = paths["t1w"]
            if t1w_path.exists():
                grid_affine = header_info["mat_affine"]
                t1w = resample_to_grid(
                    str(t1w_path), grid_affine, (N, N, N),
                    order=1, cval=0.0, dtype=np.float32,
                )

            print("  Generating Figure 2...")
            try:
                generate_fig2(mat, t1w, N, subject, profile, dx, paths["fig2"])
            except Exception as e:
                print(f"  WARNING: Figure 2 failed: {e}")

            del t1w
            t1w = None

        print(f"  Phase 3: {time.monotonic() - t0:.1f}s")
    else:
        if no_dural:
            print("\n─── Phase 3: Skipped (--no-dural) ───")
        else:
            print("\n─── Phase 3: Skipped (no u8=10 voxels) ───")
        metrics["falx_components"] = 0
        metrics["tentorium_components"] = 0

    # Free brain
    del brain

    # ── Phase 4: Fiber Texture ──
    if has_fiber:
        print("\n─── Phase 4: Fiber Texture ───")
        t0 = time.monotonic()

        fiber_img = header_info["fiber_img"]
        print("  Running fiber checks...")
        fiber_results, fiber_metrics, fiber_data = run_fiber_checks(
            mat, fiber_img, header_info["mat_affine"],
            header_info["h10_pass"], verbose,
        )
        all_results.extend(fiber_results)
        metrics.update(fiber_metrics)

        # Figure 4
        if not no_images:
            print("  Generating Figure 4...")
            try:
                generate_fig4(fiber_data, subject, profile, paths["fig4"])
            except Exception as e:
                print(f"  WARNING: Figure 4 failed: {e}")

        del fiber_data
        print(f"  Phase 4: {time.monotonic() - t0:.1f}s")
    else:
        print("\n─── Phase 4: Skipped (no fiber) ───")
        metrics["fiber_wm_coverage_pct"] = 0.0

    # Free mat
    del mat

    # Ensure domain closure violations metric
    metrics["domain_closure_violations"] = 0
    for r in all_results:
        if r["id"] == "D1" and r["status"] != "PASS":
            metrics["domain_closure_violations"] = r["value"]

    # ── Phase 5: Summary ──
    print("\n─── Phase 5: Summary ───")
    t0 = time.monotonic()

    # Figure 5
    if not no_images:
        print("  Generating Figure 5...")
        fig_paths_dict = {
            "fig1": paths["fig1"],
            "fig2": paths["fig2"],
            "fig3": paths["fig3"],
            "fig4": paths["fig4"],
        }
        try:
            generate_fig5(all_results, census, metrics, subject, profile,
                          N, dx, fig_paths_dict, paths["fig5"])
        except Exception as e:
            print(f"  WARNING: Figure 5 failed: {e}")

    # Write JSON report
    report = build_report(subject, profile, N, dx, all_results, census, metrics)
    with open(paths["report"], "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved {paths['report']}")

    # Write fsleyes script
    write_fsleyes_script(paths["fsleyes"], paths["val_dir"])

    # Console summary
    print_console_summary(all_results, census, metrics, subject, profile, N, dx)

    print(f"\n  Phase 5: {time.monotonic() - t0:.1f}s")
    print(f"  Total wall time: {time.monotonic() - t_total:.1f}s")


if __name__ == "__main__":
    main()
