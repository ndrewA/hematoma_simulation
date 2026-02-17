"""Fill subarachnoid CSF in the material map.

Identifies vacuum voxels inside the skull (sulcal CSF within the brain mask,
and shell CSF between brain surface and skull) and paints them as u8=8
(Subarachnoid CSF).  Establishes the domain closure invariant: no vacuum
remains inside the skull after this step.
"""

import argparse
import json
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label

from preprocessing.utils import PROFILES, processed_dir
from preprocessing.material_map import CLASS_NAMES, print_census


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse CLI arguments for subarachnoid_csf."""
    parser = argparse.ArgumentParser(
        description="Fill subarachnoid CSF (sulcal + shell) in the material map."
    )
    parser.add_argument("--subject", required=True, help="HCP subject ID")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        help="Named profile (default: debug)",
    )
    group.add_argument("--dx", type=float, help="Grid spacing in mm (custom)")

    parser.add_argument(
        "--grid-size", type=int,
        help="Grid size N (required with --dx, ignored with --profile)",
    )

    args = parser.parse_args(argv)

    if args.profile is None and args.dx is None:
        args.profile = "debug"

    if args.profile is not None:
        args.N, args.dx = PROFILES[args.profile]
    else:
        if args.grid_size is None:
            parser.error("--grid-size is required when using --dx")
        args.N = args.grid_size
        args.profile = f"custom_{args.N}_{args.dx}"

    return args


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_inputs(out_dir):
    """Load material map, skull SDF, brain mask, and grid metadata.

    Returns (mat, sdf, brain, affine, dx_mm).
    """
    mat_path = out_dir / "material_map.nii.gz"
    sdf_path = out_dir / "skull_sdf.nii.gz"
    brain_path = out_dir / "brain_mask.nii.gz"
    meta_path = out_dir / "grid_meta.json"

    print(f"Loading {mat_path}")
    mat_img = nib.load(str(mat_path))
    mat = np.asarray(mat_img.dataobj, dtype=np.uint8)
    affine = mat_img.affine.copy()

    print(f"Loading {sdf_path}")
    sdf_img = nib.load(str(sdf_path))
    sdf = np.asarray(sdf_img.dataobj, dtype=np.float32)

    print(f"Loading {brain_path}")
    brain_img = nib.load(str(brain_path))
    brain = np.asarray(brain_img.dataobj, dtype=np.uint8)

    print(f"Loading {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    dx_mm = float(meta["dx_mm"])

    assert mat.shape == sdf.shape == brain.shape, (
        f"Shape mismatch: mat={mat.shape}, sdf={sdf.shape}, brain={brain.shape}"
    )

    return mat, sdf, brain, affine, dx_mm


def save_material_map(out_dir, mat, affine):
    """Overwrite material_map.nii.gz as uint8."""
    img = nib.Nifti1Image(mat, affine)
    img.header.set_data_dtype(np.uint8)
    path = out_dir / "material_map.nii.gz"
    nib.save(img, str(path))
    print(f"Saved {path}  shape={mat.shape}  dtype={mat.dtype}")


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def fill_subarachnoid_csf(mat, sdf, brain):
    """Paint vacuum voxels inside the skull as subarachnoid CSF (u8=8).

    Two populations:
      - Sulcal CSF: within brain mask but unmapped (vacuum) after label remap.
      - Shell CSF:  inside skull (SDF < 0) but outside brain mask, still vacuum.

    Modifies mat in place.  Returns (n_sulcal, n_shell, sulcal, shell).
    """
    # Idempotency check
    existing = int(np.count_nonzero(mat == 8))
    if existing > 5000:
        print(f"WARNING: {existing} voxels already u8=8 — possible re-run")

    # Sulcal CSF: inside brain mask, still vacuum
    sulcal = (brain == 1) & (mat == 0)
    n_sulcal = int(np.count_nonzero(sulcal))
    mat[sulcal] = 8

    # Shell CSF: inside skull but outside brain, still vacuum
    shell = (sdf < 0) & (brain == 0) & (mat == 0)
    n_shell = int(np.count_nonzero(shell))
    mat[shell] = 8

    return n_sulcal, n_shell, sulcal, shell


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def check_domain_closure(mat, sdf):
    """CRITICAL: verify no vacuum remains inside the skull.

    Exits with code 1 on failure.
    """
    vacuum_inside = (sdf < 0) & (mat == 0)
    n_violation = int(np.count_nonzero(vacuum_inside))

    print("\n" + "=" * 60)
    print("Domain Closure Check")
    print("=" * 60)

    if n_violation > 0:
        print(f"CRITICAL FAIL: {n_violation} vacuum voxels remain inside skull")
        sys.exit(1)
    else:
        print("OK: no vacuum inside skull (domain closure satisfied)")


def print_population_volumes(n_sulcal, n_shell, dx_mm, mat):
    """Report sulcal, shell, and total subarachnoid CSF volumes."""
    voxel_vol_ml = dx_mm ** 3 / 1000.0
    total_u8_8 = int(np.count_nonzero(mat == 8))

    print("\n" + "=" * 60)
    print("Subarachnoid CSF Volumes")
    print("=" * 60)
    print(f"  Sulcal CSF (new):  {n_sulcal:>10d} voxels  "
          f"{n_sulcal * voxel_vol_ml:>8.1f} mL")
    print(f"  Shell CSF  (new):  {n_shell:>10d} voxels  "
          f"{n_shell * voxel_vol_ml:>8.1f} mL")
    print(f"  Total new:         {n_sulcal + n_shell:>10d} voxels  "
          f"{(n_sulcal + n_shell) * voxel_vol_ml:>8.1f} mL")
    print(f"  Total u8=8:        {total_u8_8:>10d} voxels  "
          f"{total_u8_8 * voxel_vol_ml:>8.1f} mL")


def print_sulcal_topology(sulcal, n_sulcal):
    """Report connected component count and largest component for sulcal CSF."""
    if n_sulcal == 0:
        print("\nSulcal topology: skipped (0 voxels)")
        return

    print("\n" + "=" * 60)
    print("Sulcal CSF Topology")
    print("=" * 60)

    labeled, n_components = cc_label(sulcal)
    if n_components > 0:
        counts = np.bincount(labeled.ravel())[1:]  # skip background (0)
        largest = int(counts.max())
        largest_pct = 100.0 * largest / n_sulcal
        print(f"  Components: {n_components}")
        print(f"  Largest:    {largest} voxels ({largest_pct:.1f}%)")
    del labeled


def print_shell_topology(shell, n_shell):
    """Report connected component count and largest component for shell CSF."""
    if n_shell == 0:
        print("\nShell topology: skipped (0 voxels)")
        return

    print("\n" + "=" * 60)
    print("Shell CSF Topology")
    print("=" * 60)

    labeled, n_components = cc_label(shell)
    if n_components > 0:
        counts = np.bincount(labeled.ravel())[1:]  # skip background (0)
        largest = int(counts.max())
        largest_pct = 100.0 * largest / n_shell
        print(f"  Components: {n_components}")
        print(f"  Largest:    {largest} voxels ({largest_pct:.1f}%)")
    del labeled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    """Orchestrate subarachnoid CSF filling."""
    args = parse_args(argv)

    print(f"Subject: {args.subject}")
    print(f"Profile: {args.profile}  (N={args.N}, dx={args.dx} mm)")
    print()

    out_dir = processed_dir(args.subject, args.profile)
    mat, sdf, brain, affine, dx_mm = load_inputs(out_dir)
    print(f"Shape: {mat.shape}  dtype: {mat.dtype}")
    print()

    # Core algorithm
    n_sulcal, n_shell, sulcal, shell = fill_subarachnoid_csf(mat, sdf, brain)
    print(f"Sulcal CSF: {n_sulcal} voxels painted")
    print(f"Shell CSF:  {n_shell} voxels painted")

    # Validation
    check_domain_closure(mat, sdf)
    del sdf
    del brain

    print_population_volumes(n_sulcal, n_shell, dx_mm, mat)
    print_census(mat, dx_mm)
    print_shell_topology(shell, n_shell)
    del shell
    print_sulcal_topology(sulcal, n_sulcal)
    del sulcal

    # Save AFTER validation passes
    print()
    save_material_map(out_dir, mat, affine)


if __name__ == "__main__":
    main()
