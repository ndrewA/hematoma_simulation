"""Construct the skull signed distance field (SDF) from the brain mask.

Uses T2w-guided growth from the brain mask to approximate the inner skull
surface: iteratively dilates the brain mask, growing only through voxels
where T2w is bright enough to be non-bone.  This naturally adapts to local
CSF thickness (thin at skull base, thick at convexity).  Then computes a
signed Euclidean distance transform and resamples to the simulation grid.

Outputs skull_sdf.nii.gz (float32, negative inside skull, positive outside).
"""

import argparse
import json
import math
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation, binary_opening, distance_transform_edt, gaussian_filter,
)

from preprocessing.profiling import step
from preprocessing.utils import PROFILES, processed_dir, raw_dir, resample_to_grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse CLI arguments for skull_sdf."""
    parser = argparse.ArgumentParser(
        description="Construct skull SDF via T2w-guided growth from brain mask."
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
        "--grid-size",
        type=int,
        help="Grid size N (required with --dx, ignored with --profile)",
    )
    parser.add_argument(
        "--bone-z",
        type=float,
        default=-1.0,
        help="T2w z-score threshold for bone classification (default: -1.0)",
    )
    parser.add_argument(
        "--max-growth",
        type=float,
        default=8.0,
        help="Maximum outward growth distance in mm (default: 8.0)",
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
# Helpers
# ---------------------------------------------------------------------------
def load_source_masks(raw):
    """Load brainmask_fs and Head masks as bool arrays.

    Returns (brain_mask, head_mask, source_affine).
    """
    brain_path = raw / "brainmask_fs.nii.gz"
    head_path = raw / "Head.nii.gz"

    print(f"Loading {brain_path}")
    brain_img = nib.load(str(brain_path))
    brain_mask = brain_img.get_fdata() > 0.5
    source_affine = brain_img.affine.copy()

    print(f"Loading {head_path}")
    head_img = nib.load(str(head_path))
    head_mask = head_img.get_fdata() > 0.5

    return brain_mask, head_mask, source_affine


def load_t2w(raw):
    """Load T2w volume from the raw subject directory.

    Returns the float32 data array, or None if the file doesn't exist.
    """
    t2w_path = raw / "T2w_acpc_dc_restore.nii.gz"
    if not t2w_path.exists():
        return None
    print(f"Loading {t2w_path}")
    t2w_img = nib.load(str(t2w_path))
    return t2w_img.get_fdata(dtype=np.float32)


def extract_voxel_size(source_affine):
    """Extract isotropic voxel size from source affine diagonal."""
    diag = np.abs(np.diag(source_affine)[:3])
    if not np.allclose(diag, diag[0], atol=0.05):
        print(f"WARNING: non-isotropic voxel sizes {diag}")
    voxel_size = float(diag[0])
    print(f"Source voxel size: {voxel_size:.2f} mm")
    return voxel_size


def load_grid_meta(out_dir):
    """Load grid_meta.json, return (grid_affine, N)."""
    meta_path = out_dir / "grid_meta.json"
    print(f"Loading {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    grid_affine = np.array(meta["affine_grid_to_phys"])
    N = int(meta["grid_size"])
    return grid_affine, N


# ---------------------------------------------------------------------------
# T2w-guided growth
# ---------------------------------------------------------------------------
def compute_bone_threshold(t2w, brain_mask, bone_z):
    """Compute T2w bone threshold using brain tissue z-score.

    Uses robust statistics (median + MAD) of brain T2w to define a
    scanner-independent threshold.  Voxels with T2w below this threshold
    are classified as bone/air and block growth.

    Returns (bone_threshold, stats_dict) or (None, None) if unreliable.
    """
    brain_vals = t2w[brain_mask]
    if len(brain_vals) < 1000:
        print("WARNING: too few brain voxels for threshold computation")
        return None, None

    brain_median = float(np.median(brain_vals))
    brain_mad = float(np.median(np.abs(brain_vals - brain_median)))
    brain_scale = brain_mad / 0.6745  # robust sigma

    if brain_scale < 1:
        print("WARNING: brain T2w scale too small")
        return None, None

    bone_threshold = brain_median + bone_z * brain_scale

    stats = {
        "brain_median": brain_median,
        "brain_scale": brain_scale,
        "bone_z": bone_z,
        "bone_threshold": bone_threshold,
    }

    print(f"Brain T2w: median={brain_median:.0f}, scale={brain_scale:.0f}")
    print(f"Bone threshold (z={bone_z:+.1f}): T2w < {bone_threshold:.0f}")

    return bone_threshold, stats


def grow_skull_interior(brain_mask, head_mask, t2w, voxel_size,
                        bone_threshold, max_growth_mm):
    """Grow outward from brain mask, stopping at bone-dark T2w voxels.

    At each iteration, expands the mask by one voxel (binary dilation)
    and keeps only new voxels where T2w >= bone_threshold.  This:
      - Fills cortical sulci (CSF between gyral banks)
      - Extends through subarachnoid CSF to the skull
      - Stops at bone (dark T2w) at spatially varying distances
      - Adapts locally: thin CSF at skull base → small growth,
        thick CSF at convexity → larger growth

    Returns (skull_interior, n_iterations).
    """
    not_bone = (t2w >= bone_threshold) & head_mask
    skull_interior = brain_mask.copy()
    max_iters = round(max_growth_mm / voxel_size)

    n_iters = 0
    for i in range(max_iters):
        grow = binary_dilation(skull_interior) & ~skull_interior & not_bone
        n = grow.sum()
        if n == 0:
            break
        skull_interior |= grow
        n_iters = i + 1

    skull_interior &= head_mask
    return skull_interior, n_iters


# ---------------------------------------------------------------------------
# Signed EDT
# ---------------------------------------------------------------------------
def compute_signed_edt(skull_interior, voxel_size):
    """Compute signed EDT: negative inside, positive outside, in mm."""
    sampling = (voxel_size, voxel_size, voxel_size)

    print("Computing EDT (outside)...")
    dt_outside = distance_transform_edt(~skull_interior, sampling=sampling)

    print("Computing EDT (inside)...")
    dt_inside = distance_transform_edt(skull_interior, sampling=sampling)

    dt_outside -= dt_inside
    del dt_inside

    sdf = dt_outside.astype(np.float32)
    del dt_outside

    sigma_mm = 2.0
    sigma_vox = sigma_mm / voxel_size
    print(f"Smoothing SDF: sigma={sigma_mm} mm ({sigma_vox:.1f} voxels)")
    sdf = gaussian_filter(sdf, sigma=sigma_vox).astype(np.float32)

    print(f"SDF range: [{sdf.min():.1f}, {sdf.max():.1f}] mm")
    return sdf


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_skull_sdf(out_dir, sdf, grid_affine):
    """Save skull_sdf.nii.gz as float32."""
    img = nib.Nifti1Image(sdf, grid_affine)
    img.header.set_data_dtype(np.float32)
    path = out_dir / "skull_sdf.nii.gz"
    nib.save(img, str(path))
    print(f"Saved {path}  shape={sdf.shape}  dtype={sdf.dtype}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def print_validation(sdf_sim, args, grid_affine, out_dir,
                     growth_stats=None):
    """Run validation checks on the final SDF."""
    N, dx = args.N, args.dx
    failed = False

    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    # 1. Intracranial volume
    n_negative = int(np.count_nonzero(sdf_sim < 0))
    icv_ml = n_negative * dx ** 3 / 1000.0
    status = "OK" if 1300 <= icv_ml <= 1600 else "WARN"
    print(f"\n1. Intracranial volume (SDF < 0):")
    print(f"   {n_negative} voxels x {dx}^3 mm^3 = {icv_ml:.1f} mL  [{status}]")

    # 2. Brain containment
    brain_path = out_dir / "brain_mask.nii.gz"
    print(f"\n2. Brain containment:")
    brain_img = nib.load(str(brain_path))
    brain_sim = np.asarray(brain_img.dataobj, dtype=np.uint8)
    brain_voxels = brain_sim > 0
    sdf_at_brain = sdf_sim[brain_voxels]

    n_outside = int(np.count_nonzero(sdf_at_brain >= 0))
    if n_outside > 0:
        print(f"   FAIL: {n_outside} brain voxels have SDF >= 0")
        failed = True
    else:
        print(f"   All {int(brain_voxels.sum())} brain voxels have SDF < 0  [OK]")
    print(f"   Min SDF at brain: {sdf_at_brain.min():.2f} mm")

    # 3. Margin at brain surface
    print(f"\n3. Margin at brain surface:")
    brain_dilated = binary_dilation(brain_voxels)
    brain_surface = brain_dilated & ~brain_voxels
    del brain_dilated
    sdf_at_surface = sdf_sim[brain_surface]
    p5 = float(np.percentile(sdf_at_surface, 5))
    p50 = float(np.median(sdf_at_surface))
    print(f"   SDF at brain surface: P5={p5:.2f}, median={p50:.2f} mm")

    del brain_sim, brain_voxels, brain_surface

    # 4. Gradient magnitude on center slices
    print(f"\n4. Gradient magnitude (center slices):")
    for name, slc in [("axial",    sdf_sim[N // 2, :, :]),
                       ("coronal",  sdf_sim[:, N // 2, :]),
                       ("sagittal", sdf_sim[:, :, N // 2])]:
        g0, g1 = np.gradient(slc, dx)
        grad_mag = np.sqrt(g0 ** 2 + g1 ** 2)
        near_brain = np.abs(slc) < 50.0
        if near_brain.any():
            mean_g = float(grad_mag[near_brain].mean())
            std_g = float(grad_mag[near_brain].std())
            print(f"   {name:10s}: mean |grad| = {mean_g:.3f}, std = {std_g:.3f}")

    # 5. Smoothness at boundary
    print(f"\n5. Smoothness at boundary:")
    shell = np.abs(sdf_sim) < dx
    n_shell = int(shell.sum())
    if n_shell > 0:
        std_shell = float(np.std(sdf_sim[shell]))
        print(f"   Shell |SDF| < {dx} mm: {n_shell} voxels, "
              f"std(SDF) = {std_shell:.3f} mm")

    # 6. Growth stats
    print(f"\n6. T2w-guided growth:")
    if growth_stats is not None:
        for k, v in growth_stats.items():
            print(f"   {k}: {v}")
    else:
        print(f"   No growth applied (T2w unavailable)")

    if failed:
        print(f"\nCRITICAL: validation failed (see FAIL above)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    """Orchestrate skull SDF construction."""
    args = parse_args(argv)

    print(f"Subject: {args.subject}")
    print(f"Profile: {args.profile}  (N={args.N}, dx={args.dx} mm)")
    print(f"Bone z-score: {args.bone_z}")
    print(f"Max growth: {args.max_growth} mm")
    print()

    # 1. Load source masks
    raw = raw_dir(args.subject)
    with step("load sources"):
        brain_mask, head_mask, source_affine = load_source_masks(raw)
        voxel_size = extract_voxel_size(source_affine)

        # 2. Load T2w and compute bone threshold
        t2w = load_t2w(raw)
        if t2w is None:
            print("FATAL: T2w required for skull SDF construction")
            sys.exit(1)

        bone_threshold, threshold_stats = compute_bone_threshold(
            t2w, brain_mask, args.bone_z,
        )
        if bone_threshold is None:
            print("FATAL: could not compute T2w threshold")
            sys.exit(1)
    print()

    # 3. T2w-guided growth
    with step("T2w-guided growth"):
        print("Growing skull interior from brain mask...")
        skull_interior, n_iters = grow_skull_interior(
            brain_mask, head_mask, t2w, voxel_size,
            bone_threshold, args.max_growth,
        )
        n_brain = int(brain_mask.sum())
        n_grown = int(skull_interior.sum()) - n_brain

        # Morphological opening: remove single-voxel protrusions from the
        # growth front that cause sawtooth artifacts in the SDF contour.
        # Re-add the brain mask so opening never erodes inside the brain.
        n_before_open = int(skull_interior.sum())
        skull_interior = binary_opening(skull_interior) | brain_mask
        n_removed = n_before_open - int(skull_interior.sum())
        print(f"Opening: removed {n_removed:,} voxels")

        growth_stats = {
            "iterations": n_iters,
            "brain_voxels": n_brain,
            "grown_voxels": n_grown,
            "opening_removed": n_removed,
            "total_voxels": int(skull_interior.sum()),
            **threshold_stats,
        }
        print(f"Growth: {n_iters} iterations, +{n_grown:,} voxels "
              f"({int(skull_interior.sum()):,} total)")
        del brain_mask, head_mask, t2w
    print()

    # 4. Signed EDT
    with step("signed EDT + smooth"):
        sdf_source = compute_signed_edt(skull_interior, voxel_size)
        del skull_interior
    print()

    # 5. Resample to simulation grid
    out_dir = processed_dir(args.subject, args.profile)
    grid_affine, N_meta = load_grid_meta(out_dir)
    assert N_meta == args.N, (
        f"grid_meta.json N={N_meta} does not match profile N={args.N}"
    )
    grid_shape = (N_meta, N_meta, N_meta)

    with step("resample to grid"):
        print(f"Resampling SDF to simulation grid ({N_meta}^3, dx={args.dx} mm)...")
        sdf_sim = resample_to_grid(
            (sdf_source, source_affine), grid_affine, grid_shape,
            order=1, cval=100.0, dtype=np.float32,
        )
        del sdf_source
    print()

    # 6. Brain containment clamp
    brain_sim = np.asarray(
        nib.load(str(out_dir / "brain_mask.nii.gz")).dataobj, dtype=np.uint8
    ) > 0
    exposed = brain_sim & (sdf_sim >= 0)
    n_exposed = int(exposed.sum())
    if n_exposed > 0:
        max_val = float(sdf_sim[exposed].max())
        sdf_sim[exposed] = -0.01
        print(f"Brain containment clamp: {n_exposed} voxels "
              f"(max was {max_val:.3f} mm)")
    else:
        print("Brain containment: no clamp needed")
    del brain_sim, exposed
    print()

    # 7. Save
    save_skull_sdf(out_dir, sdf_sim, grid_affine)

    # 8. Validation
    print_validation(sdf_sim, args, grid_affine, out_dir, growth_stats)


if __name__ == "__main__":
    main()
