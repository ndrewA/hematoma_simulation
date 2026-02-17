"""Construct the skull signed distance field (SDF) from the brain mask.

Morphologically expands the brain mask at source resolution (0.7 mm) to
approximate the inner skull surface, computes a signed Euclidean distance
transform, and resamples to the simulation grid.  Outputs skull_sdf.nii.gz
(float32, negative inside skull, positive outside).
"""

import argparse
import json
import math
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from preprocessing.utils import (
    PROFILES,
    build_ball,
    build_grid_affine,
    processed_dir,
    raw_dir,
    resample_to_grid,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse CLI arguments for skull_sdf."""
    parser = argparse.ArgumentParser(
        description="Construct skull SDF from brain mask via morphological expansion."
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
    parser.add_argument(
        "--close-radius", type=float, default=10.0,
        help="Morphological closing radius in mm (default: 10.0)",
    )
    parser.add_argument(
        "--dilate-radius", type=float, default=4.0,
        help="Outward dilation radius in mm (default: 4.0)",
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


def extract_voxel_size(source_affine):
    """Extract isotropic voxel size from source affine diagonal.

    Uses abs() since X diagonal is typically -0.7.  Verifies isotropic.
    """
    diag = np.abs(np.diag(source_affine)[:3])
    if not np.allclose(diag, diag[0], atol=0.05):
        print(f"WARNING: non-isotropic voxel sizes {diag}")
    voxel_size = float(diag[0])
    print(f"Source voxel size: {voxel_size:.2f} mm (diagonal: {np.diag(source_affine)[:3]})")
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


def pad_inferior(brain_mask, head_mask, source_affine, pad_z, voxel_size):
    """Pad both masks inferiorly on axis 2 to allow dilation below brainstem.

    Returns (brain_padded, head_padded, A_padded).
    """
    brain_padded = np.pad(brain_mask, ((0, 0), (0, 0), (pad_z, 0)),
                          constant_values=False)
    head_padded = np.pad(head_mask, ((0, 0), (0, 0), (pad_z, 0)),
                         constant_values=False)

    A_padded = source_affine.copy()
    A_padded[2, 3] -= pad_z * voxel_size

    print(f"Inferior padding: {pad_z} slices on axis 2")
    print(f"  Padded shape: {brain_padded.shape}")
    print(f"  A_padded[2,3]: {A_padded[2, 3]:.1f} (was {source_affine[2, 3]:.1f})")

    return brain_padded, head_padded, A_padded


def morphological_close(brain_mask, r_close_vox, r_step=3):
    """Morphological closing: dilation then erosion with iterated small ball.

    Fills cortical sulci and interhemispheric fissure while preserving
    convex contours.
    """
    ball = build_ball(r_step)
    n_iter = math.ceil(r_close_vox / r_step)
    effective = n_iter * r_step

    print(f"Morphological closing: r_close_vox={r_close_vox}, "
          f"r_step={r_step}, n_iter={n_iter}, effective={effective} voxels")

    print("  Dilating...")
    closed = binary_dilation(brain_mask, ball, iterations=n_iter)
    print("  Eroding...")
    closed = binary_erosion(closed, ball, iterations=n_iter)

    return closed


def dilate_outward(closed, r_dilate_vox, r_step=3):
    """Outward dilation to approximate subarachnoid space thickness."""
    ball = build_ball(r_step)
    n_iter = math.ceil(r_dilate_vox / r_step)
    effective = n_iter * r_step

    print(f"Outward dilation: r_dilate_vox={r_dilate_vox}, "
          f"n_iter={n_iter}, effective={effective} voxels")

    skull_interior = binary_dilation(closed, ball, iterations=n_iter)
    return skull_interior


def compute_signed_edt(skull_interior, voxel_size):
    """Compute signed EDT: negative inside, positive outside, in mm."""
    sampling = (voxel_size, voxel_size, voxel_size)

    print("Computing EDT (outside)...")
    dt_outside = distance_transform_edt(~skull_interior, sampling=sampling)

    print("Computing EDT (inside)...")
    dt_inside = distance_transform_edt(skull_interior, sampling=sampling)

    # Combine in-place to save memory
    dt_outside -= dt_inside
    del dt_inside

    sdf = dt_outside.astype(np.float32)
    del dt_outside

    print(f"SDF range: [{sdf.min():.1f}, {sdf.max():.1f}] mm")
    return sdf


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
def print_validation(sdf_sim, args, grid_affine, out_dir, r_dilate):
    """Run validation checks on the final SDF.

    Returns True if all critical checks pass, False otherwise.
    """
    N, dx = args.N, args.dx
    failed = False

    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    # 1. Intracranial volume
    n_negative = int(np.count_nonzero(sdf_sim < 0))
    icv_ml = n_negative * dx ** 3 / 1000.0
    status = "OK" if 1500 <= icv_ml <= 1750 else "WARN"
    print(f"\n1. Intracranial volume (SDF < 0):")
    print(f"   {n_negative} voxels × {dx}³ mm³ = {icv_ml:.1f} mL  [{status}]")

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
    print(f"   Min SDF at brain: {sdf_at_brain.min():.1f} mm")

    # 3. Margin at brain surface
    print(f"\n3. Margin at brain surface:")
    brain_dilated = binary_dilation(brain_voxels)
    brain_surface = brain_dilated & ~brain_voxels
    del brain_dilated
    sdf_at_surface = sdf_sim[brain_surface]
    p5 = float(np.percentile(sdf_at_surface, 5))
    print(f"   SDF at brain surface 5th percentile: {p5:.1f} mm "
          f"(expected ≈ -{r_dilate:.0f})")

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
            print(f"   {name:10s}: mean |∇SDF| = {mean_g:.3f}, std = {std_g:.3f}")
        else:
            print(f"   {name:10s}: no voxels within 50 mm of surface")

    # 5. Smoothness at boundary (spec Section 8.5)
    print(f"\n5. Smoothness at boundary:")
    shell_thickness = dx  # one-voxel shell in mm
    shell = np.abs(sdf_sim) < shell_thickness
    n_shell = int(shell.sum())
    if n_shell > 0:
        sdf_in_shell = sdf_sim[shell]
        std_shell = float(np.std(sdf_in_shell))
        print(f"   Shell |SDF| < {shell_thickness} mm: {n_shell} voxels, "
              f"std(SDF) = {std_shell:.3f} mm")
    else:
        print(f"   No voxels within {shell_thickness} mm of SDF = 0 surface")

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
    print(f"Close radius: {args.close_radius} mm")
    print(f"Dilate radius: {args.dilate_radius} mm")
    print()

    # 1. Load source masks
    raw = raw_dir(args.subject)
    brain_mask, head_mask, source_affine = load_source_masks(raw)
    voxel_size = extract_voxel_size(source_affine)
    print()

    # 2. Compute voxel radii and padding
    r_close_vox = math.ceil(args.close_radius / voxel_size)
    r_dilate_vox = math.ceil(args.dilate_radius / voxel_size)
    pad_z = r_close_vox + r_dilate_vox + 10

    print(f"r_close_vox={r_close_vox}, r_dilate_vox={r_dilate_vox}, pad_z={pad_z}")

    # 3. Inferior padding
    brain_mask, head_mask, A_padded = pad_inferior(
        brain_mask, head_mask, source_affine, pad_z, voxel_size
    )
    print()

    # 4. Morphological closing
    closed = morphological_close(brain_mask, r_close_vox)
    del brain_mask
    print()

    # 5. Outward dilation
    skull_interior = dilate_outward(closed, r_dilate_vox)
    del closed
    print()

    # 6. Head mask intersection
    print(f"Head mask intersection: {int(skull_interior.sum())} → ", end="")
    skull_interior &= head_mask
    print(f"{int(skull_interior.sum())} voxels")
    del head_mask
    print()

    # 7. Signed EDT
    sdf_source = compute_signed_edt(skull_interior, voxel_size)
    del skull_interior
    print()

    # 8. Load grid meta and resample to simulation grid
    out_dir = processed_dir(args.subject, args.profile)
    grid_affine, N_meta = load_grid_meta(out_dir)
    assert N_meta == args.N, (
        f"grid_meta.json N={N_meta} does not match profile N={args.N}"
    )
    grid_shape = (N_meta, N_meta, N_meta)

    print(f"Resampling SDF to simulation grid ({N_meta}³, dx={args.dx} mm)...")
    sdf_sim = resample_to_grid(
        (sdf_source, A_padded), grid_affine, grid_shape,
        order=1, cval=100.0, dtype=np.float32,
    )
    del sdf_source, A_padded
    print()

    # 9. Save
    save_skull_sdf(out_dir, sdf_sim, grid_affine)

    # 10. Validation
    print_validation(sdf_sim, args, grid_affine, out_dir, args.dilate_radius)


if __name__ == "__main__":
    main()
