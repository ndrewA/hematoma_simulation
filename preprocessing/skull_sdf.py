"""Construct the skull signed distance field (SDF) from the brain mask.

Uses atlas-guided T2w growth from the brain mask to approximate the inner
skull surface.  Two mechanisms prevent leakage (especially at skull base):

  1. Atlas pre-weighting: modulates T2w intensity by SPM TPM brain probability
     so growth is suppressed where the atlas says there's no brain (foramina,
     sinuses).  Formula: I_eff = I * (λ + (1-λ) * P_brain^α), λ=0.3, α=5.
     The power α sharpens the probability transition, eliminating ambiguous
     zones (P=0.3-0.7) that cause skull base leakage.

  2. Curvature gating: at each dilation step, only accepts candidate voxels
     with ≥8 existing mask neighbors in a 3×3×3 block, preventing growth
     through narrow channels.

  3. Morphological closing (r=1mm): fills small concavities left by strict
     curvature gating.

Then computes a signed Euclidean distance transform and resamples to the
simulation grid.

Outputs skull_sdf.nii.gz (float32, negative inside skull, positive outside).
"""

import argparse
import json
import sys
from pathlib import Path

import edt as edt_pkg
import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation, convolve, gaussian_filter,
)

from preprocessing.profiling import step
from preprocessing.utils import (
    _PROJECT_ROOT,
    add_grid_args,
    processed_dir,
    raw_dir,
    resample_to_grid,
    resolve_grid_args,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse CLI arguments for skull_sdf."""
    parser = argparse.ArgumentParser(
        description="Construct skull SDF via T2w-guided growth from brain mask."
    )
    add_grid_args(parser)
    parser.add_argument(
        "--bone-z",
        type=float,
        default=-0.5,
        help="T2w z-score threshold for bone classification (default: -0.5)",
    )
    parser.add_argument(
        "--max-growth",
        type=float,
        default=8.0,
        help="Maximum outward growth distance in mm (default: 8.0)",
    )
    parser.add_argument(
        "--atlas-lambda",
        type=float,
        default=0.3,
        help="Atlas pre-weighting: I_eff = I*(λ + (1-λ)*P_brain^α) (default: 0.3)",
    )
    parser.add_argument(
        "--atlas-alpha",
        type=float,
        default=5.0,
        help="Atlas sharpening exponent: P_brain^alpha (default: 5.0)",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=8,
        help="Curvature gating: min mask neighbors in 3x3x3 (default: 8)",
    )
    parser.add_argument(
        "--close-radius",
        type=float,
        default=1.0,
        help="Morphological closing radius in mm (0 to disable, default: 1.0)",
    )
    parser.add_argument(
        "--tpm-path",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "atlases" / "SPM_TPM.nii"),
        help="Path to brain probability atlas",
    )

    args = parser.parse_args(argv)
    resolve_grid_args(args, parser)
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
# Atlas loading
# ---------------------------------------------------------------------------
def load_atlas_brain_prob(tpm_path, target_affine, target_shape):
    """Load brain probability atlas and resample to target native space.

    Supports two formats:
      - 3D NIfTI: pre-computed P(brain) volume (e.g. ICBM152_2009c_P_brain.nii.gz)
      - 4D NIfTI: SPM-style TPM where P(brain) = channels 0+1+2 (GM+WM+CSF)

    Uses affine-only mapping (ACPC ≈ MNI).
    """
    from scipy.ndimage import affine_transform

    tpm_path = Path(tpm_path)
    if not tpm_path.exists():
        print(f"WARNING: atlas not found at {tpm_path}, skipping pre-weighting")
        return None

    print(f"Loading atlas: {tpm_path}")
    tpm_img = nib.load(str(tpm_path))
    tpm_data = tpm_img.get_fdata(dtype=np.float32)

    if tpm_data.ndim == 4:
        # SPM-style 4D TPM: Brain = GM + WM + CSF (first 3 channels)
        p_brain_mni = np.clip(
            tpm_data[:, :, :, 0] + tpm_data[:, :, :, 1] + tpm_data[:, :, :, 2],
            0, 1,
        )
    else:
        # Pre-computed 3D P(brain) volume
        p_brain_mni = np.clip(tpm_data, 0, 1)

    atlas_voxel = np.abs(np.diag(tpm_img.affine)[:3])
    print(f"Atlas voxel size: {atlas_voxel[0]:.1f}mm, shape: {p_brain_mni.shape}")

    # Affine: target voxel → world → TPM voxel
    target_to_tpm = np.linalg.inv(tpm_img.affine) @ target_affine
    p_brain = affine_transform(
        p_brain_mni, target_to_tpm[:3, :3], target_to_tpm[:3, 3],
        output_shape=target_shape, order=1, cval=0.0,
    ).astype(np.float32)
    p_brain = np.clip(p_brain, 0, 1)

    n_nonzero = int((p_brain > 0.01).sum())
    print(f"Atlas P(brain): {n_nonzero:,} nonzero voxels in native space")
    return p_brain


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
                        bone_threshold, max_growth_mm,
                        p_brain=None, atlas_lambda=0.3, atlas_alpha=5.0,
                        min_neighbors=8):
    """Grow outward from brain mask with atlas guidance and curvature gating.

    At each iteration, expands the mask by one voxel (binary dilation)
    and keeps only new voxels where the (atlas-weighted) T2w >= bone_threshold
    and the voxel has enough existing mask neighbors (curvature gate).

    Atlas pre-weighting: I_eff = I * (λ + (1-λ) * P_brain^α).  The power α
    sharpens the probability transition.  Suppresses growth at foramina and
    sinuses where P_brain ≈ 0.

    Curvature gating: require ≥ min_neighbors of 26 neighbors already in
    the mask.  Prevents growth through narrow channels.

    Returns (skull_interior, growth_stats).
    """
    # Atlas pre-weighting with non-linear sharpening
    if p_brain is not None:
        p_sharp = p_brain ** atlas_alpha
        weight = atlas_lambda + (1.0 - atlas_lambda) * p_sharp
        t2w_eff = t2w * weight
        print(f"Atlas pre-weighting: lambda={atlas_lambda}, alpha={atlas_alpha}")
    else:
        t2w_eff = t2w
        print("No atlas available, using raw T2w")

    not_bone = (t2w_eff >= bone_threshold) & head_mask
    skull_interior = brain_mask.copy()
    max_iters = round(max_growth_mm / voxel_size)

    # 3x3x3 neighbor count kernel (exclude center)
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    kernel[1, 1, 1] = 0

    n_iters = 0
    for i in range(max_iters):
        candidates = binary_dilation(skull_interior) & ~skull_interior & not_bone
        if candidates.sum() == 0:
            break

        # Curvature gating
        if min_neighbors > 1:
            neighbor_count = convolve(
                skull_interior.astype(np.int32), kernel,
                mode='constant', cval=0,
            )
            gated = candidates & (neighbor_count >= min_neighbors)
        else:
            gated = candidates

        if gated.sum() == 0:
            break
        skull_interior |= gated
        n_iters = i + 1

    skull_interior &= head_mask
    skull_interior |= brain_mask  # never lose brain interior

    n_brain = int(brain_mask.sum())
    n_grown = int(skull_interior.sum()) - n_brain
    stats = {
        "iterations": n_iters,
        "brain_voxels": n_brain,
        "grown_voxels": n_grown,
        "total_voxels": int(skull_interior.sum()),
        "atlas_lambda": atlas_lambda if p_brain is not None else None,
        "atlas_alpha": atlas_alpha if p_brain is not None else None,
        "min_neighbors": min_neighbors,
    }
    print(f"Growth: {n_iters} iterations, +{n_grown:,} voxels "
          f"({int(skull_interior.sum()):,} total)")
    return skull_interior, stats


# ---------------------------------------------------------------------------
# Morphological closing
# ---------------------------------------------------------------------------
def morphological_close(skull_interior, brain_mask, head_mask, voxel_size,
                        close_radius_mm):
    """EDT-based morphological closing to fill small concavities.

    Dilate by close_radius_mm, then erode by the same amount.
    Preserves brain interior and stays within head mask.
    """
    if close_radius_mm <= 0:
        return skull_interior

    print(f"Morphological closing: r={close_radius_mm}mm")
    dist_outside = edt_pkg.edt(
        ~skull_interior, anisotropy=(voxel_size, voxel_size, voxel_size),
    ).astype(np.float32)
    dilated = skull_interior | (dist_outside <= close_radius_mm)
    dist_inside = edt_pkg.edt(
        dilated, anisotropy=(voxel_size, voxel_size, voxel_size),
    ).astype(np.float32)
    closed = dilated & (dist_inside > close_radius_mm)
    closed |= brain_mask
    closed &= head_mask

    n_filled = int(closed.sum() - skull_interior.sum())
    print(f"Closing filled {n_filled:,} voxels")
    return closed


# ---------------------------------------------------------------------------
# Signed EDT
# ---------------------------------------------------------------------------
def compute_signed_edt(skull_interior, voxel_size, sigma_mm=2.0):
    """Compute signed EDT: negative inside, positive outside, in mm.

    Uses the fast ``edt`` package (~10x faster than scipy for large volumes).
    Applies post-EDT Gaussian smoothing (sigma_mm) to remove staircase
    artifacts from voxel-by-voxel growth.

    Note: smoothing compresses gradient magnitude (~0.77x near the surface)
    but the solver only uses SDF values at cut-cell voxels (within ~0.5 dx
    of the zero-crossing) where compression is minimal.
    """
    print("Computing signed EDT...")
    # edt.sdf returns positive inside, negative outside — negate for our convention
    sdf = -edt_pkg.sdf(
        skull_interior, anisotropy=(voxel_size, voxel_size, voxel_size),
    ).astype(np.float32)
    print(f"SDF range: [{sdf.min():.1f}, {sdf.max():.1f}] mm")

    if sigma_mm > 0:
        sigma_vox = sigma_mm / voxel_size
        print(f"Gaussian smoothing: sigma={sigma_mm}mm ({sigma_vox:.1f} voxels)")
        sdf = gaussian_filter(sdf, sigma=sigma_vox).astype(np.float32)
        print(f"Smoothed SDF range: [{sdf.min():.1f}, {sdf.max():.1f}] mm")

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
    status = "OK" if 1200 <= icv_ml <= 1700 else "WARN"
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
    print(f"Atlas lambda: {args.atlas_lambda}")
    print(f"Atlas alpha: {args.atlas_alpha}")
    print(f"Min neighbors: {args.min_neighbors}")
    print(f"Close radius: {args.close_radius} mm")
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

        # 3. Load atlas brain probability
        p_brain = load_atlas_brain_prob(
            args.tpm_path, source_affine, brain_mask.shape,
        )
    print()

    # 4. Atlas-guided T2w growth with curvature gating
    with step("atlas-guided growth"):
        print("Growing skull interior from brain mask...")
        skull_interior, growth_stats = grow_skull_interior(
            brain_mask, head_mask, t2w, voxel_size,
            bone_threshold, args.max_growth,
            p_brain=p_brain, atlas_lambda=args.atlas_lambda,
            atlas_alpha=args.atlas_alpha, min_neighbors=args.min_neighbors,
        )
        growth_stats.update(threshold_stats)
        del t2w, p_brain

        # 5. Morphological closing
        skull_interior = morphological_close(
            skull_interior, brain_mask, head_mask, voxel_size,
            args.close_radius,
        )
        del head_mask
    print()

    # 6. Signed EDT
    with step("signed EDT"):
        sdf_source = compute_signed_edt(skull_interior, voxel_size)
        del skull_interior
    print()

    # 7. Resample to simulation grid
    out_dir = processed_dir(args.subject, args.profile)
    grid_affine, N_meta = load_grid_meta(out_dir)
    if N_meta != args.N:
        raise ValueError(
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

    # 8. Brain containment clamp
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

    # 9. Save
    save_skull_sdf(out_dir, sdf_sim, grid_affine)

    # 10. Validation
    print_validation(sdf_sim, args, grid_affine, out_dir, growth_stats)


if __name__ == "__main__":
    main()
