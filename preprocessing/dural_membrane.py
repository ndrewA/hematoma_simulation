"""Reconstruct falx cerebri and tentorium cerebelli in the material map.

Identifies equidistant surfaces between tissue masses using EDT watersheds
and paints them as u8=10 (Dural Membrane).  These fibrous sheets create
a 10^9 permeability contrast that drives lateralized pressure gradients
responsible for midline shift and herniation.

Overwrites material_map.nii.gz in place (no new output files).
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

import edt as _edt
import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    label as cc_label,
)
from scipy.spatial import cKDTree

from preprocessing.utils import PROFILES, build_ball, processed_dir, raw_dir
from preprocessing.material_map import CLASS_NAMES, print_census


# ---------------------------------------------------------------------------
# FreeSurfer label sets for hemisphere classification
# ---------------------------------------------------------------------------
LEFT_CEREBRAL_LABELS = frozenset(
    {2, 3, 10, 11, 12, 13, 17, 18, 19, 20, 26, 27, 28, 78, 81}
)
LEFT_CEREBRAL_RANGE = (1001, 1035)

RIGHT_CEREBRAL_LABELS = frozenset(
    {41, 42, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 79, 82}
)
RIGHT_CEREBRAL_RANGE = (2001, 2035)

CC_LABELS = frozenset({192, 251, 252, 253, 254, 255})

_FS_LUT_SIZE = 2036  # matches material_map.LUT_SIZE


def _build_fs_luts():
    """Build boolean LUTs for left/right cerebral and CC FreeSurfer labels."""
    left_lut = np.zeros(_FS_LUT_SIZE, dtype=bool)
    for lab in LEFT_CEREBRAL_LABELS:
        left_lut[lab] = True
    left_lut[LEFT_CEREBRAL_RANGE[0]:LEFT_CEREBRAL_RANGE[1] + 1] = True

    right_lut = np.zeros(_FS_LUT_SIZE, dtype=bool)
    for lab in RIGHT_CEREBRAL_LABELS:
        right_lut[lab] = True
    right_lut[RIGHT_CEREBRAL_RANGE[0]:RIGHT_CEREBRAL_RANGE[1] + 1] = True

    cc_lut = np.zeros(_FS_LUT_SIZE, dtype=bool)
    for lab in CC_LABELS:
        cc_lut[lab] = True

    return left_lut, right_lut, cc_lut


def _compute_crop_bbox(mat, pad_vox):
    """Compute bounding box slices around non-zero voxels with padding."""
    nz = np.nonzero(mat > 0)
    lo = [max(0, int(nz[i].min()) - pad_vox) for i in range(3)]
    hi = [min(mat.shape[i], int(nz[i].max()) + pad_vox + 1) for i in range(3)]
    return tuple(slice(lo[i], hi[i]) for i in range(3))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse CLI arguments for dural_membrane."""
    parser = argparse.ArgumentParser(
        description="Reconstruct falx cerebri and tentorium cerebelli."
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
        "--watershed-threshold", type=float, default=1.0,
        help="Watershed thickness multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--notch-radius", type=float, default=5.0,
        help="Tentorial notch exclusion radius in mm (default: 5.0)",
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
    """Load material map, FS labels, and grid metadata.

    Returns (mat, fs, affine, dx_mm).
    """
    mat_path = out_dir / "material_map.nii.gz"
    fs_path = out_dir / "fs_labels_resampled.nii.gz"
    meta_path = out_dir / "grid_meta.json"

    print(f"Loading {mat_path}")
    mat_img = nib.load(str(mat_path))
    mat = np.asarray(mat_img.dataobj, dtype=np.uint8)
    affine = mat_img.affine.copy()

    print(f"Loading {fs_path}")
    fs_img = nib.load(str(fs_path))
    fs = np.asarray(fs_img.dataobj, dtype=np.int16)

    print(f"Loading {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    dx_mm = float(meta["dx_mm"])

    assert mat.shape == fs.shape, (
        f"Shape mismatch: mat={mat.shape}, fs={fs.shape}"
    )

    return mat, fs, affine, dx_mm


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
def classify_hemispheres(fs, left_lut, right_lut):
    """Classify voxels into left and right cerebral hemispheres.

    Returns (left_mask, right_mask) as boolean arrays.
    Uses pre-built LUTs for O(1) per-voxel lookup instead of np.isin.
    """
    fs_safe = np.clip(fs, 0, _FS_LUT_SIZE - 1)
    left_mask = left_lut[fs_safe]
    right_mask = right_lut[fs_safe]
    return left_mask, right_mask


def compute_cc_boundary(fs, N, cc_lut):
    """Compute the superior z-index of the corpus callosum per coronal slice.

    Returns cc_superior_z, int32 array of shape (N,).
    -1 where no CC voxels exist in that coronal slice.
    Uses pre-built LUT and vectorized max instead of per-slice loop.
    """
    fs_safe = np.clip(fs, 0, _FS_LUT_SIZE - 1)
    cc_mask = cc_lut[fs_safe]

    # any_x[y, z] = True if any CC voxel exists in column (*, y, z)
    any_x = cc_mask.any(axis=0)  # shape (Y, Z)
    Z = cc_mask.shape[2]
    z_idx = np.arange(Z, dtype=np.int32)
    # Where no CC, use -1; otherwise use z index
    z_grid = np.where(any_x, z_idx[np.newaxis, :], -1)  # shape (Y, Z)
    cc_superior_z = z_grid.max(axis=1).astype(np.int32)  # shape (Y,)

    return cc_superior_z


def reconstruct_falx(mat, left_mask, right_mask, cc_superior_z, dx_mm,
                     threshold_mult, crop_slices):
    """Reconstruct the falx cerebri via EDT watershed.

    The falx is the midline sheet between left and right cerebral hemispheres,
    constrained to CSF voxels and above the corpus callosum.

    Returns falx_mask (boolean).
    """
    sampling = (dx_mm, dx_mm, dx_mm)

    # Crop to bounding box for faster EDT
    left_crop = left_mask[crop_slices]
    right_crop = right_mask[crop_slices]

    print("Computing EDT for left+right hemispheres (cropped, threaded)...")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_l = pool.submit(_edt.edt, ~left_crop, anisotropy=sampling, parallel=1)
        fut_r = pool.submit(_edt.edt, ~right_crop, anisotropy=sampling, parallel=1)
        dist_left = fut_l.result()
        dist_right = fut_r.result()
    del left_crop, right_crop
    print(f"  EDT pair: {time.monotonic() - t0:.1f}s")

    # Watershed: equidistant surface within threshold
    diff = np.abs(dist_left - dist_right)
    del dist_left, dist_right

    mat_crop = mat[crop_slices]
    falx_crop = (diff <= threshold_mult * dx_mm) & (mat_crop == 8)
    del diff, mat_crop

    # CC inferior boundary: exclude falx at or below corpus callosum
    # Adjust z-indices for the crop offset
    z_offset = crop_slices[2].start
    y_offset = crop_slices[1].start
    crop_Y = falx_crop.shape[1]
    for y_local in range(crop_Y):
        y_full = y_local + y_offset
        z_sup = cc_superior_z[y_full]
        if z_sup >= 0:
            z_sup_crop = z_sup - z_offset
            if z_sup_crop >= 0:
                falx_crop[:, y_local, :z_sup_crop + 1] = False

    # Write back to full grid
    falx_mask = np.zeros(mat.shape, dtype=bool)
    falx_mask[crop_slices] = falx_crop
    del falx_crop

    n_falx = int(np.count_nonzero(falx_mask))
    print(f"Falx cerebri: {n_falx} voxels")

    return falx_mask


def reconstruct_tentorium(mat, dx_mm, threshold_mult, notch_radius,
                          crop_slices):
    """Reconstruct the tentorium cerebelli via EDT watershed.

    The tentorium is the horizontal sheet between the cerebrum (above) and
    cerebellum (below), with a brainstem notch exclusion.

    Returns tent_mask (boolean).
    """
    sampling = (dx_mm, dx_mm, dx_mm)
    mat_crop = mat[crop_slices]

    # Material LUT for uint8 (no clip needed)
    cerebral_lut = np.zeros(256, dtype=bool)
    cerebral_lut[[1, 2, 3, 9]] = True
    cerebral_crop = cerebral_lut[mat_crop]

    cerebellar_lut = np.zeros(256, dtype=bool)
    cerebellar_lut[[4, 5]] = True
    cerebellar_crop = cerebellar_lut[mat_crop]

    print("Computing EDT for cerebral+cerebellar tissue (cropped, threaded)...")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_c = pool.submit(_edt.edt, ~cerebral_crop, anisotropy=sampling, parallel=1)
        fut_b = pool.submit(_edt.edt, ~cerebellar_crop, anisotropy=sampling, parallel=1)
        dist_cerebral = fut_c.result()
        dist_cerebellar = fut_b.result()
    del cerebral_crop, cerebellar_crop
    print(f"  EDT pair: {time.monotonic() - t0:.1f}s")

    # Watershed: equidistant surface within threshold
    diff = np.abs(dist_cerebral - dist_cerebellar)
    del dist_cerebral, dist_cerebellar

    tent_crop = (diff <= threshold_mult * dx_mm) & (mat_crop == 8)
    del diff

    # Brainstem notch exclusion — cropped dilation
    r_notch_vox = notch_radius / dx_mm
    brainstem_crop = (mat_crop == 6)
    del mat_crop
    if r_notch_vox >= 1.0 and brainstem_crop.any():
        r_step = min(3, int(r_notch_vox))
        r_step = max(r_step, 1)
        n_iter = math.ceil(r_notch_vox / r_step)
        selem = build_ball(r_step)

        # Sub-crop around brainstem for faster dilation
        pad = int(r_notch_vox) + r_step * n_iter + 1
        bs_nz = np.nonzero(brainstem_crop)
        bs_lo = [max(0, int(bs_nz[i].min()) - pad) for i in range(3)]
        bs_hi = [min(brainstem_crop.shape[i], int(bs_nz[i].max()) + pad + 1) for i in range(3)]
        bs_slices = tuple(slice(bs_lo[i], bs_hi[i]) for i in range(3))

        bs_sub = brainstem_crop[bs_slices]
        bs_dilated_sub = binary_dilation(bs_sub, selem, iterations=n_iter)
        tent_crop[bs_slices] &= ~bs_dilated_sub
        del bs_sub, bs_dilated_sub
    del brainstem_crop

    # Write back to full grid
    tent_mask = np.zeros(mat.shape, dtype=bool)
    tent_mask[crop_slices] = tent_crop
    del tent_crop

    n_tent = int(np.count_nonzero(tent_mask))
    print(f"Tentorium cerebelli: {n_tent} voxels")

    return tent_mask


def merge_dural(mat, falx_mask, tent_mask):
    """Paint falx and tentorium voxels as u8=10 in the material map.

    Modifies mat in place.  Returns (n_falx, n_tent, n_overlap, n_total).
    """
    n_falx = int(np.count_nonzero(falx_mask))
    n_tent = int(np.count_nonzero(tent_mask))
    n_overlap = int(np.count_nonzero(falx_mask & tent_mask))

    combined = falx_mask | tent_mask
    n_total = int(np.count_nonzero(combined))
    mat[combined] = 10
    del combined

    return n_falx, n_tent, n_overlap, n_total


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def check_idempotency(mat):
    """Check for and reset pre-existing dural voxels (u8=10).

    Called BEFORE any mask computation to ensure clean state.
    Returns count of reset voxels.
    """
    existing = int(np.count_nonzero(mat == 10))
    if existing > 0:
        print(f"WARNING: {existing} pre-existing u8=10 voxels — resetting to u8=8")
        mat[mat == 10] = 8
    return existing


def print_membrane_continuity(falx_mask, tent_mask, dx_mm):
    """Report connected component analysis for each membrane."""
    print("\n" + "=" * 60)
    print("Membrane Continuity")
    print("=" * 60)

    for name, mask in [("Falx", falx_mask), ("Tentorium", tent_mask)]:
        n = int(np.count_nonzero(mask))
        if n == 0:
            print(f"  {name}: skipped (0 voxels at this resolution)")
            continue

        labeled, n_components = cc_label(mask)
        counts = np.bincount(labeled.ravel())[1:]  # skip background
        largest = int(counts.max())
        largest_pct = 100.0 * largest / n
        print(f"  {name}: {n_components} components, "
              f"largest {largest} ({largest_pct:.1f}%)")
        if n_components > 1:
            second = int(sorted(counts, reverse=True)[1])
            second_pct = 100.0 * second / n
            print(f"    second: {second} ({second_pct:.1f}%)")
        del labeled


def print_csf_components(mat, dx_mm):
    """Report connected components of remaining CSF (u8=8)."""
    csf_mask = (mat == 8)
    n_csf = int(np.count_nonzero(csf_mask))
    if n_csf == 0:
        print("\nCSF components: skipped (0 voxels)")
        return

    print("\n" + "=" * 60)
    print("Remaining CSF Components")
    print("=" * 60)

    labeled, n_components = cc_label(csf_mask)
    del csf_mask
    counts = np.bincount(labeled.ravel())[1:]
    del labeled

    voxel_vol_ml = dx_mm ** 3 / 1000.0
    top_n = min(5, len(counts))
    sorted_counts = sorted(counts, reverse=True)
    for i in range(top_n):
        c = int(sorted_counts[i])
        print(f"  #{i + 1}: {c} voxels ({c * voxel_vol_ml:.1f} mL)")
    print(f"  Total: {n_components} components, {n_csf} voxels")


def print_volumes(n_falx, n_tent, n_overlap, n_total, dx_mm):
    """Report dural membrane volumes."""
    voxel_vol_ml = dx_mm ** 3 / 1000.0

    print("\n" + "=" * 60)
    print("Dural Membrane Volumes")
    print("=" * 60)
    print(f"  Falx cerebri:        {n_falx:>10d} voxels  "
          f"{n_falx * voxel_vol_ml:>8.1f} mL")
    print(f"  Tentorium cerebelli: {n_tent:>10d} voxels  "
          f"{n_tent * voxel_vol_ml:>8.1f} mL")
    print(f"  Overlap:             {n_overlap:>10d} voxels  "
          f"{n_overlap * voxel_vol_ml:>8.1f} mL")
    print(f"  Total (union):       {n_total:>10d} voxels  "
          f"{n_total * voxel_vol_ml:>8.1f} mL")


def print_thickness_estimate(falx_mask, tent_mask, dx_mm):
    """Estimate membrane thickness via erosion."""
    print("\n" + "=" * 60)
    print("Thickness Estimate")
    print("=" * 60)

    for name, mask in [("Falx", falx_mask), ("Tentorium", tent_mask)]:
        n = int(np.count_nonzero(mask))
        if n == 0:
            print(f"  {name}: skipped (0 voxels)")
            continue

        eroded = binary_erosion(mask)
        n_interior = int(np.count_nonzero(eroded))
        n_surface = n - n_interior
        del eroded

        if n_surface > 0:
            thickness = n / (n_surface / 2.0) * dx_mm
            print(f"  {name}: ~{thickness:.2f} mm  "
                  f"(total={n}, interior={n_interior}, surface={n_surface})")
        else:
            print(f"  {name}: < 1 voxel thick  (all surface, {n} voxels)")


def check_tentorial_notch(mat, dx_mm):
    """Verify the tentorial notch is open (CSF adjacent to brainstem)."""
    print("\n" + "=" * 60)
    print("Tentorial Notch Check")
    print("=" * 60)

    brainstem = (mat == 6)
    if not brainstem.any():
        print("  Skipped: no brainstem voxels")
        return

    # Find brainstem z-range, select upper-third slice
    z_indices = np.where(brainstem.any(axis=(0, 1)))[0]
    z_min, z_max = int(z_indices.min()), int(z_indices.max())
    z_upper = z_min + 2 * (z_max - z_min) // 3

    # Check for CSF face-adjacent to brainstem in that slice
    bs_slice = brainstem[:, :, z_upper]
    csf_slice = (mat[:, :, z_upper] == 8)

    # 4-connected neighbors in 2D
    n_adjacent = 0
    if bs_slice.any() and csf_slice.any():
        # Shift brainstem mask in 4 directions and check overlap with CSF
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.zeros_like(bs_slice)
            si = slice(max(0, di), min(bs_slice.shape[0], bs_slice.shape[0] + di) or None)
            di_src = slice(max(0, -di), min(bs_slice.shape[0], bs_slice.shape[0] - di) or None)
            sj = slice(max(0, dj), min(bs_slice.shape[1], bs_slice.shape[1] + dj) or None)
            dj_src = slice(max(0, -dj), min(bs_slice.shape[1], bs_slice.shape[1] - dj) or None)
            shifted[si, sj] = bs_slice[di_src, dj_src]
            n_adjacent += int(np.count_nonzero(shifted & csf_slice))

    print(f"  Upper-third slice z={z_upper}: "
          f"{n_adjacent} CSF voxels face-adjacent to brainstem")
    if n_adjacent == 0:
        print("  WARNING: tentorial notch may be occluded (0 adjacent CSF)")
    else:
        print("  OK: tentorial notch appears open")


def check_medial_wall_proximity(falx_mask, subject, dx_mm, affine):
    """Check falx proximity to medial wall surfaces (optional).

    Skips silently if surface files don't exist.
    """
    n_falx = int(np.count_nonzero(falx_mask))
    if n_falx == 0:
        return

    native_dir = raw_dir(subject) / "Native"
    mni_native_dir = raw_dir(subject).parent / "MNINonLinear" / "Native"

    results = []
    for hemi, label in [("L", "left"), ("R", "right")]:
        surf_path = native_dir / f"{subject}.{hemi}.pial.native.surf.gii"
        roi_path = mni_native_dir / f"{subject}.{hemi}.roi.native.shape.gii"

        if not surf_path.exists() or not roi_path.exists():
            return  # skip entirely if any file missing

        surf = nib.load(str(surf_path))
        coords = surf.darrays[0].data  # (n_vertices, 3)

        roi = nib.load(str(roi_path))
        roi_data = roi.darrays[0].data  # (n_vertices,)

        # Medial wall vertices: roi == 0
        medial_idx = np.where(roi_data == 0)[0]
        if len(medial_idx) == 0:
            continue

        medial_coords = coords[medial_idx]
        tree = cKDTree(medial_coords)

        # Convert falx voxel indices to physical mm
        falx_ijk = np.argwhere(falx_mask).astype(np.float64)
        falx_mm = falx_ijk * dx_mm + affine[:3, 3]

        dists, _ = tree.query(falx_mm)
        results.append((label, float(np.median(dists)), float(np.percentile(dists, 95))))

    if results:
        print("\n" + "=" * 60)
        print("Medial Wall Proximity")
        print("=" * 60)
        for label, median_d, p95_d in results:
            print(f"  {label} medial wall: median={median_d:.1f} mm, "
                  f"95th={p95_d:.1f} mm")


def print_junction_thickness(falx_mask, tent_mask, dx_mm, mat):
    """Report overlap extent and max z-thickness at the falx-tentorium junction."""
    overlap = falx_mask & tent_mask
    n_overlap = int(np.count_nonzero(overlap))

    print("\n" + "=" * 60)
    print("Falx-Tentorium Junction")
    print("=" * 60)

    if n_overlap == 0:
        print("  No overlap between falx and tentorium")
        return

    voxel_vol_ml = dx_mm ** 3 / 1000.0
    print(f"  Overlap: {n_overlap} voxels ({n_overlap * voxel_vol_ml:.2f} mL)")

    # Max z-thickness at junction midline
    overlap_ijk = np.argwhere(overlap)
    if len(overlap_ijk) > 0:
        # Group by (x, y) and measure z-span
        xy_unique = np.unique(overlap_ijk[:, :2], axis=0)
        max_z_span = 0
        for xy in xy_unique:
            z_vals = overlap_ijk[(overlap_ijk[:, 0] == xy[0]) &
                                (overlap_ijk[:, 1] == xy[1]), 2]
            z_span = int(z_vals.max()) - int(z_vals.min()) + 1
            max_z_span = max(max_z_span, z_span)

        print(f"  Max z-thickness: {max_z_span} voxels ({max_z_span * dx_mm:.1f} mm)")
        if max_z_span > 3:
            print(f"  WARNING: junction thickness > 3 voxels")

    del overlap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    """Orchestrate dural membrane reconstruction."""
    t_total = time.monotonic()
    args = parse_args(argv)

    print(f"Subject: {args.subject}")
    print(f"Profile: {args.profile}  (N={args.N}, dx={args.dx} mm)")
    print(f"Watershed threshold: {args.watershed_threshold}")
    print(f"Notch radius: {args.notch_radius} mm")
    print()

    out_dir = processed_dir(args.subject, args.profile)
    mat, fs, affine, dx_mm = load_inputs(out_dir)
    print(f"Shape: {mat.shape}  dtype: {mat.dtype}")
    print()

    # Idempotency: reset any pre-existing dural voxels
    check_idempotency(mat)

    # Build LUTs once
    left_lut, right_lut, cc_lut = _build_fs_luts()

    # Classify hemispheres and compute CC boundary
    t0 = time.monotonic()
    left_mask, right_mask = classify_hemispheres(fs, left_lut, right_lut)
    cc_superior_z = compute_cc_boundary(fs, mat.shape[1], cc_lut)
    del fs
    print(f"Hemisphere classification + CC boundary: {time.monotonic() - t0:.1f}s")

    print(f"Left hemisphere:  {int(np.count_nonzero(left_mask))} voxels")
    print(f"Right hemisphere: {int(np.count_nonzero(right_mask))} voxels")
    print()

    # Compute crop bounding box for EDT
    pad_vox = math.ceil(args.notch_radius / dx_mm) + 2
    crop_slices = _compute_crop_bbox(mat, pad_vox)
    crop_shape = tuple(s.stop - s.start for s in crop_slices)
    print(f"EDT crop: {crop_shape} (pad={pad_vox} voxels)")
    print()

    # Reconstruct falx cerebri
    t0 = time.monotonic()
    falx_mask = reconstruct_falx(
        mat, left_mask, right_mask, cc_superior_z,
        dx_mm, args.watershed_threshold, crop_slices,
    )
    print(f"  Falx total: {time.monotonic() - t0:.1f}s")
    del left_mask, right_mask, cc_superior_z

    # Reconstruct tentorium cerebelli
    t0 = time.monotonic()
    tent_mask = reconstruct_tentorium(
        mat, dx_mm, args.watershed_threshold, args.notch_radius, crop_slices,
    )
    print(f"  Tentorium total: {time.monotonic() - t0:.1f}s")

    # Merge into material map
    n_falx, n_tent, n_overlap, n_total = merge_dural(mat, falx_mask, tent_mask)
    print(f"\nMerged: {n_total} total dural voxels "
          f"(falx={n_falx}, tent={n_tent}, overlap={n_overlap})")

    # Validation
    print_volumes(n_falx, n_tent, n_overlap, n_total, dx_mm)
    print_membrane_continuity(falx_mask, tent_mask, dx_mm)
    print_thickness_estimate(falx_mask, tent_mask, dx_mm)
    check_tentorial_notch(mat, dx_mm)
    print_junction_thickness(falx_mask, tent_mask, dx_mm, mat)
    check_medial_wall_proximity(falx_mask, args.subject, dx_mm, affine)
    print_csf_components(mat, dx_mm)
    print_census(mat, dx_mm)

    del falx_mask, tent_mask

    # Save
    print()
    save_material_map(out_dir, mat, affine)

    print(f"\nTotal wall time: {time.monotonic() - t_total:.1f}s")


if __name__ == "__main__":
    main()
