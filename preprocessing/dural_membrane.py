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

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
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
def classify_hemispheres(fs):
    """Classify voxels into left and right cerebral hemispheres.

    Returns (left_mask, right_mask) as boolean arrays.
    """
    left_mask = (
        np.isin(fs, list(LEFT_CEREBRAL_LABELS))
        | ((fs >= LEFT_CEREBRAL_RANGE[0]) & (fs <= LEFT_CEREBRAL_RANGE[1]))
    )
    right_mask = (
        np.isin(fs, list(RIGHT_CEREBRAL_LABELS))
        | ((fs >= RIGHT_CEREBRAL_RANGE[0]) & (fs <= RIGHT_CEREBRAL_RANGE[1]))
    )
    return left_mask, right_mask


def compute_cc_boundary(fs, N):
    """Compute the superior z-index of the corpus callosum per coronal slice.

    Returns cc_superior_z, int32 array of shape (N,).
    -1 where no CC voxels exist in that coronal slice.
    """
    cc_mask = np.isin(fs, list(CC_LABELS))
    cc_superior_z = np.full(N, -1, dtype=np.int32)

    for y in range(N):
        cc_slice = cc_mask[:, y, :]
        if cc_slice.any():
            z_indices = np.where(cc_slice.any(axis=0))[0]
            cc_superior_z[y] = int(z_indices.max())

    return cc_superior_z


def reconstruct_falx(mat, left_mask, right_mask, cc_superior_z, dx_mm,
                     threshold_mult):
    """Reconstruct the falx cerebri via EDT watershed.

    The falx is the midline sheet between left and right cerebral hemispheres,
    constrained to CSF voxels and above the corpus callosum.

    Returns falx_mask (boolean).
    """
    sampling = (dx_mm, dx_mm, dx_mm)

    print("Computing EDT for left hemisphere...")
    dist_left = distance_transform_edt(~left_mask, sampling=sampling).astype(
        np.float32
    )
    print("Computing EDT for right hemisphere...")
    dist_right = distance_transform_edt(~right_mask, sampling=sampling).astype(
        np.float32
    )

    # Watershed: equidistant surface within threshold
    diff = np.abs(dist_left - dist_right)
    del dist_left, dist_right

    falx_mask = (diff <= threshold_mult * dx_mm) & (mat == 8)
    del diff

    # CC inferior boundary: exclude falx at or below corpus callosum
    N = mat.shape[1]
    for y in range(N):
        z_sup = cc_superior_z[y]
        if z_sup >= 0:
            falx_mask[:, y, :z_sup + 1] = False

    n_falx = int(np.count_nonzero(falx_mask))
    print(f"Falx cerebri: {n_falx} voxels")

    return falx_mask


def reconstruct_tentorium(mat, dx_mm, threshold_mult, notch_radius):
    """Reconstruct the tentorium cerebelli via EDT watershed.

    The tentorium is the horizontal sheet between the cerebrum (above) and
    cerebellum (below), with a brainstem notch exclusion.

    Returns tent_mask (boolean).
    """
    sampling = (dx_mm, dx_mm, dx_mm)

    # Cerebral tissue: WM (1), cortical GM (2), deep GM (3), choroid (9)
    cerebral_mask = np.isin(mat, [1, 2, 3, 9])
    print("Computing EDT for cerebral tissue...")
    dist_cerebral = distance_transform_edt(~cerebral_mask, sampling=sampling).astype(
        np.float32
    )
    del cerebral_mask

    # Cerebellar tissue: cerebellar WM (4), cerebellar cortex (5)
    cerebellar_mask = np.isin(mat, [4, 5])
    print("Computing EDT for cerebellar tissue...")
    dist_cerebellar = distance_transform_edt(~cerebellar_mask, sampling=sampling).astype(
        np.float32
    )
    del cerebellar_mask

    # Watershed: equidistant surface within threshold
    diff = np.abs(dist_cerebral - dist_cerebellar)
    del dist_cerebral, dist_cerebellar

    tent_mask = (diff <= threshold_mult * dx_mm) & (mat == 8)
    del diff

    # Brainstem notch exclusion
    brainstem_mask = (mat == 6)
    r_notch_vox = notch_radius / dx_mm
    if r_notch_vox >= 1.0 and brainstem_mask.any():
        r_step = min(3, int(r_notch_vox))
        r_step = max(r_step, 1)
        n_iter = math.ceil(r_notch_vox / r_step)
        selem = build_ball(r_step)
        brainstem_dilated = binary_dilation(brainstem_mask, selem, iterations=n_iter)
        tent_mask &= ~brainstem_dilated
        del brainstem_dilated
    del brainstem_mask

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

    # Classify hemispheres and compute CC boundary
    left_mask, right_mask = classify_hemispheres(fs)
    cc_superior_z = compute_cc_boundary(fs, mat.shape[1])
    del fs

    print(f"Left hemisphere:  {int(np.count_nonzero(left_mask))} voxels")
    print(f"Right hemisphere: {int(np.count_nonzero(right_mask))} voxels")
    print()

    # Reconstruct falx cerebri
    falx_mask = reconstruct_falx(
        mat, left_mask, right_mask, cc_superior_z,
        dx_mm, args.watershed_threshold,
    )
    del left_mask, right_mask, cc_superior_z

    # Reconstruct tentorium cerebelli
    tent_mask = reconstruct_tentorium(
        mat, dx_mm, args.watershed_threshold, args.notch_radius,
    )

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


if __name__ == "__main__":
    main()
