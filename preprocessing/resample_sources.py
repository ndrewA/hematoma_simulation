"""Resample source NIfTI volumes onto the simulation grid.

Loads aparc+aseg.nii.gz and brainmask_fs.nii.gz from HCP T1w space,
resamples both onto the simulation grid via nearest-neighbor interpolation,
computes brain bounding box metadata, and saves outputs:
  - fs_labels_resampled.nii.gz  (int16)
  - brain_mask.nii.gz           (uint8)
  - grid_meta.json
"""

import argparse
import json
import sys

import nibabel as nib
import numpy as np

from preprocessing.profiling import step
from preprocessing.utils import (
    add_grid_args,
    build_grid_affine,
    processed_dir,
    raw_dir,
    resample_to_grid,
    resolve_grid_args,
)


# Critical FreeSurfer labels that should survive resampling
CRITICAL_LABELS = [2, 41, 3, 42, 4, 43, 10, 49, 12, 51, 16, 31, 63]


def parse_args(argv=None):
    """Parse CLI arguments for resample_sources."""
    parser = argparse.ArgumentParser(
        description="Resample HCP source volumes onto the simulation grid."
    )
    add_grid_args(parser)
    args = parser.parse_args(argv)
    resolve_grid_args(args, parser)
    return args


def load_source_volumes(raw):
    """Load and dtype-cast aparc+aseg and brainmask_fs from raw directory.

    Returns
    -------
    labels_data : ndarray, int16
    mask_data : ndarray, uint8
    source_affine : ndarray (4, 4)
    """
    labels_path = raw / "aparc+aseg.nii.gz"
    mask_path = raw / "brainmask_fs.nii.gz"

    print(f"Loading {labels_path}")
    labels_img = nib.load(str(labels_path))
    labels_data = np.round(labels_img.get_fdata()).astype(np.int16)
    source_affine = labels_img.affine.copy()

    print(f"Loading {mask_path}")
    mask_img = nib.load(str(mask_path))
    mask_data = (mask_img.get_fdata() > 0.5).astype(np.uint8)

    return labels_data, mask_data, source_affine


def verify_source_affine(source_affine):
    """Check that the source affine diagonal is approximately (-0.7, +0.7, +0.7)."""
    diag = np.diag(source_affine)[:3]
    expected = np.array([-0.7, 0.7, 0.7])
    if not np.allclose(diag, expected, atol=0.05):
        print(f"WARNING: source affine diagonal {diag} != expected {expected}")
    else:
        print(f"Source affine diagonal: {diag}  [OK]")


def compute_brain_bbox(mask):
    """Compute bounding box, volume, and centroid from resampled brain mask.

    Returns
    -------
    bbox_min, bbox_max, centroid : ndarray (each shape (3,))
    n_voxels : int
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        print("FATAL: resampled brain mask has zero nonzero voxels.")
        sys.exit(1)

    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    centroid = coords.mean(axis=0)
    return bbox_min, bbox_max, centroid, len(coords)


def build_grid_meta(args, grid_affine, source_affine, source_shape,
                    bbox_min, bbox_max, centroid, n_voxels):
    """Assemble the grid_meta.json dictionary."""
    N, dx = args.N, args.dx
    phys_to_grid = np.linalg.inv(grid_affine)
    source_voxel_mm = float(abs(source_affine[1, 1]))
    brain_volume_ml = float(n_voxels * dx ** 3 / 1000.0)

    meta = {
        "subject_id": args.subject,
        "profile": args.profile,
        "grid_size": int(N),
        "dx_mm": float(dx),
        "domain_extent_mm": float(N * dx),
        "affine_grid_to_phys": grid_affine.tolist(),
        "affine_phys_to_grid": phys_to_grid.tolist(),
        "source_shape": [int(s) for s in source_shape],
        "source_voxel_mm": source_voxel_mm,
        "source_affine": source_affine.tolist(),
        "brain_bbox_grid": {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist(),
        },
        "brain_volume_ml": round(brain_volume_ml, 1),
        "brain_centroid_grid": [round(float(c), 1) for c in centroid],
    }
    return meta


def save_outputs(out_dir, grid_affine, labels_resampled, mask_resampled, meta):
    """Save resampled volumes and metadata to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # fs_labels_resampled.nii.gz
    labels_img = nib.Nifti1Image(labels_resampled, grid_affine)
    labels_img.header.set_data_dtype(np.int16)
    labels_path = out_dir / "fs_labels_resampled.nii.gz"
    nib.save(labels_img, str(labels_path))
    print(f"Saved {labels_path}  shape={labels_resampled.shape}  dtype={labels_resampled.dtype}")

    # brain_mask.nii.gz
    mask_img = nib.Nifti1Image(mask_resampled, grid_affine)
    mask_img.header.set_data_dtype(np.uint8)
    mask_path = out_dir / "brain_mask.nii.gz"
    nib.save(mask_img, str(mask_path))
    print(f"Saved {mask_path}  shape={mask_resampled.shape}  dtype={mask_resampled.dtype}")

    # grid_meta.json
    meta_path = out_dir / "grid_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")


def print_validation(source_unique_labels, n_source_mask_voxels,
                     labels_resampled, mask_resampled,
                     source_affine, args, bbox_min, bbox_max, centroid):
    """Print the four validation checks from spec Section 5.4."""
    N, dx = args.N, args.dx
    source_voxel_mm = float(abs(source_affine[1, 1]))

    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    # 1. Volume conservation (mask vs mask)
    n_resampled = int(np.count_nonzero(mask_resampled > 0))
    vol_source_ml = n_source_mask_voxels * source_voxel_mm ** 3 / 1000.0
    vol_resampled_ml = n_resampled * dx ** 3 / 1000.0
    if vol_source_ml > 0:
        pct_diff = abs(vol_resampled_ml - vol_source_ml) / vol_source_ml * 100
    else:
        pct_diff = float("inf")
    status = "OK" if pct_diff < 3.0 else "WARN"
    print(f"\n1. Volume conservation:")
    print(f"   Source:    {vol_source_ml:.1f} mL  ({n_source_mask_voxels} voxels × {source_voxel_mm:.2f}³ mm³)")
    print(f"   Resampled: {vol_resampled_ml:.1f} mL  ({n_resampled} voxels × {dx:.2f}³ mm³)")
    print(f"   Difference: {pct_diff:.1f}%  [{status}]")

    # 2. Label preservation
    source_labels = set(source_unique_labels)
    resampled_labels = set(np.unique(labels_resampled))
    invented = resampled_labels - source_labels
    print(f"\n2. Label preservation:")
    print(f"   Source unique labels:    {len(source_labels)}")
    print(f"   Resampled unique labels: {len(resampled_labels)}")
    if invented:
        print(f"   WARNING: invented labels: {sorted(invented)}")
    else:
        print(f"   No invented labels  [OK]")

    # Check critical labels (only those present in source)
    missing_critical = []
    for lab in CRITICAL_LABELS:
        if lab in source_labels and lab not in resampled_labels:
            missing_critical.append(lab)
    if missing_critical:
        print(f"   WARNING: critical labels missing from resampled: {missing_critical}")
    else:
        present = [lab for lab in CRITICAL_LABELS if lab in source_labels]
        print(f"   Critical labels present: {present}  [OK]")

    # 3. Centering
    grid_center = np.array([N / 2.0, N / 2.0, N / 2.0])
    offset_vox = centroid - grid_center
    offset_mm = offset_vox * dx
    print(f"\n3. Centering:")
    print(f"   Brain centroid (grid): ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
    print(f"   Grid center:          ({grid_center[0]:.1f}, {grid_center[1]:.1f}, {grid_center[2]:.1f})")
    print(f"   Offset (voxels):      ({offset_vox[0]:.1f}, {offset_vox[1]:.1f}, {offset_vox[2]:.1f})")
    print(f"   Offset (mm):          ({offset_mm[0]:.1f}, {offset_mm[1]:.1f}, {offset_mm[2]:.1f})")

    # 4. Bbox margins
    print(f"\n4. Bounding box margins:")
    axis_names = ["X (L-R)", "Y (P-A)", "Z (I-S)"]
    any_tight = False
    for ax in range(3):
        margin_low = int(bbox_min[ax]) * dx
        margin_high = (N - 1 - int(bbox_max[ax])) * dx
        flag_low = " *** <30mm" if margin_low < 30 else ""
        flag_high = " *** <30mm" if margin_high < 30 else ""
        if margin_low < 30 or margin_high < 30:
            any_tight = True
        print(f"   {axis_names[ax]:10s}  low: {margin_low:6.1f} mm{flag_low}  "
              f"high: {margin_high:6.1f} mm{flag_high}")
    if not any_tight:
        print("   All margins >= 30 mm  [OK]")


def main(argv=None):
    """Orchestrate source volume resampling."""
    args = parse_args(argv)
    N, dx = args.N, args.dx
    grid_shape = (N, N, N)

    print(f"Subject: {args.subject}")
    print(f"Profile: {args.profile}  (N={N}, dx={dx} mm)")
    print(f"Domain extent: {N * dx:.0f} mm")
    print()

    # Load source volumes
    raw = raw_dir(args.subject)
    labels_data, mask_data, source_affine = load_source_volumes(raw)
    source_shape = labels_data.shape
    verify_source_affine(source_affine)
    print()

    # Precompute source stats for validation (before freeing arrays)
    source_unique_labels = np.unique(labels_data)
    n_source_mask_voxels = int(np.count_nonzero(mask_data))

    # Build grid affine
    grid_affine = build_grid_affine(N, dx)

    # Resample labels
    with step("resample aparc+aseg"):
        print("Resampling labels (nearest-neighbor)...")
        labels_resampled = resample_to_grid(
            (labels_data, source_affine), grid_affine, grid_shape,
            order=0, cval=0, dtype=np.int16,
        )
        del labels_data  # free memory before next resample

    # Resample mask
    with step("resample brainmask"):
        print("Resampling brain mask (nearest-neighbor)...")
        mask_resampled = resample_to_grid(
            (mask_data, source_affine), grid_affine, grid_shape,
            order=0, cval=0, dtype=np.uint8,
        )
        del mask_data

    # Compute brain bounding box
    bbox_min, bbox_max, centroid, n_voxels = compute_brain_bbox(mask_resampled)
    print(f"Brain bbox: min={bbox_min.tolist()}, max={bbox_max.tolist()}")
    print(f"Brain voxels: {n_voxels}  volume: {n_voxels * dx**3 / 1000:.1f} mL")

    # Build metadata
    meta = build_grid_meta(
        args, grid_affine, source_affine, source_shape,
        bbox_min, bbox_max, centroid, n_voxels,
    )

    # Save outputs
    out_dir = processed_dir(args.subject, args.profile)
    print()
    save_outputs(out_dir, grid_affine, labels_resampled, mask_resampled, meta)

    # Validation
    print_validation(
        source_unique_labels, n_source_mask_voxels,
        labels_resampled, mask_resampled,
        source_affine, args, bbox_min, bbox_max, centroid,
    )


if __name__ == "__main__":
    main()
