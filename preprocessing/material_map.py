"""Remap FreeSurfer aparc+aseg labels to 12 simulation material classes.

Loads fs_labels_resampled.nii.gz (int16, produced by resample_sources.py),
applies a lookup-table mapping from ~120 FS labels to 12 u8 material classes
(0–11), and saves material_map.nii.gz (uint8).
"""

import argparse
import sys

import nibabel as nib
import numpy as np

from preprocessing.utils import (
    FS_LUT_SIZE,
    add_grid_args,
    processed_dir,
    resolve_grid_args,
)


# ---------------------------------------------------------------------------
# Material class names (u8 index → human-readable)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "Vacuum",
    1: "Cerebral White Matter",
    2: "Cortical Gray Matter",
    3: "Deep Gray Matter",
    4: "Cerebellar White Matter",
    5: "Cerebellar Cortex",
    6: "Brainstem",
    7: "Ventricular CSF",
    8: "Subarachnoid CSF",
    9: "Choroid Plexus",
    10: "Dural Membrane",
    11: "Vessel / Venous Sinus",
    255: "Air Halo",
}


# ---------------------------------------------------------------------------
# Direct mapping: FS label → u8 class
# ---------------------------------------------------------------------------
DIRECT_MAP = {
    # u8 = 0: Vacuum
    0: 0,
    # u8 = 1: Cerebral White Matter
    2: 1, 41: 1,
    77: 1, 78: 1, 79: 1,                       # WM hypointensities
    85: 1,                                       # Optic-Chiasm
    192: 1,                                      # Corpus_Callosum
    250: 1,                                      # Fornix
    251: 1, 252: 1, 253: 1, 254: 1, 255: 1,     # CC sub-regions
    # u8 = 2: Cortical Gray Matter
    3: 2, 42: 2,
    19: 2, 55: 2,                               # Insula
    20: 2, 56: 2,                               # Operculum
    80: 2, 81: 2, 82: 2,                        # non-WM hypointensities
    # u8 = 3: Deep Gray Matter
    10: 3, 49: 3,                               # Thalamus
    11: 3, 50: 3,                               # Caudate
    12: 3, 51: 3,                               # Putamen
    13: 3, 52: 3,                               # Pallidum
    17: 3, 53: 3,                               # Hippocampus
    18: 3, 54: 3,                               # Amygdala
    26: 3, 58: 3,                               # Accumbens
    27: 3, 59: 3,                               # Substancia-Nigra
    28: 3, 60: 3,                               # VentralDC
    # u8 = 4: Cerebellar White Matter
    7: 4, 46: 4,
    # u8 = 5: Cerebellar Cortex
    8: 5, 47: 5,
    # u8 = 6: Brainstem
    16: 6, 75: 6, 76: 6,
    # u8 = 7: Ventricular CSF
    4: 7, 43: 7,                                # Lateral ventricles
    5: 7, 44: 7,                                # Inf-Lat-Vent
    14: 7, 15: 7,                               # 3rd / 4th ventricle
    72: 7,                                       # 5th ventricle
    # u8 = 8: Subarachnoid CSF
    24: 8,
    # u8 = 9: Choroid Plexus
    31: 9, 63: 9,
    # u8 = 11: Vessel / Venous Sinus
    30: 11, 62: 11,
}

# Cortical parcellation ranges (Desikan-Killiany atlas)
DIRECT_MAP.update({fs: 2 for fs in range(1001, 1036)})   # ctx-lh-*
DIRECT_MAP.update({fs: 2 for fs in range(2001, 2036)})   # ctx-rh-*


# ---------------------------------------------------------------------------
# Fallback mapping: rare/deprecated FS labels → u8 class (warns on hit)
# ---------------------------------------------------------------------------
FALLBACK_MAP = {
    1: 2, 40: 2,                                # Cerebral Exterior → cortex
    6: 5, 45: 5,                                # Cerebellum Exterior → cerebellar cortex
    9: 3, 48: 3,                                # Thalamus-unused → deep GM
    21: 0, 22: 0, 23: 0,                        # Line placeholders → vacuum
    25: 2, 57: 2,                                # Lesion → cortex
    29: 2, 61: 2,                                # Undetermined → cortex
    73: 2, 74: 2,                                # Interior → cortex
    83: 2, 84: 2,                                # F1 → cortex
}
FALLBACK_MAP.update({fs: 2 for fs in range(32, 40)})     # Left gyral labels
FALLBACK_MAP.update({fs: 2 for fs in range(64, 72)})      # Right gyral labels


# ---------------------------------------------------------------------------
# LUT parameters
# ---------------------------------------------------------------------------
_SENTINEL = 128       # marks unmapped LUT slots (must be outside 0-11 and 255)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def build_lut():
    """Build the FS label → u8 material lookup table.

    Returns
    -------
    lut : ndarray, uint8, shape (FS_LUT_SIZE,)
    direct_labels : set of int
        The FS labels covered by DIRECT_MAP (for warning logic).
    """
    lut = np.full(FS_LUT_SIZE, _SENTINEL, dtype=np.uint8)
    for fs_label, u8_class in FALLBACK_MAP.items():
        if fs_label < FS_LUT_SIZE:
            lut[fs_label] = u8_class
    # DIRECT_MAP applied second so it always wins over FALLBACK_MAP
    for fs_label, u8_class in DIRECT_MAP.items():
        if fs_label < FS_LUT_SIZE:
            lut[fs_label] = u8_class
    return lut, set(DIRECT_MAP.keys())


def apply_mapping(fs_labels, lut):
    """Apply LUT to remap FS labels to u8 material classes.

    Parameters
    ----------
    fs_labels : ndarray, int16
    lut : ndarray, uint8, shape (FS_LUT_SIZE,)

    Returns
    -------
    material : ndarray, uint8, same shape as fs_labels
    """
    # Clamp negatives to 0
    if np.any(fs_labels < 0):
        n_neg = int(np.count_nonzero(fs_labels < 0))
        print(f"WARNING: {n_neg} voxels have negative FS labels; clamping to 0")

    # Out-of-range mask (labels >= FS_LUT_SIZE)
    oor_mask = fs_labels >= FS_LUT_SIZE

    # Safe indexing: clip to valid LUT range
    labels_safe = np.clip(fs_labels, 0, FS_LUT_SIZE - 1)
    material = lut[labels_safe]

    # Fix out-of-range voxels: cortical GM fallback
    material[oor_mask] = 2

    # Fix sentinel hits (unmapped in-range labels): cortical GM fallback
    sentinel_mask = material == _SENTINEL
    material[sentinel_mask] = 2

    return material


def collect_warnings(fs_labels, direct_labels):
    """Scan for fallback and unknown FS labels, print warnings.

    Parameters
    ----------
    fs_labels : ndarray, int16
    direct_labels : set of int
    """
    unique_labels, label_counts = np.unique(fs_labels, return_counts=True)
    counts_by_label = dict(zip(unique_labels.tolist(), label_counts.tolist()))
    fallback_labels = set(FALLBACK_MAP.keys())

    fallback_hits = []
    unknown_hits = []

    for lab in unique_labels:
        lab = int(lab)
        if lab in direct_labels:
            continue
        if lab < 0:
            continue  # already warned in apply_mapping; mapped to u8=0 via clamp
        count = counts_by_label[lab]
        if lab in fallback_labels:
            fallback_hits.append((lab, count, FALLBACK_MAP[lab]))
        else:
            # Unknown: mapped to u8=2 by sentinel/oor fallback
            unknown_hits.append((lab, count, 2))

    if fallback_hits:
        print(f"\nFallback labels encountered ({len(fallback_hits)} types):")
        for lab, count, u8 in sorted(fallback_hits):
            print(f"  FS {lab:5d} → u8={u8} ({CLASS_NAMES[u8]}):  {count} voxels")

    if unknown_hits:
        print(f"\nUnknown labels (not in any map) ({len(unknown_hits)} types):")
        for lab, count, u8 in sorted(unknown_hits):
            print(f"  FS {lab:5d} → u8={u8} (fallback):  {count} voxels")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse CLI arguments for material_map."""
    parser = argparse.ArgumentParser(
        description="Remap FreeSurfer labels to simulation material classes."
    )
    add_grid_args(parser)
    args = parser.parse_args(argv)
    resolve_grid_args(args, parser)
    return args


def load_fs_labels(out_dir):
    """Load fs_labels_resampled.nii.gz from out_dir, cast to int16.

    Returns
    -------
    fs_labels : ndarray, int16
    affine : ndarray (4, 4)
    """
    path = out_dir / "fs_labels_resampled.nii.gz"
    print(f"Loading {path}")
    img = nib.load(str(path))
    fs_labels = np.asarray(img.dataobj, dtype=np.int16)
    affine = img.affine.copy()
    return fs_labels, affine


def save_material_map(out_dir, material, affine):
    """Save material_map.nii.gz as uint8."""
    img = nib.Nifti1Image(material, affine)
    img.header.set_data_dtype(np.uint8)
    path = out_dir / "material_map.nii.gz"
    nib.save(img, str(path))
    print(f"Saved {path}  shape={material.shape}  dtype={material.dtype}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def print_census(material, dx):
    """Print voxel count and volume per material class."""
    voxel_vol_ml = dx ** 3 / 1000.0

    print("\n" + "=" * 60)
    print("Material Census")
    print("=" * 60)
    print(f"{'u8':>3s}  {'Class':<28s}  {'Voxels':>10s}  {'Volume (mL)':>11s}")
    print("-" * 60)

    counts = np.bincount(material.ravel(), minlength=12)

    total_voxels = 0
    total_tissue_ml = 0.0

    for u8 in range(12):
        count = int(counts[u8])
        vol_ml = count * voxel_vol_ml
        name = CLASS_NAMES.get(u8, "???")
        print(f"{u8:3d}  {name:<28s}  {count:10d}  {vol_ml:11.1f}")
        total_voxels += count
        if u8 >= 1:  # non-vacuum tissue
            total_tissue_ml += vol_ml

    print("-" * 60)
    print(f"{'':3s}  {'Total (all classes)':<28s}  {total_voxels:10d}  "
          f"{total_voxels * voxel_vol_ml:11.1f}")
    print(f"{'':3s}  {'Total tissue (u8 1–11)':<28s}  {'':>10s}  "
          f"{total_tissue_ml:11.1f}")


def print_validation(material):
    """Run per-step validation checks M1–M3."""
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    unique_vals = set(int(v) for v in np.unique(material))

    # M1: all unique values in {0..11}
    valid_range = set(range(12))
    out_of_range = unique_vals - valid_range
    if out_of_range:
        print(f"\nM1 [CRITICAL FAIL]: values outside {{0..11}}: {sorted(out_of_range)}")
        sys.exit(1)
    else:
        print(f"\nM1 [OK]: All unique values in {{0..11}}: {sorted(unique_vals)}")

    # M2: 255 not present
    if 255 in unique_vals:
        print("M2 [CRITICAL FAIL]: u8=255 found in material map")
        sys.exit(1)
    else:
        print("M2 [OK]: u8=255 not present")

    # M3: report missing classes
    missing = valid_range - unique_vals
    if missing:
        print(f"M3 [INFO]: Missing classes: {sorted(missing)}")
        for m in sorted(missing):
            print(f"   u8={m} ({CLASS_NAMES.get(m, '???')}) — 0 voxels")
    else:
        print("M3 [INFO]: All 12 classes present")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    """Orchestrate material map creation."""
    args = parse_args(argv)

    print(f"Subject: {args.subject}")
    print(f"Profile: {args.profile}  (N={args.N}, dx={args.dx} mm)")
    print()

    # Load FS labels
    out_dir = processed_dir(args.subject, args.profile)
    fs_labels, affine = load_fs_labels(out_dir)
    print(f"Shape: {fs_labels.shape}  dtype: {fs_labels.dtype}")
    print(f"Unique FS labels: {len(np.unique(fs_labels))}")
    print()

    # Build LUT and apply mapping
    lut, direct_labels = build_lut()
    material = apply_mapping(fs_labels, lut)

    # Warnings
    collect_warnings(fs_labels, direct_labels)

    # Census
    print_census(material, args.dx)

    # Save
    print()
    save_material_map(out_dir, material, affine)

    # Validation
    print_validation(material)


if __name__ == "__main__":
    main()
