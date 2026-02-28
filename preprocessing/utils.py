"""Shared utilities for the preprocessing pipeline.

Provides grid construction, slab-based resampling, structuring elements,
profile configs, and path helpers used by all preprocessing steps.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

# ---------------------------------------------------------------------------
# Profile configs: name -> (N, dx_mm)
# ---------------------------------------------------------------------------
PROFILES = {"debug": (128, 2.0), "dev": (256, 1.0), "prod": (512, 0.5)}

# ---------------------------------------------------------------------------
# FreeSurfer LUT size: covers FS labels 0..2035 (max = ctx-rh-insula = 2035)
# ---------------------------------------------------------------------------
FS_LUT_SIZE = 2036


# ---------------------------------------------------------------------------
# Common CLI argument helpers
# ---------------------------------------------------------------------------
def add_grid_args(parser):
    """Add --subject, --profile, --dx, --grid-size to an argparse parser."""
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


def resolve_grid_args(args, parser):
    """Resolve N, dx, profile from parsed --profile/--dx/--grid-size.

    Modifies args in place, setting args.N, args.dx, and args.profile.
    """
    if args.profile is None and args.dx is None:
        args.profile = "debug"

    if args.profile is not None:
        args.N, args.dx = PROFILES[args.profile]
    else:
        if args.grid_size is None:
            parser.error("--grid-size is required when using --dx")
        args.N = args.grid_size
        args.profile = f"custom_{args.N}_{args.dx}"


# ---------------------------------------------------------------------------
# Grid affine construction
# ---------------------------------------------------------------------------
def build_grid_affine(N, dx_mm):
    """Build 4x4 grid-to-physical affine (RAS+, ACPC-centered).

    Grid center (N/2, N/2, N/2) maps to physical (0, 0, 0).
    """
    affine = np.eye(4)
    affine[0, 0] = dx_mm
    affine[1, 1] = dx_mm
    affine[2, 2] = dx_mm
    affine[0, 3] = -N / 2 * dx_mm
    affine[1, 3] = -N / 2 * dx_mm
    affine[2, 3] = -N / 2 * dx_mm
    return affine


# ---------------------------------------------------------------------------
# Slab-based resampling
# ---------------------------------------------------------------------------
def resample_to_grid(source, grid_affine, grid_shape, order=0, cval=0.0,
                     dtype=None):
    """Resample a volume onto the simulation grid.

    Uses scipy.ndimage.affine_transform (single C call) for the
    composite transform: grid voxel → source voxel.

    Parameters
    ----------
    source : str, Path, or (ndarray, affine) tuple
        Either a path to a NIfTI file, or a (data, 4x4_affine) tuple.
    grid_affine : ndarray (4, 4)
        Grid-to-physical affine (A_g->p).
    grid_shape : tuple of int
        Output shape (Ni, Nj, Nk).
    order : int
        Interpolation order (0=nearest, 1=trilinear).
    cval : float
        Fill value for out-of-bounds voxels.
    dtype : numpy dtype or None
        Output dtype. If None, inferred from source.

    Returns
    -------
    ndarray of shape grid_shape with the requested dtype.
    """
    # --- Load source data and affine ---
    if isinstance(source, (str, Path)):
        img = nib.load(str(source))
        source_data = img.get_fdata(dtype=np.float32)
        source_affine = img.affine
    else:
        source_data, source_affine = source

    if dtype is None:
        dtype = source_data.dtype

    target_dtype = np.dtype(dtype)
    round_int = order == 0 and np.issubdtype(target_dtype, np.integer)

    # Composite transform: grid voxel -> source voxel
    M = np.linalg.inv(source_affine) @ grid_affine

    out = affine_transform(
        source_data, M[:3, :3], M[:3, 3],
        output_shape=grid_shape, order=order,
        mode='constant', cval=float(cval),
    )

    if round_int:
        np.round(out, out=out)
    return out.astype(target_dtype)


# ---------------------------------------------------------------------------
# Spherical structuring element
# ---------------------------------------------------------------------------
def build_ball(radius_vox):
    """Build a spherical boolean structuring element.

    Returns an array of shape (2*r+1,) per axis, True where
    Euclidean distance from center <= radius_vox.  The half-width r
    is ceil(radius_vox) so that fractional radii are not silently
    truncated (e.g. 1.5 produces a diameter-5 kernel, not diameter-3).
    """
    r = int(np.ceil(radius_vox))
    diameter = 2 * r + 1
    ax = np.arange(diameter) - r
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    return (x * x + y * y + z * z) <= radius_vox * radius_vox


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def raw_dir(subject_id):
    """Return absolute path to raw HCP data for a subject."""
    return _PROJECT_ROOT / "data" / "raw" / subject_id / "T1w"


def processed_dir(subject_id, profile):
    """Return absolute path to processed output for a subject/profile."""
    return _PROJECT_ROOT / "data" / "processed" / subject_id / profile


def validation_dir(subject_id):
    """Return absolute path to validation ground-truth data for a subject."""
    return _PROJECT_ROOT / "data" / "validation" / subject_id
