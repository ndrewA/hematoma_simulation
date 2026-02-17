"""Shared utilities for the preprocessing pipeline.

Provides grid construction, slab-based resampling, structuring elements,
profile configs, and path helpers used by all preprocessing steps.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

# ---------------------------------------------------------------------------
# Profile configs: name -> (N, dx_mm)
# ---------------------------------------------------------------------------
PROFILES = {"debug": (256, 2.0), "dev": (512, 1.0), "prod": (512, 0.5)}


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
                     dtype=None, slab_size=32):
    """Resample a volume onto the simulation grid using slab-based processing.

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
    slab_size : int
        Number of slices per slab along axis 0.

    Returns
    -------
    ndarray of shape grid_shape with the requested dtype.
    """
    # --- Load source data and affine ---
    if isinstance(source, (str, Path)):
        img = nib.load(str(source))
        source_data = img.get_fdata()
        source_affine = img.affine
    else:
        source_data, source_affine = source

    if dtype is None:
        dtype = source_data.dtype

    target_dtype = np.dtype(dtype)
    round_int = order == 0 and np.issubdtype(target_dtype, np.integer)

    # --- Composite transform: grid voxel -> source voxel ---
    M = np.linalg.inv(source_affine) @ grid_affine

    Ni, Nj, Nk = grid_shape
    out = np.empty(grid_shape, dtype=target_dtype)

    for i_start in range(0, Ni, slab_size):
        i_end = min(i_start + slab_size, Ni)

        # Build grid coordinates for this slab
        ii = np.arange(i_start, i_end, dtype=np.float64)
        jj = np.arange(Nj, dtype=np.float64)
        kk = np.arange(Nk, dtype=np.float64)
        gi, gj, gk = np.meshgrid(ii, jj, kk, indexing='ij')

        # Apply affine: source_coords = M @ [i, j, k, 1]^T
        si = M[0, 0] * gi + M[0, 1] * gj + M[0, 2] * gk + M[0, 3]
        sj = M[1, 0] * gi + M[1, 1] * gj + M[1, 2] * gk + M[1, 3]
        sk = M[2, 0] * gi + M[2, 1] * gj + M[2, 2] * gk + M[2, 3]

        del gi, gj, gk

        coords = np.array([si, sj, sk])
        del si, sj, sk

        slab = map_coordinates(source_data, coords, order=order,
                               mode='constant', cval=cval)
        del coords

        if round_int:
            np.round(slab, out=slab)
        out[i_start:i_end] = slab.astype(target_dtype)
        del slab

    return out


# ---------------------------------------------------------------------------
# Spherical structuring element
# ---------------------------------------------------------------------------
def build_ball(radius_vox):
    """Build a spherical boolean structuring element.

    Returns an array of shape (2*radius_vox+1,) per axis, True where
    Euclidean distance from center <= radius_vox.
    """
    r = int(radius_vox)
    diameter = 2 * r + 1
    ax = np.arange(diameter) - r
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    return (x * x + y * y + z * z) <= r * r


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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("preprocessing/utils.py  —  self-test")
    print("=" * 60)

    # 1. PROFILES
    assert PROFILES["debug"] == (256, 2.0)
    assert PROFILES["dev"] == (512, 1.0)
    assert PROFILES["prod"] == (512, 0.5)
    print("[PASS] PROFILES values correct")

    # 2. Grid affines — center maps to physical (0, 0, 0)
    for name, (N, dx) in PROFILES.items():
        A = build_grid_affine(N, dx)
        center = np.array([N / 2, N / 2, N / 2, 1.0])
        phys = A @ center
        assert np.allclose(phys[:3], 0.0), f"{name}: center -> {phys[:3]}"
        # Diagonal positive (RAS+)
        assert A[0, 0] > 0 and A[1, 1] > 0 and A[2, 2] > 0, \
            f"{name}: diagonal not positive"
        print(f"[PASS] {name:5s}  N={N}, dx={dx}  center->(0,0,0), RAS+")

    # 3. build_ball
    ball = build_ball(3)
    assert ball.shape == (7, 7, 7), f"ball shape {ball.shape}"
    n_true = int(ball.sum())
    assert 100 <= n_true <= 150, f"ball voxels {n_true}"
    assert ball[3, 3, 3], "center must be True"
    assert not ball[0, 0, 0], "corner must be False"
    assert not ball[6, 6, 6], "corner must be False"
    print(f"[PASS] build_ball(3): shape (7,7,7), {n_true} True voxels, "
          "center True, corners False")

    # 4. Path helpers
    assert raw_dir("157336") == _PROJECT_ROOT / "data/raw/157336/T1w"
    assert processed_dir("157336", "dev") == _PROJECT_ROOT / "data/processed/157336/dev"
    assert raw_dir("157336").is_absolute(), "raw_dir should return absolute path"
    assert processed_dir("157336", "dev").is_absolute(), "processed_dir should return absolute path"
    print(f"[PASS] path helpers (root={_PROJECT_ROOT})")

    # 5. resample_to_grid — identity transform (small synthetic volume)
    print("\n--- resample_to_grid tests ---")

    # 5a. Identity: source and grid share the same affine
    src_data = np.arange(8 * 8 * 8, dtype=np.float64).reshape(8, 8, 8)
    src_affine = np.eye(4)
    grid_affine_id = np.eye(4)
    result = resample_to_grid((src_data, src_affine), grid_affine_id,
                              (8, 8, 8), order=1)
    assert np.allclose(result, src_data), "identity resample mismatch"
    print("[PASS] identity resample (float64, order=1)")

    # 5b. Identity with integer dtype + order=0
    src_int = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int16)
    src_aff = np.eye(4)
    result_int = resample_to_grid((src_int, src_aff), np.eye(4),
                                  (2, 2, 2), order=0, dtype=np.int16)
    assert result_int.dtype == np.int16
    assert np.array_equal(result_int, src_int), "int16 identity mismatch"
    print("[PASS] identity resample (int16, order=0)")

    # 5c. Shifted transform — grid shifted by 1 voxel in all axes
    shifted_affine = np.eye(4)
    shifted_affine[:3, 3] = [1.0, 1.0, 1.0]  # grid origin shifted +1mm
    result_shift = resample_to_grid((src_data, src_affine), shifted_affine,
                                    (8, 8, 8), order=0, cval=-1.0,
                                    dtype=np.float64)
    # Grid voxel (0,0,0) -> physical (1,1,1) -> source voxel (1,1,1)
    assert result_shift[0, 0, 0] == src_data[1, 1, 1], \
        f"shifted (0,0,0): got {result_shift[0, 0, 0]}, expected {src_data[1, 1, 1]}"
    # Grid voxel (7,7,7) -> physical (8,8,8) -> source voxel (8,8,8) = OOB -> cval
    assert result_shift[7, 7, 7] == -1.0, \
        f"shifted OOB: got {result_shift[7, 7, 7]}"
    print("[PASS] shifted resample (order=0, cval=-1.0)")

    # 5d. Verify mode='constant' — OOB should NOT reflect
    src_3 = np.zeros((4, 4, 4), dtype=np.float64)
    src_3[0, 0, 0] = 99.0  # value at corner
    big_shift = np.eye(4)
    big_shift[:3, 3] = [10.0, 10.0, 10.0]  # shift grid far outside source
    result_oob = resample_to_grid((src_3, np.eye(4)), big_shift,
                                  (4, 4, 4), order=0, cval=0.0,
                                  dtype=np.float64)
    assert np.all(result_oob == 0.0), \
        "OOB voxels should be 0 (mode='constant'), got non-zero — reflect bug?"
    print("[PASS] out-of-bounds uses mode='constant' (no reflection)")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
