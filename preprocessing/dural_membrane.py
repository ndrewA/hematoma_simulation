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
from concurrent.futures import ThreadPoolExecutor

import edt as _edt
import nibabel as nib
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    label as cc_label,
)
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line

from preprocessing.profiling import step
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

# Kayalioglu falx height targets at CC landmarks (mm)
_H_GENU = 21.3
_H_BODY = 25.7
_H_SPLENIUM = 45.6

# FreeSurfer CC sub-region labels
_CC_GENU = 255
_CC_BODY = 253
_CC_SPLENIUM = 251

# Kayalioglu Type I ratios: falx_height / (falx_height + FC-CC gap)
_RATIO_GENU = _H_GENU / (_H_GENU + 14.1)
_RATIO_BODY = _H_BODY / (_H_BODY + 12.4)
_RATIO_SPLENIUM = _H_SPLENIUM / (_H_SPLENIUM + 2.1)

# Frassanito 2020 dimensionless ratios (3D CT reconstruction, n=40)
_NOTCH_AREA_RATIO = 28.8 / 56.5
_NOTCH_ASPECT_RATIO = 41.8 / 96.9

# Tentorial notch aspect ratio (length / width)
# Adler 2002 (n=100): NL 54-62mm / MNW 27-32mm ≈ 1.8
# Arrambide-Garza 2022 (n=60): NL 55.6mm / MNW 31.3mm ≈ 1.78
_TENT_NOTCH_AR = 1.8


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
# Surface-map construction
# ---------------------------------------------------------------------------
def _extract_membrane(phi, eligible, t_target_mm):
    """Extract a dural membrane surface with target physical thickness.

    1. Finds the 1-voxel-thick sign-change surface (pick-closer) as a
       barrier floor — guarantees every sign-change face has at least one
       marked voxel, sufficient for the face-centered finite-volume solver.
    2. Thickens by including all eligible voxels where |phi| < t_target_mm.
       Since |phi| ≈ 2 × distance_from_midplane, this selects a slab of
       approximately t_target_mm total thickness centred on the zero-level-set.

    At coarse grids the sign-change base dominates (1 voxel).  At fine grids
    the |phi| threshold adds voxels, naturally approaching the target.

    Parameters
    ----------
    phi : ndarray, 3-D float
        Signed distance field (dist_A - dist_B).
    eligible : ndarray, 3-D bool
        Non-vacuum mask. Only eligible voxels are marked.
    t_target_mm : float
        Target membrane thickness in mm.

    Returns
    -------
    surface : ndarray, 3-D bool
        The membrane mask.
    """
    # Base: 1-voxel sign-change surface (barrier guarantee)
    positive = phi >= 0
    abs_phi = np.abs(phi)
    surface = np.zeros(phi.shape, dtype=bool)

    for ax in range(3):
        lo = [slice(None)] * 3; hi = [slice(None)] * 3
        lo[ax] = slice(None, -1); hi[ax] = slice(1, None)

        sign_flip = positive[tuple(lo)] != positive[tuple(hi)]
        both_elig = eligible[tuple(lo)] & eligible[tuple(hi)]
        faces = sign_flip & both_elig

        s_lo = [slice(None)] * 3; s_lo[ax] = slice(None, -1)
        s_hi = [slice(None)] * 3; s_hi[ax] = slice(1, None)

        lo_closer = faces & (abs_phi[tuple(lo)] <= abs_phi[tuple(hi)])
        hi_closer = faces & (abs_phi[tuple(lo)] > abs_phi[tuple(hi)])

        surface[tuple(s_lo)] |= lo_closer
        surface[tuple(s_hi)] |= hi_closer

    # Thicken: include eligible voxels within t_target of the midplane
    surface |= (abs_phi < t_target_mm) & eligible

    return surface



def _get_edges(proj, Y, Z):
    """Get top/bottom z for each y column of a 2D projection."""
    top = np.full(Y, -1, dtype=int)
    bot = np.full(Y, Z, dtype=int)
    for y in range(Y):
        zz = np.where(proj[y])[0]
        if len(zz) > 0:
            top[y] = zz.max()
            bot[y] = zz.min()
    exists = top >= 0
    return top, bot, exists


def _shoelace_area(ys, zs):
    """Area of a closed polygon (shoelace formula)."""
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    return 0.5 * abs(np.dot(ys, np.roll(zs, -1)) - np.dot(zs, np.roll(ys, -1)))


def _draw_line_safe(y0, z0, y1, z1, target, Y, Z):
    """Draw a Bresenham line on a 2D grid with bounds clamping."""
    y0, z0 = np.clip(y0, 0, Y - 1), np.clip(z0, 0, Z - 1)
    y1, z1 = np.clip(y1, 0, Y - 1), np.clip(z1, 0, Z - 1)
    rr, cc = draw_line(int(y0), int(z0), int(y1), int(z1))
    target[rr, cc] = True


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
        "--notch-radius", type=float, default=5.0,
        help="Tentorial notch exclusion radius in mm (default: 5.0)",
    )
    parser.add_argument(
        "--falx-thickness", type=float, default=1.0,
        help="Falx cerebri target thickness in mm (default: 1.0)",
    )
    parser.add_argument(
        "--tent-thickness", type=float, default=0.5,
        help="Tentorium cerebelli target thickness in mm (default: 0.5)",
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
    """Load material map, FS labels, skull SDF, and grid metadata.

    Returns (mat, fs, skull_sdf, affine, dx_mm).
    """
    mat_path = out_dir / "material_map.nii.gz"
    fs_path = out_dir / "fs_labels_resampled.nii.gz"
    skull_path = out_dir / "skull_sdf.nii.gz"
    meta_path = out_dir / "grid_meta.json"

    print(f"Loading {mat_path}")
    mat_img = nib.load(str(mat_path))
    mat = np.asarray(mat_img.dataobj, dtype=np.uint8)
    affine = mat_img.affine.copy()

    print(f"Loading {fs_path}")
    fs_img = nib.load(str(fs_path))
    fs = np.asarray(fs_img.dataobj, dtype=np.int16)

    print(f"Loading {skull_path}")
    skull_img = nib.load(str(skull_path))
    skull_sdf = np.asarray(skull_img.dataobj, dtype=np.float32)

    print(f"Loading {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    dx_mm = float(meta["dx_mm"])

    assert mat.shape == fs.shape, (
        f"Shape mismatch: mat={mat.shape}, fs={fs.shape}"
    )

    return mat, fs, skull_sdf, affine, dx_mm


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


def reconstruct_falx(mat, fs, skull_sdf, dx_mm, crop_slices,
                     thickness_mm=1.0, tent_mask=None):
    """Reconstruct the falx cerebri via PCHIP + Bezier free edge.

    Full phi=0 midplane membrane through all intracranial tissue,
    shaped by a PCHIP-interpolated free edge with Bezier genu-crista
    segment, using Kayalioglu ratios and Frassanito notch constraint.
    Rasterized via flood fill.

    Returns falx_mask (boolean).
    """
    sampling = (dx_mm, dx_mm, dx_mm)

    # Crop to bounding box
    fs_crop = fs[crop_slices]
    skull_crop = skull_sdf[crop_slices]

    # Classify hemispheres on crop
    left_lut, right_lut, cc_lut = _build_fs_luts()
    left_crop, right_crop = classify_hemispheres(fs_crop, left_lut, right_lut)

    # EDT pair on crop (threaded)
    print("Computing EDT for left+right hemispheres (cropped, threaded)...")
    with step("EDT pair (L/R hemispheres)"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_l = pool.submit(_edt.edt, ~left_crop, anisotropy=sampling, parallel=1)
            fut_r = pool.submit(_edt.edt, ~right_crop, anisotropy=sampling, parallel=1)
            dist_left = fut_l.result()
            dist_right = fut_r.result()
    del left_crop, right_crop

    # Signed diff field
    phi = dist_left - dist_right
    del dist_left, dist_right

    # Full midplane membrane through all intracranial tissue
    intracranial = skull_crop < 0
    membrane = _extract_membrane(phi, intracranial, thickness_mm)
    n_membrane = int(membrane.sum())
    print(f"  Full midplane membrane: {n_membrane} voxels")
    del phi, intracranial  # keep skull_crop for crista galli detection

    X, Y, Z = membrane.shape

    # CC landmark y-positions from FS sub-region labels
    fs_safe = np.clip(fs_crop, 0, _FS_LUT_SIZE - 1)
    cc_landmarks = {}
    for name, label, height in [
        ("splenium", _CC_SPLENIUM, _H_SPLENIUM),
        ("body", _CC_BODY, _H_BODY),
        ("genu", _CC_GENU, _H_GENU),
    ]:
        mask = (fs_safe == label)
        if mask.any():
            ys = np.where(mask.any(axis=(0, 2)))[0]
            cc_landmarks[name] = (int(np.median(ys)), height)

    # CC top/bottom edges for ratio-based control points
    cc_proj = cc_lut[fs_safe].any(axis=0)
    cc_top_z, _, cc_exists = _get_edges(cc_proj, Y, Z)
    del fs_crop, fs_safe

    # Membrane edges per y-column (projected onto y-z plane)
    mem_yz = membrane.any(axis=0)
    mem_top_z, mem_bot_z, mem_exists = _get_edges(mem_yz, Y, Z)

    # Tentorium top per y-column
    tent_top_z = np.full(Y, -1, dtype=int)
    tent_exists = np.zeros(Y, dtype=bool)
    if tent_mask is not None:
        tent_crop = tent_mask[crop_slices]
        tent_yz = tent_crop.any(axis=0)
        tent_top_z, _, tent_exists = _get_edges(tent_yz, Y, Z)
        del tent_crop, tent_yz

    mem_y_min = int(np.where(mem_exists)[0].min())
    mem_y_max = int(np.where(mem_exists)[0].max())

    # Junction: where tent_top_z peaks (straight sinus)
    junction_y = mem_y_min  # fallback if no tentorium
    if tent_exists.any():
        valid = tent_exists & mem_exists
        if valid.any():
            valid_y = np.where(valid)[0]
            junction_y = int(valid_y[np.argmax(tent_top_z[valid_y])])

    splenium_y = cc_landmarks.get("splenium", (junction_y, 0))[0]
    body_y = cc_landmarks.get("body", (splenium_y, 0))[0]
    genu_y = cc_landmarks.get("genu", (body_y, 0))[0]

    # Midline tentorium analysis for anchor point
    mid_x = X // 2
    tent_mid_top = np.full(Y, -1, dtype=int)
    tent_mid_exists = np.zeros(Y, dtype=bool)
    if tent_mask is not None:
        tent_crop = tent_mask[crop_slices]
        tent_mid = tent_crop[mid_x]
        for y in range(Y):
            zz = np.where(tent_mid[y])[0]
            if len(zz) > 0:
                tent_mid_top[y] = zz.max()
                tent_mid_exists[y] = True
        del tent_crop

    tent_mid_ys = np.where(tent_mid_exists)[0]
    if len(tent_mid_ys) > 0:
        anchor_y = int(tent_mid_ys.max())
        anchor_z = float(tent_mid_top[anchor_y])
    else:
        anchor_y = junction_y
        anchor_z = (float(tent_top_z[junction_y]) if tent_exists[junction_y]
                    else float(mem_bot_z[junction_y]))
    print(f"  Anchor (midline tent end): y={anchor_y}, z={anchor_z:.0f}")

    # Crista galli detection: lowest skull floor anterior to genu
    crista_y = None
    crista_z = None
    if len(cc_landmarks) >= 3:
        skull_mid = skull_crop[mid_x]
        skull_floor_z = np.full(Y, -1, dtype=int)
        for y in range(genu_y, mem_y_max + 1):
            if not mem_exists[y]:
                continue
            for z in range(0, Z):
                if skull_mid[y, z] < 0:
                    skull_floor_z[y] = z
                    break
        floor_valid = skull_floor_z > 0
        search_end = mem_y_max - 5
        best_floor = Z
        for y in range(genu_y + 5, search_end + 1):
            if not floor_valid[y]:
                continue
            mem_h = (mem_top_z[y] - mem_bot_z[y]) * dx_mm
            if mem_h < 15:
                continue
            if skull_floor_z[y] < best_floor:
                best_floor = skull_floor_z[y]
                crista_y = y
        if crista_y is not None:
            crista_z = skull_floor_z[crista_y]
            print(f"  Crista galli: y={crista_y}, z={crista_z}")
        else:
            crista_y = mem_y_max
            crista_z = mem_bot_z[crista_y]
            print(f"  Crista galli not found, using membrane tip y={crista_y}")
    else:
        crista_y = mem_y_max
        crista_z = mem_bot_z[crista_y]
        print(f"  Insufficient CC landmarks, using membrane tip y={crista_y}")
    del skull_crop

    # PCHIP control points: anchor → splenium → body → genu
    ctrl_y = [float(anchor_y)]
    ctrl_z = [anchor_z]
    for name, ratio in [("splenium", _RATIO_SPLENIUM),
                         ("body", _RATIO_BODY),
                         ("genu", _RATIO_GENU)]:
        if name in cc_landmarks:
            y_cc, _ = cc_landmarks[name]
            skull_top = mem_top_z[y_cc]
            cc_top_val = (cc_top_z[y_cc] if cc_exists[y_cc]
                          else mem_bot_z[y_cc])
            z_val = skull_top - ratio * (skull_top - cc_top_val)
            ctrl_y.append(float(y_cc))
            ctrl_z.append(float(z_val))

    ctrl_y = np.array(ctrl_y)
    ctrl_z = np.array(ctrl_z)
    pchip = PchipInterpolator(ctrl_y, ctrl_z)
    pchip_ys = np.arange(anchor_y, genu_y, dtype=float)
    pchip_zs = pchip(pchip_ys)

    # Outer boundary polyline (skull top + frontal wrap + skull bottom)
    outer_ys = []
    outer_zs = []
    for y in range(anchor_y, mem_y_max + 1):
        if mem_exists[y]:
            outer_ys.append(float(y))
            outer_zs.append(float(mem_top_z[y]))
    outer_ys.append(float(mem_y_max))
    outer_zs.append(float(mem_bot_z[mem_y_max]))
    for y in range(mem_y_max - 1, crista_y - 1, -1):
        if mem_exists[y]:
            outer_ys.append(float(y))
            outer_zs.append(float(mem_bot_z[y]))
    outer_y = np.array(outer_ys)
    outer_z = np.array(outer_zs)

    # Posterior attached area (above tentorium, y < anchor)
    posterior_area_vox2 = 0.0
    for y in range(mem_y_min, anchor_y):
        if mem_exists[y]:
            top = mem_top_z[y]
            bot = tent_top_z[y] if tent_exists[y] else mem_bot_z[y]
            posterior_area_vox2 += max(0.0, float(top - bot))

    # Similarity transform: map skull contour to free edge endpoints
    skull_pts_y = []
    skull_pts_z = []
    for y in range(genu_y, mem_y_max + 1):
        if mem_exists[y]:
            skull_pts_y.append(float(y))
            skull_pts_z.append(float(mem_top_z[y]))
    skull_pts_y.append(float(mem_y_max))
    skull_pts_z.append(float(mem_bot_z[mem_y_max]))
    for y in range(mem_y_max - 1, crista_y - 1, -1):
        if mem_exists[y]:
            skull_pts_y.append(float(y))
            skull_pts_z.append(float(mem_bot_z[y]))
    skull_pts_y = np.array(skull_pts_y)
    skull_pts_z = np.array(skull_pts_z)

    genu_ctrl_z = float(ctrl_z[-1])
    skull_start = np.array([skull_pts_y[0], skull_pts_z[0]])
    skull_end = np.array([skull_pts_y[-1], skull_pts_z[-1]])
    free_start = np.array([float(genu_y), genu_ctrl_z])
    free_end = np.array([float(crista_y), float(crista_z)])

    v_skull = skull_start - skull_end
    v_free = free_start - free_end
    sim_scale = np.linalg.norm(v_free) / np.linalg.norm(v_skull)
    angle_skull = np.arctan2(v_skull[1], v_skull[0])
    angle_free = np.arctan2(v_free[1], v_free[0])
    dtheta = angle_free - angle_skull
    cos_d, sin_d = np.cos(dtheta), np.sin(dtheta)

    tx_y = np.zeros_like(skull_pts_y)
    tx_z = np.zeros_like(skull_pts_z)
    for i in range(len(skull_pts_y)):
        dy = skull_pts_y[i] - skull_end[0]
        dz = skull_pts_z[i] - skull_end[1]
        tx_y[i] = free_end[0] + (cos_d * dy - sin_d * dz) * sim_scale
        tx_z[i] = free_end[1] + (sin_d * dy + cos_d * dz) * sim_scale

    # Arc-length parameterize for endpoint tangent extraction
    skull_arc = np.zeros(len(tx_y))
    for i in range(1, len(tx_y)):
        skull_arc[i] = skull_arc[i - 1] + np.hypot(
            tx_y[i] - tx_y[i - 1], tx_z[i] - tx_z[i - 1])
    skull_t = skull_arc / skull_arc[-1]
    skull_y_of_t = interp1d(skull_t, tx_y, kind="linear")
    skull_z_of_t = interp1d(skull_t, tx_z, kind="linear")

    eps = 1e-4
    T0_y = float(skull_y_of_t(eps) - skull_y_of_t(0)) / eps
    T0_z = float(skull_z_of_t(eps) - skull_z_of_t(0)) / eps
    T1_y = float(skull_y_of_t(1) - skull_y_of_t(1 - eps)) / eps
    T1_z = float(skull_z_of_t(1) - skull_z_of_t(1 - eps)) / eps
    T0_len = np.hypot(T0_y, T0_z)
    T0_dir = np.array([T0_y / T0_len, T0_z / T0_len])
    T1_len = np.hypot(T1_y, T1_z)
    T1_dir = np.array([T1_y / T1_len, T1_z / T1_len])

    # Bezier alpha solve: match Frassanito notch area ratio
    P0 = free_start
    P3 = free_end
    chord = np.linalg.norm(P3 - P0)

    def _bezier_curve(alpha):
        d = alpha * chord
        p1 = P0 + d * T0_dir
        p2 = P3 - d * T1_dir
        t = np.linspace(0, 1, 300)
        by = ((1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * p1[0]
              + 3 * (1 - t) * t**2 * p2[0] + t**3 * P3[0])
        bz = ((1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * p1[1]
              + 3 * (1 - t) * t**2 * p2[1] + t**3 * P3[1])
        return by, bz

    def _notch_ratio(alpha):
        by, bz = _bezier_curve(alpha)
        iy = np.concatenate([pchip_ys, by])
        iz = np.concatenate([pchip_zs, bz])
        n_y = np.concatenate([iy, [float(anchor_y)]])
        n_z = np.concatenate([iz, [anchor_z]])
        notch = _shoelace_area(n_y, n_z)
        f_y = np.concatenate([outer_y, iy[::-1]])
        f_z = np.concatenate([outer_z, iz[::-1]])
        falx_area = _shoelace_area(f_y, f_z) + posterior_area_vox2
        return notch / falx_area if falx_area > 0 else 0

    alphas = np.linspace(0.05, 0.8, 200)
    ratios_sweep = np.array([_notch_ratio(a) for a in alphas])
    idx = int(np.argmin(np.abs(ratios_sweep - _NOTCH_AREA_RATIO)))
    alpha_solved = float(alphas[idx])
    d_solved = alpha_solved * chord
    P1 = P0 + d_solved * T0_dir
    P2 = P3 - d_solved * T1_dir
    print(f"  Bezier: \u03b1={alpha_solved:.3f}, d={d_solved * dx_mm:.1f}mm")

    # Dense Bezier sampling (genu -> crista)
    n_bez = 2000
    t_bez = np.linspace(0, 1, n_bez)
    bez_y = ((1 - t_bez)**3 * P0[0] + 3 * (1 - t_bez)**2 * t_bez * P1[0]
             + 3 * (1 - t_bez) * t_bez**2 * P2[0] + t_bez**3 * P3[0])
    bez_z = ((1 - t_bez)**3 * P0[1] + 3 * (1 - t_bez)**2 * t_bez * P1[1]
             + 3 * (1 - t_bez) * t_bez**2 * P2[1] + t_bez**3 * P3[1])
    inner_ys = np.concatenate([pchip_ys, bez_y])
    inner_zs = np.concatenate([pchip_zs, bez_z])

    # Flood fill rasterization
    barrier = np.zeros((Y, Z), dtype=bool)

    # 1. Free edge curve (PCHIP + Bezier)
    for i in range(len(inner_ys) - 1):
        _draw_line_safe(round(inner_ys[i]), round(inner_zs[i]),
                        round(inner_ys[i + 1]), round(inner_zs[i + 1]),
                        barrier, Y, Z)

    # 2. Tentorium top (anchor backward to posterior extent)
    tent_ys_arr = np.where(tent_exists)[0]
    tent_post_y = int(tent_ys_arr.min()) if len(tent_ys_arr) > 0 else anchor_y
    for y in range(anchor_y, tent_post_y, -1):
        if tent_exists[y] and tent_exists[y - 1]:
            _draw_line_safe(y, tent_top_z[y], y - 1, tent_top_z[y - 1],
                            barrier, Y, Z)

    # 3. Posterior close: tent end -> skull bottom backward -> vertical
    _draw_line_safe(tent_post_y, tent_top_z[tent_post_y],
                    tent_post_y, mem_bot_z[tent_post_y], barrier, Y, Z)
    for y in range(tent_post_y, mem_y_min, -1):
        if mem_exists[y] and mem_exists[y - 1]:
            _draw_line_safe(y, mem_bot_z[y], y - 1, mem_bot_z[y - 1],
                            barrier, Y, Z)
    if mem_exists[mem_y_min]:
        _draw_line_safe(mem_y_min, mem_bot_z[mem_y_min],
                        mem_y_min, mem_top_z[mem_y_min], barrier, Y, Z)

    # 4. Skull top (mem_y_min -> frontal pole)
    for y in range(mem_y_min, mem_y_max):
        if mem_exists[y] and mem_exists[y + 1]:
            _draw_line_safe(y, mem_top_z[y], y + 1, mem_top_z[y + 1],
                            barrier, Y, Z)

    # 5. Frontal pole vertical drop
    if mem_exists[mem_y_max]:
        _draw_line_safe(mem_y_max, mem_top_z[mem_y_max],
                        mem_y_max, mem_bot_z[mem_y_max], barrier, Y, Z)

    # 6. Skull bottom (frontal pole back to crista)
    for y in range(mem_y_max, crista_y, -1):
        if mem_exists[y] and mem_exists[y - 1]:
            _draw_line_safe(y, mem_bot_z[y], y - 1, mem_bot_z[y - 1],
                            barrier, Y, Z)

    # 7. Connect crista skull bottom -> free edge end
    _draw_line_safe(crista_y, mem_bot_z[crista_y],
                    int(round(inner_ys[-1])), int(round(inner_zs[-1])),
                    barrier, Y, Z)

    # Flood fill from midpoint at body CC
    seed_y = body_y
    seed_z = (int(float(pchip(float(body_y)))) + mem_top_z[body_y]) // 2
    if barrier[seed_y, seed_z]:
        for dz in range(1, 10):
            if seed_z + dz < Z and not barrier[seed_y, seed_z + dz]:
                seed_z = seed_z + dz
                break

    labeled, n_labels = cc_label(~barrier)
    seed_label = labeled[seed_y, seed_z]
    flood = labeled == seed_label
    cookie = binary_fill_holes(flood | barrier)
    del labeled, flood

    # Intersect cookie with membrane
    cookie_3d = np.broadcast_to(cookie[np.newaxis, :, :], (X, Y, Z))
    falx_crop = membrane & cookie_3d
    del membrane, cookie_3d, cookie

    n_falx_crop = int(falx_crop.sum())
    print(f"  After cookie-cut: {n_falx_crop} voxels "
          f"(from {n_membrane} membrane)")

    # Cut falx below the tentorium surface
    if tent_mask is not None:
        tent_crop = tent_mask[crop_slices]
        tent_below = np.zeros((Y, Z), dtype=bool)
        for y in range(Y):
            if tent_exists[y]:
                tent_below[y, :tent_top_z[y]] = True
        tent_below_3d = np.broadcast_to(
            tent_below[np.newaxis, :, :], (X, Y, Z))
        n_below = int((falx_crop & tent_below_3d).sum())
        falx_crop &= ~tent_below_3d
        del tent_crop, tent_below, tent_below_3d
        print(f"  Tentorium cut: removed {n_below} voxels below tent surface")

    # Write back to full grid
    falx_mask = np.zeros(mat.shape, dtype=bool)
    falx_mask[crop_slices] = falx_crop
    del falx_crop

    n_falx = int(np.count_nonzero(falx_mask))
    print(f"Falx cerebri: {n_falx} voxels")

    return falx_mask


def _measure_notch_ellipse(mat_crop, dx_mm):
    """Measure tentorial notch dimensions from cerebellar gap anatomy.

    Scans the gap between left and right cerebellar medial edges at every
    AP position across 20mm of axial slices below the tentorial level.

    Width  = maximum median gap (Maximum Notch Width from literature).
    Length = width × 1.8 (literature aspect ratio, robust across subjects).

    Returns a 2D boolean ellipse mask (X, Y) or None if measurement fails.
    Also prints diagnostic info.
    """
    # Find brainstem at tentorial level
    brainstem = (mat_crop == 6)
    if not brainstem.any():
        return None

    z_indices = np.where(brainstem.any(axis=(0, 1)))[0]
    z_min_bs, z_max_bs = int(z_indices.min()), int(z_indices.max())
    tent_z = z_min_bs + 2 * (z_max_bs - z_min_bs) // 3

    bs_2d = brainstem[:, :, tent_z]
    if not bs_2d.any():
        return None
    bs_ij = np.argwhere(bs_2d)
    bs_centroid = bs_ij.mean(axis=0)
    mid_x = bs_centroid[0]

    X, Y, Z = mat_crop.shape

    # Measure gap at every y across z-levels below tentorial level
    z_start = max(0, tent_z - round(20.0 / dx_mm))
    gap_by_y = {}
    for z in range(z_start, tent_z + 1):
        cereb_slice = (mat_crop[:, :, z] == 4) | (mat_crop[:, :, z] == 5)
        for y in range(Y):
            col = cereb_slice[:, y]
            left_x = np.where(col[:int(mid_x)])[0]
            right_x = np.where(col[int(mid_x):])[0]
            if len(left_x) > 0 and len(right_x) > 0:
                left_medial = left_x.max()
                right_medial = int(mid_x) + right_x.min()
                gap_w = (right_medial - left_medial) * dx_mm
                if 5 < gap_w < 80:
                    gap_by_y.setdefault(y, []).append(gap_w)

    if not gap_by_y:
        return None

    # MNW = max of median gap at each y
    median_gaps = {y: float(np.median(ws)) for y, ws in gap_by_y.items()}
    mnw = max(median_gaps.values())
    length = mnw * _TENT_NOTCH_AR
    buffer_mm = 2.0

    semi_x = (mnw / 2 + buffer_mm) / dx_mm
    semi_y = (length / 2 + buffer_mm) / dx_mm

    xx, yy = np.mgrid[:X, :Y]
    ellipse = (
        (xx - bs_centroid[0]) ** 2 / max(semi_x ** 2, 1)
        + (yy - bs_centroid[1]) ** 2 / max(semi_y ** 2, 1)
    ) <= 1.0

    area_cm2 = ellipse.sum() * dx_mm ** 2 / 100
    print(f"  Notch ellipse: MNW={mnw:.1f}mm, NL={length:.1f}mm, "
          f"area={area_cm2:.1f}cm² (lit: ~13cm²)")

    return ellipse


def reconstruct_tentorium(mat, dx_mm, notch_radius, crop_slices,
                          thickness_mm=0.5):
    """Reconstruct the tentorium cerebelli via sign-change EDT watershed.

    Computes signed distance field between cerebral and cerebellar tissue
    using EDT, then extracts the zero-level-set via sign-change detection
    with target physical thickness.

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
    with step("EDT pair (cerebral/cerebellar)"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_c = pool.submit(_edt.edt, ~cerebral_crop, anisotropy=sampling, parallel=1)
            fut_b = pool.submit(_edt.edt, ~cerebellar_crop, anisotropy=sampling, parallel=1)
            dist_cerebral = fut_c.result()
            dist_cerebellar = fut_b.result()
    del cerebral_crop, cerebellar_crop

    phi = dist_cerebral - dist_cerebellar
    del dist_cerebral, dist_cerebellar

    # Eligible: non-vacuum minus notch exclusion
    eligible = (mat_crop > 0)
    notch_ellipse = _measure_notch_ellipse(mat_crop, dx_mm)
    if notch_ellipse is not None:
        eligible &= ~notch_ellipse[:, :, np.newaxis]
    else:
        # Fallback: brainstem dilation (original method)
        brainstem_crop = (mat_crop == 6)
        r_notch_vox = notch_radius / dx_mm
        if r_notch_vox >= 1.0 and brainstem_crop.any():
            z_indices = np.where(brainstem_crop.any(axis=(0, 1)))[0]
            z_min_bs, z_max_bs = int(z_indices.min()), int(z_indices.max())
            tent_z_local = z_min_bs + 2 * (z_max_bs - z_min_bs) // 3
            bs_xy = brainstem_crop[:, :, tent_z_local]
            n_iter = max(1, round(r_notch_vox))
            bs_dilated = binary_dilation(bs_xy, iterations=n_iter)
            eligible &= ~bs_dilated[:, :, np.newaxis]
            del bs_xy, bs_dilated
        del brainstem_crop

    # Membrane extraction with target thickness
    tent_crop = _extract_membrane(phi, eligible, thickness_mm)
    del phi, eligible

    n_csf = int((tent_crop & (mat_crop == 8)).sum())
    n_tissue = int(tent_crop.sum()) - n_csf
    print(f"  Tentorium sign-change: {int(tent_crop.sum())} voxels "
          f"({n_csf} CSF, {n_tissue} tissue)")
    del mat_crop

    # Write back to full grid
    tent_mask = np.zeros(mat.shape, dtype=bool)
    tent_mask[crop_slices] = tent_crop
    del tent_crop

    n_tent = int(np.count_nonzero(tent_mask))
    print(f"Tentorium cerebelli: {n_tent} voxels (sign-change)")

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
def check_idempotency(mat, fs):
    """Reset pre-existing dural voxels (u8=10) to their original material class.

    Called BEFORE any mask computation to ensure clean state.  Uses FS labels
    to restore the correct material class instead of blindly setting u8=8,
    which would corrupt tissue voxels (brainstem, cerebellar cortex, etc.)
    that were overwritten by a previous dural membrane run.

    Returns count of reset voxels.
    """
    existing = int(np.count_nonzero(mat == 10))
    if existing > 0:
        from preprocessing.material_map import build_lut, apply_mapping
        lut, _ = build_lut()
        dural_mask = (mat == 10)
        restored = apply_mapping(fs[dural_mask], lut)
        mat[dural_mask] = restored
        # Voxels whose FS label maps to vacuum (u8=0) were originally sulcal
        # CSF (filled by subarachnoid_csf step) — restore them as CSF.
        mat[dural_mask & (mat == 0)] = 8
        n_tissue = int(np.count_nonzero(restored > 0))
        n_csf = existing - n_tissue
        print(f"Reset {existing} pre-existing u8=10 voxels "
              f"({n_tissue} to tissue, {n_csf} to CSF)")
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
    """Check falx proximity to pial surfaces (optional).

    Uses all pial vertices (not just the HCP medial wall ROI, which only
    covers ~5k CC-adjacent vertices). The falx sits in the interhemispheric
    fissure between the full medial pial surfaces of each hemisphere.

    Skips silently if surface files don't exist.
    """
    n_falx = int(np.count_nonzero(falx_mask))
    if n_falx == 0:
        return

    native_dir = raw_dir(subject) / "Native"

    # Convert falx voxel indices to physical mm (once for both hemispheres)
    falx_ijk = np.argwhere(falx_mask).astype(np.float64)
    falx_mm = falx_ijk * dx_mm + affine[:3, 3]

    results = []
    for hemi, label in [("L", "left"), ("R", "right")]:
        surf_path = native_dir / f"{subject}.{hemi}.pial.native.surf.gii"

        if not surf_path.exists():
            return  # skip entirely if any file missing

        surf = nib.load(str(surf_path))
        coords = surf.darrays[0].data  # (n_vertices, 3)

        tree = cKDTree(coords)
        dists, _ = tree.query(falx_mm)
        results.append((label, float(np.median(dists)), float(np.percentile(dists, 95))))

    if results:
        print("\n" + "=" * 60)
        print("Medial Wall Proximity")
        print("=" * 60)
        for label, median_d, p95_d in results:
            print(f"  {label} pial surface: median={median_d:.1f} mm, "
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

    # Max contiguous z-run at junction
    overlap_ijk = np.argwhere(overlap)
    if len(overlap_ijk) > 0:
        # Group by (x, y) and measure max contiguous z-run
        xy_unique = np.unique(overlap_ijk[:, :2], axis=0)
        max_z_run = 0
        for xy in xy_unique:
            z_vals = np.sort(overlap_ijk[(overlap_ijk[:, 0] == xy[0]) &
                                         (overlap_ijk[:, 1] == xy[1]), 2])
            # Find max contiguous run length
            if len(z_vals) == 1:
                run = 1
            else:
                gaps = np.diff(z_vals)
                runs = np.split(z_vals, np.where(gaps > 1)[0] + 1)
                run = max(len(r) for r in runs)
            max_z_run = max(max_z_run, run)

        print(f"  Max z-run: {max_z_run} voxels ({max_z_run * dx_mm:.1f} mm)")
        if max_z_run > 3:
            print(f"  WARNING: junction z-run > 3 voxels")

    del overlap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    """Orchestrate dural membrane reconstruction."""
    args = parse_args(argv)

    print(f"Subject: {args.subject}")
    print(f"Profile: {args.profile}  (N={args.N}, dx={args.dx} mm)")
    print(f"Notch radius: {args.notch_radius} mm")
    print(f"Falx thickness: {args.falx_thickness} mm")
    print(f"Tentorium thickness: {args.tent_thickness} mm")
    print()

    out_dir = processed_dir(args.subject, args.profile)

    with step("load inputs"):
        mat, fs, skull_sdf, affine, dx_mm = load_inputs(out_dir)

    print(f"Shape: {mat.shape}  dtype: {mat.dtype}")
    print()

    # Idempotency: reset any pre-existing dural voxels using FS labels
    check_idempotency(mat, fs)

    # Compute crop bounding box for EDT (used by both falx and tentorium)
    pad_vox = math.ceil(args.notch_radius / dx_mm) + 2
    crop_slices = _compute_crop_bbox(mat, pad_vox)
    crop_shape = tuple(s.stop - s.start for s in crop_slices)
    print(f"EDT crop: {crop_shape} (pad={pad_vox} voxels)")
    print()

    # Reconstruct tentorium cerebelli (needed by falx for posterior junction)
    with step("reconstruct tentorium"):
        tent_mask = reconstruct_tentorium(
            mat, dx_mm, args.notch_radius, crop_slices,
            args.tent_thickness,
        )

    # Reconstruct falx cerebri (midplane cookie-cutter with spline edge)
    with step("reconstruct falx"):
        falx_mask = reconstruct_falx(mat, fs, skull_sdf, dx_mm, crop_slices,
                                     args.falx_thickness, tent_mask)
    del fs, skull_sdf

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
