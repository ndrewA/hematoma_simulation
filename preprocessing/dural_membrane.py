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
from dataclasses import dataclass

import edt as _edt
import nibabel as nib
import numpy as np
from scipy.interpolate import PchipInterpolator, make_interp_spline
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    label as cc_label,
)
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line, polygon as draw_polygon

from preprocessing.profiling import step
from preprocessing.utils import (
    FS_LUT_SIZE,
    add_grid_args,
    processed_dir,
    raw_dir,
    resolve_grid_args,
    section,
)
from preprocessing.material_map import CLASS_NAMES, print_census, save_material_map


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

# Medial orbitofrontal cortex (gyrus rectus) FS labels for crista galli
_MOF_LEFT = 1014
_MOF_RIGHT = 2014


@dataclass
class _FalxGeometry:
    """Sagittal-plane geometry for falx cookie-cutter.

    Bundles the intermediate edge profiles, landmarks, and anchor points
    detected during falx reconstruction so they can flow between phases
    without 16+ loose variables.
    """
    # Membrane edge profiles (length Y)
    mem_top_z: np.ndarray
    mem_bot_z: np.ndarray
    mem_exists: np.ndarray
    mem_y_min: int
    mem_y_max: int
    # Tentorium edge profiles
    tent_top_z: np.ndarray
    tent_exists: np.ndarray
    # CC landmarks
    cc_landmarks: dict
    cc_top_z: np.ndarray
    cc_exists: np.ndarray
    # Key points
    anchor_y: int
    anchor_z: float
    crista_y: int
    crista_z: float
    genu_y: int
    mid_x: int


def _build_fs_luts():
    """Build boolean LUTs for left/right cerebral and CC FreeSurfer labels."""
    left_lut = np.zeros(FS_LUT_SIZE, dtype=bool)
    for lab in LEFT_CEREBRAL_LABELS:
        left_lut[lab] = True
    left_lut[LEFT_CEREBRAL_RANGE[0]:LEFT_CEREBRAL_RANGE[1] + 1] = True

    right_lut = np.zeros(FS_LUT_SIZE, dtype=bool)
    for lab in RIGHT_CEREBRAL_LABELS:
        right_lut[lab] = True
    right_lut[RIGHT_CEREBRAL_RANGE[0]:RIGHT_CEREBRAL_RANGE[1] + 1] = True

    cc_lut = np.zeros(FS_LUT_SIZE, dtype=bool)
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
    any_row = proj.any(axis=1)  # (Y,) bool
    # argmax/argmin on flipped array to get last/first True per row
    top = np.where(any_row, Z - 1 - np.argmax(proj[:, ::-1], axis=1), -1)
    bot = np.where(any_row, np.argmax(proj, axis=1), Z)
    return top, bot, any_row


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


def _collect_falx_polygon(inner_ys, inner_zs, geo, tent_post_y):
    """Collect ordered polygon vertices for falx cookie-cutter region.

    Traces the closed boundary of the falx region in the sagittal (Y-Z) plane:
    free edge → skull bottom → frontal wrap → skull top → posterior close →
    tent top → auto-close to start.  Used with skimage.draw.polygon for
    scanline fill (even-odd rule) instead of seed-based flood fill.
    """
    poly_y = []
    poly_z = []

    # Free edge (PCHIP + Bezier): anchor → crista
    for i in range(len(inner_ys)):
        poly_y.append(float(inner_ys[i]))
        poly_z.append(float(inner_zs[i]))

    # Free edge end → skull bottom at crista
    poly_y.append(float(geo.crista_y))
    poly_z.append(float(geo.mem_bot_z[geo.crista_y]))

    # Skull bottom crista → frontal pole
    for y in range(geo.crista_y + 1, geo.mem_y_max + 1):
        if geo.mem_exists[y]:
            poly_y.append(float(y))
            poly_z.append(float(geo.mem_bot_z[y]))

    # Frontal pole vertical (bottom → top)
    if geo.mem_exists[geo.mem_y_max]:
        poly_y.append(float(geo.mem_y_max))
        poly_z.append(float(geo.mem_top_z[geo.mem_y_max]))

    # Skull top frontal → posterior
    for y in range(geo.mem_y_max - 1, geo.mem_y_min - 1, -1):
        if geo.mem_exists[y]:
            poly_y.append(float(y))
            poly_z.append(float(geo.mem_top_z[y]))

    # Posterior vertical (top → bottom)
    if geo.mem_exists[geo.mem_y_min]:
        poly_y.append(float(geo.mem_y_min))
        poly_z.append(float(geo.mem_bot_z[geo.mem_y_min]))

    # Skull bottom posterior → tent start
    for y in range(geo.mem_y_min + 1, tent_post_y + 1):
        if geo.mem_exists[y]:
            poly_y.append(float(y))
            poly_z.append(float(geo.mem_bot_z[y]))

    # Tent start vertical (bottom → top)
    if geo.tent_exists[tent_post_y]:
        poly_y.append(float(tent_post_y))
        poly_z.append(float(geo.tent_top_z[tent_post_y]))

    # Tent top tent_post → anchor
    anchor_y = int(round(inner_ys[0]))
    for y in range(tent_post_y + 1, anchor_y + 1):
        if geo.tent_exists[y]:
            poly_y.append(float(y))
            poly_z.append(float(geo.tent_top_z[y]))

    # skimage.draw.polygon auto-closes (last vertex → first vertex)
    return np.array(poly_y), np.array(poly_z)


def _save_falx_debug_figure(mem_yz, poly_y, poly_z, cookie,
                             inner_ys, inner_zs, save_path):
    """Save debug visualization of polygon fill result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: polygon outline on membrane projection
    axes[0].imshow(mem_yz.T, origin="lower", cmap="gray", aspect="equal")
    axes[0].plot(poly_y, poly_z, "r-", linewidth=0.5, alpha=0.8)
    axes[0].plot(inner_ys, inner_zs, "b-", linewidth=1.0, alpha=0.8,
                 label="free edge")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Polygon outline (red) + free edge (blue)")
    axes[0].set_xlabel("Y (anterior-posterior)")
    axes[0].set_ylabel("Z (inferior-superior)")

    # Right: cookie overlaid on membrane
    overlay = np.zeros((*mem_yz.shape, 3))
    overlay[mem_yz, :] = 0.4
    overlay[cookie, 1] = 0.9
    overlay[mem_yz & cookie, 0] = 0.0
    overlay[mem_yz & cookie, 2] = 0.0
    axes[1].imshow(np.transpose(overlay, (1, 0, 2)), origin="lower",
                   aspect="equal")
    axes[1].set_title("Cookie (green) on membrane (gray)")
    axes[1].set_xlabel("Y (anterior-posterior)")
    axes[1].set_ylabel("Z (inferior-superior)")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Debug figure saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse CLI arguments for dural_membrane."""
    parser = argparse.ArgumentParser(
        description="Reconstruct falx cerebri and tentorium cerebelli."
    )
    add_grid_args(parser)
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
    resolve_grid_args(args, parser)
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

    if mat.shape != fs.shape:
        raise ValueError(f"Shape mismatch: mat={mat.shape}, fs={fs.shape}")

    return mat, fs, skull_sdf, affine, dx_mm


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def classify_hemispheres(fs, left_lut, right_lut):
    """Classify voxels into left and right cerebral hemispheres.

    Returns (left_mask, right_mask) as boolean arrays.
    Uses pre-built LUTs for O(1) per-voxel lookup instead of np.isin.
    """
    fs_safe = np.clip(fs, 0, FS_LUT_SIZE - 1)
    left_mask = left_lut[fs_safe]
    right_mask = right_lut[fs_safe]
    return left_mask, right_mask


def _detect_crista_galli(fs_crop, mid_x, genu_y):
    """Find crista galli via inferior extent of medial orbitofrontal cortex.

    The gyrus rectus (medial orbitofrontal, FS labels 1014/2014) sits
    directly on the cribriform plate adjacent to the crista galli.  Its
    most inferior voxel near the midline is a reliable proxy for the
    anterior falx attachment point.

    Returns (crista_y, crista_z).
    """
    band = fs_crop[mid_x - 2 : mid_x + 3]
    mof = (band == _MOF_LEFT) | (band == _MOF_RIGHT)
    mof_yz = mof.any(axis=0)  # project to sagittal plane
    ys, zs = np.where(mof_yz)

    anterior = ys > genu_y
    if not anterior.any():
        raise ValueError(
            "No medial orbitofrontal cortex (1014/2014) found anterior to genu")
    ys, zs = ys[anterior], zs[anterior]

    idx = int(np.argmin(zs))
    crista_y, crista_z = int(ys[idx]), int(zs[idx])
    print(f"  Crista galli (medial orbitofrontal): y={crista_y}, z={crista_z}")
    return crista_y, crista_z


def _build_free_edge_controls(geo):
    """Build PCHIP control points and interpolate the free edge curve.

    Control points: anchor → splenium → body → genu, using Kayalioglu
    height ratios to place each point between skull top and CC top.

    Returns (pchip_ys, pchip_zs, genu_ctrl_z).
    """
    ctrl_y = [float(geo.anchor_y)]
    ctrl_z = [geo.anchor_z]
    for name, ratio in [("splenium", _RATIO_SPLENIUM),
                         ("body", _RATIO_BODY),
                         ("genu", _RATIO_GENU)]:
        if name in geo.cc_landmarks:
            y_cc, _ = geo.cc_landmarks[name]
            skull_top = geo.mem_top_z[y_cc]
            cc_top_val = (geo.cc_top_z[y_cc] if geo.cc_exists[y_cc]
                          else geo.mem_bot_z[y_cc])
            z_val = skull_top - ratio * (skull_top - cc_top_val)
            ctrl_y.append(float(y_cc))
            ctrl_z.append(float(z_val))

    ctrl_y = np.array(ctrl_y)
    ctrl_z = np.array(ctrl_z)
    pchip = PchipInterpolator(ctrl_y, ctrl_z)
    pchip_ys = np.arange(geo.anchor_y, geo.genu_y, dtype=float)
    pchip_zs = pchip(pchip_ys)
    genu_ctrl_z = float(ctrl_z[-1])
    return pchip_ys, pchip_zs, genu_ctrl_z


def _collect_boundary_polylines(geo):
    """Collect outer boundary polyline and posterior attached area.

    Outer boundary traces skull top (anchor→anterior) then wraps back
    along skull bottom.  Posterior area is the falx above the tentorium.

    Returns (outer_y, outer_z, posterior_area_vox2).
    """
    outer_ys = []
    outer_zs = []
    for y in range(geo.anchor_y, geo.mem_y_max + 1):
        if geo.mem_exists[y]:
            outer_ys.append(float(y))
            outer_zs.append(float(geo.mem_top_z[y]))
    outer_ys.append(float(geo.mem_y_max))
    outer_zs.append(float(geo.mem_bot_z[geo.mem_y_max]))
    for y in range(geo.mem_y_max - 1, geo.crista_y - 1, -1):
        if geo.mem_exists[y]:
            outer_ys.append(float(y))
            outer_zs.append(float(geo.mem_bot_z[y]))
    outer_y = np.array(outer_ys)
    outer_z = np.array(outer_zs)

    posterior_area_vox2 = 0.0
    for y in range(geo.mem_y_min, geo.anchor_y):
        if geo.mem_exists[y]:
            top = geo.mem_top_z[y]
            bot = geo.tent_top_z[y] if geo.tent_exists[y] else geo.mem_bot_z[y]
            posterior_area_vox2 += max(0.0, float(top - bot))

    return outer_y, outer_z, posterior_area_vox2


def _solve_bezier_shape(pchip_ys, pchip_zs, genu_ctrl_z,
                        outer_y, outer_z, posterior_area_vox2,
                        geo, dx_mm):
    """Solve Bezier free edge shape matching Frassanito notch area ratio.

    Applies a similarity transform of the skull contour to the free edge
    endpoints, extracts endpoint tangents, then sweeps Bezier alpha to
    match the target notch-to-falx area ratio.

    Returns (inner_ys, inner_zs) combining PCHIP + Bezier curves,
    or None if the geometry is degenerate.
    """
    # Collect skull contour points (genu → anterior tip → crista)
    skull_pts_y = []
    skull_pts_z = []
    for y in range(geo.genu_y, geo.mem_y_max + 1):
        if geo.mem_exists[y]:
            skull_pts_y.append(float(y))
            skull_pts_z.append(float(geo.mem_top_z[y]))
    skull_pts_y.append(float(geo.mem_y_max))
    skull_pts_z.append(float(geo.mem_bot_z[geo.mem_y_max]))
    for y in range(geo.mem_y_max - 1, geo.crista_y - 1, -1):
        if geo.mem_exists[y]:
            skull_pts_y.append(float(y))
            skull_pts_z.append(float(geo.mem_bot_z[y]))
    skull_pts_y = np.array(skull_pts_y)
    skull_pts_z = np.array(skull_pts_z)

    # Similarity transform: map skull contour to free edge endpoints
    skull_start = np.array([skull_pts_y[0], skull_pts_z[0]])
    skull_end = np.array([skull_pts_y[-1], skull_pts_z[-1]])
    free_start = np.array([float(geo.genu_y), genu_ctrl_z])
    free_end = np.array([float(geo.crista_y), float(geo.crista_z)])

    v_skull = skull_start - skull_end
    v_free = free_start - free_end
    skull_len = np.linalg.norm(v_skull)
    if skull_len < 1e-8:
        return None  # degenerate: genu and crista at same location
    sim_scale = np.linalg.norm(v_free) / skull_len
    dtheta = np.arctan2(v_free[1], v_free[0]) - np.arctan2(v_skull[1], v_skull[0])
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
    if skull_arc[-1] < 1e-8:
        return None  # degenerate: zero arc length
    skull_t = skull_arc / skull_arc[-1]
    skull_y_of_t = make_interp_spline(skull_t, tx_y, k=1)
    skull_z_of_t = make_interp_spline(skull_t, tx_z, k=1)

    eps = 1e-4
    T0_y = float(skull_y_of_t(eps) - skull_y_of_t(0)) / eps
    T0_z = float(skull_z_of_t(eps) - skull_z_of_t(0)) / eps
    T1_y = float(skull_y_of_t(1) - skull_y_of_t(1 - eps)) / eps
    T1_z = float(skull_z_of_t(1) - skull_z_of_t(1 - eps)) / eps
    T0_len = np.hypot(T0_y, T0_z)
    T0_dir = np.array([T0_y / T0_len, T0_z / T0_len])
    T1_len = np.hypot(T1_y, T1_z)
    T1_dir = np.array([T1_y / T1_len, T1_z / T1_len])

    # Bezier alpha sweep: match Frassanito notch area ratio
    P0 = free_start
    P3 = free_end
    chord = np.linalg.norm(P3 - P0)

    def _eval_bezier(p1, p2, n_pts):
        t = np.linspace(0, 1, n_pts)
        s = 1 - t
        by = s**3 * P0[0] + 3 * s**2 * t * p1[0] + 3 * s * t**2 * p2[0] + t**3 * P3[0]
        bz = s**3 * P0[1] + 3 * s**2 * t * p1[1] + 3 * s * t**2 * p2[1] + t**3 * P3[1]
        return by, bz

    def _notch_ratio(alpha):
        d = alpha * chord
        by, bz = _eval_bezier(P0 + d * T0_dir, P3 - d * T1_dir, 300)
        iy = np.concatenate([pchip_ys, by])
        iz = np.concatenate([pchip_zs, bz])
        n_y = np.concatenate([iy, [float(geo.anchor_y)]])
        n_z = np.concatenate([iz, [geo.anchor_z]])
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

    # Dense Bezier sampling (genu → crista)
    bez_y, bez_z = _eval_bezier(P1, P2, 2000)
    inner_ys = np.concatenate([pchip_ys, bez_y])
    inner_zs = np.concatenate([pchip_zs, bez_z])
    return inner_ys, inner_zs


def _compute_midplane_membrane(fs_crop, skull_crop, dx_mm, thickness_mm):
    """EDT pair on left/right hemispheres → sign-change membrane + CC info.

    Returns (membrane, n_membrane, cc_landmarks, cc_top_z, cc_exists).
    """
    sampling = (dx_mm, dx_mm, dx_mm)

    left_lut, right_lut, cc_lut = _build_fs_luts()
    left_crop, right_crop = classify_hemispheres(fs_crop, left_lut, right_lut)

    print("Computing EDT for left+right hemispheres (cropped, threaded)...")
    with step("EDT pair (L/R hemispheres)"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_l = pool.submit(_edt.edt, ~left_crop, anisotropy=sampling, parallel=1)
            fut_r = pool.submit(_edt.edt, ~right_crop, anisotropy=sampling, parallel=1)
            dist_left = fut_l.result()
            dist_right = fut_r.result()
    del left_crop, right_crop

    phi = dist_left - dist_right
    del dist_left, dist_right
    intracranial = skull_crop < 0
    membrane = _extract_membrane(phi, intracranial, thickness_mm)
    n_membrane = int(membrane.sum())
    print(f"  Full midplane membrane: {n_membrane} voxels")
    del phi, intracranial

    # CC landmark y-positions from FS sub-region labels
    _, Y, Z = membrane.shape
    fs_safe = np.clip(fs_crop, 0, FS_LUT_SIZE - 1)
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

    cc_proj = cc_lut[fs_safe].any(axis=0)
    cc_top_z, _, cc_exists = _get_edges(cc_proj, Y, Z)
    del fs_safe

    return membrane, n_membrane, cc_landmarks, cc_top_z, cc_exists


def _detect_falx_geometry(membrane, skull_crop, tent_crop,
                          cc_landmarks, cc_top_z, cc_exists,
                          dx_mm, fs_crop):
    """Detect edge profiles, anchor, junction, and crista galli.

    Returns a populated _FalxGeometry with all landmark data.
    """
    X, Y, Z = membrane.shape

    # Membrane edges
    mem_yz = membrane.any(axis=0)
    mem_top_z, mem_bot_z, mem_exists = _get_edges(mem_yz, Y, Z)

    # Tentorium edges
    tent_top_z = np.full(Y, -1, dtype=int)
    tent_exists = np.zeros(Y, dtype=bool)
    if tent_crop is not None:
        tent_yz = tent_crop.any(axis=0)
        tent_top_z, _, tent_exists = _get_edges(tent_yz, Y, Z)
        del tent_yz

    mem_y_min = int(np.where(mem_exists)[0].min())
    mem_y_max = int(np.where(mem_exists)[0].max())

    # Junction and anchor detection
    junction_y = mem_y_min
    if tent_exists.any():
        valid = tent_exists & mem_exists
        if valid.any():
            valid_y = np.where(valid)[0]
            junction_y = int(valid_y[np.argmax(tent_top_z[valid_y])])

    splenium_y = cc_landmarks.get("splenium", (junction_y, 0))[0]
    body_y = cc_landmarks.get("body", (splenium_y, 0))[0]
    genu_y = cc_landmarks.get("genu", (body_y, 0))[0]

    # Tent midline anchor (vectorized via _get_edges)
    mid_x = X // 2
    if tent_crop is not None:
        tent_mid = tent_crop[mid_x]
        tent_mid_top, _, tent_mid_exists = _get_edges(tent_mid, Y, Z)
        tent_mid_ys = np.where(tent_mid_exists)[0]
    else:
        tent_mid_ys = np.array([], dtype=int)

    if len(tent_mid_ys) > 0:
        anchor_y = int(tent_mid_ys.max())
        anchor_z = float(tent_mid_top[anchor_y])
    else:
        anchor_y = junction_y
        anchor_z = (float(tent_top_z[junction_y]) if tent_exists[junction_y]
                    else float(mem_bot_z[junction_y]))
    print(f"  Anchor (midline tent end): y={anchor_y}, z={anchor_z:.0f}")

    # Crista galli
    crista_y, crista_z = _detect_crista_galli(fs_crop, mid_x, genu_y)

    return _FalxGeometry(
        mem_top_z=mem_top_z, mem_bot_z=mem_bot_z, mem_exists=mem_exists,
        mem_y_min=mem_y_min, mem_y_max=mem_y_max,
        tent_top_z=tent_top_z, tent_exists=tent_exists,
        cc_landmarks=cc_landmarks, cc_top_z=cc_top_z, cc_exists=cc_exists,
        anchor_y=anchor_y, anchor_z=anchor_z,
        crista_y=crista_y, crista_z=crista_z,
        genu_y=genu_y, mid_x=mid_x,
    )


def _rasterize_falx_cookie(membrane, inner_ys, inner_zs, geo, tent_crop,
                           crop_slices, full_shape, n_membrane):
    """Polygon fill + tentorium cut → falx_mask on full grid.

    Returns falx_mask (boolean, full grid shape).
    """
    X, Y, Z = membrane.shape

    # Collect and rasterize polygon
    tent_ys_arr = np.where(geo.tent_exists)[0]
    tent_post_y = (int(tent_ys_arr.min()) if len(tent_ys_arr) > 0
                   else geo.anchor_y)

    poly_y, poly_z = _collect_falx_polygon(
        inner_ys, inner_zs, geo, tent_post_y)

    rr, cc = draw_polygon(poly_y, poly_z, shape=(Y, Z))
    cookie = np.zeros((Y, Z), dtype=bool)
    cookie[rr, cc] = True
    for i in range(len(poly_y) - 1):
        rr_e, cc_e = draw_line(
            np.clip(int(round(poly_y[i])), 0, Y - 1),
            np.clip(int(round(poly_z[i])), 0, Z - 1),
            np.clip(int(round(poly_y[i + 1])), 0, Y - 1),
            np.clip(int(round(poly_z[i + 1])), 0, Z - 1))
        cookie[rr_e, cc_e] = True

    cookie_3d = np.broadcast_to(cookie[np.newaxis, :, :], (X, Y, Z))
    falx_crop = membrane & cookie_3d
    del cookie_3d, cookie

    n_falx_crop = int(falx_crop.sum())
    print(f"  After cookie-cut: {n_falx_crop} voxels "
          f"(from {n_membrane} membrane)")

    # Cut falx below the tentorium surface
    if tent_crop is not None:
        tent_below = np.zeros((Y, Z), dtype=bool)
        for y in range(Y):
            if geo.tent_exists[y]:
                tent_below[y, :geo.tent_top_z[y]] = True
        tent_below_3d = np.broadcast_to(
            tent_below[np.newaxis, :, :], (X, Y, Z))
        n_below = int((falx_crop & tent_below_3d).sum())
        falx_crop &= ~tent_below_3d
        del tent_below, tent_below_3d
        print(f"  Tentorium cut: removed {n_below} voxels below tent surface")

    # Write back to full grid
    falx_mask = np.zeros(full_shape, dtype=bool)
    falx_mask[crop_slices] = falx_crop
    del falx_crop

    n_falx = int(np.count_nonzero(falx_mask))
    print(f"Falx cerebri: {n_falx} voxels")

    return falx_mask


def reconstruct_falx(mat, fs, skull_sdf, dx_mm, crop_slices,
                     thickness_mm=1.0, tent_mask=None):
    """Reconstruct the falx cerebri via PCHIP + Bezier free edge.

    Full phi=0 midplane membrane through all intracranial tissue,
    shaped by a PCHIP-interpolated free edge with Bezier genu-crista
    segment, using Kayalioglu ratios and Frassanito notch constraint.
    Rasterized via polygon fill.

    Returns falx_mask (boolean).
    """
    # Crop all inputs once
    fs_crop = fs[crop_slices]
    skull_crop = skull_sdf[crop_slices]
    tent_crop = tent_mask[crop_slices] if tent_mask is not None else None

    # Phase 1: EDT pair → midplane membrane + CC landmarks
    membrane, n_membrane, cc_landmarks, cc_top_z, cc_exists = \
        _compute_midplane_membrane(fs_crop, skull_crop, dx_mm, thickness_mm)

    # Phase 2: Edge profiles, anchor, crista galli → geometry bundle
    geo = _detect_falx_geometry(
        membrane, skull_crop, tent_crop,
        cc_landmarks, cc_top_z, cc_exists, dx_mm, fs_crop)
    del fs_crop, skull_crop

    # Phase 3: Free edge curve (PCHIP + Bezier)
    pchip_ys, pchip_zs, genu_ctrl_z = _build_free_edge_controls(geo)

    outer_y, outer_z, posterior_area_vox2 = _collect_boundary_polylines(geo)

    result = _solve_bezier_shape(
        pchip_ys, pchip_zs, genu_ctrl_z,
        outer_y, outer_z, posterior_area_vox2, geo, dx_mm)
    if result is None:
        return None
    inner_ys, inner_zs = result

    # Phase 4: Polygon rasterization + tentorium cut
    return _rasterize_falx_cookie(
        membrane, inner_ys, inner_zs, geo, tent_crop,
        crop_slices, mat.shape, n_membrane)


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
    # Tentorial notch sits at the superior third of the brainstem
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
    mid_xi = int(mid_x)

    # Cerebellar mask for the z-slab: (X, Y, n_z)
    cereb_slab = (mat_crop[:, :, z_start:tent_z + 1] == 4) | \
                 (mat_crop[:, :, z_start:tent_z + 1] == 5)

    # Left hemisphere: any cerebellar voxel in x < mid_x per (y, z)?
    left_any = cereb_slab[:mid_xi, :, :].any(axis=0)        # (Y, n_z)
    # Right hemisphere: any cerebellar voxel in x >= mid_x per (y, z)?
    right_any = cereb_slab[mid_xi:, :, :].any(axis=0)       # (Y, n_z)
    both_present = left_any & right_any                      # (Y, n_z)

    if not both_present.any():
        return None

    # Left medial edge = last (max) x in left half; right medial = mid_x + first (min) x in right half
    # Flip left half so argmax gives last True
    left_medial = mid_xi - 1 - np.argmax(cereb_slab[:mid_xi, :, :][::-1, :, :], axis=0)  # (Y, n_z)
    right_medial = mid_xi + np.argmax(cereb_slab[mid_xi:, :, :], axis=0)                  # (Y, n_z)

    gap_w = (right_medial - left_medial).astype(np.float32) * dx_mm  # (Y, n_z)

    # Reject noise (<5mm) and leakage (>80mm); literature MNW is ~25-35mm
    valid = both_present & (gap_w > 5) & (gap_w < 80)

    if not valid.any():
        return None

    # MNW = max of median gap at each y (across z-levels)
    valid_ys = np.where(valid.any(axis=1))[0]
    mnw = 0.0
    for y in valid_ys:
        m = valid[y]
        mnw = max(mnw, float(np.median(gap_w[y, m])))
    length = mnw * _TENT_NOTCH_AR
    buffer_mm = 2.0  # pad so exclusion ellipse doesn't clip the notch edge

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
    section("Membrane Continuity")

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

    section("Remaining CSF Components")

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

    section("Dural Membrane Volumes")
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
    section("Thickness Estimate")

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
    falx_mm = (affine[:3, :3] @ falx_ijk.T).T + affine[:3, 3]

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
        section("Medial Wall Proximity")
        for label, median_d, p95_d in results:
            print(f"  {label} pial surface: median={median_d:.1f} mm, "
                  f"95th={p95_d:.1f} mm")


def print_junction_thickness(falx_mask, tent_mask, dx_mm, mat):
    """Report overlap extent and max z-thickness at the falx-tentorium junction."""
    overlap = falx_mask & tent_mask
    n_overlap = int(np.count_nonzero(overlap))

    section("Falx-Tentorium Junction")

    if n_overlap == 0:
        print("  No overlap between falx and tentorium")
        return

    voxel_vol_ml = dx_mm ** 3 / 1000.0
    print(f"  Overlap: {n_overlap} voxels ({n_overlap * voxel_vol_ml:.2f} mL)")

    # Max contiguous z-run at junction
    overlap_ijk = np.argwhere(overlap)
    if len(overlap_ijk) > 0:
        # Sort by (x, y, z) so same-column voxels are adjacent
        order = np.lexsort((overlap_ijk[:, 2], overlap_ijk[:, 1], overlap_ijk[:, 0]))
        sorted_ijk = overlap_ijk[order]

        # Column breaks: where x or y changes between consecutive rows
        col_break = np.diff(sorted_ijk[:, :2], axis=0).any(axis=1)
        # Z-gaps within same column
        z_gap = np.diff(sorted_ijk[:, 2]) > 1
        # A run ends at column breaks or z-gaps
        run_break = col_break | z_gap

        # Run lengths from break positions
        break_idx = np.where(run_break)[0]
        starts = np.concatenate([[0], break_idx + 1])
        ends = np.concatenate([break_idx + 1, [len(sorted_ijk)]])
        max_z_run = int((ends - starts).max())

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

    if falx_mask is None:
        raise ValueError("Falx reconstruction failed (degenerate geometry)")

    # Merge into material map
    n_falx, n_tent, n_overlap, n_total = merge_dural(mat, falx_mask, tent_mask)
    print(f"\nMerged: {n_total} total dural voxels "
          f"(falx={n_falx}, tent={n_tent}, overlap={n_overlap})")

    # Validation
    print_volumes(n_falx, n_tent, n_overlap, n_total, dx_mm)
    print_membrane_continuity(falx_mask, tent_mask, dx_mm)
    print_thickness_estimate(falx_mask, tent_mask, dx_mm)
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
