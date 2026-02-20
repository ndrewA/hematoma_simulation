"""Prototype multiple skull SDF estimation approaches.

Each approach produces a skull_interior binary mask at source resolution.
All are evaluated against SimNIBS ground truth using the same metrics.

Usage:
    python -m preprocessing.prototype_skull_approaches --subject 157336 --profile debug
"""

import argparse
import json
import math
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
)

from preprocessing.utils import (
    PROFILES,
    build_ball,
    processed_dir,
    raw_dir,
    resample_to_grid,
)


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════
def load_data(subject_id, profile):
    """Load all shared data needed for prototyping."""
    raw = raw_dir(subject_id)
    out = processed_dir(subject_id, profile)
    _, dx = PROFILES[profile]

    brain_img = nib.load(str(raw / "brainmask_fs.nii.gz"))
    brain_mask = brain_img.get_fdata() > 0.5
    source_affine = brain_img.affine.copy()
    head_mask = nib.load(str(raw / "Head.nii.gz")).get_fdata() > 0.5
    t2w = nib.load(str(raw / "T2w_acpc_dc_restore.nii.gz")).get_fdata(
        dtype=np.float32
    )

    diag = np.abs(np.diag(source_affine)[:3])
    voxel_size = float(diag[0])

    with open(out / "grid_meta.json") as f:
        meta = json.load(f)
    grid_affine = np.array(meta["affine_grid_to_phys"])
    N = int(meta["grid_size"])

    # T2w thresholds (brain-referenced z-scores)
    brain_vals = t2w[brain_mask]
    brain_median = float(np.median(brain_vals))
    brain_mad = float(np.median(np.abs(brain_vals - brain_median)))
    brain_scale = brain_mad / 0.6745

    simnibs_path = raw.parent / "final_tissues.nii.gz"
    simnibs_img = nib.load(str(simnibs_path))

    print(f"Subject: {subject_id}, Profile: {profile}")
    print(f"Source shape: {brain_mask.shape}, voxel: {voxel_size:.2f} mm")
    print(f"Brain T2w: median={brain_median:.0f}, scale={brain_scale:.0f}")
    print()

    return {
        "brain_mask": brain_mask,
        "head_mask": head_mask,
        "source_affine": source_affine,
        "t2w": t2w,
        "voxel_size": voxel_size,
        "grid_affine": grid_affine,
        "N": N,
        "dx": dx,
        "out_dir": out,
        "simnibs_img": simnibs_img,
        "brain_median": brain_median,
        "brain_scale": brain_scale,
    }


def compute_reference(data):
    """Compute SimNIBS inner skull boundary at sim grid. Called once."""
    t0 = time.time()
    N = data["N"]
    dx = data["dx"]
    gs = (N, N, N)

    simnibs_img = data["simnibs_img"]
    simnibs_data = np.asarray(simnibs_img.dataobj, dtype=np.int16)
    if simnibs_data.ndim == 4 and simnibs_data.shape[3] == 1:
        simnibs_data = simnibs_data[:, :, :, 0]

    labels_sim = resample_to_grid(
        (simnibs_data, simnibs_img.affine),
        data["grid_affine"],
        gs,
        order=0,
        cval=0,
        dtype=np.int16,
    )
    del simnibs_data

    intracranial = np.isin(labels_sim, [1, 2, 3, 9])
    bone = np.isin(labels_sim, [4, 7, 8])
    boundary = intracranial & binary_dilation(bone)

    # Regional z-splits
    zs = np.where(boundary.any(axis=(0, 1)))[0]
    z_min, z_max = int(zs[0]), int(zs[-1])
    z_span = z_max - z_min
    z_cuts = [z_min, z_min + z_span // 3, z_min + 2 * z_span // 3, z_max + 1]

    # Brain mask at sim grid (for containment check)
    brain_sim = (
        np.asarray(
            nib.load(str(data["out_dir"] / "brain_mask.nii.gz")).dataobj,
            dtype=np.uint8,
        )
        > 0
    )

    elapsed = time.time() - t0
    n_bnd = int(boundary.sum())
    print(f"SimNIBS reference: {n_bnd:,} boundary voxels ({elapsed:.1f}s)")

    return {
        "boundary": boundary,
        "z_cuts": z_cuts,
        "brain_sim": brain_sim,
    }


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════
def evaluate(skull_interior, src_affine, data, ref, label):
    """Compute SDF from skull_interior mask, compare to SimNIBS."""
    t0 = time.time()
    vs = data["voxel_size"]
    N = data["N"]
    dx = data["dx"]
    gs = (N, N, N)
    samp = (vs, vs, vs)

    # EDT → signed SDF
    dt_out = distance_transform_edt(~skull_interior, sampling=samp)
    dt_in = distance_transform_edt(skull_interior, sampling=samp)
    sdf = (dt_out - dt_in).astype(np.float32)
    del dt_out, dt_in
    sdf = gaussian_filter(sdf, sigma=1.0 / vs).astype(np.float32)

    # Resample to sim grid
    sdf_sim = resample_to_grid(
        (sdf, src_affine),
        data["grid_affine"],
        gs,
        order=1,
        cval=100.0,
        dtype=np.float32,
    )
    del sdf

    boundary = ref["boundary"]
    brain_sim = ref["brain_sim"]
    z_cuts = ref["z_cuts"]

    # Metrics at SimNIBS boundary
    sdf_at_bnd = sdf_sim[boundary]
    median_err = float(np.median(sdf_at_bnd))
    mae = float(np.mean(np.abs(sdf_at_bnd)))

    # Brain containment
    n_outside = int((brain_sim & (sdf_sim >= 0)).sum())

    # ICV
    icv_ml = int((sdf_sim < 0).sum()) * dx**3 / 1000.0

    # Regional MAE
    regional = {}
    for name, z_lo, z_hi in [
        ("inf", z_cuts[0], z_cuts[1]),
        ("mid", z_cuts[1], z_cuts[2]),
        ("sup", z_cuts[2], z_cuts[3]),
    ]:
        rmask = boundary.copy()
        rmask[:, :, :z_lo] = False
        rmask[:, :, z_hi:] = False
        if rmask.any():
            regional[name] = float(np.mean(np.abs(sdf_sim[rmask])))

    elapsed = time.time() - t0

    return {
        "label": label,
        "median": median_err,
        "mae": mae,
        "p5": float(np.percentile(sdf_at_bnd, 5)),
        "p95": float(np.percentile(sdf_at_bnd, 95)),
        "brain_out": n_outside,
        "icv": icv_ml,
        "regional": regional,
        "time": elapsed,
    }


def print_table(results):
    """Print comparison table."""
    hdr = (
        f"{'Approach':<40} {'Median':>7} {'MAE':>6} {'P5':>7} {'P95':>6}"
        f" {'Brn∉':>5} {'ICV':>7}"
        f" {'inf':>5} {'mid':>5} {'sup':>5} {'T':>4}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        reg = r["regional"]
        print(
            f"{r['label']:<40} {r['median']:>+7.2f} {r['mae']:>6.2f}"
            f" {r['p5']:>+7.2f} {r['p95']:>+6.2f}"
            f" {r['brain_out']:>5d} {r['icv']:>7.1f}"
            f" {reg.get('inf', 0):>5.2f} {reg.get('mid', 0):>5.2f}"
            f" {reg.get('sup', 0):>5.2f} {r['time']:>4.0f}"
        )
    print("=" * len(hdr))


# ══════════════════════════════════════════════════════════════════════
# Shared building blocks
# ══════════════════════════════════════════════════════════════════════
def morph_close_edt(mask, r_vox):
    """EDT-based morphological closing."""
    dt = distance_transform_edt(~mask)
    dilated = dt <= r_vox
    del dt
    dt = distance_transform_edt(dilated)
    del dilated
    return dt > r_vox


def bone_threshold(data, bone_z):
    """Compute T2w bone threshold from z-score."""
    return data["brain_median"] + bone_z * data["brain_scale"]


def pad_for_closing(data, r_close_vox, extra=10):
    """Pad brain + head masks inferiorly for morphological closing."""
    vs = data["voxel_size"]
    pad_z = r_close_vox + extra
    bp = np.pad(
        data["brain_mask"], ((0, 0), (0, 0), (pad_z, 0)), constant_values=False
    )
    hp = np.pad(
        data["head_mask"], ((0, 0), (0, 0), (pad_z, 0)), constant_values=False
    )
    A_pad = data["source_affine"].copy()
    A_pad[2, 3] -= pad_z * vs
    return bp, hp, A_pad, pad_z


# ══════════════════════════════════════════════════════════════════════
# Approach implementations
# ══════════════════════════════════════════════════════════════════════


# --- A: Baseline (current pipeline) ---
def approach_baseline(data):
    """Morph close 10mm + dilate 0.5mm + T2w conditional erosion."""
    vs = data["voxel_size"]
    r_close = math.ceil(10.0 / vs)
    r_dilate = math.ceil(0.5 / vs)
    bp, hp, A_pad, pad_z = pad_for_closing(data, r_close + r_dilate)

    closed = morph_close_edt(bp, r_close)
    del bp
    skull_interior = binary_dilation(closed, build_ball(r_dilate))
    del closed
    skull_interior &= hp

    # T2w erosion
    bthresh = bone_threshold(data, -1.5)
    csf_thresh = data["brain_median"]
    t2w = data["t2w"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    extracranial = ~brain & head
    is_csf = (t2w > csf_thresh) & extracranial
    is_bone = (t2w < bthresh) & extracranial
    brain_guard = binary_dilation(brain)
    csf_adj = binary_dilation(is_csf)

    si = skull_interior[:, :, pad_z:]
    for _ in range(round(3.0 / vs)):
        surface = si & ~binary_erosion(si)
        to_rm = surface & ~brain_guard & is_bone & csf_adj
        if to_rm.sum() == 0:
            break
        si[to_rm] = False

    return skull_interior, A_pad, "A: baseline (close+dil+erode)"


# --- B: T2w-guided growth from brain mask ---
def approach_growth(data, bone_z=-1.5, max_mm=8.0, dilate_mm=0.0):
    """Grow outward from brain mask, stop at bone-dark T2w voxels."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]
    bthresh = bone_threshold(data, bone_z)

    not_bone = (t2w >= bthresh) & head
    skull_interior = brain.copy()
    max_iters = round(max_mm / vs)

    for i in range(max_iters):
        expanded = binary_dilation(skull_interior)
        grow = expanded & ~skull_interior & not_bone
        if grow.sum() == 0:
            break
        skull_interior |= grow

    skull_interior &= head

    # Optional safety dilation
    if dilate_mm > 0:
        r = math.ceil(dilate_mm / vs)
        skull_interior = binary_dilation(skull_interior, build_ball(r))
        skull_interior &= head

    suffix = f"z={bone_z}"
    if dilate_mm > 0:
        suffix += f", +{dilate_mm}mm"
    return skull_interior, data["source_affine"], f"B: growth ({suffix})"


# --- C: Growth from morphologically closed mask ---
def approach_growth_from_closed(data, bone_z=-1.5, max_mm=6.0, close_mm=10.0):
    """Close first, then grow outward through non-bone T2w."""
    vs = data["voxel_size"]
    r_close = math.ceil(close_mm / vs)
    bp, hp, A_pad, pad_z = pad_for_closing(data, r_close)

    closed = morph_close_edt(bp, r_close)
    del bp
    closed &= hp

    bthresh = bone_threshold(data, bone_z)
    t2w_pad = np.pad(
        data["t2w"] >= bthresh, ((0, 0), (0, 0), (pad_z, 0)), constant_values=False
    )
    t2w_pad &= hp

    skull_interior = closed.copy()
    max_iters = round(max_mm / vs)

    for i in range(max_iters):
        expanded = binary_dilation(skull_interior)
        grow = expanded & ~skull_interior & t2w_pad
        if grow.sum() == 0:
            break
        skull_interior |= grow

    skull_interior &= hp
    return skull_interior, A_pad, f"C: close+growth (z={bone_z})"


# --- D: Ray casting along surface normals ---
def approach_raycast(data, bone_z=-1.5, max_mm=8.0, step_mm=0.5, close_mm=10.0):
    """Per-surface-voxel T2w ray casting along EDT normals."""
    vs = data["voxel_size"]
    r_close = math.ceil(close_mm / vs)
    bp, hp, A_pad, pad_z = pad_for_closing(data, r_close)

    closed = morph_close_edt(bp, r_close)
    del bp
    closed &= hp

    # Surface voxels
    surface = closed & ~binary_erosion(closed)
    sc = np.argwhere(surface).astype(np.float64)  # (M, 3)
    M = len(sc)
    print(f"  D: raycast — {M:,} surface voxels")

    # Outward normals via EDT gradient
    dt = distance_transform_edt(~closed, sampling=(vs, vs, vs))
    shape = np.array(dt.shape)
    normals = np.zeros((M, 3), dtype=np.float64)
    si, sj, sk = sc[:, 0].astype(int), sc[:, 1].astype(int), sc[:, 2].astype(int)

    for ax in range(3):
        idx = sc[:, ax].astype(int)
        idx_m = np.clip(idx - 1, 0, shape[ax] - 1)
        idx_p = np.clip(idx + 1, 0, shape[ax] - 1)

        cm = [si.copy(), sj.copy(), sk.copy()]
        cp = [si.copy(), sj.copy(), sk.copy()]
        cm[ax] = idx_m
        cp[ax] = idx_p

        normals[:, ax] = (dt[cp[0], cp[1], cp[2]] - dt[cm[0], cm[1], cm[2]]) / (
            2 * vs
        )
    del dt

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-8)

    # Pad T2w
    t2w_pad = np.pad(data["t2w"], ((0, 0), (0, 0), (pad_z, 0)), constant_values=0.0)
    bthresh = bone_threshold(data, bone_z)

    # Cast rays
    max_steps = round(max_mm / step_mm)
    step_vox = step_mm / vs
    detected = np.full(M, max_mm, dtype=np.float32)
    t2w_shape = np.array(t2w_pad.shape, dtype=np.float64)

    for s in range(1, max_steps + 1):
        coords = sc + s * step_vox * normals
        in_bounds = np.all((coords >= 0) & (coords < t2w_shape[None, :] - 1), axis=1)

        vals = np.full(M, 0.0)
        if in_bounds.any():
            vc = coords[in_bounds].T
            vals[in_bounds] = map_coordinates(t2w_pad, vc, order=1, cval=0.0)

        crossed = (vals < bthresh) & in_bounds & (detected >= max_mm)
        detected[crossed] = s * step_mm

    del t2w_pad

    # Propagate detected distances to all voxels via nearest-surface EDT
    offset_vol = np.zeros(closed.shape, dtype=np.float32)
    offset_vol[si, sj, sk] = detected

    _, indices = distance_transform_edt(~surface, return_indices=True)
    offset_prop = offset_vol[indices[0], indices[1], indices[2]]
    del offset_vol, indices

    # skull_interior = distance from closed surface < detected offset
    brain_dist = distance_transform_edt(~closed, sampling=(vs, vs, vs))
    skull_interior = (brain_dist <= offset_prop) | closed
    del brain_dist, offset_prop
    skull_interior &= hp

    return skull_interior, A_pad, f"D: raycast (z={bone_z})"


# --- E: Ray casting with gradient detection (threshold-free) ---
def approach_raycast_gradient(
    data, max_mm=8.0, step_mm=0.5, close_mm=10.0, smooth_mm=1.0
):
    """Ray cast, detect steepest T2w drop instead of fixed threshold."""
    vs = data["voxel_size"]
    r_close = math.ceil(close_mm / vs)
    bp, hp, A_pad, pad_z = pad_for_closing(data, r_close)

    closed = morph_close_edt(bp, r_close)
    del bp
    closed &= hp

    surface = closed & ~binary_erosion(closed)
    sc = np.argwhere(surface).astype(np.float64)
    M = len(sc)
    print(f"  E: raycast_gradient — {M:,} surface voxels")

    # Normals
    dt = distance_transform_edt(~closed, sampling=(vs, vs, vs))
    shape = np.array(dt.shape)
    normals = np.zeros((M, 3), dtype=np.float64)
    si, sj, sk = sc[:, 0].astype(int), sc[:, 1].astype(int), sc[:, 2].astype(int)

    for ax in range(3):
        idx = sc[:, ax].astype(int)
        idx_m = np.clip(idx - 1, 0, shape[ax] - 1)
        idx_p = np.clip(idx + 1, 0, shape[ax] - 1)
        cm = [si.copy(), sj.copy(), sk.copy()]
        cp = [si.copy(), sj.copy(), sk.copy()]
        cm[ax] = idx_m
        cp[ax] = idx_p
        normals[:, ax] = (dt[cp[0], cp[1], cp[2]] - dt[cm[0], cm[1], cm[2]]) / (
            2 * vs
        )
    del dt

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-8)

    # Pad T2w
    t2w_pad = np.pad(data["t2w"], ((0, 0), (0, 0), (pad_z, 0)), constant_values=0.0)

    # Sample profiles
    max_steps = round(max_mm / step_mm)
    step_vox = step_mm / vs
    t2w_shape = np.array(t2w_pad.shape, dtype=np.float64)
    profiles = np.zeros((M, max_steps + 1), dtype=np.float32)

    for s in range(max_steps + 1):
        coords = sc + s * step_vox * normals
        in_bounds = np.all((coords >= 0) & (coords < t2w_shape[None, :] - 1), axis=1)
        if in_bounds.any():
            vc = coords[in_bounds].T
            profiles[in_bounds, s] = map_coordinates(t2w_pad, vc, order=1, cval=0.0)

    del t2w_pad

    # Smooth profiles along ray direction
    smooth_steps = max(1, round(smooth_mm / step_mm))
    kernel = np.ones(smooth_steps) / smooth_steps
    from scipy.ndimage import convolve1d

    profiles_smooth = convolve1d(profiles, kernel, axis=1, mode="nearest")

    # Find steepest negative gradient (largest T2w drop)
    grad = np.diff(profiles_smooth, axis=1)  # (M, max_steps)
    # Only consider drops from bright-enough signal (avoid noise in bone)
    min_signal = data["brain_median"] * 0.3
    bright_enough = profiles_smooth[:, :-1] > min_signal

    grad_masked = np.where(bright_enough, grad, 0.0)
    min_grad_step = np.argmin(grad_masked, axis=1)  # step with steepest drop
    detected = (min_grad_step + 0.5) * step_mm  # midpoint of the drop

    # Clamp to [1mm, max_mm]
    detected = np.clip(detected, 1.0, max_mm).astype(np.float32)

    del profiles, profiles_smooth, grad, grad_masked

    # Propagate and build mask (same as approach_raycast)
    offset_vol = np.zeros(closed.shape, dtype=np.float32)
    offset_vol[si, sj, sk] = detected

    _, indices = distance_transform_edt(~surface, return_indices=True)
    offset_prop = offset_vol[indices[0], indices[1], indices[2]]
    del offset_vol, indices

    brain_dist = distance_transform_edt(~closed, sampling=(vs, vs, vs))
    skull_interior = (brain_dist <= offset_prop) | closed
    del brain_dist, offset_prop
    skull_interior &= hp

    return skull_interior, A_pad, "E: raycast_gradient"


# --- F: GMM growth ---
def approach_gmm_growth(data, max_mm=6.0, close_mm=10.0, n_components=3):
    """Fit GMM to extracranial T2w, use bone posterior for growth."""
    vs = data["voxel_size"]
    r_close = math.ceil(close_mm / vs)
    bp, hp, A_pad, pad_z = pad_for_closing(data, r_close)

    closed = morph_close_edt(bp, r_close)
    del bp
    closed &= hp

    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]

    # Fit GMM on extracranial voxels
    extracranial = ~brain & head
    vals = t2w[extracranial].ravel()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(vals), min(200_000, len(vals)), replace=False)

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(vals[idx].reshape(-1, 1))

    order = np.argsort(gmm.means_.flatten())
    means = gmm.means_.flatten()[order]
    stds = np.sqrt(gmm.covariances_.flatten()[order])
    weights = gmm.weights_[order]

    print(f"  F: GMM components:")
    for k in range(n_components):
        print(f"    {k}: mean={means[k]:.0f}, std={stds[k]:.0f}, w={weights[k]:.2f}")

    # Find bone-tissue crossing point
    x = np.linspace(means[0], means[1], 1000)
    p0 = weights[0] * np.exp(-0.5 * ((x - means[0]) / stds[0]) ** 2) / stds[0]
    p1 = weights[1] * np.exp(-0.5 * ((x - means[1]) / stds[1]) ** 2) / stds[1]
    cross_idx = np.argmin(np.abs(p0 - p1))
    bthresh = float(x[cross_idx])
    print(f"    Bone threshold (crossing): {bthresh:.0f}")

    # Growth from closed surface
    not_bone = np.pad(
        t2w >= bthresh, ((0, 0), (0, 0), (pad_z, 0)), constant_values=False
    )
    not_bone &= hp
    skull_interior = closed.copy()
    max_iters = round(max_mm / vs)

    for i in range(max_iters):
        expanded = binary_dilation(skull_interior)
        grow = expanded & ~skull_interior & not_bone
        if grow.sum() == 0:
            break
        skull_interior |= grow

    skull_interior &= hp
    return skull_interior, A_pad, f"F: gmm_growth (n={n_components})"


# --- G: Growth from brain + small closing to smooth ---
def approach_growth_then_smooth(data, bone_z=-1.5, max_mm=8.0, smooth_r_mm=3.0):
    """Grow from brain, then light morphological closing for smoothness."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]
    bthresh = bone_threshold(data, bone_z)

    not_bone = (t2w >= bthresh) & head
    skull_interior = brain.copy()
    max_iters = round(max_mm / vs)

    for i in range(max_iters):
        expanded = binary_dilation(skull_interior)
        grow = expanded & ~skull_interior & not_bone
        if grow.sum() == 0:
            break
        skull_interior |= grow

    skull_interior &= head

    # Light closing to smooth rough edges
    r_smooth = math.ceil(smooth_r_mm / vs)
    skull_interior = morph_close_edt(skull_interior, r_smooth)
    skull_interior &= head

    return (
        skull_interior,
        data["source_affine"],
        f"G: growth+smooth (z={bone_z}, r={smooth_r_mm})",
    )


# --- H: Growth + brain margin (ensures containment without dilation everywhere) ---
def approach_growth_margin(data, bone_z=-1.5, max_mm=8.0, margin_vox=1):
    """Grow from brain, then union with dilated brain for minimum margin."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]
    bthresh = bone_threshold(data, bone_z)

    not_bone = (t2w >= bthresh) & head
    skull_interior = brain.copy()
    max_iters = round(max_mm / vs)

    for i in range(max_iters):
        expanded = binary_dilation(skull_interior)
        grow = expanded & ~skull_interior & not_bone
        if grow.sum() == 0:
            break
        skull_interior |= grow

    # Ensure minimum margin around brain (guarantees containment)
    brain_margin = brain.copy()
    for _ in range(margin_vox):
        brain_margin = binary_dilation(brain_margin)
    skull_interior |= brain_margin
    del brain_margin

    skull_interior &= head
    margin_mm = margin_vox * vs
    return (
        skull_interior,
        data["source_affine"],
        f"H: growth+margin (z={bone_z}, m={margin_mm:.1f}mm)",
    )


# --- I: Growth from brain with ray-cast along EDT gradient ---
def approach_growth_raycast(data, bone_z=-1.5, max_mm=8.0, step_mm=0.5):
    """Growth from brain mask, using ray-cast to find bone per surface voxel."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]
    bthresh = bone_threshold(data, bone_z)

    # Get brain surface voxels
    surface = brain & ~binary_erosion(brain)
    sc = np.argwhere(surface).astype(np.float64)
    M = len(sc)
    print(f"  I: growth_raycast — {M:,} brain surface voxels")

    # Outward normals from brain EDT
    dt = distance_transform_edt(~brain, sampling=(vs, vs, vs))
    shape = np.array(dt.shape)
    normals = np.zeros((M, 3), dtype=np.float64)
    si, sj, sk = sc[:, 0].astype(int), sc[:, 1].astype(int), sc[:, 2].astype(int)

    for ax in range(3):
        idx = sc[:, ax].astype(int)
        idx_m = np.clip(idx - 1, 0, shape[ax] - 1)
        idx_p = np.clip(idx + 1, 0, shape[ax] - 1)
        cm = [si.copy(), sj.copy(), sk.copy()]
        cp = [si.copy(), sj.copy(), sk.copy()]
        cm[ax] = idx_m
        cp[ax] = idx_p
        normals[:, ax] = (dt[cp[0], cp[1], cp[2]] - dt[cm[0], cm[1], cm[2]]) / (
            2 * vs
        )
    del dt

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-8)

    # Cast rays from brain surface outward
    max_steps = round(max_mm / step_mm)
    step_vox = step_mm / vs
    detected = np.full(M, max_mm, dtype=np.float32)
    t2w_shape = np.array(t2w.shape, dtype=np.float64)

    for s in range(1, max_steps + 1):
        coords = sc + s * step_vox * normals
        in_bounds = np.all(
            (coords >= 0) & (coords < t2w_shape[None, :] - 1), axis=1
        )
        vals = np.full(M, 0.0)
        if in_bounds.any():
            vc = coords[in_bounds].T
            vals[in_bounds] = map_coordinates(t2w, vc, order=1, cval=0.0)
        crossed = (vals < bthresh) & in_bounds & (detected >= max_mm)
        detected[crossed] = s * step_mm

    # Minimum detected distance = 1 voxel (safety margin)
    detected = np.maximum(detected, vs)

    # Propagate detected distances: for every voxel, use nearest brain
    # surface voxel's detected offset
    offset_vol = np.zeros(brain.shape, dtype=np.float32)
    offset_vol[si, sj, sk] = detected

    _, indices = distance_transform_edt(~surface, return_indices=True)
    offset_prop = offset_vol[indices[0], indices[1], indices[2]]
    del offset_vol, indices

    brain_dist = distance_transform_edt(~brain, sampling=(vs, vs, vs))
    skull_interior = (brain_dist <= offset_prop) | brain
    del brain_dist, offset_prop
    skull_interior &= head

    return (
        skull_interior,
        data["source_affine"],
        f"I: growth_raycast (z={bone_z})",
    )


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main(argv=None):
    parser = argparse.ArgumentParser(description="Prototype skull SDF approaches.")
    parser.add_argument("--subject", default="157336")
    parser.add_argument("--profile", default="debug", choices=list(PROFILES.keys()))
    args = parser.parse_args(argv)

    data = load_data(args.subject, args.profile)
    ref = compute_reference(data)

    approaches = [
        # A: Current pipeline baseline
        (approach_baseline, {}),
        # B: Growth from brain mask — sweep bone_z
        (approach_growth, {"bone_z": -2.0}),
        (approach_growth, {"bone_z": -1.5}),
        (approach_growth, {"bone_z": -1.0}),
        (approach_growth, {"bone_z": -0.5}),
        # B+: Growth + safety dilation
        (approach_growth, {"bone_z": -1.5, "dilate_mm": 0.5}),
        (approach_growth, {"bone_z": -1.0, "dilate_mm": 0.5}),
        # C: Close + growth
        (approach_growth_from_closed, {"bone_z": -1.5}),
        (approach_growth_from_closed, {"bone_z": -1.0}),
        # D: Ray casting from closed surface
        (approach_raycast, {"bone_z": -1.5}),
        (approach_raycast, {"bone_z": -1.0}),
        # E: Ray casting with gradient detection
        (approach_raycast_gradient, {}),
        # G: Growth + light closing for smoothness
        (approach_growth_then_smooth, {"bone_z": -1.5}),
        (approach_growth_then_smooth, {"bone_z": -1.0, "smooth_r_mm": 2.0}),
        # H: Growth + brain margin (best of both: MAE + containment)
        (approach_growth_margin, {"bone_z": -1.5, "margin_vox": 1}),
        (approach_growth_margin, {"bone_z": -1.0, "margin_vox": 1}),
        (approach_growth_margin, {"bone_z": -0.5, "margin_vox": 1}),
        (approach_growth_margin, {"bone_z": -1.0, "margin_vox": 2}),
        # I: Ray casting from brain surface (no closing)
        (approach_growth_raycast, {"bone_z": -1.5}),
        (approach_growth_raycast, {"bone_z": -1.0}),
    ]

    # F: GMM growth (optional, needs sklearn)
    try:
        import sklearn  # noqa: F401

        approaches.append((approach_gmm_growth, {}))
    except ImportError:
        print("sklearn not available, skipping GMM approach")

    results = []
    for fn, kwargs in approaches:
        name = fn.__name__
        print(f"\nRunning {name} ({kwargs})...")
        t0 = time.time()
        si, affine, label = fn(data, **kwargs)
        build_time = time.time() - t0
        print(f"  Built in {build_time:.1f}s, mask sum={int(si.sum()):,}")

        metrics = evaluate(si, affine, data, ref, label)
        results.append(metrics)
        print(
            f"  => median={metrics['median']:+.2f}, MAE={metrics['mae']:.2f}, "
            f"brain∉={metrics['brain_out']}, ICV={metrics['icv']:.1f}"
        )
        del si

    print_table(results)


if __name__ == "__main__":
    main()
