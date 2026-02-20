"""Focused skull SDF experiments — round 2.

Tests more aggressive growth thresholds, GMM, and combination approaches.
Reuses data loading and evaluation from prototype_skull_approaches.

Usage:
    python -m preprocessing.prototype_skull_v2 --subject 157336 --profile debug
"""

import math
import time

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    map_coordinates,
)
from sklearn.mixture import GaussianMixture

from preprocessing.prototype_skull_approaches import (
    bone_threshold,
    compute_reference,
    evaluate,
    load_data,
    print_table,
)
from preprocessing.utils import build_ball


# ══════════════════════════════════════════════════════════════════════
# Approaches
# ══════════════════════════════════════════════════════════════════════


def growth(data, bone_z=-1.0, max_mm=8.0):
    """Pure T2w growth from brain mask."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]
    bthresh = bone_threshold(data, bone_z)

    not_bone = (t2w >= bthresh) & head
    si = brain.copy()
    for _ in range(round(max_mm / vs)):
        grow = binary_dilation(si) & ~si & not_bone
        if grow.sum() == 0:
            break
        si |= grow
    si &= head
    return si, data["source_affine"], f"growth (z={bone_z})"


def growth_margin(data, bone_z=-1.0, max_mm=8.0, margin_vox=1):
    """Growth + minimum brain margin for containment."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]
    bthresh = bone_threshold(data, bone_z)

    not_bone = (t2w >= bthresh) & head
    si = brain.copy()
    for _ in range(round(max_mm / vs)):
        grow = binary_dilation(si) & ~si & not_bone
        if grow.sum() == 0:
            break
        si |= grow

    # Union with dilated brain for minimum margin
    bm = brain.copy()
    for _ in range(margin_vox):
        bm = binary_dilation(bm)
    si |= bm
    si &= head
    mm = margin_vox * vs
    return si, data["source_affine"], f"growth+margin (z={bone_z}, m={mm:.1f})"


def growth_twophase(data, csf_z=0.0, bone_z=-1.5, max_mm=8.0, dura_vox=1):
    """Two-phase: grow through CSF (bright), then add dura margin."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]
    csf_thresh = data["brain_median"] + csf_z * data["brain_scale"]

    # Phase 1: grow through CSF-bright voxels only
    is_csf = (t2w >= csf_thresh) & head
    si = brain.copy()
    for _ in range(round(max_mm / vs)):
        grow = binary_dilation(si) & ~si & is_csf
        if grow.sum() == 0:
            break
        si |= grow

    # Phase 2: dura margin — grow N more voxels unconditionally (but within head)
    for _ in range(dura_vox):
        si = binary_dilation(si) & head

    si &= head
    mm = dura_vox * vs
    return (
        si,
        data["source_affine"],
        f"twophase (csf_z={csf_z}, bone_z={bone_z}, dura={mm:.1f}mm)",
    )


def growth_gmm(data, max_mm=8.0, n_components=3):
    """Growth using GMM-derived bone threshold."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]

    # Fit GMM on extracranial T2w
    extra = ~brain & head
    vals = t2w[extra].ravel()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(vals), min(200_000, len(vals)), replace=False)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(vals[idx].reshape(-1, 1))

    order = np.argsort(gmm.means_.flatten())
    means = gmm.means_.flatten()[order]
    stds = np.sqrt(gmm.covariances_.flatten()[order])
    weights = gmm.weights_[order]

    for k in range(n_components):
        print(
            f"    GMM[{k}]: mean={means[k]:.0f}, std={stds[k]:.0f}, w={weights[k]:.2f}"
        )

    # Bone threshold = crossing of bone and tissue Gaussians
    x = np.linspace(means[0], means[1], 1000)
    p0 = weights[0] * np.exp(-0.5 * ((x - means[0]) / stds[0]) ** 2) / stds[0]
    p1 = weights[1] * np.exp(-0.5 * ((x - means[1]) / stds[1]) ** 2) / stds[1]
    bthresh = float(x[np.argmin(np.abs(p0 - p1))])
    print(f"    Bone threshold: {bthresh:.0f}")

    not_bone = (t2w >= bthresh) & head
    si = brain.copy()
    for _ in range(round(max_mm / vs)):
        grow = binary_dilation(si) & ~si & not_bone
        if grow.sum() == 0:
            break
        si |= grow
    si &= head
    return si, data["source_affine"], f"gmm_growth (thresh={bthresh:.0f})"


def growth_raycast_brain(data, bone_z=-1.0, max_mm=8.0, step_mm=0.5, min_mm=0.7):
    """Ray cast from brain surface along normals to detect bone distance."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]
    bthresh = bone_threshold(data, bone_z)

    surface = brain & ~binary_erosion(brain)
    sc = np.argwhere(surface).astype(np.float64)
    M = len(sc)
    si_idx = sc[:, 0].astype(int)
    sj_idx = sc[:, 1].astype(int)
    sk_idx = sc[:, 2].astype(int)
    print(f"    {M:,} brain surface voxels")

    # Normals from brain EDT gradient
    dt = distance_transform_edt(~brain, sampling=(vs, vs, vs))
    shape = np.array(dt.shape)
    normals = np.zeros((M, 3), dtype=np.float64)

    for ax in range(3):
        idx = sc[:, ax].astype(int)
        cm = [si_idx.copy(), sj_idx.copy(), sk_idx.copy()]
        cp = [si_idx.copy(), sj_idx.copy(), sk_idx.copy()]
        cm[ax] = np.clip(idx - 1, 0, shape[ax] - 1)
        cp[ax] = np.clip(idx + 1, 0, shape[ax] - 1)
        normals[:, ax] = (dt[cp[0], cp[1], cp[2]] - dt[cm[0], cm[1], cm[2]]) / (
            2 * vs
        )
    del dt

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-8)

    # Cast rays
    max_steps = round(max_mm / step_mm)
    step_vox = step_mm / vs
    detected = np.full(M, max_mm, dtype=np.float32)
    t2w_shape = np.array(t2w.shape, dtype=np.float64)

    for s in range(1, max_steps + 1):
        coords = sc + s * step_vox * normals
        ib = np.all((coords >= 0) & (coords < t2w_shape[None, :] - 1), axis=1)
        vals = np.full(M, 0.0)
        if ib.any():
            vals[ib] = map_coordinates(t2w, coords[ib].T, order=1, cval=0.0)
        crossed = (vals < bthresh) & ib & (detected >= max_mm)
        detected[crossed] = s * step_mm

    # Enforce minimum distance
    detected = np.maximum(detected, min_mm)

    # Propagate to volume
    offset_vol = np.zeros(brain.shape, dtype=np.float32)
    offset_vol[si_idx, sj_idx, sk_idx] = detected

    _, indices = distance_transform_edt(~surface, return_indices=True)
    offset_prop = offset_vol[indices[0], indices[1], indices[2]]
    del offset_vol, indices

    brain_dist = distance_transform_edt(~brain, sampling=(vs, vs, vs))
    si = (brain_dist <= offset_prop) | brain
    del brain_dist, offset_prop
    si &= head
    return si, data["source_affine"], f"raycast_brain (z={bone_z}, min={min_mm})"


def growth_adaptive(data, max_mm=8.0):
    """Growth with distance-adaptive threshold: tighter near brain, looser far."""
    vs = data["voxel_size"]
    brain = data["brain_mask"]
    head = data["head_mask"]
    t2w = data["t2w"]
    bmed = data["brain_median"]
    bscale = data["brain_scale"]

    # Compute brain distance field
    brain_dist = distance_transform_edt(~brain, sampling=(vs, vs, vs))

    # Near brain (0-2mm): strict threshold (z=-0.5, only grow through bright CSF)
    # Far from brain (>4mm): relaxed threshold (z=-2.0, grow through intermediate tissue)
    z_near = -0.5
    z_far = -2.0
    d_near = 1.0  # mm
    d_far = 4.0  # mm

    # Spatially varying threshold
    alpha = np.clip((brain_dist - d_near) / (d_far - d_near), 0, 1)
    z_local = z_near + alpha * (z_far - z_near)
    thresh_map = bmed + z_local * bscale
    del alpha, z_local

    not_bone = (t2w >= thresh_map) & head
    del thresh_map

    si = brain.copy()
    for _ in range(round(max_mm / vs)):
        grow = binary_dilation(si) & ~si & not_bone
        if grow.sum() == 0:
            break
        si |= grow
    si &= head
    return si, data["source_affine"], "adaptive (z=-0.5...-2.0)"


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="157336")
    parser.add_argument("--profile", default="debug", choices=["debug", "dev", "prod"])
    args = parser.parse_args(argv)

    data = load_data(args.subject, args.profile)
    ref = compute_reference(data)

    approaches = [
        # Reference: best from round 1
        (growth, {"bone_z": -1.0}),
        (growth, {"bone_z": -0.5}),
        (growth_margin, {"bone_z": -0.5, "margin_vox": 1}),
        # More aggressive thresholds
        (growth, {"bone_z": 0.0}),
        (growth, {"bone_z": 0.5}),
        (growth_margin, {"bone_z": 0.0, "margin_vox": 1}),
        (growth_margin, {"bone_z": 0.5, "margin_vox": 1}),
        # Two-phase: CSF growth + dura margin
        (growth_twophase, {"csf_z": 0.0, "dura_vox": 1}),
        (growth_twophase, {"csf_z": 0.0, "dura_vox": 2}),
        (growth_twophase, {"csf_z": 0.5, "dura_vox": 1}),
        (growth_twophase, {"csf_z": 0.5, "dura_vox": 2}),
        # GMM
        (growth_gmm, {}),
        # Ray cast from brain surface
        (growth_raycast_brain, {"bone_z": -1.0, "min_mm": 0.7}),
        (growth_raycast_brain, {"bone_z": -0.5, "min_mm": 0.7}),
        # Adaptive (distance-dependent threshold)
        (growth_adaptive, {}),
    ]

    results = []
    for fn, kwargs in approaches:
        print(f"\n--- {fn.__name__} ({kwargs}) ---")
        t0 = time.time()
        si, affine, label = fn(data, **kwargs)
        build_t = time.time() - t0
        print(f"  Built: {build_t:.1f}s, voxels={int(si.sum()):,}")

        m = evaluate(si, affine, data, ref, label)
        results.append(m)
        print(
            f"  median={m['median']:+.2f} MAE={m['mae']:.2f} "
            f"brain∉={m['brain_out']} ICV={m['icv']:.1f}"
        )
        del si

    print_table(results)


if __name__ == "__main__":
    main()
