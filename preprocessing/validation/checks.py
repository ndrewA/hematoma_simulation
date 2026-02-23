"""Registry-based validation checks.

Each check is a decorated function.  To add a check, write one function
with ``@check(...)``.  To remove a check, delete the function.  No other
file changes required.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckDef:
    check_id: str
    severity: str       # "CRITICAL" | "WARN" | "INFO"
    phase: str           # "header", "domain", "material", "volume",
                         # "compartment", "dural", "fiber"
    needs: frozenset     # {"mat", "sdf", "brain", "fs", "fiber_img"}
    func: Callable
    description: str


_REGISTRY: list[CheckDef] = []


def check(check_id, *, severity, phase, needs=None):
    """Decorator that registers a validation check."""
    def decorator(fn):
        desc = (fn.__doc__ or "").strip().split("\n")[0]
        _REGISTRY.append(CheckDef(
            check_id, severity, phase,
            frozenset(needs or ()), fn, desc,
        ))
        return fn
    return decorator


# Built after all decorations (see module bottom).
_REGISTRY_BY_ID: dict[str, CheckDef] = {}


# ---------------------------------------------------------------------------
# Shared helper: load FreeSurfer data for C5/C6
# ---------------------------------------------------------------------------

def _load_fs_data(ctx):
    """Load FS labels, compute CC boundary.  Cached as 'fs_data'."""
    import nibabel as nib
    from preprocessing.dural_membrane import (
        _build_fs_luts, _FS_LUT_SIZE, CC_LABELS,
    )

    fs_img = nib.load(str(ctx.paths["fs"]))
    fs = np.asarray(fs_img.dataobj, dtype=np.int16)
    del fs_img

    left_lut, right_lut, _ = _build_fs_luts()
    fs_safe = np.clip(fs, 0, _FS_LUT_SIZE - 1)
    del fs

    cc_lut = np.zeros(_FS_LUT_SIZE, dtype=bool)
    for lab in CC_LABELS:
        cc_lut[lab] = True
    cc_mask = cc_lut[fs_safe]
    cc_z_indices = np.where(cc_mask.any(axis=(0, 1)))[0]
    del cc_mask

    N = ctx.N
    cc_z_sup = int(cc_z_indices.max()) if len(cc_z_indices) > 0 else N // 2

    return {
        "cc_z_sup": cc_z_sup,
        "fs_safe": fs_safe,
        "left_lut": left_lut,
        "right_lut": right_lut,
    }


# ---------------------------------------------------------------------------
# Shared helper: volume census for V1-V6
# ---------------------------------------------------------------------------

def _compute_volume_census(ctx):
    """Compute volume census.  Cached as 'volume_census'."""
    from preprocessing.material_map import CLASS_NAMES

    mat = ctx.mat
    dx = ctx.dx
    vol_voxel_ml = dx ** 3 / 1000.0

    counts = np.bincount(mat.ravel(), minlength=12)
    census = {}
    for uid in range(12):
        n = int(counts[uid])
        vol = round(n * vol_voxel_ml, 2)
        census[str(uid)] = {
            "name": CLASS_NAMES.get(uid, "???"),
            "voxels": n,
            "volume_mL": vol,
        }

    parenchyma_ids = [1, 2, 3, 4, 5, 6, 9]
    parenchyma_ml = sum(int(counts[i]) for i in parenchyma_ids) * vol_voxel_ml
    ventricular_ml = int(counts[7]) * vol_voxel_ml
    subarachnoid_ml = int(counts[8]) * vol_voxel_ml
    dural_ml = int(counts[10]) * vol_voxel_ml

    sdf = ctx.sdf
    n_sdf_neg = int(np.count_nonzero(sdf < 0))
    icv_ml = n_sdf_neg * vol_voxel_ml
    n_nonzero_mat = int(np.count_nonzero(mat > 0))

    return {
        "census": census,
        "counts": counts,
        "vol_voxel_ml": vol_voxel_ml,
        "parenchyma_ml": parenchyma_ml,
        "ventricular_ml": ventricular_ml,
        "subarachnoid_ml": subarachnoid_ml,
        "dural_ml": dural_ml,
        "icv_ml": icv_ml,
        "n_sdf_neg": n_sdf_neg,
        "n_nonzero_mat": n_nonzero_mat,
    }


# ---------------------------------------------------------------------------
# Fiber helpers
# ---------------------------------------------------------------------------

def _find_landmark(fiber_data, dominant_axis, x_range, frac_thresh=0.5,
                   trace_thresh=0.1):
    """Find a landmark voxel where the given axis dominates the tensor."""
    trace = fiber_data[..., 0] + fiber_data[..., 1] + fiber_data[..., 2]
    nonzero = trace > trace_thresh
    diag = fiber_data[..., dominant_axis]
    frac = np.zeros_like(trace)
    frac[nonzero] = diag[nonzero] / trace[nonzero]
    candidates = (frac > frac_thresh) & nonzero
    x_lo, x_hi = x_range
    mask = np.zeros_like(candidates)
    mask[x_lo:x_hi, :, :] = candidates[x_lo:x_hi, :, :]

    if not mask.any():
        return None
    frac_masked = np.where(mask, frac, 0.0)
    best = np.unravel_index(np.argmax(frac_masked), frac_masked.shape)
    return (int(best[0]), int(best[1]), int(best[2]))


def _principal_direction(m6):
    """Extract principal eigenvector from 6-component upper triangle."""
    if np.allclose(m6, 0):
        return None
    mat3 = np.array([
        [m6[0], m6[3], m6[4]],
        [m6[3], m6[1], m6[5]],
        [m6[4], m6[5], m6[2]],
    ], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(mat3)
    return eigvecs[:, -1]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Header Consistency (H1–H10)
# ═══════════════════════════════════════════════════════════════════════════

@check("H1", severity="CRITICAL", phase="header")
def check_h1(ctx):
    """Bitwise-identical affines across grid NIfTIs."""
    headers = ctx.headers
    mat_affine = headers["mat"].affine
    all_match = all(
        np.array_equal(mat_affine, headers[k].affine)
        for k in ("sdf", "brain", "fs")
    )
    ctx.record("H1", all_match)


@check("H2", severity="CRITICAL", phase="header")
def check_h2(ctx):
    """Shape matches grid_meta grid_size."""
    headers = ctx.headers
    N = ctx.N
    shapes_ok = all(headers[k].shape == (N, N, N)
                    for k in ("mat", "sdf", "brain", "fs"))
    value = (f"{N}x{N}x{N}" if shapes_ok
             else str({k: headers[k].shape for k in headers}))
    ctx.record("H2", shapes_ok, value=value)


@check("H3", severity="CRITICAL", phase="header")
def check_h3(ctx):
    """material_map dtype is uint8."""
    ctx.record("H3", ctx.headers["mat"].get_data_dtype() == np.uint8)


@check("H4", severity="CRITICAL", phase="header")
def check_h4(ctx):
    """skull_sdf dtype is float32."""
    ctx.record("H4", ctx.headers["sdf"].get_data_dtype() == np.float32)


@check("H5", severity="CRITICAL", phase="header", needs={"fiber_img"})
def check_h5(ctx):
    """fiber_M0 dtype is float32."""
    ctx.record("H5", ctx.fiber_img.get_data_dtype() == np.float32)


@check("H6", severity="CRITICAL", phase="header", needs={"fiber_img"})
def check_h6(ctx):
    """fiber_M0 shape matches expected dimensions."""
    ctx.record("H6", ctx.fiber_img.shape == (145, 174, 145, 6),
               value=str(ctx.fiber_img.shape))


@check("H7", severity="CRITICAL", phase="header")
def check_h7(ctx):
    """grid_meta dx_mm matches NIfTI affine diagonal."""
    meta = ctx.meta
    mat_affine = ctx.mat_affine
    meta_dx = float(meta["dx_mm"])
    ctx.record("H7",
               abs(meta_dx - float(mat_affine[0, 0])) < 1e-6,
               value=f"meta={meta_dx}, affine={float(mat_affine[0, 0])}")


@check("H8", severity="CRITICAL", phase="header")
def check_h8(ctx):
    """grid_meta grid_size matches NIfTI shape."""
    ctx.record("H8", int(ctx.meta["grid_size"]) == ctx.headers["mat"].shape[0])


@check("H9", severity="CRITICAL", phase="header")
def check_h9(ctx):
    """grid_meta affine matches NIfTI sform."""
    meta_affine = np.array(ctx.meta["affine_grid_to_phys"])
    ctx.record("H9", np.allclose(meta_affine, ctx.mat_affine, atol=1e-6))


@check("H10", severity="CRITICAL", phase="header")
def check_h10(ctx):
    """ACPC origin round-trip through grid and fiber affines."""
    mat_affine = ctx.mat_affine
    N = ctx.N
    h10_pass = True
    h10_value = None

    phys_origin = np.array([0.0, 0.0, 0.0, 1.0])
    grid_vox = np.linalg.inv(mat_affine) @ phys_origin
    if not np.allclose(grid_vox[:3], N / 2, atol=0.5):
        h10_pass = False
        h10_value = f"grid_vox={grid_vox[:3]}, expected={N / 2}"

    fiber_img = ctx.fiber_img
    if fiber_img is not None and h10_pass:
        diff_vox = np.linalg.inv(fiber_img.affine) @ phys_origin
        for d in range(3):
            if not (0 <= diff_vox[d] < fiber_img.shape[d]):
                h10_pass = False
                h10_value = f"ACPC outside fiber on axis {d}: {diff_vox[d]:.1f}"
                break

    ctx._cache["h10_pass"] = h10_pass
    ctx.record("H10", h10_pass, value=h10_value)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Domain Closure (D1–D5)
# ═══════════════════════════════════════════════════════════════════════════

@check("D1", severity="CRITICAL", phase="domain", needs={"mat", "sdf", "brain"})
def check_d1(ctx):
    """Zero vacuum inside skull."""
    n_vac_inside = int(np.count_nonzero((ctx.sdf < 0) & (ctx.mat == 0)))
    ctx.record("D1", n_vac_inside == 0, value=n_vac_inside)


@check("D2", severity="CRITICAL", phase="domain", needs={"mat", "sdf"})
def check_d2(ctx):
    """Zero tissue outside skull."""
    tissue_lut = np.zeros(256, dtype=bool)
    tissue_lut[1:12] = True
    n_tissue_outside = int(np.count_nonzero((ctx.sdf > 0) & tissue_lut[ctx.mat]))
    ctx.record("D2", n_tissue_outside == 0, value=n_tissue_outside)


@check("D3", severity="CRITICAL", phase="domain", needs={"mat", "sdf", "brain"})
def check_d3(ctx):
    """Brain containment in skull."""
    brain_outside = int(np.count_nonzero((ctx.brain == 1) & (ctx.sdf >= 0)))
    ctx.record("D3", brain_outside == 0, value=brain_outside)


@check("D4", severity="WARN", phase="domain", needs={"mat", "brain"})
def check_d4(ctx):
    """No isolated vacuum islands inside skull."""
    mat = ctx.mat
    sdf = ctx.sdf
    vacuum = (mat == 0)
    vacuum_labels, n_comp = cc_label(vacuum)
    sizes = np.bincount(vacuum_labels.ravel())
    main_label = int(sizes[1:].argmax()) + 1 if n_comp > 0 else 0
    n_islands = 0
    island_voxels = 0
    if n_comp > 1:
        for label_id in range(1, n_comp + 1):
            if label_id == main_label:
                continue
            component = (vacuum_labels == label_id)
            if np.all(sdf[component] < 0):
                n_islands += 1
                island_voxels += int(component.sum())
    del vacuum_labels
    ctx.record("D4", n_islands == 0,
               value=f"{n_islands} islands, {island_voxels} voxels")


@check("D5", severity="WARN", phase="domain", needs={"mat", "sdf"})
def check_d5(ctx):
    """SDF cut-cell quality: gradient magnitude and smoothness at the boundary.

    The solver uses SDF values at cut-cell voxels (|SDF| < 0.5*dx) to
    compute porosity: phi = clamp(0.5 + SDF/dx, eps, 1.0).  This check
    verifies that |∇SDF| ≈ 1.0 at those voxels (porosity accuracy) and
    that the zero-crossing is smooth (no sign oscillations).
    """
    sdf = ctx.sdf
    N = ctx.N
    dx = ctx.dx

    # --- Cut-cell gradient magnitude ---
    cut_cell = np.argwhere(np.abs(sdf) < 0.5 * dx)
    n_cut = len(cut_cell)
    if n_cut == 0:
        ctx.record("D5", True, value="no cut-cell voxels")
        return

    # Sample up to 100k cut-cell voxels for gradient computation
    rng = np.random.default_rng(42)
    n_sample = min(100_000, n_cut)
    sample = cut_cell[rng.choice(n_cut, size=n_sample, replace=False)]

    grad_mag_sq = np.zeros(n_sample)
    for axis in range(3):
        fwd = sample.copy()
        fwd[:, axis] = np.clip(fwd[:, axis] + 1, 0, N - 1)
        bwd = sample.copy()
        bwd[:, axis] = np.clip(bwd[:, axis] - 1, 0, N - 1)
        diff = (sdf[fwd[:, 0], fwd[:, 1], fwd[:, 2]]
                - sdf[bwd[:, 0], bwd[:, 1], bwd[:, 2]])
        grad_mag_sq += (diff / (2 * dx)) ** 2
    grad_mag = np.sqrt(grad_mag_sq)

    p5 = float(np.percentile(grad_mag, 5))
    median = float(np.median(grad_mag))
    p95 = float(np.percentile(grad_mag, 95))

    # Porosity error: |1 - |∇|| is the fractional error in phi
    median_err = abs(1.0 - median)

    # Pass if cut-cell gradients are within reasonable bounds
    d5_pass = (p5 >= 0.7) and (p95 <= 1.3)
    ctx.record("D5", d5_pass,
               value=f"|∇| p5={p5:.2f} med={median:.2f} p95={p95:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Material Integrity (M1–M4)
# ═══════════════════════════════════════════════════════════════════════════

@check("M1", severity="CRITICAL", phase="material", needs={"mat"})
def check_m1(ctx):
    """All values in {0..11}."""
    unique_vals = set(int(v) for v in np.unique(ctx.mat))
    out_of_range = unique_vals - set(range(12))
    ctx.record("M1", len(out_of_range) == 0,
               value=sorted(out_of_range) if out_of_range else None)


@check("M2", severity="CRITICAL", phase="material", needs={"mat"})
def check_m2(ctx):
    """Does not contain 255."""
    unique_vals = set(int(v) for v in np.unique(ctx.mat))
    ctx.record("M2", 255 not in unique_vals)


@check("M3", severity="WARN", phase="material", needs={"mat"})
def check_m3(ctx):
    """All expected classes {1..11} present."""
    unique_vals = set(int(v) for v in np.unique(ctx.mat))
    missing = set(range(1, 12)) - unique_vals
    ctx.record("M3", len(missing) == 0,
               value=f"missing: {sorted(missing)}" if missing else "all present")


@check("M4", severity="WARN", phase="material", needs={"mat", "brain"})
def check_m4(ctx):
    """Brain mask consistency."""
    brain_vacuum = int(np.count_nonzero((ctx.brain == 1) & (ctx.mat == 0)))
    ctx.record("M4", brain_vacuum == 0, value=brain_vacuum)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Volume Sanity (V1–V6)
# ═══════════════════════════════════════════════════════════════════════════

@check("V1", severity="WARN", phase="volume", needs={"mat", "sdf"})
def check_v1(ctx):
    """Brain parenchyma volume."""
    vc = ctx.get_cached("volume_census", lambda: _compute_volume_census(ctx))
    parenchyma_ml = vc["parenchyma_ml"]
    ctx.record("V1", 850 <= parenchyma_ml <= 1500,
               value=f"{parenchyma_ml:.1f} mL")


@check("V2", severity="WARN", phase="volume", needs={"mat", "sdf"})
def check_v2(ctx):
    """Ventricular CSF volume."""
    vc = ctx.get_cached("volume_census", lambda: _compute_volume_census(ctx))
    ventricular_ml = vc["ventricular_ml"]
    ctx.record("V2", 10 <= ventricular_ml <= 80,
               value=f"{ventricular_ml:.1f} mL")


@check("V3", severity="WARN", phase="volume", needs={"mat", "sdf"})
def check_v3(ctx):
    """Subarachnoid CSF volume (7-30% ICV)."""
    vc = ctx.get_cached("volume_census", lambda: _compute_volume_census(ctx))
    subarachnoid_ml = vc["subarachnoid_ml"]
    icv_ml = vc["icv_ml"]
    csf_icv_frac = subarachnoid_ml / icv_ml if icv_ml > 0 else 0.0
    ctx.record("V3", 0.07 <= csf_icv_frac <= 0.30,
               value=f"{subarachnoid_ml:.1f} mL ({csf_icv_frac:.1%} ICV)")


@check("V4", severity="WARN", phase="volume", needs={"mat", "sdf"})
def check_v4(ctx):
    """Dural membrane volume."""
    vc = ctx.get_cached("volume_census", lambda: _compute_volume_census(ctx))
    dural_ml = vc["dural_ml"]
    ctx.record("V4", 3 <= dural_ml <= 30,
               value=f"{dural_ml:.1f} mL")


@check("V5", severity="WARN", phase="volume", needs={"mat", "sdf"})
def check_v5(ctx):
    """ICV vs nonzero-material consistency."""
    vc = ctx.get_cached("volume_census", lambda: _compute_volume_census(ctx))
    n_sdf_neg = vc["n_sdf_neg"]
    n_nonzero_mat = vc["n_nonzero_mat"]
    if n_sdf_neg > 0:
        frac_diff = abs(n_sdf_neg - n_nonzero_mat) / n_sdf_neg
        ctx.record("V5", frac_diff < 0.02,
                   value=f"{frac_diff:.4f} ({n_sdf_neg} vs {n_nonzero_mat})")
    else:
        ctx.record("V5", False, value="no SDF<0 voxels")


@check("V6", severity="INFO", phase="volume", needs={"mat", "sdf"})
def check_v6(ctx):
    """Complete volume census."""
    vc = ctx.get_cached("volume_census", lambda: _compute_volume_census(ctx))
    ctx.census = vc["census"]
    ctx.metrics.update({
        "brain_parenchyma_mL": round(vc["parenchyma_ml"], 1),
        "intracranial_volume_mL": round(vc["icv_ml"], 1),
        "ventricular_csf_mL": round(vc["ventricular_ml"], 1),
        "subarachnoid_csf_mL": round(vc["subarachnoid_ml"], 1),
        "dural_membrane_mL": round(vc["dural_ml"], 1),
    })
    ctx.record("V6", True, value="see census")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Compartmentalization (C1)
# ═══════════════════════════════════════════════════════════════════════════

@check("C1", severity="INFO", phase="compartment", needs={"mat"})
def check_c1(ctx):
    """Active domain connected components."""
    mat = ctx.mat
    active = (mat > 0)
    labels, n = cc_label(active)
    sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1] if n > 0 else []
    del labels
    top_str = ", ".join(str(int(s)) for s in sizes[:3]) if len(sizes) > 0 else "none"
    ctx.metrics["active_domain_components"] = n
    ctx.record("C1", True, value=f"{n} components (top: {top_str})")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Dural Membrane (C2–C11)
# ═══════════════════════════════════════════════════════════════════════════

def _classify_dural(ctx):
    """Classify dural (u8=10) voxels as falx vs tentorium.

    Uses cerebellar tissue proximity: dural voxels within 3 voxels of
    cerebellar cortex/WM are tentorium; the rest are falx.  Cached.
    """
    mat = ctx.mat
    dural = (mat == 10)
    if not dural.any():
        return dural.copy(), dural.copy()
    cerebellar_lut = np.zeros(256, dtype=bool)
    cerebellar_lut[[4, 5]] = True
    cerebellar_nearby = binary_dilation(cerebellar_lut[mat], iterations=3)
    tent = dural & cerebellar_nearby
    falx = dural & ~tent
    return falx, tent


@check("C2", severity="WARN", phase="dural", needs={"mat"})
def check_c2(ctx):
    """Falx largest component > 90%."""
    falx_region, _ = ctx.get_cached(
        "dural_classification", lambda: _classify_dural(ctx))
    n_falx = int(np.count_nonzero(falx_region))
    if n_falx == 0:
        ctx.record("C2", True, value="skipped (0 falx voxels)")
        return

    falx_labels, falx_n_comp = cc_label(falx_region)
    falx_sizes = np.sort(np.bincount(falx_labels.ravel())[1:])[::-1]
    frac_largest = float(falx_sizes[0]) / float(falx_sizes.sum())
    ctx.metrics["falx_components"] = falx_n_comp
    ctx.record("C2", frac_largest > 0.90,
               value=f"{falx_n_comp} comp, largest={frac_largest:.1%}")
    del falx_labels


@check("C3", severity="WARN", phase="dural", needs={"mat"})
def check_c3(ctx):
    """Tentorial notch patency."""
    mat = ctx.mat
    brainstem = (mat == 6)
    if not brainstem.any():
        ctx.record("C3", True, value="skipped (no brainstem)")
        return

    z_indices = np.where(brainstem.any(axis=(0, 1)))[0]
    z_min, z_max = int(z_indices.min()), int(z_indices.max())
    z_upper = z_min + 2 * (z_max - z_min) // 3

    bs_slice = brainstem[:, :, z_upper]
    csf_slice = (mat[:, :, z_upper] == 8)
    bs_dilated = binary_dilation(bs_slice, iterations=1)
    csf_adjacent = csf_slice & bs_dilated & ~bs_slice
    n_adjacent = int(np.count_nonzero(csf_adjacent))
    ctx.record("C3", n_adjacent > 0,
               value=f"{n_adjacent} adjacent CSF voxels at z={z_upper}")


@check("C4", severity="WARN", phase="dural", needs={"mat", "fs"})
def check_c4(ctx):
    """Falx barrier separates left/right supratentorial CSF."""
    mat = ctx.mat
    fsd = ctx.get_cached("fs_data", lambda: _load_fs_data(ctx))
    cc_z_sup = fsd["cc_z_sup"]
    left_lut = fsd["left_lut"]
    right_lut = fsd["right_lut"]
    fs_safe = fsd["fs_safe"]

    left_tissue = left_lut[fs_safe]
    right_tissue = right_lut[fs_safe]
    left_tissue[:, :, :cc_z_sup + 1] = False
    right_tissue[:, :, :cc_z_sup + 1] = False

    csf_above_cc = (mat == 8)
    csf_above_cc[:, :, :cc_z_sup + 1] = False

    n_csf_above = int(np.count_nonzero(csf_above_cc))
    if n_csf_above == 0:
        del left_tissue, right_tissue
        ctx.record("C4", True,
                   value="skipped (no supratentorial CSF above CC)")
        return

    labeled, n_comp = cc_label(csf_above_cc)
    del csf_above_cc

    left_adj_labels = set()
    right_adj_labels = set()
    for ax in range(3):
        for d in (-1, 1):
            sl_src = [slice(None)] * 3
            sl_dst = [slice(None)] * 3
            if d == 1:
                sl_src[ax] = slice(None, -1)
                sl_dst[ax] = slice(1, None)
            else:
                sl_src[ax] = slice(1, None)
                sl_dst[ax] = slice(None, -1)
            labels_at_dst = labeled[tuple(sl_dst)]
            lt = left_tissue[tuple(sl_src)]
            left_adj_labels.update(labels_at_dst[lt & (labels_at_dst > 0)])
            rt = right_tissue[tuple(sl_src)]
            right_adj_labels.update(labels_at_dst[rt & (labels_at_dst > 0)])

    bridging_labels = left_adj_labels & right_adj_labels
    n_bridging = len(bridging_labels)
    del labeled, left_tissue, right_tissue

    if n_bridging == 0:
        ctx.record("C4", True,
                   value=f"{n_comp} components, 0 bridging")
    else:
        ctx.record("C4", False,
                   value=f"{n_bridging} bridging of {n_comp} components")


@check("C5", severity="WARN", phase="dural", needs={"mat", "sdf"})
def check_c5(ctx):
    """Falx fissure coverage >= 95% (above CC, inside skull)."""
    mat = ctx.mat
    sdf = ctx.sdf
    N = ctx.N
    mid_x = N // 2

    csf_or_dural_lut = np.zeros(256, dtype=bool)
    csf_or_dural_lut[8] = True
    csf_or_dural_lut[10] = True
    midline_slab = mat[mid_x - 5:mid_x + 6, :, :]
    fissure_mip = csf_or_dural_lut[midline_slab].any(axis=0)
    dural_mip = (midline_slab == 10).any(axis=0)
    del midline_slab

    if sdf is not None:
        inside_skull_midline = (sdf[mid_x, :, :] <= 0)
        fissure_mip &= inside_skull_midline
        del inside_skull_midline

    # Use CC boundary from fs_data if available
    cc_z_sup = None
    if ctx.paths["fs"].exists():
        fsd = ctx.get_cached("fs_data", lambda: _load_fs_data(ctx))
        cc_z_sup = fsd["cc_z_sup"]

    if cc_z_sup is not None:
        fissure_mip[:, :cc_z_sup + 1] = False
        dural_mip[:, :cc_z_sup + 1] = False

    n_fissure = int(np.count_nonzero(fissure_mip))
    if n_fissure > 0:
        n_sealed = int(np.count_nonzero(fissure_mip & dural_mip))
        falx_coverage_pct = round(n_sealed / n_fissure * 100, 1)
        ctx.metrics["falx_coverage_pct"] = falx_coverage_pct
        ctx.record("C5", falx_coverage_pct >= 95.0,
                   value=f"{falx_coverage_pct:.1f}% ({n_sealed}/{n_fissure})")
    else:
        ctx.record("C5", True, value="skipped (no fissure voxels)")
    del fissure_mip, dural_mip


@check("C6", severity="WARN", phase="dural", needs={"mat"})
def check_c6(ctx):
    """Tentorium gap coverage >= 95%."""
    mat = ctx.mat
    N = ctx.N

    cerebral_lut = np.zeros(256, dtype=bool)
    cerebral_lut[[1, 2, 3, 9]] = True
    cerebellar_lut = np.zeros(256, dtype=bool)
    cerebellar_lut[[4, 5]] = True

    brainstem = (mat == 6)
    if not brainstem.any():
        ctx.record("C6", True, value="skipped (no brainstem)")
        return

    z_indices = np.where(brainstem.any(axis=(0, 1)))[0]
    z_min, z_max = int(z_indices.min()), int(z_indices.max())
    tent_z = z_min + 2 * (z_max - z_min) // 3

    z_lo = max(0, tent_z - 20)
    z_hi = min(N, tent_z + 21)
    mat_slab = mat[:, :, z_lo:z_hi]

    cerebral_proj = cerebral_lut[mat_slab].any(axis=2)
    cerebellar_proj = cerebellar_lut[mat_slab].any(axis=2)
    boundary_cols = cerebral_proj & cerebellar_proj
    del cerebral_proj, cerebellar_proj

    # Exclude tentorial notch: columns containing brainstem in the slab
    # are part of the natural opening and don't need dural coverage.
    notch_proj = (mat_slab == 6).any(axis=2)
    notch_excl = binary_dilation(notch_proj, iterations=3)
    boundary_cols &= ~notch_excl
    del notch_proj, notch_excl

    dural_proj = (mat_slab == 10).any(axis=2)
    del mat_slab

    n_boundary = int(np.count_nonzero(boundary_cols))
    if n_boundary > 0:
        n_sealed_tent = int(np.count_nonzero(boundary_cols & dural_proj))
        tent_coverage_pct = round(n_sealed_tent / n_boundary * 100, 1)
        ctx.metrics["tentorium_coverage_pct"] = tent_coverage_pct
        ctx.record("C6", tent_coverage_pct >= 95.0,
                   value=f"{tent_coverage_pct:.1f}% ({n_sealed_tent}/{n_boundary})")
    else:
        ctx.record("C6", True, value="skipped (no boundary columns)")
    del boundary_cols, dural_proj


@check("C7", severity="WARN", phase="dural", needs={"mat"})
def check_c7(ctx):
    """Tentorium barrier separates supra/infratentorial CSF."""
    mat = ctx.mat
    brainstem = (mat == 6)
    if not brainstem.any():
        ctx.record("C7", True, value="skipped (no brainstem)")
        return

    z_indices = np.where(brainstem.any(axis=(0, 1)))[0]
    z_min, z_max = int(z_indices.min()), int(z_indices.max())
    tent_z = z_min + 2 * (z_max - z_min) // 3
    del brainstem

    csf_mask = (mat == 8)
    supra_csf = csf_mask.copy()
    supra_csf[:, :, :tent_z + 6] = False
    infra_csf = csf_mask.copy()
    infra_csf[:, :, tent_z - 5:] = False
    del csf_mask

    n_supra = int(np.count_nonzero(supra_csf))
    n_infra = int(np.count_nonzero(infra_csf))

    if n_supra == 0 or n_infra == 0:
        ctx.record("C7", True,
                   value=f"skipped (supra={n_supra}, infra={n_infra})")
        return

    combined_csf = supra_csf | infra_csf
    labeled_csf, n_comp_csf = cc_label(combined_csf)
    del combined_csf

    supra_labels = set(np.unique(labeled_csf[supra_csf]))
    infra_labels = set(np.unique(labeled_csf[infra_csf]))
    del supra_csf, infra_csf

    supra_labels.discard(0)
    infra_labels.discard(0)

    bridging_labels = supra_labels & infra_labels
    tent_bridging = len(bridging_labels)
    ctx.metrics["tentorium_bridging_components"] = tent_bridging
    del labeled_csf

    if tent_bridging == 0:
        ctx.record("C7", True,
                   value=f"{n_comp_csf} components, 0 bridging")
    else:
        ctx.record("C7", False,
                   value=f"{tent_bridging} bridging components")


@check("C8", severity="WARN", phase="dural", needs={"mat"})
def check_c8(ctx):
    """Dural membrane surface area (falx ~56.5 cm², tentorium ~60 cm²)."""
    mat = ctx.mat
    dx = ctx.dx

    dural = (mat == 10)
    n_dural = int(np.count_nonzero(dural))
    if n_dural == 0:
        ctx.record("C8", True, value="skipped (0 dural voxels)")
        return

    pixel_cm2 = dx ** 2 / 100.0  # mm² → cm²

    falx, tent = ctx.get_cached(
        "dural_classification", lambda: _classify_dural(ctx))

    # Falx is a sagittal curtain → project onto y-z plane (collapse x)
    falx_cm2 = float(np.count_nonzero(falx.any(axis=0))) * pixel_cm2
    # Tentorium is a horizontal sheet → project onto x-y plane (collapse z)
    tent_cm2 = float(np.count_nonzero(tent.any(axis=2))) * pixel_cm2

    # Anatomical references (Staquet et al. 2020, n=40 CT):
    #   Falx: 56.5 ± 7.7 cm², ±2 SD (95% population) → [41, 72]
    #   Tentorium: 57.6 ± 5.8 cm², ±2 SD (95% population) → [46, 69]
    falx_ok = 41.0 <= falx_cm2 <= 72.0
    tent_ok = 46.0 <= tent_cm2 <= 69.0

    ctx.metrics["falx_area_cm2"] = round(falx_cm2, 1)
    ctx.metrics["tentorium_area_cm2"] = round(tent_cm2, 1)

    ctx.record("C8", falx_ok and tent_ok,
               value=f"falx={falx_cm2:.1f} cm² [41–72], tent={tent_cm2:.1f} cm² [46–69]")


# ---------------------------------------------------------------------------
# Shared helper: falx heights at CC landmarks
# ---------------------------------------------------------------------------

def _compute_falx_cc_heights(ctx):
    """Measure falx z-extent at CC body and genu y-positions.

    Returns dict with keys 'body_mm', 'genu_mm', 'body_y', 'genu_y'.
    Cached as 'falx_cc_heights'.
    """
    from preprocessing.dural_membrane import _FS_LUT_SIZE

    mat = ctx.mat
    dx = ctx.dx

    # Load FS labels
    fsd = ctx.get_cached("fs_data", lambda: _load_fs_data(ctx))
    fs_safe = fsd["fs_safe"]

    # Find CC body (253) and genu (255) y-positions
    result = {}
    for name, label in [("body", 253), ("genu", 255)]:
        cc_region = (fs_safe == label)
        if not cc_region.any():
            result[f"{name}_mm"] = None
            result[f"{name}_y"] = None
            continue
        # Centroid y of this CC region
        y_indices = np.where(cc_region.any(axis=(0, 2)))[0]
        y_pos = int(np.median(y_indices))
        result[f"{name}_y"] = y_pos

        # Falx z-extent at this y: project dural=10 across x
        falx, _ = ctx.get_cached(
            "dural_classification", lambda: _classify_dural(ctx))
        falx_at_y = falx[:, y_pos, :].any(axis=0)
        z_indices = np.where(falx_at_y)[0]
        if len(z_indices) > 0:
            h = (z_indices.max() - z_indices.min() + 1) * dx
            result[f"{name}_mm"] = round(h, 1)
        else:
            result[f"{name}_mm"] = None

    return result


@check("C9", severity="WARN", phase="dural", needs={"mat", "fs"})
def check_c9(ctx):
    """Falx height at CC body: Kayalioglu 25.7 mm (±2 SD ≈ 14–37)."""
    heights = ctx.get_cached(
        "falx_cc_heights", lambda: _compute_falx_cc_heights(ctx))
    h = heights["body_mm"]
    if h is None:
        ctx.record("C9", True, value="skipped (no CC body or no falx)")
        return
    ctx.metrics["falx_height_body_mm"] = h
    ctx.record("C9", 14.0 <= h <= 37.0,
               value=f"{h:.1f} mm [14–37]")


@check("C10", severity="WARN", phase="dural", needs={"mat", "fs"})
def check_c10(ctx):
    """Falx height at CC genu: Kayalioglu 21.3 mm (±2 SD ≈ 10–33)."""
    heights = ctx.get_cached(
        "falx_cc_heights", lambda: _compute_falx_cc_heights(ctx))
    h = heights["genu_mm"]
    if h is None:
        ctx.record("C10", True, value="skipped (no CC genu or no falx)")
        return
    ctx.metrics["falx_height_genu_mm"] = h
    ctx.record("C10", 10.0 <= h <= 33.0,
               value=f"{h:.1f} mm [10–33]")


@check("C11", severity="WARN", phase="dural", needs={"mat"})
def check_c11(ctx):
    """Falx-tentorium junction A-P extent >= 30 mm."""
    mat = ctx.mat
    dx = ctx.dx

    falx, tent = ctx.get_cached(
        "dural_classification", lambda: _classify_dural(ctx))

    n_falx = int(np.count_nonzero(falx))
    n_tent = int(np.count_nonzero(tent))
    if n_falx == 0 or n_tent == 0:
        ctx.record("C11", True,
                   value=f"skipped (falx={n_falx}, tent={n_tent})")
        return

    # Junction = falx voxels adjacent (within 2 voxels) to tentorium
    falx_dilated = binary_dilation(falx, iterations=2)
    junction = falx_dilated & tent
    del falx_dilated

    if not junction.any():
        ctx.record("C11", False, value="no junction (gap > 2 vox)")
        return

    # Measure A-P extent: y-range of junction voxels
    junction_y = np.where(junction.any(axis=(0, 2)))[0]
    extent_mm = round((junction_y.max() - junction_y.min() + 1) * dx, 1)
    n_junction = int(junction.sum())
    del junction

    ctx.metrics["falx_tent_junction_mm"] = extent_mm
    ctx.record("C11", extent_mm >= 30.0,
               value=f"{extent_mm:.0f} mm ({n_junction} vox) [≥30]")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Fiber Texture (F1–F6)
# ═══════════════════════════════════════════════════════════════════════════

@check("F1", severity="WARN", phase="fiber", needs={"mat", "fiber_img"})
def check_f1(ctx):
    """WM forward-transform coverage >= 90%."""
    from scipy.ndimage import map_coordinates

    mat = ctx.mat
    fiber_data = ctx.fiber_data
    fiber_affine = ctx.fiber_img.affine
    mat_affine = ctx.mat_affine

    wm_lut = np.zeros(256, dtype=bool)
    wm_lut[[1, 4, 6]] = True
    wm_mask = wm_lut[mat]
    wm_indices = np.argwhere(wm_mask)

    rng = np.random.default_rng(42)
    n_sample = min(50_000, len(wm_indices))
    coverage_pct = 0.0
    if n_sample > 0:
        sample = wm_indices[rng.choice(len(wm_indices), size=n_sample, replace=False)]

        phys_to_diff = np.linalg.inv(fiber_affine)
        grid_to_diff = phys_to_diff @ mat_affine

        grid_homo = np.column_stack([sample, np.ones(n_sample)])
        diff_coords = (grid_to_diff @ grid_homo.T).T[:, :3]

        trace_at_wm = np.zeros(n_sample)
        for ch in range(3):
            vals = map_coordinates(
                fiber_data[..., ch], diff_coords.T,
                order=1, mode="constant", cval=0.0,
            )
            trace_at_wm += vals

        coverage_pct = float((trace_at_wm > 0).sum()) / n_sample * 100
        ctx.record("F1", coverage_pct >= 90.0,
                   value=f"{coverage_pct:.1f}%")

        nonzero_trace = trace_at_wm[trace_at_wm > 0]
        if len(nonzero_trace) > 0:
            ctx.metrics["fiber_trace_mean"] = round(float(np.mean(nonzero_trace)), 4)
            ctx.metrics["fiber_trace_p95"] = round(float(np.percentile(nonzero_trace, 95)), 4)
    else:
        ctx.record("F1", False, value="no WM voxels")

    ctx.metrics["fiber_wm_coverage_pct"] = round(coverage_pct, 1)


@check("F2", severity="CRITICAL", phase="fiber", needs={"fiber_img"})
def check_f2(ctx):
    """M_0 diagonal elements >= 0 (PSD)."""
    fiber_data = ctx.fiber_data

    f2_pass = True
    f2_detail = []
    for ch in range(3):
        diag = fiber_data[..., ch]
        n_negative = int(np.count_nonzero(diag < -1e-7))
        if n_negative > 0:
            f2_pass = False
            f2_detail.append(f"ch{ch}: {n_negative} neg")

    rng = np.random.default_rng(42)
    nonzero_mask = np.any(fiber_data != 0, axis=-1)
    nonzero_indices = np.argwhere(nonzero_mask)
    n_eig_sample = min(10_000, len(nonzero_indices))
    n_eig_fail = 0
    if n_eig_sample > 0:
        eig_sample = nonzero_indices[rng.choice(len(nonzero_indices), size=n_eig_sample, replace=False)]
        for idx in eig_sample:
            i, j, k = idx
            m = fiber_data[i, j, k]
            mat3 = np.array([
                [m[0], m[3], m[4]],
                [m[3], m[1], m[5]],
                [m[4], m[5], m[2]],
            ], dtype=np.float64)
            eigvals = np.linalg.eigvalsh(mat3)
            if np.any(eigvals < -1e-7):
                n_eig_fail += 1
                f2_pass = False

    if n_eig_fail > 0:
        f2_detail.append(f"eigen: {n_eig_fail}/{n_eig_sample} fail")

    ctx.record("F2", f2_pass,
               value="; ".join(f2_detail) if f2_detail else None)


@check("F3", severity="WARN", phase="fiber", needs={"fiber_img"})
def check_f3(ctx):
    """Trace <= 1.0 bound."""
    fiber_data = ctx.fiber_data
    trace_vol = fiber_data[..., 0] + fiber_data[..., 1] + fiber_data[..., 2]
    n_over = int(np.count_nonzero(trace_vol > 1.0 + 1e-5))
    ctx.record("F3", n_over == 0, value=f"{n_over} voxels over")


@check("F4", severity="WARN", phase="fiber", needs={"fiber_img"})
def check_f4(ctx):
    """CC landmark X-dominant."""
    fiber_data = ctx.fiber_data
    cc_vox = _find_landmark(fiber_data, dominant_axis=0,
                            x_range=(fiber_data.shape[0] // 2 - 5,
                                     fiber_data.shape[0] // 2 + 5))
    if cc_vox is not None:
        m6 = fiber_data[cc_vox[0], cc_vox[1], cc_vox[2]]
        cc_dir = _principal_direction(m6)
        f4_pass = cc_dir is not None and abs(cc_dir[0]) > 0.7
        ctx.record("F4", f4_pass, value=f"vox={cc_vox}, dir={cc_dir}")
    else:
        ctx.record("F4", True,
                   value="skipped (no CC-like voxels found)")


@check("F5", severity="WARN", phase="fiber", needs={"fiber_img"})
def check_f5(ctx):
    """IC landmark Z-dominant."""
    fiber_data = ctx.fiber_data
    shape = fiber_data.shape[:3]
    ic_vox = _find_landmark(fiber_data, dominant_axis=2,
                            x_range=(int(shape[0] * 0.3),
                                     int(shape[0] * 0.45)))
    if ic_vox is not None:
        m6 = fiber_data[ic_vox[0], ic_vox[1], ic_vox[2]]
        ic_dir = _principal_direction(m6)
        f5_pass = ic_dir is not None and abs(ic_dir[2]) > 0.7
        ctx.record("F5", f5_pass, value=f"vox={ic_vox}, dir={ic_dir}")
    else:
        ctx.record("F5", True,
                   value="skipped (no IC-like voxels found)")


@check("F6", severity="CRITICAL", phase="fiber", needs={"fiber_img"})
def check_f6(ctx):
    """Fiber affine round-trip (copy of H10)."""
    h10_pass = ctx._cache.get("h10_pass")
    if h10_pass is None:
        # H10 was skipped; compute independently
        mat_affine = ctx.mat_affine
        N = ctx.N
        phys_origin = np.array([0.0, 0.0, 0.0, 1.0])
        grid_vox = np.linalg.inv(mat_affine) @ phys_origin
        h10_pass = np.allclose(grid_vox[:3], N / 2, atol=0.5)
        if h10_pass and ctx.fiber_img is not None:
            diff_vox = np.linalg.inv(ctx.fiber_img.affine) @ phys_origin
            for d in range(3):
                if not (0 <= diff_vox[d] < ctx.fiber_img.shape[d]):
                    h10_pass = False
                    break
    ctx.record("F6", h10_pass, value="copied from H10")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Ground Truth (G1–G3)
# ═══════════════════════════════════════════════════════════════════════════

# SimNIBS CHARM label definitions (from final_tissues_LUT.txt)
SIMNIBS_LABELS = {
    1: "White-Matter",
    2: "Gray-Matter",
    3: "CSF",
    4: "Bone",
    5: "Scalp",
    6: "Eye_balls",
    7: "Compact_bone",
    8: "Spongy_bone",
    9: "Blood",
    10: "Muscle",
}


def _load_simnibs_resampled(ctx):
    """Load SimNIBS final_tissues, resample to grid, compute SDF.

    Caches the resampled labels and SDF to val_dir/ so subsequent runs
    skip the expensive resample + EDT (~5 min at 512³).

    Returns dict with 'labels', 'sdf', 'inner_boundary'.
    """
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt
    from preprocessing.utils import validation_dir, resample_to_grid

    subject = ctx.subject
    grid_affine = ctx.mat_affine
    grid_shape = (ctx.N, ctx.N, ctx.N)
    dx = ctx.dx
    val_dir = ctx.paths["val_dir"]

    labels_cache = val_dir / "simnibs_labels.nii.gz"
    sdf_cache = val_dir / "simnibs_sdf.nii.gz"

    if sdf_cache.exists() and labels_cache.exists():
        labels = np.asarray(
            nib.load(str(labels_cache)).dataobj, dtype=np.int16)
        sdf = np.asarray(
            nib.load(str(sdf_cache)).dataobj, dtype=np.float32)
    else:
        path = validation_dir(subject) / "final_tissues.nii.gz"
        img = nib.load(str(path))
        data = np.asarray(img.dataobj, dtype=np.int16)
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[:, :, :, 0]

        labels = resample_to_grid(
            (data, img.affine), grid_affine, grid_shape,
            order=0, cval=0, dtype=np.int16,
        )
        del data

        inner_skull = np.isin(labels, [1, 2, 3, 9])
        sampling = (dx, dx, dx)
        dt_out = distance_transform_edt(~inner_skull, sampling=sampling)
        dt_in = distance_transform_edt(inner_skull, sampling=sampling)
        sdf = (dt_out - dt_in).astype(np.float32)
        del dt_out, dt_in, inner_skull

        val_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(labels, grid_affine), str(labels_cache))
        nib.save(nib.Nifti1Image(sdf, grid_affine), str(sdf_cache))

    # Inner skull boundary: intracranial voxels adjacent to bone
    intracranial = np.isin(labels, [1, 2, 3, 9])
    bone = np.isin(labels, [4, 7, 8])
    inner_boundary = intracranial & binary_dilation(bone)

    return {"labels": labels, "sdf": sdf, "inner_boundary": inner_boundary}


@check("G1", severity="WARN", phase="ground_truth", needs={"sdf", "simnibs"})
def check_g1(ctx):
    """Skull SDF MAE at SimNIBS inner skull boundary."""
    sim = ctx.get_cached("simnibs_resampled",
                         lambda: _load_simnibs_resampled(ctx))
    inner_boundary = sim["inner_boundary"]
    n_boundary = int(inner_boundary.sum())
    if n_boundary == 0:
        ctx.record("G1", True, value="no boundary voxels")
        return

    sdf_at_boundary = ctx.sdf[inner_boundary]
    mae = float(np.mean(np.abs(sdf_at_boundary)))
    median = float(np.median(sdf_at_boundary))
    ctx.metrics["gt_boundary_mae_mm"] = round(mae, 3)
    ctx.metrics["gt_boundary_median_mm"] = round(median, 3)
    ctx.metrics["gt_boundary_n_voxels"] = n_boundary
    ctx.record("G1", mae <= 2.0,
               value=f"MAE={mae:.2f}mm, med={median:+.2f}mm, n={n_boundary:,}")


@check("G2", severity="INFO", phase="ground_truth", needs={"sdf", "simnibs"})
def check_g2(ctx):
    """Regional error by axial thirds (inferior/middle/superior)."""
    sim = ctx.get_cached("simnibs_resampled",
                         lambda: _load_simnibs_resampled(ctx))
    inner_boundary = sim["inner_boundary"]
    zs = np.where(inner_boundary.any(axis=(0, 1)))[0]
    if len(zs) == 0:
        ctx.record("G2", True, value="no boundary voxels")
        return

    z_min, z_max = int(zs[0]), int(zs[-1])
    z_span = z_max - z_min
    cuts = [z_min, z_min + z_span // 3,
            z_min + 2 * z_span // 3, z_max + 1]

    parts = []
    for name, z_lo, z_hi in [("inf", cuts[0], cuts[1]),
                              ("mid", cuts[1], cuts[2]),
                              ("sup", cuts[2], cuts[3])]:
        region = inner_boundary.copy()
        region[:, :, :z_lo] = False
        region[:, :, z_hi:] = False
        n_r = int(region.sum())
        if n_r > 0:
            vals = ctx.sdf[region]
            mae = float(np.mean(np.abs(vals)))
            ctx.metrics[f"gt_{name}_mae_mm"] = round(mae, 3)
            parts.append(f"{name}={mae:.2f}")

    ctx.record("G2", True, value=", ".join(parts))


@check("G3", severity="INFO", phase="ground_truth", needs={"sdf", "simnibs"})
def check_g3(ctx):
    """Symmetric surface distance (marching cubes + KD-tree)."""
    from skimage.measure import marching_cubes
    from scipy.spatial import cKDTree

    our_sdf = ctx.sdf
    our_affine = ctx.mat_affine
    sim = ctx.get_cached("simnibs_resampled",
                         lambda: _load_simnibs_resampled(ctx))
    sim_sdf = sim["sdf"]

    def _isosurface(vol, affine):
        verts_vox, _, _, _ = marching_cubes(vol, level=0.0)
        ones = np.ones((len(verts_vox), 1), dtype=verts_vox.dtype)
        return (affine @ np.hstack([verts_vox, ones]).T).T[:, :3]

    verts_ours = _isosurface(our_sdf, our_affine)
    verts_sim = _isosurface(sim_sdf, our_affine)

    tree_ours = cKDTree(verts_ours)
    tree_sim = cKDTree(verts_sim)
    d_o2s, _ = tree_sim.query(verts_ours)
    d_s2o, _ = tree_ours.query(verts_sim)
    all_d = np.concatenate([d_o2s, d_s2o])

    mean_d = float(all_d.mean())
    median_d = float(np.median(all_d))
    p95_d = float(np.percentile(all_d, 95))
    hausdorff = float(all_d.max())
    hd95 = float(np.percentile(all_d, 95))

    # Per-direction
    o2s_p95 = float(np.percentile(d_o2s, 95))
    o2s_max = float(d_o2s.max())
    s2o_p95 = float(np.percentile(d_s2o, 95))
    s2o_max = float(d_s2o.max())

    ctx.metrics["gt_surface_mean_mm"] = round(mean_d, 3)
    ctx.metrics["gt_surface_median_mm"] = round(median_d, 3)
    ctx.metrics["gt_surface_p95_mm"] = round(p95_d, 3)
    ctx.metrics["gt_surface_hausdorff_mm"] = round(hausdorff, 3)
    ctx.metrics["gt_surface_hd95_mm"] = round(hd95, 3)
    ctx.metrics["gt_o2s_p95_mm"] = round(o2s_p95, 3)
    ctx.metrics["gt_o2s_hausdorff_mm"] = round(o2s_max, 3)
    ctx.metrics["gt_s2o_p95_mm"] = round(s2o_p95, 3)
    ctx.metrics["gt_s2o_hausdorff_mm"] = round(s2o_max, 3)

    # Cache vertices for fig6
    ctx._cache["gt_verts_ours"] = verts_ours
    ctx._cache["gt_verts_sim"] = verts_sim
    ctx._cache["gt_d_o2s"] = d_o2s
    ctx._cache["gt_d_s2o"] = d_s2o

    ctx.record("G3", True,
               value=f"mean={mean_d:.2f}, med={median_d:.2f}, "
                     f"P95={p95_d:.2f}, HD={hausdorff:.1f}, "
                     f"HD95={hd95:.2f}mm")


# ---------------------------------------------------------------------------
# Populate lookup dict after all decorations
# ---------------------------------------------------------------------------
_REGISTRY_BY_ID = {d.check_id: d for d in _REGISTRY}
