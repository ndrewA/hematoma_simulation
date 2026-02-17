"""Compute per-voxel fiber structure tensor M_0 from bedpostX diffusion data.

M_0 encodes white matter fiber architecture used by the Darcy solver
(anisotropic permeability) and HGO model (anisotropic elasticity) at runtime.

Output is profile-independent — saved to data/processed/{subject_id}/ at native
bedpostX resolution (1.25 mm, 145x174x145).

    python -m preprocessing.fiber_orientation --subject 157336
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Anisotropic (white-matter-like) FreeSurfer labels  (Section 4.4)
# ---------------------------------------------------------------------------
_ANISO_LABELS = frozenset({
    2, 41,                              # Cerebral WM L/R
    77, 78, 79,                         # WM hypointensities
    85,                                 # Optic chiasm
    192,                                # Corpus callosum
    250,                                # Fornix
    251, 252, 253, 254, 255,            # CC sub-regions
    7, 46,                              # Cerebellar WM L/R
    16, 75, 76,                         # Brainstem
})

_FS_LUT_SIZE = 2036  # covers FS labels 0..2035


def _build_aniso_lut():
    """Boolean LUT (size 2036) for anisotropic FS labels."""
    lut = np.zeros(_FS_LUT_SIZE, dtype=bool)
    for lab in _ANISO_LABELS:
        lut[lab] = True
    return lut


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """CLI: --subject (required), --f-threshold (default 0.05)."""
    parser = argparse.ArgumentParser(
        description="Compute fiber structure tensor M_0 from bedpostX data."
    )
    parser.add_argument("--subject", required=True, help="HCP subject ID")
    parser.add_argument(
        "--f-threshold", type=float, default=0.05,
        help="Volume fraction threshold (default: 0.05)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_bedpostx(bedpostx_dir):
    """Load 6 bedpostX files + brain mask.

    Returns (dyads, fracs, brain_mask, diff_affine) where:
        dyads: list of 3 arrays, each (I, J, K, 3) float32
        fracs: list of 3 arrays, each (I, J, K) float32
        brain_mask: (I, J, K) bool
        diff_affine: (4, 4) float64
    """
    bedpostx_dir = Path(bedpostx_dir)

    dyads = []
    fracs = []
    for n in range(1, 4):
        dyad_path = bedpostx_dir / f"dyads{n}.nii.gz"
        frac_path = bedpostx_dir / f"mean_f{n}samples.nii.gz"
        print(f"Loading {dyad_path.name} + {frac_path.name}")

        dyad_img = nib.load(str(dyad_path))
        dyads.append(dyad_img.get_fdata(dtype=np.float32))
        if n == 1:
            diff_affine = dyad_img.affine.copy()

        frac_img = nib.load(str(frac_path))
        fracs.append(frac_img.get_fdata(dtype=np.float32))

    mask_path = bedpostx_dir / "nodif_brain_mask.nii.gz"
    print(f"Loading {mask_path.name}")
    mask_img = nib.load(str(mask_path))
    brain_mask = np.asarray(mask_img.dataobj, dtype=bool)

    return dyads, fracs, brain_mask, diff_affine


def load_fs_labels(t1w_dir):
    """Load aparc+aseg.nii.gz -> (data, affine)."""
    path = Path(t1w_dir) / "aparc+aseg.nii.gz"
    print(f"Loading {path.name}")
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.int16)
    return data, img.affine.copy()


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------
def threshold_fractions(fracs, threshold, brain_mask):
    """In-place zeroing of volume fractions below threshold."""
    for n, f in enumerate(fracs):
        below = (f < threshold) & brain_mask
        n_below = int(np.count_nonzero(below))
        f[below] = 0.0
        print(f"  f{n + 1}: zeroed {n_below} voxels below {threshold}")


def compute_structure_tensor(dyads, fracs):
    """Compute M_0 = sum_n f_n * (v_n outer v_n), upper triangle.

    Returns M0, shape (I, J, K, 6) float32 with components
    [M_00, M_11, M_22, M_01, M_02, M_12].

    Frees each dyad array after accumulation to save memory.
    """
    shape3 = dyads[0].shape[:3]
    M0 = np.zeros(shape3 + (6,), dtype=np.float32)

    for n in range(3):
        v = dyads[n]   # (I, J, K, 3)
        f = fracs[n]   # (I, J, K)

        # Outer product upper triangle, weighted by f
        M0[..., 0] += f * v[..., 0] * v[..., 0]  # M_00
        M0[..., 1] += f * v[..., 1] * v[..., 1]  # M_11
        M0[..., 2] += f * v[..., 2] * v[..., 2]  # M_22
        M0[..., 3] += f * v[..., 0] * v[..., 1]  # M_01
        M0[..., 4] += f * v[..., 0] * v[..., 2]  # M_02
        M0[..., 5] += f * v[..., 1] * v[..., 2]  # M_12

        # Free dyad memory
        dyads[n] = None

    return M0


def build_wm_mask(diff_affine, fs_data, fs_affine, shape, aniso_lut):
    """Cross-resolution nearest-neighbor lookup -> boolean mask.

    Maps each bedpostX voxel into FreeSurfer voxel space via
    inv(fs_affine) @ diff_affine, looks up the FS label, and checks
    if it belongs to the anisotropic set.
    """
    # Composite transform: diff voxel -> physical -> FS voxel
    M = np.linalg.inv(fs_affine) @ diff_affine

    I, J, K = shape
    ii = np.arange(I, dtype=np.float64)
    jj = np.arange(J, dtype=np.float64)
    kk = np.arange(K, dtype=np.float64)
    gi, gj, gk = np.meshgrid(ii, jj, kk, indexing='ij')

    # Transform to FS voxel coordinates
    fi = M[0, 0] * gi + M[0, 1] * gj + M[0, 2] * gk + M[0, 3]
    fj = M[1, 0] * gi + M[1, 1] * gj + M[1, 2] * gk + M[1, 3]
    fk = M[2, 0] * gi + M[2, 1] * gj + M[2, 2] * gk + M[2, 3]
    del gi, gj, gk

    # Nearest-neighbor: round and clip
    fi = np.clip(np.round(fi).astype(np.intp), 0, fs_data.shape[0] - 1)
    fj = np.clip(np.round(fj).astype(np.intp), 0, fs_data.shape[1] - 1)
    fk = np.clip(np.round(fk).astype(np.intp), 0, fs_data.shape[2] - 1)

    # Lookup FS labels
    fs_labels = fs_data[fi, fj, fk]
    del fi, fj, fk

    # Map through aniso LUT
    fs_safe = np.clip(fs_labels, 0, _FS_LUT_SIZE - 1)
    is_aniso = aniso_lut[fs_safe]

    return is_aniso, fs_labels


def apply_wm_mask(M0, is_aniso):
    """Zero M_0 outside WM (non-anisotropic voxels)."""
    outside = ~is_aniso
    n_zeroed = int(np.count_nonzero(
        np.any(M0 != 0, axis=-1) & outside
    ))
    M0[outside] = 0.0
    print(f"  WM mask: zeroed {n_zeroed} non-aniso voxels with non-zero M0")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_fiber_m0(out_dir, M0, diff_affine):
    """Save fiber_M0.nii.gz (float32, diffusion affine)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = nib.Nifti1Image(M0, diff_affine)
    img.header.set_data_dtype(np.float32)
    path = out_dir / "fiber_M0.nii.gz"
    nib.save(img, str(path))
    print(f"Saved {path}  shape={M0.shape}  dtype={M0.dtype}")


# ---------------------------------------------------------------------------
# Validation printers
# ---------------------------------------------------------------------------
def print_trace_stats(M0, is_aniso):
    """Trace stats in WM voxels."""
    print("\n" + "=" * 60)
    print("Trace Statistics (WM voxels)")
    print("=" * 60)
    trace = M0[..., 0] + M0[..., 1] + M0[..., 2]
    wm_trace = trace[is_aniso]
    nonzero = wm_trace[wm_trace > 0]
    if len(nonzero) == 0:
        print("  WARNING: no non-zero trace values in WM")
        return
    print(f"  WM voxels:     {int(np.count_nonzero(is_aniso))}")
    print(f"  Non-zero trace: {len(nonzero)}")
    print(f"  Mean:  {float(np.mean(nonzero)):.4f}")
    print(f"  Std:   {float(np.std(nonzero)):.4f}")
    print(f"  Min:   {float(np.min(nonzero)):.4f}")
    print(f"  Max:   {float(np.max(nonzero)):.4f}")
    print(f"  Median: {float(np.median(nonzero)):.4f}")


def print_psd_check(M0, is_aniso, n_samples=10000):
    """Check positive semi-definiteness of M0 at random WM voxels."""
    print("\n" + "=" * 60)
    print("PSD Check (random WM voxels)")
    print("=" * 60)

    wm_indices = np.argwhere(is_aniso)
    if len(wm_indices) == 0:
        print("  No WM voxels to check")
        return

    rng = np.random.default_rng(42)
    n = min(n_samples, len(wm_indices))
    sample_idx = rng.choice(len(wm_indices), size=n, replace=False)

    n_fail = 0
    for idx in sample_idx:
        i, j, k = wm_indices[idx]
        m = M0[i, j, k]
        # Reconstruct symmetric 3x3 matrix
        mat = np.array([
            [m[0], m[3], m[4]],
            [m[3], m[1], m[5]],
            [m[4], m[5], m[2]],
        ], dtype=np.float64)
        eigvals = np.linalg.eigvalsh(mat)
        if np.any(eigvals < -1e-7):
            n_fail += 1

    print(f"  Sampled: {n}")
    print(f"  Failures: {n_fail}")


def print_zero_check(M0, is_aniso):
    """Verify M0 is zero outside WM."""
    print("\n" + "=" * 60)
    print("Zero Outside WM Check")
    print("=" * 60)
    outside = ~is_aniso
    nonzero_outside = int(np.count_nonzero(np.any(M0[outside] != 0, axis=-1)))
    print(f"  Non-zero voxels outside WM: {nonzero_outside}")


def print_coverage(fracs, brain_mask):
    """Report fraction coverage (% of brain voxels with f_n > 0)."""
    print("\n" + "=" * 60)
    print("Fraction Coverage")
    print("=" * 60)
    n_brain = int(np.count_nonzero(brain_mask))
    for n, f in enumerate(fracs):
        n_active = int(np.count_nonzero((f > 0) & brain_mask))
        pct = 100.0 * n_active / n_brain if n_brain > 0 else 0.0
        print(f"  f{n + 1}: {n_active}/{n_brain} ({pct:.1f}%)")


_CC_LABELS = frozenset({192, 251, 252, 253, 254, 255})


def print_principal_directions(M0, diff_labels):
    """Report principal fiber direction in corpus callosum (expect X-dominant).

    Uses FS labels mapped to diffusion space to identify CC voxels.
    """
    print("\n" + "=" * 60)
    print("Principal Directions (CC region)")
    print("=" * 60)

    # Build CC mask from FS labels in diffusion space
    cc_mask = np.zeros(diff_labels.shape, dtype=bool)
    for lab in _CC_LABELS:
        cc_mask |= (diff_labels == lab)

    n_cc = int(np.count_nonzero(cc_mask))
    if n_cc == 0:
        print("  No CC voxels found")
        return

    m0_cc = M0[cc_mask]
    avg_diag = np.array([
        float(np.mean(m0_cc[:, 0])),
        float(np.mean(m0_cc[:, 1])),
        float(np.mean(m0_cc[:, 2])),
    ])
    axis_names = ["X (L-R)", "Y (A-P)", "Z (I-S)"]
    dominant = int(np.argmax(avg_diag))
    print(f"  CC voxels: {n_cc}")
    print(f"  Avg M_00 (X): {avg_diag[0]:.4f}")
    print(f"  Avg M_11 (Y): {avg_diag[1]:.4f}")
    print(f"  Avg M_22 (Z): {avg_diag[2]:.4f}")
    print(f"  Dominant axis: {axis_names[dominant]}")


def print_smoothness(M0, is_aniso):
    """Mean gradient magnitude of trace in WM — expect < 0.5."""
    print("\n" + "=" * 60)
    print("Smoothness Check")
    print("=" * 60)

    trace = M0[..., 0] + M0[..., 1] + M0[..., 2]

    # Gradient magnitude (central differences)
    gx = np.diff(trace, axis=0)
    gy = np.diff(trace, axis=1)
    gz = np.diff(trace, axis=2)

    # Trim to common shape
    s = (
        min(gx.shape[0], gy.shape[0], gz.shape[0]),
        min(gx.shape[1], gy.shape[1], gz.shape[1]),
        min(gx.shape[2], gy.shape[2], gz.shape[2]),
    )
    grad_mag = np.sqrt(
        gx[:s[0], :s[1], :s[2]] ** 2 +
        gy[:s[0], :s[1], :s[2]] ** 2 +
        gz[:s[0], :s[1], :s[2]] ** 2
    )
    del gx, gy, gz

    # Mask to WM (trim is_aniso to match)
    wm_trimmed = is_aniso[:s[0], :s[1], :s[2]]
    wm_grad = grad_mag[wm_trimmed]
    del grad_mag

    if len(wm_grad) == 0:
        print("  No WM voxels for smoothness check")
        return

    mean_grad = float(np.mean(wm_grad))
    print(f"  Mean gradient: {mean_grad:.4f}")
    print(f"  Max gradient:  {float(np.max(wm_grad)):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    """Orchestrate fiber orientation preprocessing."""
    args = parse_args(argv)
    subject = args.subject
    f_threshold = args.f_threshold

    print(f"Subject: {subject}")
    print(f"Fraction threshold: {f_threshold}")
    print()

    # Paths
    t1w_dir = _PROJECT_ROOT / "data" / "raw" / subject / "T1w"
    bedpostx_dir = t1w_dir / "Diffusion.bedpostX"
    out_dir = _PROJECT_ROOT / "data" / "processed" / subject

    # 1. Load bedpostX
    print("--- Loading bedpostX data ---")
    dyads, fracs, brain_mask, diff_affine = load_bedpostx(bedpostx_dir)
    shape = dyads[0].shape[:3]
    print(f"Shape: {shape}  Brain voxels: {int(np.count_nonzero(brain_mask))}")
    print()

    # 2. Threshold fractions
    print("--- Thresholding fractions ---")
    threshold_fractions(fracs, f_threshold, brain_mask)
    print()

    # Coverage (before structure tensor computation frees fracs)
    print_coverage(fracs, brain_mask)

    # 3. Compute structure tensor
    print("\n--- Computing structure tensor M_0 ---")
    M0 = compute_structure_tensor(dyads, fracs)
    print(f"M0 shape: {M0.shape}  dtype: {M0.dtype}")
    print()

    # 4. WM mask via FS labels
    print("--- Building WM mask ---")
    fs_data, fs_affine = load_fs_labels(t1w_dir)
    aniso_lut = _build_aniso_lut()
    is_aniso, diff_labels = build_wm_mask(
        diff_affine, fs_data, fs_affine, shape, aniso_lut,
    )
    del fs_data, fs_affine
    n_aniso = int(np.count_nonzero(is_aniso))
    print(f"  Anisotropic voxels: {n_aniso}")

    apply_wm_mask(M0, is_aniso)
    print()

    # 5. Save
    print("--- Saving ---")
    save_fiber_m0(out_dir, M0, diff_affine)

    # 6. Validation
    print_trace_stats(M0, is_aniso)
    print_psd_check(M0, is_aniso)
    print_zero_check(M0, is_aniso)
    print_principal_directions(M0, diff_labels)
    print_smoothness(M0, is_aniso)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
