"""Diagnostic figure generation for validation.

Figures 1-4 visualize material maps, dural membranes, skull SDF,
and fiber orientation (DEC) maps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from preprocessing.material_map import CLASS_NAMES
from preprocessing.utils import resample_to_grid

# Material colormap: u8 index -> RGBA
MATERIAL_COLORS = [
    (0.0, 0.0, 0.0, 0.0),    # 0: Vacuum (transparent)
    (1.0, 1.0, 1.0, 1.0),    # 1: Cerebral WM (white)
    (0.3, 0.3, 0.3, 1.0),    # 2: Cortical GM (dark gray)
    (1.0, 1.0, 0.0, 1.0),    # 3: Deep GM (yellow)
    (1.0, 1.0, 0.94, 1.0),   # 4: Cerebellar WM (ivory)
    (0.5, 0.5, 0.0, 1.0),    # 5: Cerebellar Cortex (olive)
    (1.0, 0.65, 0.0, 1.0),   # 6: Brainstem (orange)
    (0.0, 0.0, 1.0, 1.0),    # 7: Ventricular CSF (blue)
    (0.0, 1.0, 1.0, 1.0),    # 8: Subarachnoid CSF (cyan)
    (1.0, 0.0, 1.0, 1.0),    # 9: Choroid Plexus (magenta)
    (1.0, 0.0, 0.0, 1.0),    # 10: Dural Membrane (red)
    (0.0, 1.0, 0.0, 1.0),    # 11: Vessel / Venous Sinus (green)
]
_MAT_CMAP = ListedColormap(MATERIAL_COLORS)
_MAT_NORM = BoundaryNorm(np.arange(-0.5, 12.5, 1.0), _MAT_CMAP.N)


def generate_all_figures(ctx, which=None):
    """Generate diagnostic figures, resampling T1w once.

    Parameters
    ----------
    which : set of int or None
        Figure numbers to generate (e.g. {1, 2}).  None means all.
    """
    from scipy.ndimage import binary_dilation

    mat = ctx.mat
    N = ctx.N
    paths = ctx.paths

    # Resample T1w to grid (shared across figs 1-3)
    need_t1w = which is None or bool(which & {1, 2, 3})
    t1w = None
    if need_t1w:
        t1w_path = paths["t1w"]
        if t1w_path.exists():
            t1w = resample_to_grid(
                str(t1w_path), ctx.mat_affine, (N, N, N),
                order=1, cval=0.0, dtype=np.float32,
            )
        else:
            print(f"  WARNING: T1w not found, figures will lack underlay")

    if which is None or 1 in which:
        print("  Generating Figure 1...")
        try:
            generate_fig1(mat, t1w, N, ctx.subject, ctx.profile, ctx.dx, paths["fig1"])
        except Exception as e:
            print(f"  WARNING: Figure 1 failed: {e}")

    if which is None or 2 in which:
        print("  Generating Figure 2...")
        try:
            generate_fig2(mat, t1w, N, ctx.subject, ctx.profile, ctx.dx, paths["fig2"])
        except Exception as e:
            print(f"  WARNING: Figure 2 failed: {e}")

    if which is None or 3 in which:
        sdf = ctx.sdf
        brain = ctx.brain
        if sdf is not None and brain is not None:
            print("  Generating Figure 3...")
            try:
                generate_fig3(mat, sdf, t1w, brain, N, ctx.subject, ctx.profile,
                              ctx.dx, paths["fig3"])
            except Exception as e:
                print(f"  WARNING: Figure 3 failed: {e}")

    if which is None or 4 in which:
        if ctx.fiber_data is not None:
            print("  Generating Figure 4...")
            try:
                generate_fig4(ctx.fiber_data, ctx.subject, ctx.profile, paths["fig4"])
            except Exception as e:
                print(f"  WARNING: Figure 4 failed: {e}")

    del t1w


# ---------------------------------------------------------------------------
# Figure 1: Material Map Triplanar
# ---------------------------------------------------------------------------

def generate_fig1(mat, t1w, N, subject, profile, dx, path):
    """Material map triplanar: 2 rows x 3 columns."""
    mid = N // 2

    slices_mat = [
        mat[mid, :, :],
        mat[:, mid, :],
        mat[:, :, mid],
    ]
    slices_t1w = [
        t1w[mid, :, :],
        t1w[:, mid, :],
        t1w[:, :, mid],
    ] if t1w is not None else [None, None, None]

    titles = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Material Map — {subject} / {profile} ({N}\u00b3, {dx} mm)",
                 fontsize=14, fontweight="bold")

    for col in range(3):
        ax = axes[0, col]
        ax.imshow(slices_mat[col].T, origin="lower", cmap=_MAT_CMAP,
                  norm=_MAT_NORM, interpolation="nearest")
        ax.set_title(f"{titles[col]} — material")
        ax.axis("off")

        ax = axes[1, col]
        if slices_t1w[col] is not None:
            ax.imshow(slices_t1w[col].T, origin="lower", cmap="gray",
                      interpolation="nearest")
        ax.imshow(slices_mat[col].T, origin="lower", cmap=_MAT_CMAP,
                  norm=_MAT_NORM, interpolation="nearest", alpha=0.4)
        ax.set_title(f"{titles[col]} — overlay on T1w")
        ax.axis("off")

    _add_material_legend(fig)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _add_material_legend(fig):
    """Add material class legend to the right side of the figure."""
    handles = []
    for uid in range(12):
        color = MATERIAL_COLORS[uid]
        if uid == 0:
            continue
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color[:3],
                                     ec="black", linewidth=0.5))
    labels = [f"{uid}: {CLASS_NAMES.get(uid, '???')}" for uid in range(1, 12)]
    fig.legend(handles, labels, loc="center right", fontsize=8,
               title="Material Classes", title_fontsize=9)


# ---------------------------------------------------------------------------
# Figure 2: Dural Membrane Detail
# ---------------------------------------------------------------------------

def generate_fig2(mat, t1w, N, subject, profile, dx, path):
    """Dural membrane detail: 3x3 panels."""
    mid_x = N // 2
    dural = (mat == 10)

    if not dural.any():
        print(f"  Skipping fig2 (no dural voxels)")
        return

    brainstem = (mat == 6)
    if brainstem.any():
        z_indices = np.where(brainstem.any(axis=(0, 1)))[0]
        z_min, z_max = int(z_indices.min()), int(z_indices.max())
        z_tent = z_min + 2 * (z_max - z_min) // 3
    else:
        z_tent = N // 2

    wm_mid = (mat[mid_x - 2:mid_x + 2, :, :] == 1)
    if wm_mid.any():
        y_indices = np.where(wm_mid.any(axis=(0, 2)))[0]
        y_ant = int(y_indices.min()) + (int(y_indices.max()) - int(y_indices.min())) // 4
        y_mid = (int(y_indices.min()) + int(y_indices.max())) // 2
        y_post = int(y_indices.max()) - (int(y_indices.max()) - int(y_indices.min())) // 4
    else:
        y_ant, y_mid, y_post = N // 3, N // 2, 2 * N // 3

    cerebellar = (mat == 5)
    if cerebellar.any():
        cb_y = int(np.mean(np.where(cerebellar.any(axis=(0, 2)))[0]))
    else:
        cb_y = y_post

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle(f"Dural Membrane Detail — {subject} / {profile}",
                 fontsize=14, fontweight="bold")

    def _show_dural_panel(ax, mat_slice, t1w_slice, title):
        if t1w_slice is not None:
            ax.imshow(t1w_slice.T, origin="lower", cmap="gray",
                      interpolation="nearest")

        mat_display = mat_slice.copy().astype(float)
        mat_display[mat_slice == 10] = np.nan
        ax.imshow(mat_display.T, origin="lower", cmap=_MAT_CMAP,
                  norm=_MAT_NORM, interpolation="nearest", alpha=0.2)

        dural_mask = (mat_slice == 10)
        if dural_mask.any():
            dural_overlay = np.zeros(mat_slice.shape + (4,))
            dural_overlay[dural_mask, :] = [1.0, 0.0, 1.0, 1.0]
            ax.imshow(np.transpose(dural_overlay, (1, 0, 2)),
                      origin="lower", interpolation="nearest")

        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Row 1
    _show_dural_panel(axes[0, 0], mat[mid_x, :, :],
                      t1w[mid_x, :, :] if t1w is not None else None,
                      f"Midsagittal x={mid_x}")

    falx_region = dural.copy()
    falx_region[:mid_x - 5, :, :] = False
    falx_region[mid_x + 5:, :, :] = False
    ax = axes[0, 1]
    ax.imshow(falx_region[mid_x, :, :].T, origin="lower", cmap="Reds",
              interpolation="nearest")
    ax.set_title(f"Falx mask, midsag x={mid_x}", fontsize=9)
    ax.axis("off")
    del falx_region

    _show_dural_panel(axes[0, 2], mat[:, y_mid, :],
                      t1w[:, y_mid, :] if t1w is not None else None,
                      f"Coronal y={y_mid} (mid CC)")

    # Row 2
    _show_dural_panel(axes[1, 0], mat[:, :, z_tent],
                      t1w[:, :, z_tent] if t1w is not None else None,
                      f"Axial z={z_tent} (tentorial)")

    if brainstem.any():
        bs_x = int(np.mean(np.where(brainstem[:, :, z_tent].any(axis=1))[0]))
        bs_y = int(np.mean(np.where(brainstem[:, :, z_tent].any(axis=0))[0]))
        pad = 30
        x_lo, x_hi = max(0, bs_x - pad), min(N, bs_x + pad)
        y_lo, y_hi = max(0, bs_y - pad), min(N, bs_y + pad)
        _show_dural_panel(axes[1, 1], mat[x_lo:x_hi, y_lo:y_hi, z_tent],
                          t1w[x_lo:x_hi, y_lo:y_hi, z_tent] if t1w is not None else None,
                          f"Notch zoom z={z_tent}")
    else:
        axes[1, 1].set_title("No brainstem", fontsize=9)
        axes[1, 1].axis("off")

    _show_dural_panel(axes[1, 2], mat[:, cb_y, :],
                      t1w[:, cb_y, :] if t1w is not None else None,
                      f"Coronal y={cb_y} (post. fossa)")

    # Row 3
    for col_idx, (y_slice, label) in enumerate([
        (y_ant, "anterior CC"),
        (y_mid, "mid CC"),
        (y_post, "posterior CC"),
    ]):
        _show_dural_panel(axes[2, col_idx], mat[:, y_slice, :],
                          t1w[:, y_slice, :] if t1w is not None else None,
                          f"Coronal y={y_slice} ({label})")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Skull SDF + Domain Boundary
# ---------------------------------------------------------------------------

def generate_fig3(mat, sdf, t1w, brain, N, subject, profile, dx, path):
    """Skull SDF + domain boundary: multi-slice triplanar grid.

    Row 0-2: Axial slices (inferior → superior)
    Row 3-5: Coronal slices (posterior → anterior)
    Row 6-8: Sagittal slices (left → right)
    Each row has 3 evenly spaced slices through the brain extent.
    """
    contour_levels = [-20, -10, -5, 0, 5, 10]
    contour_colors = ["#0000FF", "#4444FF", "#8888FF", "#00FF00", "#FF8888", "#FF0000"]
    contour_lw = [0.3, 0.3, 0.3, 1.0, 0.3, 0.3]

    # Find brain extent on each axis
    def _brain_range(axis):
        slices = np.where(brain.any(axis=tuple(i for i in range(3) if i != axis)))[0]
        return int(slices[0]), int(slices[-1])

    def _pick_slices(lo, hi, n=3):
        margin = (hi - lo) // (2 * (n + 1))
        return np.linspace(lo + margin, hi - margin, n).astype(int).tolist()

    ax_lo, ax_hi = _brain_range(2)  # axial = axis 2
    cor_lo, cor_hi = _brain_range(1)  # coronal = axis 1
    sag_lo, sag_hi = _brain_range(0)  # sagittal = axis 0

    axial_zs = _pick_slices(ax_lo, ax_hi)
    coronal_ys = _pick_slices(cor_lo, cor_hi)
    sagittal_xs = _pick_slices(sag_lo, sag_hi)

    nrows = 3
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 14))
    fig.suptitle(f"Skull SDF + Domain Boundary — {subject} / {profile}",
                 fontsize=14, fontweight="bold")

    def _plot_sdf(ax, sdf_slc, t1w_slc, brain_slc, title):
        if t1w_slc is not None:
            ax.imshow(t1w_slc.T, origin="lower", cmap="gray",
                      interpolation="nearest")
        ax.contour(sdf_slc.T, levels=contour_levels,
                   colors=contour_colors, linewidths=contour_lw)
        ax.contour(brain_slc.astype(float).T, levels=[0.5],
                   colors=["#4444FF"], linewidths=[0.5], linestyles="dotted")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Row 0: Axial slices
    for col, z in enumerate(axial_zs):
        _plot_sdf(axes[0, col],
                  sdf[:, :, z], t1w[:, :, z] if t1w is not None else None,
                  brain[:, :, z], f"Axial z={z}")

    # Row 1: Coronal slices
    for col, y in enumerate(coronal_ys):
        _plot_sdf(axes[1, col],
                  sdf[:, y, :], t1w[:, y, :] if t1w is not None else None,
                  brain[:, y, :], f"Coronal y={y}")

    # Row 2: Sagittal slices
    for col, x in enumerate(sagittal_xs):
        _plot_sdf(axes[2, col],
                  sdf[x, :, :], t1w[x, :, :] if t1w is not None else None,
                  brain[x, :, :], f"Sagittal x={x}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Fiber DEC Map
# ---------------------------------------------------------------------------

def generate_fig4(fiber_data, subject, profile, path):
    """Direction-Encoded Color map at native fiber resolution: 2x2 + legend."""
    shape = fiber_data.shape[:3]

    slices_info = [
        ("Axial z=72 (CC level)", 2, 72),
        ("Coronal y=100", 1, 100),
        ("Sagittal x=72", 0, 72),
        ("Axial z=82 (corona radiata)", 2, 82),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                             gridspec_kw={"width_ratios": [1, 1, 0.3]})
    fig.suptitle(f"Fiber Orientation DEC Map — {subject} / {profile}",
                 fontsize=14, fontweight="bold")

    panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, ((title, axis, slice_idx), (row, col)) in enumerate(
            zip(slices_info, panel_positions)):
        ax = axes[row, col]

        if slice_idx >= shape[axis]:
            ax.set_title(f"{title} — OOB", fontsize=9)
            ax.axis("off")
            continue

        if axis == 0:
            tensor_slice = fiber_data[slice_idx, :, :, :]
        elif axis == 1:
            tensor_slice = fiber_data[:, slice_idx, :, :]
        else:
            tensor_slice = fiber_data[:, :, slice_idx, :]

        dec_rgb = _compute_dec_slice(tensor_slice)
        ax.imshow(np.transpose(dec_rgb, (1, 0, 2)), origin="lower",
                  interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for row in range(2):
        axes[row, 2].axis("off")

    ax_leg = axes[0, 2]
    ax_leg.text(0.5, 0.8, "DEC Convention", ha="center", va="top",
                fontsize=11, fontweight="bold", transform=ax_leg.transAxes)
    ax_leg.text(0.5, 0.6, "R = Left-Right (X)", ha="center", va="top",
                fontsize=10, color="red", transform=ax_leg.transAxes)
    ax_leg.text(0.5, 0.45, "G = Ant-Post (Y)", ha="center", va="top",
                fontsize=10, color="green", transform=ax_leg.transAxes)
    ax_leg.text(0.5, 0.3, "B = Sup-Inf (Z)", ha="center", va="top",
                fontsize=10, color="blue", transform=ax_leg.transAxes)
    ax_leg.text(0.5, 0.1, "Brightness = trace", ha="center", va="top",
                fontsize=9, transform=ax_leg.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _compute_dec_slice(tensor_slice):
    """Compute DEC RGB image from a 2D slice of 6-component tensors.

    tensor_slice: (W, H, 6)
    Returns: (W, H, 3) float32 RGB in [0, 1]
    """
    W, H = tensor_slice.shape[:2]
    trace = tensor_slice[..., 0] + tensor_slice[..., 1] + tensor_slice[..., 2]
    trace_max = trace.max()
    if trace_max == 0:
        return np.zeros((W, H, 3), dtype=np.float32)

    rgb = np.zeros((W, H, 3), dtype=np.float32)
    nonzero = trace > 0

    for i in range(W):
        nz_cols = np.where(nonzero[i])[0]
        if len(nz_cols) == 0:
            continue

        m = tensor_slice[i, nz_cols]
        mats = np.zeros((len(nz_cols), 3, 3), dtype=np.float64)
        mats[:, 0, 0] = m[:, 0]
        mats[:, 1, 1] = m[:, 1]
        mats[:, 2, 2] = m[:, 2]
        mats[:, 0, 1] = mats[:, 1, 0] = m[:, 3]
        mats[:, 0, 2] = mats[:, 2, 0] = m[:, 4]
        mats[:, 1, 2] = mats[:, 2, 1] = m[:, 5]

        eigvals, eigvecs = np.linalg.eigh(mats)
        principal = eigvecs[:, :, -1]
        abs_principal = np.abs(principal)

        brightness = (trace[i, nz_cols] / trace_max) ** 0.4
        rgb[i, nz_cols] = abs_principal * brightness[:, np.newaxis]

    return np.clip(rgb, 0, 1)
