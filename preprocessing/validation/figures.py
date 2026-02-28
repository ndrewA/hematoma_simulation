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

    if which is None or 5 in which:
        if ctx.has_simnibs:
            print("  Generating Figure 5...")
            try:
                from preprocessing.validation.checks import _load_simnibs_resampled
                sim = ctx.get_cached("simnibs_resampled",
                                     lambda: _load_simnibs_resampled(ctx))
                generate_fig5(ctx.sdf, sim["sdf"], sim["labels"],
                              sim["inner_boundary"], t1w, N,
                              ctx.subject, ctx.profile, ctx.dx,
                              paths["fig5"])
            except Exception as e:
                print(f"  WARNING: Figure 5 failed: {e}")

    if which is None or 6 in which:
        if ctx.has_simnibs:
            # Ensure surface distances are computed (may not be if checks were skipped)
            if "gt_verts_ours" not in ctx._cache:
                from preprocessing.validation.checks import _compute_surface_distances
                sd = ctx.get_cached("surface_distances",
                                    lambda: _compute_surface_distances(ctx))
                ctx._cache["gt_verts_ours"] = sd["phys_ours"]
                ctx._cache["gt_verts_sim"] = sd["phys_sim"]
                ctx._cache["gt_d_o2s"] = sd["d_o2s"]
                ctx._cache["gt_d_s2o"] = sd["d_s2o"]
            print("  Generating Figure 6...")
            try:
                generate_fig6(
                    ctx._cache["gt_verts_ours"], ctx._cache["gt_verts_sim"],
                    ctx._cache["gt_d_o2s"], ctx._cache["gt_d_s2o"],
                    ctx.subject, ctx.profile, paths["fig6"])
            except Exception as e:
                print(f"  WARNING: Figure 6 failed: {e}")

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

    titles = ["Sagittal (x)", "Coronal (y)", "Axial (z)"]

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


# ---------------------------------------------------------------------------
# Figure 5: Skull SDF vs SimNIBS Ground Truth
# ---------------------------------------------------------------------------

def generate_fig5(our_sdf, simnibs_sdf, labels_sim, inner_boundary,
                  t1w, N, subject, profile, dx, path):
    """SDF contour comparison + error histogram vs SimNIBS ground truth."""
    from matplotlib.colors import TwoSlopeNorm

    mid = N // 2

    # SDF values at SimNIBS boundary
    sdf_at_boundary = our_sdf[inner_boundary]
    mae = float(np.mean(np.abs(sdf_at_boundary)))
    med = float(np.median(sdf_at_boundary))

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Skull SDF vs SimNIBS Ground Truth \u2014 {subject} / {profile} "
        f"({N}\u00b3, {dx} mm)",
        fontsize=14, fontweight="bold",
    )
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)

    # Row 1: Triplanar with both contours
    slice_specs = [
        ("Sagittal (x)", our_sdf[mid, :, :], simnibs_sdf[mid, :, :],
         t1w[mid, :, :] if t1w is not None else None),
        ("Coronal (y)", our_sdf[:, mid, :], simnibs_sdf[:, mid, :],
         t1w[:, mid, :] if t1w is not None else None),
        ("Axial (z)", our_sdf[:, :, mid], simnibs_sdf[:, :, mid],
         t1w[:, :, mid] if t1w is not None else None),
    ]

    for col, (title, our_slc, sim_slc, t1w_slc) in enumerate(slice_specs):
        ax = fig.add_subplot(gs[0, col])
        if t1w_slc is not None:
            ax.imshow(t1w_slc.T, origin="lower", cmap="gray",
                      interpolation="nearest")
        ax.contour(our_slc.T, levels=[0], colors=["#00FF00"],
                   linewidths=[2.0])
        ax.contour(sim_slc.T, levels=[0], colors=["#FF0000"],
                   linewidths=[2.0], linestyles="dashed")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Row 2: SDF difference in near-boundary shell
    diff_vol = our_sdf - simnibs_sdf
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    row2_specs = [
        ("Sagittal", diff_vol[mid, :, :], simnibs_sdf[mid, :, :],
         t1w[mid, :, :] if t1w is not None else None),
        ("Coronal", diff_vol[:, mid, :], simnibs_sdf[:, mid, :],
         t1w[:, mid, :] if t1w is not None else None),
        ("Axial", diff_vol[:, :, mid], simnibs_sdf[:, :, mid],
         t1w[:, :, mid] if t1w is not None else None),
    ]

    for col, (title, diff_slc, sim_slc, t1w_slc) in enumerate(row2_specs):
        ax = fig.add_subplot(gs[1, col])
        if t1w_slc is not None:
            ax.imshow(t1w_slc.T, origin="lower", cmap="gray",
                      interpolation="nearest", alpha=0.5)
        near_boundary = np.abs(sim_slc) < 3.0
        diff_masked = np.where(near_boundary, diff_slc, np.nan)
        im = ax.imshow(diff_masked.T, origin="lower", cmap="RdBu_r",
                       norm=norm, interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.7, label="SDF diff (mm)")
        ax.set_title(f"{title} \u2014 SDF difference", fontsize=10)
        ax.axis("off")
    del diff_vol

    # Row 3: Histogram + stats
    ax_hist = fig.add_subplot(gs[2, 0:2])
    bins = np.arange(-8, 8.25, 0.25)
    ax_hist.hist(sdf_at_boundary, bins=bins, alpha=0.7, color="#2196F3",
                 edgecolor="white", linewidth=0.3,
                 label="Our SDF at SimNIBS boundary")
    ax_hist.axvline(0, color="black", linewidth=1.5, linestyle="--",
                    label="Ideal (0 mm)")
    ax_hist.axvline(med, color="#FF5722", linewidth=2,
                    label=f"Median: {med:+.2f} mm")
    ax_hist.set_xlabel("Signed distance error (mm)", fontsize=11)
    ax_hist.set_ylabel("Voxel count", fontsize=11)
    ax_hist.set_title("Error distribution at SimNIBS inner skull boundary",
                      fontsize=11)
    ax_hist.legend(fontsize=9)
    ax_hist.set_xlim(-8, 8)

    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis("off")
    rmse = float(np.sqrt(np.mean(sdf_at_boundary ** 2)))
    stats_text = (
        f"Error at SimNIBS boundary\n"
        f"{'=' * 28}\n"
        f"N voxels:  {len(sdf_at_boundary):,}\n\n"
        f"Median:    {med:+.2f} mm\n"
        f"Mean:      {np.mean(sdf_at_boundary):+.2f} mm\n"
        f"Std:       {np.std(sdf_at_boundary):.2f} mm\n"
        f"MAE:       {mae:.2f} mm\n"
        f"RMSE:      {rmse:.2f} mm\n\n"
        f"P5:        {np.percentile(sdf_at_boundary, 5):+.2f} mm\n"
        f"P25:       {np.percentile(sdf_at_boundary, 25):+.2f} mm\n"
        f"P75:       {np.percentile(sdf_at_boundary, 75):+.2f} mm\n"
        f"P95:       {np.percentile(sdf_at_boundary, 95):+.2f} mm\n\n"
        f"Green = ours, Red = SimNIBS"
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, fontfamily="monospace", verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                            edgecolor="gray"))

    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 6: Surface Distance Spatial Error
# ---------------------------------------------------------------------------

def generate_fig6(verts_ours, verts_sim, d_o2s, d_s2o,
                  subject, profile, path):
    """Surface distance spatial error: scatter projections + histograms + CDF."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Surface Distance: Our SDF=0 vs SimNIBS Inner Skull \u2014 "
        f"{subject} / {profile}",
        fontsize=14, fontweight="bold")

    vmax = 8.0

    # Row 1: scatter projections (ours -> simnibs)
    projs = [
        ("Axial (X vs Y)", 0, 1),
        ("Coronal (X vs Z)", 0, 2),
        ("Sagittal (Y vs Z)", 1, 2),
    ]
    for col, (title, ax_i, ax_j) in enumerate(projs):
        ax = axes[0, col]
        n = len(verts_ours)
        step = max(1, n // 50000)
        idx = np.arange(0, n, step)
        sc = ax.scatter(verts_ours[idx, ax_i], verts_ours[idx, ax_j],
                        c=d_o2s[idx], s=0.3, cmap="hot_r",
                        vmin=0, vmax=vmax, rasterized=True)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("XYZ"[ax_i] + " (mm)")
        ax.set_ylabel("XYZ"[ax_j] + " (mm)")
        plt.colorbar(sc, ax=ax, label="Distance (mm)", shrink=0.8)

    # Row 2: histograms + CDF
    ax = axes[1, 0]
    ax.hist(d_o2s, bins=100, range=(0, 15), color="#2196F3",
            edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(d_o2s), color="red", linewidth=2,
               label=f"Mean: {np.mean(d_o2s):.2f}mm")
    ax.axvline(np.median(d_o2s), color="orange", linewidth=2,
               label=f"Median: {np.median(d_o2s):.2f}mm")
    ax.set_xlabel("Ours\u2192SimNIBS distance (mm)")
    ax.set_ylabel("Vertex count")
    ax.set_title("Ours\u2192SimNIBS")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.hist(d_s2o, bins=100, range=(0, 15), color="#4CAF50",
            edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(d_s2o), color="red", linewidth=2,
               label=f"Mean: {np.mean(d_s2o):.2f}mm")
    ax.axvline(np.median(d_s2o), color="orange", linewidth=2,
               label=f"Median: {np.median(d_s2o):.2f}mm")
    ax.set_xlabel("SimNIBS\u2192Ours distance (mm)")
    ax.set_ylabel("Vertex count")
    ax.set_title("SimNIBS\u2192Ours")
    ax.legend(fontsize=9)

    ax = axes[1, 2]
    thresholds = np.linspace(0, 15, 300)
    d_o2s_sorted = np.sort(d_o2s)
    d_s2o_sorted = np.sort(d_s2o)
    cdf_o2s = np.searchsorted(d_o2s_sorted, thresholds) / len(d_o2s) * 100
    cdf_s2o = np.searchsorted(d_s2o_sorted, thresholds) / len(d_s2o) * 100
    ax.plot(thresholds, cdf_o2s, label="Ours\u2192SimNIBS", linewidth=2)
    ax.plot(thresholds, cdf_s2o, label="SimNIBS\u2192Ours", linewidth=2)
    ax.axhline(95, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance threshold (mm)")
    ax.set_ylabel("% of surface within threshold")
    ax.set_title("Cumulative Distribution")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
