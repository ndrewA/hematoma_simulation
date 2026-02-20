"""Compare skull SDF inner boundary against SimNIBS ground truth.

Loads the pipeline's skull_sdf.nii.gz and SimNIBS final_tissues.nii.gz,
extracts the inner skull surface from SimNIBS, and measures how far
our SDF zero-crossing deviates from the SimNIBS boundary.

Usage:
    python -m preprocessing.compare_skull_sdf --subject 157336 --profile dev
"""

import argparse
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from preprocessing.utils import PROFILES, processed_dir, raw_dir, resample_to_grid


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare skull SDF against SimNIBS ground truth."
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--profile", required=True, choices=list(PROFILES.keys()))
    return parser.parse_args(argv)


def load_our_sdf(out_dir):
    """Load our skull_sdf.nii.gz and return (data, affine, dx)."""
    path = out_dir / "skull_sdf.nii.gz"
    print(f"Loading {path}")
    img = nib.load(str(path))
    sdf = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine.copy()
    dx = float(affine[0, 0])
    return sdf, affine, dx


def load_simnibs(raw, grid_affine, grid_shape, dx, out_dir=None):
    """Resample SimNIBS final_tissues to our simulation grid.

    Caches the resampled labels and SDF to out_dir/validation/ so
    subsequent runs skip the expensive resample + EDT (~5 min at 512³).

    Returns (labels_sim, simnibs_sdf).
    """
    cache_dir = out_dir / "validation" if out_dir is not None else None
    labels_cache = cache_dir / "simnibs_labels.nii.gz" if cache_dir else None
    sdf_cache = cache_dir / "simnibs_sdf.nii.gz" if cache_dir else None

    # Try loading from cache
    if sdf_cache and sdf_cache.exists() and labels_cache and labels_cache.exists():
        print(f"Loading cached {sdf_cache}")
        labels_sim = np.asarray(
            nib.load(str(labels_cache)).dataobj, dtype=np.int16
        )
        simnibs_sdf = np.asarray(
            nib.load(str(sdf_cache)).dataobj, dtype=np.float32
        )
        print(f"SimNIBS SDF range: [{simnibs_sdf.min():.1f}, {simnibs_sdf.max():.1f}] mm")
        return labels_sim, simnibs_sdf

    # Compute from scratch
    path = raw.parent / "final_tissues.nii.gz"
    if not path.exists():
        print(f"FATAL: SimNIBS not found at {path}")
        sys.exit(1)

    print(f"Loading {path}")
    simnibs_img = nib.load(str(path))
    simnibs_data = np.asarray(simnibs_img.dataobj, dtype=np.int16)
    # Squeeze trailing singleton (SimNIBS outputs 4D with shape[3]=1)
    if simnibs_data.ndim == 4 and simnibs_data.shape[3] == 1:
        simnibs_data = simnibs_data[:, :, :, 0]
    simnibs_affine = simnibs_img.affine

    print(f"Resampling SimNIBS to simulation grid ({grid_shape[0]}^3, dx={dx} mm)...")
    labels_sim = resample_to_grid(
        (simnibs_data, simnibs_affine), grid_affine, grid_shape,
        order=0, cval=0, dtype=np.int16,
    )
    del simnibs_data

    # Inner skull = everything inside the bone: WM(1) + GM(2) + CSF(3) + Blood(9)
    inner_skull = np.isin(labels_sim, [1, 2, 3, 9])

    print("Computing SimNIBS signed EDT...")
    sampling = (dx, dx, dx)
    dt_out = distance_transform_edt(~inner_skull, sampling=sampling)
    dt_in = distance_transform_edt(inner_skull, sampling=sampling)
    simnibs_sdf = (dt_out - dt_in).astype(np.float32)
    del dt_out, dt_in

    print(f"SimNIBS SDF range: [{simnibs_sdf.min():.1f}, {simnibs_sdf.max():.1f}] mm")

    # Cache for next time
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(labels_sim, grid_affine), str(labels_cache))
        nib.save(nib.Nifti1Image(simnibs_sdf, grid_affine), str(sdf_cache))
        print(f"Cached to {cache_dir}")

    return labels_sim, simnibs_sdf


def extract_inner_skull_boundary(labels_sim):
    """Extract SimNIBS inner skull surface voxels.

    Inner skull boundary = any intracranial voxel (WM, GM, CSF, Blood)
    adjacent to bone (labels 4, 7, 8).  Using all intracranial labels
    ensures full coverage — CSF-only misses convexities where the thin
    CSF layer is lost to nearest-neighbor resampling, and misses the
    midline vertex where the superior sagittal sinus (Blood) sits
    between brain and bone.
    """
    intracranial = np.isin(labels_sim, [1, 2, 3, 9])
    bone = np.isin(labels_sim, [4, 7, 8])
    bone_dilated = binary_dilation(bone)
    inner_boundary = intracranial & bone_dilated
    return inner_boundary


def analyze_error(our_sdf, simnibs_sdf, inner_boundary, dx):
    """Compute and print error statistics.

    Two complementary analyses:
    1. Our SDF sampled at SimNIBS inner skull boundary points
       (ideal: SDF = 0 there, actual value = signed distance error)
    2. Voxel-wise SDF difference in the near-boundary shell
    """
    print()
    print("=" * 65)
    print("  Error Analysis: Our Skull SDF vs SimNIBS Ground Truth")
    print("=" * 65)

    # --- Analysis 1: SDF value at SimNIBS boundary ---
    n_boundary = int(inner_boundary.sum())
    print(f"\n1. Our SDF at SimNIBS inner skull boundary ({n_boundary:,} voxels)")
    print(f"   (ideal = 0mm; negative = our boundary farther from brain)")

    sdf_at_boundary = our_sdf[inner_boundary]
    print(f"   Median: {np.median(sdf_at_boundary):+.2f} mm")
    print(f"   Mean:   {np.mean(sdf_at_boundary):+.2f} mm")
    print(f"   Std:    {np.std(sdf_at_boundary):.2f} mm")
    print(f"   P5:     {np.percentile(sdf_at_boundary, 5):+.2f} mm")
    print(f"   P25:    {np.percentile(sdf_at_boundary, 25):+.2f} mm")
    print(f"   P75:    {np.percentile(sdf_at_boundary, 75):+.2f} mm")
    print(f"   P95:    {np.percentile(sdf_at_boundary, 95):+.2f} mm")
    print(f"   MAE:    {np.mean(np.abs(sdf_at_boundary)):.2f} mm")

    # --- Analysis 2: SDF difference in near-boundary shell ---
    shell = np.abs(simnibs_sdf) < 5.0  # within 5mm of SimNIBS boundary
    n_shell = int(shell.sum())
    sdf_diff = our_sdf[shell] - simnibs_sdf[shell]

    print(f"\n2. SDF difference in near-boundary shell (|SimNIBS SDF| < 5mm, {n_shell:,} voxels)")
    print(f"   (positive = our boundary closer to brain than SimNIBS)")
    print(f"   Median: {np.median(sdf_diff):+.2f} mm")
    print(f"   Mean:   {np.mean(sdf_diff):+.2f} mm")
    print(f"   Std:    {np.std(sdf_diff):.2f} mm")
    print(f"   MAE:    {np.mean(np.abs(sdf_diff)):.2f} mm")
    print(f"   RMSE:   {np.sqrt(np.mean(sdf_diff ** 2)):.2f} mm")
    print(f"   P5:     {np.percentile(sdf_diff, 5):+.2f} mm")
    print(f"   P95:    {np.percentile(sdf_diff, 95):+.2f} mm")

    # --- Analysis 3: Regional breakdown by brain-relative axial thirds ---
    print(f"\n3. Regional error (brain-relative axial thirds)")
    zs = np.where(inner_boundary.any(axis=(0, 1)))[0]
    if len(zs) > 0:
        z_min, z_max = int(zs[0]), int(zs[-1])
        z_span = z_max - z_min
        cuts = [z_min, z_min + z_span // 3,
                z_min + 2 * z_span // 3, z_max + 1]
        for name, z_lo, z_hi in [("Inferior", cuts[0], cuts[1]),
                                  ("Middle", cuts[1], cuts[2]),
                                  ("Superior", cuts[2], cuts[3])]:
            region = inner_boundary.copy()
            region[:, :, :z_lo] = False
            region[:, :, z_hi:] = False
            n_r = int(region.sum())
            if n_r > 0:
                vals = our_sdf[region]
                print(f"   {name:10s} (z {z_lo}-{z_hi}): "
                      f"median={np.median(vals):+.2f}, "
                      f"MAE={np.mean(np.abs(vals)):.2f}, "
                      f"n={n_r:,}")
            else:
                print(f"   {name:10s} (z {z_lo}-{z_hi}): no boundary voxels")

    return sdf_at_boundary, sdf_diff


def generate_figure(our_sdf, simnibs_sdf, labels_sim, inner_boundary,
                    sdf_at_boundary, sdf_diff,
                    t1w_sim, dx, N, subject, profile, out_path):
    """Generate comparison figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mid = N // 2

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Skull SDF vs SimNIBS Ground Truth \u2014 {subject} / {profile} "
        f"({N}\u00b3, {dx} mm)",
        fontsize=14, fontweight="bold",
    )

    # Use gridspec for flexible layout: 3 rows
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)

    # ── Row 1: Triplanar with both boundaries ──
    slice_specs = [
        ("Axial (z)", our_sdf[mid, :, :], simnibs_sdf[mid, :, :],
         t1w_sim[mid, :, :] if t1w_sim is not None else None),
        ("Coronal (y)", our_sdf[:, mid, :], simnibs_sdf[:, mid, :],
         t1w_sim[:, mid, :] if t1w_sim is not None else None),
        ("Sagittal (x)", our_sdf[:, :, mid], simnibs_sdf[:, :, mid],
         t1w_sim[:, :, mid] if t1w_sim is not None else None),
    ]

    for col, (title, our_slc, sim_slc, t1w_slc) in enumerate(slice_specs):
        ax = fig.add_subplot(gs[0, col])
        if t1w_slc is not None:
            ax.imshow(t1w_slc.T, origin="lower", cmap="gray",
                      interpolation="nearest")

        # Our SDF contours
        ax.contour(our_slc.T, levels=[0], colors=["#00FF00"],
                   linewidths=[2.0])
        # SimNIBS SDF contours
        ax.contour(sim_slc.T, levels=[0], colors=["#FF0000"],
                   linewidths=[2.0], linestyles="dashed")

        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # ── Row 2: SDF difference maps near the boundary ──
    # Show (our_sdf - simnibs_sdf) in a thin shell around the SimNIBS boundary.
    # This gives continuous coverage over the entire skull surface.
    diff_vol = our_sdf - simnibs_sdf
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)

    row2_specs = [
        ("Sagittal", diff_vol[mid, :, :], simnibs_sdf[mid, :, :],
         t1w_sim[mid, :, :] if t1w_sim is not None else None),
        ("Coronal", diff_vol[:, mid, :], simnibs_sdf[:, mid, :],
         t1w_sim[:, mid, :] if t1w_sim is not None else None),
        ("Axial", diff_vol[:, :, mid], simnibs_sdf[:, :, mid],
         t1w_sim[:, :, mid] if t1w_sim is not None else None),
    ]

    for col, (title, diff_slc, sim_slc, t1w_slc) in enumerate(row2_specs):
        ax = fig.add_subplot(gs[1, col])
        if t1w_slc is not None:
            ax.imshow(t1w_slc.T, origin="lower", cmap="gray",
                      interpolation="nearest", alpha=0.5)
        # Mask to shell within 3mm of SimNIBS boundary
        near_boundary = np.abs(sim_slc) < 3.0
        diff_masked = np.where(near_boundary, diff_slc, np.nan)
        im = ax.imshow(diff_masked.T, origin="lower", cmap="RdBu_r",
                        norm=norm, interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.7, label="SDF diff (mm)")
        ax.set_title(f"{title} \u2014 SDF difference", fontsize=10)
        ax.axis("off")

    del diff_vol

    # ── Row 3: Histogram + stats ──
    # Panel [2,0:2]: Error histogram (wide)
    ax_hist = fig.add_subplot(gs[2, 0:2])
    bins = np.arange(-8, 8.25, 0.25)

    ax_hist.hist(sdf_at_boundary, bins=bins, alpha=0.7, color="#2196F3",
                 edgecolor="white", linewidth=0.3,
                 label="Our SDF at SimNIBS boundary")
    ax_hist.axvline(0, color="black", linewidth=1.5, linestyle="--",
                    label="Ideal (0 mm)")
    med = np.median(sdf_at_boundary)
    ax_hist.axvline(med, color="#FF5722", linewidth=2,
                    label=f"Median: {med:+.2f} mm")

    ax_hist.set_xlabel("Signed distance error (mm)", fontsize=11)
    ax_hist.set_ylabel("Voxel count", fontsize=11)
    ax_hist.set_title("Error distribution at SimNIBS inner skull boundary",
                      fontsize=11)
    ax_hist.legend(fontsize=9)
    ax_hist.set_xlim(-8, 8)

    # Panel [2,2]: Stats text
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis("off")

    mae = np.mean(np.abs(sdf_at_boundary))
    rmse = np.sqrt(np.mean(sdf_at_boundary ** 2))
    stats_text = (
        f"Error at SimNIBS boundary\n"
        f"{'=' * 28}\n"
        f"N voxels:  {len(sdf_at_boundary):,}\n"
        f"\n"
        f"Median:    {med:+.2f} mm\n"
        f"Mean:      {np.mean(sdf_at_boundary):+.2f} mm\n"
        f"Std:       {np.std(sdf_at_boundary):.2f} mm\n"
        f"MAE:       {mae:.2f} mm\n"
        f"RMSE:      {rmse:.2f} mm\n"
        f"\n"
        f"P5:        {np.percentile(sdf_at_boundary, 5):+.2f} mm\n"
        f"P25:       {np.percentile(sdf_at_boundary, 25):+.2f} mm\n"
        f"P75:       {np.percentile(sdf_at_boundary, 75):+.2f} mm\n"
        f"P95:       {np.percentile(sdf_at_boundary, 95):+.2f} mm\n"
        f"\n"
        f"Green = ours, Red = SimNIBS"
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, fontfamily="monospace", verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                            edgecolor="gray"))

    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")

    args = parse_args(argv)
    subject = args.subject
    profile = args.profile
    N, dx = PROFILES[profile]

    raw = raw_dir(subject)
    out_dir = processed_dir(subject, profile)

    # Load our SDF
    our_sdf, grid_affine, _ = load_our_sdf(out_dir)
    grid_shape = our_sdf.shape

    # Load and resample SimNIBS
    labels_sim, simnibs_sdf = load_simnibs(raw, grid_affine, grid_shape, dx, out_dir)

    # Extract inner skull boundary
    inner_boundary = extract_inner_skull_boundary(labels_sim)
    print(f"SimNIBS inner skull boundary: {int(inner_boundary.sum()):,} voxels")

    # Error analysis
    sdf_at_boundary, sdf_diff = analyze_error(
        our_sdf, simnibs_sdf, inner_boundary, dx
    )

    # Load T1w for underlay
    t1w_path = raw / "T1w_acpc_dc_restore_brain.nii.gz"
    t1w_sim = None
    if t1w_path.exists():
        print(f"\nResampling T1w for figure underlay...")
        t1w_sim = resample_to_grid(
            str(t1w_path), grid_affine, grid_shape,
            order=1, cval=0.0, dtype=np.float32,
        )

    # Generate figure
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    fig_path = val_dir / "fig_skull_sdf_vs_simnibs.png"
    generate_figure(
        our_sdf, simnibs_sdf, labels_sim, inner_boundary,
        sdf_at_boundary, sdf_diff,
        t1w_sim, dx, N, subject, profile, fig_path,
    )


if __name__ == "__main__":
    main()
