# Preprocessing Step 7: Validation and Diagnostic Visualization

This document specifies the cross-cutting validation and publication-quality diagnostic visualization for all preprocessing outputs (Tasks 5–10). It is the final preprocessing step — a quality gate that confirms the anatomical model is correct before the solver runs.

## 1. Purpose

The per-step validation checks embedded in Tasks 5–10 verify local invariants (brain volume conservation, label preservation, SDF smoothness, membrane continuity). This step validates **cross-cutting** invariants that span multiple outputs and cannot be checked in isolation:

1. **Header consistency.** All grid-resolution NIfTIs must share identical affines. The fiber texture affine must map to the same physical space. Grid metadata must agree with NIfTI headers.
2. **Domain closure.** The material map and skull SDF must form a sealed domain — no vacuum inside the skull, no tissue outside it.
3. **Cross-resolution fiber coverage.** The forward transform from the material map grid through physical space to diffusion voxel coordinates must produce valid M_0 values at white matter locations, matching the runtime sampling path.

The step also produces **diagnostic figures** — spatially-localized error detection that statistics cannot provide. A brain volume within normal range does not prove the material map is correct; a triplanar overlay on T1w shows whether the classes are in the right anatomical locations.

**Dual role:** These outputs serve both engineering QC (catch bugs before the solver runs) and research methodology evidence (publication-quality figures for the paper, JSON metrics for cross-subject aggregation in Table 1).

## 2. Inputs

All preprocessing outputs from Tasks 5–10, plus the source T1w for visualization overlay:

| File | Source | Shape | Dtype | Content |
|------|--------|-------|-------|---------|
| `material_map.nii.gz` | Task 9 (final) | N³ | uint8 | Material classes 0–11 |
| `skull_sdf.nii.gz` | Task 7 | N³ | float32 | Signed distance to inner skull (mm) |
| `brain_mask.nii.gz` | Task 5 | N³ | uint8 | Binary brain mask |
| `fs_labels_resampled.nii.gz` | Task 5 | N³ | int16 | Original FreeSurfer labels |
| `grid_meta.json` | Task 5 | — | JSON | Grid parameters (N, dx_mm, affines) |
| `fiber_M0.nii.gz` | Task 10 | 145×174×145×6 | float32 | Structure tensor (profile-independent) |
| `T1w_acpc_dc_restore_brain.nii.gz` | Raw HCP | 260×311×260 | float32 | T1w structural image for overlay |

All grid-resolution files are in `data/processed/{subject_id}/{profile}/`. The fiber texture is in `data/processed/{subject_id}/` (profile-independent). The T1w source is in `data/raw/{subject_id}/T1w/`.

## 3. Automated Validation Checks

Each check has a severity level:
- **CRITICAL** — Solver will produce incorrect results. Pipeline must halt.
- **WARN** — Likely error or unusual value. Investigate before proceeding.
- **INFO** — Reported for completeness. No expected failure.

All checks run to completion regardless of failures — the script never short-circuits, ensuring a complete diagnostic picture.

### 3.1 Module: Header Consistency (10 checks, all CRITICAL)

These verify that all preprocessing outputs are in the same coordinate system and that metadata is self-consistent. No volume data is loaded — only NIfTI headers and JSON.

| ID | Check | Criterion |
|----|-------|-----------|
| H1 | All grid-resolution NIfTIs have bitwise-identical affines | `np.array_equal(mat.affine, sdf.affine)` etc. for all 4 grid-resolution files |
| H2 | All grid-resolution NIfTIs have shape (N,N,N) matching `grid_meta.grid_size` | `img.shape == (N, N, N)` for `material_map`, `skull_sdf`, `brain_mask`, `fs_labels_resampled` |
| H3 | `material_map` dtype is uint8 | `img.get_data_dtype() == np.uint8` |
| H4 | `skull_sdf` dtype is float32 | `img.get_data_dtype() == np.float32` |
| H5 | `fiber_M0` dtype is float32 | `img.get_data_dtype() == np.float32` |
| H6 | `fiber_M0` shape matches expected diffusion dimensions | `img.shape == (145, 174, 145, 6)` |
| H7 | `grid_meta.json` dx_mm matches NIfTI affine diagonal | `abs(meta['dx_mm'] - affine[0,0]) < 1e-6` |
| H8 | `grid_meta.json` grid_size matches NIfTI shape[0] | `meta['grid_size'] == img.shape[0]` |
| H9 | `grid_meta.json` affine_grid_to_phys matches NIfTI sform | `np.allclose(meta['affine_grid_to_phys'], img.affine, atol=1e-6)` |
| H10 | Fiber texture affine round-trip: physical(0,0,0) maps to consistent location | See Section 3.1.1 |

**H10 detail:** The ACPC origin (physical 0,0,0) should map to the grid center (N/2, N/2, N/2) via the grid affine and to the corresponding diffusion voxel via the fiber affine. Verify:

```python
grid_affine = mat_img.affine
fiber_affine = fiber_img.affine

# Physical origin → grid voxel
phys_origin = np.array([0.0, 0.0, 0.0, 1.0])
grid_vox = np.linalg.inv(grid_affine) @ phys_origin
assert np.allclose(grid_vox[:3], N/2, atol=0.5), "Grid affine does not center on ACPC"

# Physical origin → diffusion voxel (should land inside the diffusion volume)
diff_vox = np.linalg.inv(fiber_affine) @ phys_origin
for d in range(3):
    assert 0 <= diff_vox[d] < fiber_img.shape[d], \
        f"ACPC origin maps outside fiber texture on axis {d}: {diff_vox[d]}"
```

### 3.2 Module: Domain Closure (5 checks)

These verify the sealed-domain invariant — the material map and skull SDF together form a closed domain with no gaps. Requires loading `material_map`, `skull_sdf`, and `brain_mask`.

| ID | Severity | Check | Criterion |
|----|----------|-------|-----------|
| D1 | CRITICAL | Zero vacuum voxels inside skull | `count(sdf < 0 AND mat == 0) == 0` |
| D2 | CRITICAL | Zero tissue voxels outside skull | `count(sdf > 0 AND mat in {1..11}) == 0` |
| D3 | CRITICAL | Brain containment | `all(sdf[brain_mask == 1] < 0)` |
| D4 | WARN | No isolated vacuum islands inside the active domain | See below |
| D5 | WARN | SDF Eikonal property: gradient magnitude ≈ 1.0 | `np.percentile(sampled_grad_mag, [5, 95])` in [0.8, 1.2] |

**D1** is the strongest invariant of the preprocessing pipeline. If vacuum exists inside the skull, the runtime solver encounters undefined material at active voxels. This was already checked in Task 8's per-step validation; the cross-cutting check here confirms the invariant survives the subsequent dural membrane painting (Task 9), which modifies the material map.

**D2** catches the inverse failure: tissue labels that leaked outside the skull. This can occur if the SDF and material map were resampled with inconsistent affines or if nearest-neighbor resampling placed a tissue label at a boundary voxel where the SDF is positive.

**D3** is a weaker form of D1 — every brain voxel must be inside the skull. This is guaranteed by construction (the skull SDF was built from the brain mask with outward dilation), but verifying it here catches affine mismatches.

**D4 detail:** Compute connected components of vacuum (mat == 0) and check for isolated islands completely surrounded by tissue/CSF (sdf < 0):

```python
vacuum = (mat == 0)
vacuum_labels, n_comp = ndimage.label(vacuum)
# The main vacuum component is the exterior (largest)
sizes = np.bincount(vacuum_labels.ravel())
main_label = sizes[1:].argmax() + 1

# Any vacuum component entirely inside the skull?
for label_id in range(1, n_comp + 1):
    if label_id == main_label:
        continue
    component = (vacuum_labels == label_id)
    if np.all(sdf[component] < 0):
        n_island = component.sum()
        print(f"  WARN D4: Vacuum island of {n_island} voxels inside skull")
```

**D5 detail:** A true signed distance field satisfies the Eikonal equation |∇φ| = 1.0 everywhere. Deviations indicate resampling distortion, clipping, or incorrect SDF construction. Since computing the full gradient volume (3 × 512 MB float32) would spike memory, D5 samples 100k random interior voxels and checks the gradient magnitude using central differences:

```python
# Sample 100k interior voxels (away from the boundary to avoid natural curvature effects)
interior = np.argwhere(sdf < -1.0)
sample = interior[rng.choice(len(interior), size=min(100_000, len(interior)), replace=False)]

# Central-difference gradient magnitude at sampled points
dx = meta['dx_mm']
grad_mag_sq = np.zeros(len(sample))
for axis in range(3):
    fwd = sample.copy(); fwd[:, axis] = np.clip(fwd[:, axis] + 1, 0, N - 1)
    bwd = sample.copy(); bwd[:, axis] = np.clip(bwd[:, axis] - 1, 0, N - 1)
    diff = sdf[fwd[:, 0], fwd[:, 1], fwd[:, 2]] - sdf[bwd[:, 0], bwd[:, 1], bwd[:, 2]]
    grad_mag_sq += (diff / (2 * dx)) ** 2
grad_mag = np.sqrt(grad_mag_sq)

p5, p95 = np.percentile(grad_mag, [5, 95])
# WARN if p5 < 0.8 or p95 > 1.2
```

Memory cost is negligible: 100k × 3 × 8 bytes (indices) + 100k × 8 bytes (gradient) ≈ 3.2 MB. The sampling excludes voxels near the zero-contour (sdf > -1.0 mm) where curvature naturally causes |∇φ| to deviate from 1.0.

### 3.3 Module: Material Integrity (4 checks)

These verify the material map contains only valid values and is consistent with the brain mask.

| ID | Severity | Check | Criterion |
|----|----------|-------|-----------|
| M1 | CRITICAL | All unique values in {0..11} | `set(np.unique(mat)) ⊆ {0,1,...,11}` |
| M2 | CRITICAL | Does not contain 255 | `255 not in np.unique(mat)` — air halo is runtime-only |
| M3 | WARN | Every expected class {1..11} is represented | `for c in range(1,12): count(mat==c) > 0` |
| M4 | WARN | Brain mask consistency | `all(mat[brain_mask==1] in {1..11})` — every brain voxel has tissue/fluid |

**M2 rationale:** Material ID 255 (air halo) is assigned at runtime by the topology update kernel. If it appears in the preprocessed material map, something wrote to the wrong output or an old runtime artifact leaked back.

**M3:** At debug resolution (256³, 2.0 mm), some thin structures (dural membrane, vessels) may have zero voxels. The check warns but does not fail, since the debug profile is not used for production simulation.

**M4:** A brain mask voxel with mat == 0 indicates a labeling gap — the brain mask considers it intracranial, the FS labels did not label it, and the subarachnoid CSF gap-fill did not reach it. This should be impossible after Task 8 but would indicate a serious bug if present.

### 3.4 Module: Volume Sanity (6 checks)

These verify that anatomical volumes are within physiologically plausible ranges. All thresholds are for healthy adult brains (HCP cohort: young adults, 22–35 years).

| ID | Severity | Check | Range | Rationale |
|----|----------|-------|-------|-----------|
| V1 | WARN | Brain parenchyma volume | [800, 2000] mL | u8 ∈ {1,2,3,4,5,6,9}: all solid tissue. Healthy adult: 1200–1500 mL. |
| V2 | WARN | Ventricular CSF | [10, 60] mL | u8 = 7. Healthy young adult: 15–30 mL. |
| V3 | WARN | Subarachnoid CSF | [100, 500] mL | u8 = 8. Includes meninges at sim resolution. |
| V4 | WARN | Dural membrane | [2, 50] mL | u8 = 10. Resolution-dependent (1–2 voxel sheets). |
| V5 | WARN | ICV ≈ sum of all nonzero-material voxels | `abs(n_sdf_neg - n_nonzero_mat) / n_sdf_neg < 0.02` |
| V6 | INFO | Complete volume census | — | Report all 12 classes + vacuum |

Volume computation:

```python
dx = meta['dx_mm']
vol_voxel_mL = (dx ** 3) / 1000.0  # mm³ → mL

for uid in range(12):
    n = np.count_nonzero(mat == uid)
    vol = n * vol_voxel_mL
    census[uid] = {'name': CLASS_NAMES[uid], 'voxels': int(n), 'volume_mL': round(vol, 2)}

brain_parenchyma = np.isin(mat, [1, 2, 3, 4, 5, 6, 9]).sum() * vol_voxel_mL
ventricular_csf = (mat == 7).sum() * vol_voxel_mL
subarachnoid_csf = (mat == 8).sum() * vol_voxel_mL
dural_membrane = (mat == 10).sum() * vol_voxel_mL
```

**V5 rationale:** The intracranial volume (voxels with SDF < 0) should approximately equal the sum of all nonzero-material voxels (everything that isn't vacuum). A discrepancy > 2% indicates misalignment between the SDF and material map. Note: if D1 and D2 both pass (zero violations in the strict interior/exterior), V5 can only fail due to voxels at exactly `sdf == 0` — the boundary layer where the SDF zero-contour crosses voxel centers. A small discrepancy from this layer is expected; V5's 2% tolerance accommodates it.

**V6 output format** (for JSON aggregation):

```json
{
  "volume_census": {
    "0":  {"name": "Vacuum",              "voxels": 132710000, "volume_mL": 132710.0},
    "1":  {"name": "Cerebral WM",         "voxels": 550000,    "volume_mL": 550.0},
    "2":  {"name": "Cortical GM",         "voxels": 520000,    "volume_mL": 520.0},
    ...
  }
}
```

### 3.5 Module: Compartmentalization (4 checks)

These verify the structural integrity of the pressure compartments created by the dural membranes.

| ID | Severity | Check | Criterion |
|----|----------|-------|-----------|
| C1 | INFO | Active domain connected components | Report count and sizes for mat ∈ {1..11} |
| C2 | WARN | Falx forms 1 connected component | >90% of u8=10 voxels within the falx region in largest component |
| C3 | WARN | Tentorium forms 1 connected component | >90% of u8=10 voxels within the tentorial region in largest component |
| C4 | WARN | Tentorial notch patency | CSF (u8=8) adjacent to brainstem (u8=6) at tentorial level |

**C1 detail:** The active domain (all nonzero-material voxels) should ideally be one connected component — the sealed intracranial volume. Multiple large components indicate a gap in the domain (the dural membranes create permeability barriers, not topological barriers). Small fragments (< 100 voxels) at the domain boundary are harmless.

```python
active = (mat > 0) & (mat != 255)
labels, n = ndimage.label(active)
sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1]
print(f"Active domain: {n} components, largest: {sizes[0]:,}")
```

**C2/C3 detail:** The dural membrane (u8=10) includes both the falx and tentorium. To check each separately, split by anatomy: falx voxels are near the midsagittal plane (x ≈ N/2), tentorium voxels are at the cerebral-cerebellar boundary (z below the tentorial level).

```python
dural = (mat == 10)
N = mat.shape[0]
mid_x = N // 2

# Falx: dural voxels near midline (within ±5 voxels of x = N/2)
falx_region = dural.copy()
falx_region[:mid_x - 5, :, :] = False
falx_region[mid_x + 5:, :, :] = False

falx_labels, n_falx = ndimage.label(falx_region)
if n_falx > 0:
    falx_sizes = np.sort(np.bincount(falx_labels.ravel())[1:])[::-1]
    frac_largest = falx_sizes[0] / falx_sizes.sum()
    check_pass = frac_largest > 0.90

# Tentorium: dural voxels away from midline
tent_region = dural & ~falx_region
tent_labels, n_tent = ndimage.label(tent_region)
# ... similar size analysis
```

**C4 detail:** At the tentorial level, CSF should exist immediately adjacent to the brainstem, forming the perimesencephalic cisterns (the tentorial notch). If the tentorium sealed the notch, no CSF would be adjacent to brainstem voxels at this level.

```python
brainstem = (mat == 6)
bs_slices = np.where(brainstem.any(axis=(0, 1)))[0]
bs_upper_z = bs_slices[len(bs_slices) * 2 // 3]  # upper third ≈ tentorial level

bs_slice = brainstem[:, :, bs_upper_z]
csf_slice = (mat[:, :, bs_upper_z] == 8)
bs_dilated = ndimage.binary_dilation(bs_slice, iterations=1)
csf_adjacent = csf_slice & bs_dilated & ~bs_slice
n_adjacent = np.count_nonzero(csf_adjacent)

# WARN if no CSF around brainstem at tentorial level
assert n_adjacent > 0, "Tentorial notch may be sealed"
```

### 3.6 Module: Fiber Coverage (6 checks, cross-resolution)

These verify the fiber texture is valid and correctly aligned with the simulation grid. The critical test (F1) performs the **full forward-transform** matching the runtime sampling path: material map grid → physical → diffusion voxel coordinates → trilinear interpolation of M_0.

| ID | Severity | Check | Criterion |
|----|----------|-------|-----------|
| F1 | WARN | ≥90% of sampled WM voxels have tr(M_0) > 0 | Forward-transform coverage |
| F2 | CRITICAL | All M_0 diagonal elements ≥ 0 | PSD property (all 6 channels checked) |
| F3 | WARN | tr(M_0) ≤ 1.0 everywhere in fiber texture | Unnormalized fractions bounded by sum ≤ 1 |
| F4 | WARN | CC landmark principal direction is X-dominant | `|eigvec[0]| > 0.7` at corpus callosum |
| F5 | WARN | Internal capsule landmark principal direction is Z-dominant | `|eigvec[2]| > 0.7` at internal capsule |
| F6 | CRITICAL | Fiber texture affine round-trip | Physical(0,0,0) maps correctly in both spaces |

**F1: Cross-resolution forward-transform coverage.**

This is the most important fiber check. It verifies that the runtime sampling path — grid voxel → physical coordinate → diffusion voxel → trilinear interpolation of M_0 — produces valid data at WM locations.

```python
# Identify WM voxels in the material map
wm_mask = np.isin(mat, [1, 4, 6])  # cerebral WM, cerebellar WM, brainstem
wm_indices = np.argwhere(wm_mask)

# Sample 50,000 random WM voxels
rng = np.random.default_rng(42)
n_sample = min(50000, len(wm_indices))
sample = wm_indices[rng.choice(len(wm_indices), size=n_sample, replace=False)]

# Forward transform: grid voxel → physical → diffusion voxel
grid_affine = mat_img.affine                      # grid → physical
fiber_affine = fiber_img.affine                    # diffusion → physical
phys_to_diff = np.linalg.inv(fiber_affine)         # physical → diffusion
grid_to_diff = phys_to_diff @ grid_affine          # composite

# Transform sample grid indices to diffusion voxel coordinates
grid_homo = np.column_stack([sample, np.ones(n_sample)])  # (n, 4)
diff_coords = (grid_to_diff @ grid_homo.T).T[:, :3]      # (n, 3)

# Trilinear interpolation of M_0 at these coordinates
from scipy.ndimage import map_coordinates
fiber_data = fiber_img.get_fdata()  # (145, 174, 145, 6)
trace_at_wm = np.zeros(n_sample)
for ch in range(3):  # diagonal elements: M_00, M_11, M_22
    vals = map_coordinates(
        fiber_data[..., ch], diff_coords.T,
        order=1, mode='constant', cval=0.0
    )
    trace_at_wm += vals

coverage = (trace_at_wm > 0).sum() / n_sample
# WARN if coverage < 0.90
```

**F2: Positive semi-definiteness.** M_0 is a sum of weighted outer products — it must be PSD by construction. Check the diagonal elements (which are the eigenvalue lower bound for diagonal-dominant matrices):

```python
for ch in range(3):
    diag = fiber_data[..., ch]
    n_negative = np.count_nonzero(diag < -1e-7)
    # CRITICAL if n_negative > 0
```

The full PSD check (eigenvalue decomposition) is expensive at every voxel. The diagonal check catches the most common failure (sign error in the outer product). For a thorough check, sample 10k nonzero voxels and verify all eigenvalues ≥ -1e-7.

**F3: Trace bound.** Since volume fractions are unnormalized bedpostX fractions with f1 + f2 + f3 ≤ 1 (the remainder is the isotropic compartment), tr(M_0) = Σ f_n should be ≤ 1.0 at every voxel. Values > 1.0 + epsilon indicate corrupted input data.

```python
trace = fiber_data[..., 0] + fiber_data[..., 1] + fiber_data[..., 2]
n_over = np.count_nonzero(trace > 1.0 + 1e-5)
# WARN if n_over > 0
```

**F4/F5: Landmark direction checks.** These verify that the fiber orientations are anatomically plausible at known landmarks.

For F4, the corpus callosum at midsagittal level has fibers running left-right (X-dominant). The landmark voxel is approximately at bedpostX voxel (72, 100, 72) — the midline, slightly anterior to center, at CC level.

```python
def get_principal_direction(M0_6):
    """Extract principal eigenvector from 6-component upper triangle."""
    m = np.array([[M0_6[0], M0_6[3], M0_6[4]],
                  [M0_6[3], M0_6[1], M0_6[5]],
                  [M0_6[4], M0_6[5], M0_6[2]]])
    eigvals, eigvecs = np.linalg.eigh(m)
    return eigvecs[:, -1]  # largest eigenvalue

cc_vox = fiber_data[72, 100, 72]
cc_dir = get_principal_direction(cc_vox)
# WARN if |cc_dir[0]| < 0.7 (expect X-dominant)

ic_vox = fiber_data[52, 110, 62]  # internal capsule (approximate)
ic_dir = get_principal_direction(ic_vox)
# WARN if |ic_dir[2]| < 0.7 (expect Z-dominant)
```

These checks are approximate — the exact voxel coordinates depend on the subject's anatomy. The landmark coordinates given are for subject 157336. For multi-subject use, the implementation should derive landmark voxels automatically: transform the CC centroid (from FS labels 251–255 in the material map grid) and the internal capsule centroid (from FS labels 58/60 in the left/right posterior limb) through the grid-to-diffusion affine to obtain bedpostX voxel coordinates. This replaces the hardcoded coordinates with anatomy-driven landmarks that generalize across subjects.

**F6:** Cross-references H10 — not a unique check. The ACPC origin must map to a valid location in both the grid and diffusion coordinate systems. Listed here so that the fiber module's check inventory is self-contained; the implementation runs the test once (in Phase 1 as H10) and copies the result to F6.

## 4. Diagnostic Figures

All figures are 300 DPI PNGs with proper axis labels (L/R, A/P, S/I), colorbars/legends, and subject/profile annotation text. The matplotlib Agg backend is used (no display server required).

### 4.1 Figure 1: Material Map Triplanar (6 panels)

**Purpose:** Verify that all 11 material classes are in the correct anatomical locations. The T1w underlay provides spatial reference — misaligned classes or mislabeled regions are immediately visible.

**Layout:** 2 rows × 3 columns.

| | Axial | Coronal | Sagittal |
|---|---|---|---|
| **Row 1** | Material map alone | Material map alone | Material map alone |
| **Row 2** | Material map at 40% opacity over T1w | Material map at 40% opacity over T1w | Material map at 40% opacity over T1w |

**Slice selection:** All slices pass through the brain center (approximately the ACPC origin), at grid index (N/2, N/2, N/2).

**Colormap:** A categorical colormap with 12 distinct colors (one per u8 class). Vacuum (u8=0) is transparent. The colormap must distinguish adjacent classes that occupy nearby anatomical regions (e.g., WM=1 vs GM=2 vs deep GM=3).

| u8 | Class | Color |
|---:|-------|-------|
| 0 | Vacuum | Transparent |
| 1 | Cerebral WM | White |
| 2 | Cortical GM | Dark gray |
| 3 | Deep GM | Yellow |
| 4 | Cerebellar WM | Ivory |
| 5 | Cerebellar Cortex | Olive |
| 6 | Brainstem | Orange |
| 7 | Ventricular CSF | Blue |
| 8 | Subarachnoid CSF | Cyan |
| 9 | Choroid Plexus | Magenta |
| 10 | Dural Membrane | Red |
| 11 | Vessel / Sinus | Green |

**T1w underlay:** The T1w source image (0.7 mm) is resampled to the simulation grid in-memory at full N³ resolution using the `resample_to_grid` utility with trilinear interpolation. This resampled volume is used for all figure overlays but is **not saved to disk** — it is a transient visualization aid.

```python
t1w_sim = resample_to_grid(
    source_nifti_path=t1w_path,
    grid_affine=grid_affine,
    grid_shape=(N, N, N),
    order=1,  # trilinear
    cval=0.0,
    dtype=np.float32,
    slab_size=32
)
```

**Annotation:** Subject ID, profile name, grid size, and voxel spacing in the figure title.

### 4.2 Figure 2: Dural Membrane Detail (9 panels)

**Purpose:** Verify the falx and tentorium are anatomically correct — the right thickness, in the right location, with the tentorial notch open. These are 1–2 voxel structures; statistical checks cannot localize errors.

**Layout:** 3 rows × 3 columns.

| | Column 1 | Column 2 | Column 3 |
|---|---|---|---|
| **Row 1** | Midsagittal: mat map with dural highlighted | Binary falx mask (dural near midline) | Coronal through CC: falx above, CC below |
| **Row 2** | Axial at tentorial level | Zoomed: tentorial notch region | Coronal through posterior fossa |
| **Row 3** | Coronal at y = anterior CC | Coronal at y = mid CC | Coronal at y = posterior CC |

**Dural highlighting:** Dural membrane voxels (u8=10) are rendered in magenta at full opacity. All other material classes are shown at 20% opacity on the T1w underlay. This makes the 1–2 voxel membrane clearly visible against the anatomy.

**Slice selection:**
- Midsagittal: x = N/2
- Tentorial level: z at the upper third of the brainstem's axial extent (the cerebral-cerebellar transition)
- CC coronal slices: at the anterior, mid, and posterior extent of corpus callosum voxels (u8=1 with FS labels 251–255)
- Posterior fossa: coronal slice through the cerebellum center

**Zoomed notch panel:** The tentorial notch panel is cropped to ±30 voxels around the brainstem center at the tentorial level, showing the thin CSF ring (cyan) between the brainstem (orange) and dural membrane (magenta).

### 4.3 Figure 3: Skull SDF + Domain Boundary (6 panels)

**Purpose:** Verify the SDF is smooth, correctly signed, and aligned with the anatomy. The zero-contour defines the domain boundary — it must wrap the brain without dipping into sulci or leaving gaps at the convexity.

**Layout:** 2 rows × 3 columns.

| | Axial | Coronal | Sagittal |
|---|---|---|---|
| **Row 1** | SDF contours on T1w | SDF contours on T1w | SDF contours on T1w |
| **Row 2** | Filled SDF colormap (RdBu_r) | Brain mask boundary on SDF | Sylvian fissure close-up |

**Row 1 contours:** SDF isolines at {-20, -10, -5, 0, +5, +10} mm overlaid on the T1w underlay. The zero-contour is drawn in thick black (linewidth=2). Negative contours in blue, positive in red. Slices through the brain center.

```python
contour_levels = [-20, -10, -5, 0, 5, 10]
colors = ['#0000FF', '#4444FF', '#8888FF', '#000000', '#FF8888', '#FF0000']
linewidths = [0.5, 0.5, 0.5, 2.0, 0.5, 0.5]
```

**Row 2, panel 1:** SDF values as a filled colormap (RdBu_r, centered at 0, range ±30 mm). Blue = inside skull, red = outside.

**Row 2, panel 2:** The brain mask boundary (1-voxel-dilated minus original) overlaid on the SDF. This shows the margin between the brain surface and the SDF = 0 isosurface — it should be approximately R_dilate (4 mm) everywhere.

**Row 2, panel 3:** A zoomed coronal view of the Sylvian fissure region. The Sylvian fissure is the deepest cortical concavity — if the morphological closing in Task 7 failed, the SDF zero-contour would dip into the fissure. The zoom region is ±40 voxels around the lateral fissure (approximately y = N/2, lateral to the brain center).

### 4.4 Figure 4: Fiber Orientation DEC Map (4 panels + legend)

**Purpose:** Verify fiber orientations are anatomically plausible. The Direction-Encoded Color (DEC) convention makes errors immediately visible — the corpus callosum should be red (L/R), internal capsule blue (S/I), cingulum green (A/P).

**Layout:** 2 rows × 2 columns + color sphere legend.

| | Column 1 | Column 2 |
|---|---|---|
| **Row 1** | Axial at CC level | Coronal mid-ventricular |
| **Row 2** | Sagittal midline | Axial at corona radiata level |

**DEC convention:** RGB = |principal eigenvector|, brightness = trace(M_0). Specifically:

```python
# For each voxel with trace > 0:
eigvals, eigvecs = np.linalg.eigh(M0_3x3)
principal = eigvecs[:, -1]
rgb = np.abs(principal)  # [|x|, |y|, |z|] → [R, G, B]
brightness = trace / trace_max  # normalize to [0, 1]
dec_color = rgb * brightness
```

- Red = Left-Right (X)
- Green = Anterior-Posterior (Y)
- Blue = Superior-Inferior (Z)

**Resolution:** These panels show the fiber texture at **native bedpostX resolution** (145×174×145). No upsampling is performed — the DEC map reflects what the runtime will actually sample. The slice indices are in diffusion voxel coordinates.

**Slice selection (approximate, for subject 157336):**
- CC level axial: z ≈ 72 (bedpostX voxel, approximately at the body of the corpus callosum)
- Mid-ventricular coronal: y ≈ 100
- Midline sagittal: x ≈ 72
- Corona radiata axial: z ≈ 82 (above the lateral ventricles, where the corona radiata fans out)

**Color sphere legend:** A small inset showing a sphere colored by |eigenvector| with axis labels: R=L/R, G=A/P, B=S/I. This follows the standard DEC convention used in DTI visualization.

### 4.5 Figure 5: Validation Summary (text + thumbnails)

**Purpose:** A single-page overview with all check results and key metrics, suitable for quick review or inclusion as a supplementary figure.

**Layout:** 3 sections arranged vertically.

**Section 1: Check Results Table.** All checks from Sections 3.1–3.6 in a table with columns: ID, Severity, Description, Result (PASS/WARN/FAIL), Value. Color-coded: green=PASS, yellow=WARN, red=FAIL.

**Section 2: Material Map Census Table.** Columns: u8, Class Name, Voxel Count, Volume (mL). All 12 classes + vacuum.

**Section 3: Key Metrics + Thumbnails.** A row of key numbers:
- Brain parenchyma volume (mL)
- Intracranial volume (mL)
- Ventricular CSF (mL)
- Subarachnoid CSF (mL)
- Dural membrane volume (mL)
- Fiber WM coverage (%)

Below the metrics, 4 small thumbnails (one from each of Figures 1–4) as a visual index. Each thumbnail is a single representative panel reduced to ~2 inches wide.

## 5. Architecture

### 5.1 Single Script

All validation and visualization is implemented in a single script: `validate_preprocessing.py`. This avoids double-loading the large grid volumes (~1.4 GB combined for material_map + skull_sdf + brain_mask + fs_labels at 512³). The script runs all checks and generates all figures in a single pass through the data.

### 5.2 Phase-Based Memory Management

The script loads and frees data in phases to control peak memory. The dev profile (512³) has the largest volumes.

**Phase 1: Header Checks (peak ~0 MB data)**

Only NIfTI headers and `grid_meta.json` are read. No volume data is loaded. All 10 header checks (H1–H10) run here.

**Phase 2: Grid Volumes (peak ~1,280 MB)**

Load material_map (128 MB), skull_sdf (512 MB), brain_mask (128 MB). Optionally load T1w resampled in-memory (512 MB float32 at 512³, computed via slab-based `resample_to_grid`).

Run checks: D1–D5, M1–M4, V1–V6, C1.

Generate: Figure 1 (material map triplanar), Figure 3 (skull SDF).

After figure generation, free skull_sdf and T1w (no longer needed).

| Array | Dtype | Size (512³) |
|-------|-------|:-----------:|
| material_map | uint8 | 128 MB |
| skull_sdf | float32 | 512 MB |
| brain_mask | uint8 | 128 MB |
| T1w resampled | float32 | 512 MB |
| **Peak** | | **~1,280 MB** |

The T1w resampling uses slab-based processing (32-slice slabs), so the peak during resampling itself is lower — the 512 MB output array is filled incrementally. The peak of 1,280 MB occurs when the completed T1w array coexists with the other three volumes for figure generation.

**Phase 3: Dural Membrane (peak ~768 MB)**

Material_map (128 MB) and brain_mask (128 MB) are still in memory from Phase 2. The T1w was freed at the end of Phase 2.

This phase has two sub-peaks that do not overlap:

**Sub-phase 3a — checks (C2–C4):** Connected component analysis allocates an int32 label array (512 MB at 512³). CC labels are freed after checks complete.

| Array | Size (512³) |
|-------|:-----------:|
| material_map | 128 MB |
| brain_mask | 128 MB |
| Connected component labels (int32) | 512 MB |
| **Sub-peak (checks)** | **~768 MB** |

**Sub-phase 3b — Figure 2:** T1w is re-resampled for dural overlay (~10–20s cost, same slab-based `resample_to_grid`). The CC labels are already freed, so the T1w replaces them in the memory budget.

| Array | Size (512³) |
|-------|:-----------:|
| material_map | 128 MB |
| brain_mask | 128 MB |
| T1w resampled | 512 MB |
| **Sub-peak (figures)** | **~768 MB** |

Free T1w and brain_mask after this phase.

Note: scipy's `ndimage.label` allocates an int32 output (512 MB at 512³). This is the dominant cost. The connected component analysis runs on the dural mask (much smaller), but if run on the full active domain (C1), the int32 output covers the full grid.

**Phase 4: Fiber Texture (peak ~480 MB)**

Load fiber_M0 (88 MB). Material_map (128 MB) still in memory for WM identification.

Run checks: F1–F6.

Generate: Figure 4 (DEC map).

Free fiber_M0 and material_map after this phase.

| Array | Size |
|-------|:----:|
| material_map | 128 MB |
| fiber_M0 (145×174×145×6, float32) | 88 MB |
| Coordinate arrays for forward transform | ~175 MB |
| Interpolated trace values (50k floats) | negligible |
| **Peak** | **~391 MB** |

The coordinate arrays for the 50k-sample forward transform (F1) are modest: 50000 × 3 × 8 bytes = 1.2 MB. The DEC map computation requires eigen-decomposition per voxel but only on 2D slices (145 × 174 per axial slice = 25k voxels), which is fast.

**Phase 5: Summary (peak ~50 MB)**

All checks are complete. Assemble results into Figure 5 and the JSON report. Only small data structures (check results, census tables) are in memory.

### 5.3 Overall Peak Memory

**~1,280 MB** during Phase 2 (grid volumes + T1w overlay). With OS and Python overhead (~500 MB), total system usage ~1.8 GB of 5.7 GB available. Comfortable.

At debug resolution (256³): all grid volumes are 8× smaller. Peak ~160 MB. Trivial.

## 6. Outputs

All outputs are saved to `data/processed/{subject_id}/{profile}/validation/`:

```
data/processed/{subject_id}/{profile}/validation/
  validation_report.json
  fig1_material_map.png
  fig2_dural_detail.png
  fig3_skull_sdf.png
  fig4_fiber_dec.png
  fig5_summary.png
  launch_fsleyes.sh
```

### 6.1 validation_report.json

Machine-readable report for cross-subject aggregation. Schema:

```json
{
  "subject_id": "157336",
  "profile": "dev",
  "grid_size": 512,
  "dx_mm": 1.0,
  "timestamp": "2026-02-16T14:30:00Z",
  "overall_status": "PASS",

  "checks": {
    "H1": {"severity": "CRITICAL", "description": "Bitwise-identical affines", "status": "PASS", "value": null},
    "H2": {"severity": "CRITICAL", "description": "Shape matches grid_meta", "status": "PASS", "value": "512x512x512"},
    "D1": {"severity": "CRITICAL", "description": "Zero vacuum inside skull", "status": "PASS", "value": 0},
    "D2": {"severity": "CRITICAL", "description": "Zero tissue outside skull", "status": "PASS", "value": 0},
    "V1": {"severity": "WARN",     "description": "Brain parenchyma volume", "status": "PASS", "value": 1332.1},
    ...
  },

  "volume_census": {
    "0":  {"name": "Vacuum",              "voxels": 132710000, "volume_mL": 132710.0},
    "1":  {"name": "Cerebral WM",         "voxels": 550000,    "volume_mL": 550.0},
    "2":  {"name": "Cortical GM",         "voxels": 520000,    "volume_mL": 520.0},
    "3":  {"name": "Deep GM",             "voxels": 85000,     "volume_mL": 85.0},
    "4":  {"name": "Cerebellar WM",       "voxels": 25000,     "volume_mL": 25.0},
    "5":  {"name": "Cerebellar Cortex",   "voxels": 62000,     "volume_mL": 62.0},
    "6":  {"name": "Brainstem",           "voxels": 22000,     "volume_mL": 22.0},
    "7":  {"name": "Ventricular CSF",     "voxels": 25000,     "volume_mL": 25.0},
    "8":  {"name": "Subarachnoid CSF",    "voxels": 280000,    "volume_mL": 280.0},
    "9":  {"name": "Choroid Plexus",      "voxels": 1200,      "volume_mL": 1.2},
    "10": {"name": "Dural Membrane",      "voxels": 15000,     "volume_mL": 15.0},
    "11": {"name": "Vessel / Sinus",      "voxels": 800,       "volume_mL": 0.8}
  },

  "key_metrics": {
    "brain_parenchyma_mL": 1332.1,
    "intracranial_volume_mL": 1620.0,
    "ventricular_csf_mL": 25.0,
    "subarachnoid_csf_mL": 280.0,
    "dural_membrane_mL": 15.0,
    "fiber_wm_coverage_pct": 94.2,
    "fiber_trace_mean": 0.38,
    "fiber_trace_p95": 0.72,
    "domain_closure_violations": 0,
    "active_domain_components": 1,
    "falx_components": 1,
    "tentorium_components": 1
  },

  "figures": [
    "fig1_material_map.png",
    "fig2_dural_detail.png",
    "fig3_skull_sdf.png",
    "fig4_fiber_dec.png",
    "fig5_summary.png"
  ]
}
```

The `overall_status` is:
- `"PASS"` — no CRITICAL failures, no WARN failures
- `"WARN"` — no CRITICAL failures, one or more WARN failures
- `"FAIL"` — one or more CRITICAL failures

### 6.2 launch_fsleyes.sh

A convenience script for interactive 3D inspection:

```bash
#!/bin/bash
# Launch FSLeyes with all preprocessing outputs overlaid
DIR="$(dirname "$0")/.."
fsleyes \
  "$DIR/skull_sdf.nii.gz" -cm blue-lightblue -dr -30 30 \
  "$DIR/material_map.nii.gz" -cm random -ot label \
  "$DIR/brain_mask.nii.gz" -cm green -a 30 \
  "$DIR/fs_labels_resampled.nii.gz" -cm random -ot label -a 0 &
```

### 6.3 Figure Files

Five PNG files at 300 DPI. Typical sizes:
- fig1 (6 panels): ~2–4 MB
- fig2 (9 panels): ~3–5 MB
- fig3 (6 panels): ~2–4 MB
- fig4 (4 panels + legend): ~1–3 MB
- fig5 (text + thumbnails): ~1–2 MB

## 7. Implementation

### 7.1 Stack

Python 3, numpy, scipy (`scipy.ndimage` for connected components and `map_coordinates` for interpolation), nibabel (NIfTI I/O), matplotlib (Agg backend, for all figures). Same dependencies as Tasks 5–10 — no additional packages.

### 7.2 CLI Interface

```
python validate_preprocessing.py \
    --subject 157336 \
    --profile dev \
    [--no-images]    \
    [--no-fiber]     \
    [--no-dural]     \
    [--verbose]
```

| Flag | Effect |
|------|--------|
| `--subject` | Subject ID (required) |
| `--profile` | Simulation profile: debug/dev/prod (required) |
| `--no-images` | Skip all figure generation. Run checks only. Useful for batch validation of many subjects. |
| `--no-fiber` | Skip Phase 4 (fiber checks and Figure 4). For runs where bedpostX data is unavailable. |
| `--no-dural` | Skip Phase 3 (dural compartmentalization checks and Figure 2). For runs where dural membranes were not reconstructed. |
| `--verbose` | Print per-check detail to console (default: summary only). |

**Auto-detection:** The script checks for the existence of each input file before each phase. If `fiber_M0.nii.gz` is missing, Phase 4 is skipped with a warning (equivalent to `--no-fiber`). If `material_map.nii.gz` is missing, the script aborts immediately — no checks are possible without the material map.

### 7.3 Algorithm Summary

```
Input:  subject_id, profile, flags
Output: validation_report.json, fig1–fig5.png, launch_fsleyes.sh

Phase 1 — Header checks:
  1. Read NIfTI headers (no data) for all grid-resolution files
  2. Read grid_meta.json
  3. Read fiber_M0 header (if available)
  4. Run checks H1–H10
  5. Abort if any CRITICAL header check fails (all subsequent phases
     depend on consistent headers)

Phase 2 — Grid volumes:
  6. Load material_map.nii.gz → mat[N,N,N] (uint8, 128 MB)
  7. Load skull_sdf.nii.gz → sdf[N,N,N] (float32, 512 MB)
  8. Load brain_mask.nii.gz → brain[N,N,N] (uint8, 128 MB)
  9. Run checks D1–D5, M1–M4, V1–V6, C1
  10. Resample T1w to grid in-memory (slab-based, 512 MB output)
  11. Generate Figure 1 (material map triplanar)
  12. Generate Figure 3 (skull SDF + domain boundary)
  13. Free sdf, T1w resampled

Phase 3 — Dural membrane (unless --no-dural):
  14. Run checks C2–C4 using mat and brain (CC labels allocated/freed within)
  15. Re-resample T1w for overlay (~10–20s; freed at end of Phase 2)
  16. Generate Figure 2 (dural membrane detail)
  17. Free T1w, brain

Phase 4 — Fiber texture (unless --no-fiber):
  18. Load fiber_M0.nii.gz → fiber[145,174,145,6] (float32, 88 MB)
  19. Run checks F1–F6 using mat and fiber
  20. Generate Figure 4 (DEC map)
  21. Free fiber, mat

Phase 5 — Summary:
  22. Assemble all check results
  23. Generate Figure 5 (summary)
  24. Write validation_report.json
  25. Write launch_fsleyes.sh
  26. Print human-readable summary to console
```

### 7.4 Console Output

The script prints a human-readable table to stdout:

```
═══════════════════════════════════════════════════════════════
  Validation Report: 157336 / dev (512³, 1.0 mm)
═══════════════════════════════════════════════════════════════

  Header Consistency
  ─────────────────────────────────────────────────────────────
  H1  Bitwise-identical affines ............... PASS
  H2  Shape matches grid_meta ................ PASS  (512³)
  H3  material_map dtype uint8 ............... PASS
  ...

  Domain Closure
  ─────────────────────────────────────────────────────────────
  D1  Zero vacuum inside skull ............... PASS  (0 violations)
  D2  Zero tissue outside skull .............. PASS  (0 violations)
  ...

  Volume Census
  ─────────────────────────────────────────────────────────────
  u8=0   Vacuum             132,710,000 vox   132,710.0 mL
  u8=1   Cerebral WM            550,000 vox       550.0 mL
  u8=2   Cortical GM            520,000 vox       520.0 mL
  ...

  ═══════════════════════════════════════════════════════════════
  OVERALL: PASS  (35/35 checks passed, 0 warnings, 0 failures)
  ═══════════════════════════════════════════════════════════════
```

### 7.5 T1w Resampling Strategy

The T1w image is needed for Figures 1, 2, and 3. Rather than resampling to the full 512³ grid (which requires 512 MB and the slab-based utility), the script can extract individual 2D slices more efficiently by resampling only the needed slice planes. However, for simplicity and consistency with the existing `resample_to_grid` utility, the implementation resamples the full volume once and caches it for all three figures.

If `--no-images` is specified, the T1w resampling is skipped entirely. This is the primary memory saving when running in batch mode.

**Memory optimization:** If the T1w resampled volume must be freed between phases (to accommodate Phase 3's connected component analysis), it can be regenerated. The slab-based resampling takes ~10–20 seconds — acceptable for a validation script that runs once per subject.

## 8. Expected Output for Subject 157336 at Dev Profile

These are approximate expected values for the reference subject. They serve as regression targets — if values drift significantly from these ranges after code changes, the change should be investigated.

### 8.1 Key Metrics

| Metric | Expected Value | Notes |
|--------|:--------------:|-------|
| Brain parenchyma volume | ~1,280–1,350 mL | Healthy adult male, HCP |
| Intracranial volume (SDF < 0) | ~1,550–1,700 mL | Includes subarachnoid space |
| Ventricular CSF | ~20–30 mL | Young healthy adult |
| Subarachnoid CSF | ~200–350 mL | Includes meninges at 1 mm resolution |
| Dural membrane | ~10–25 mL | 1–2 voxel sheets at 1 mm |
| Fiber WM coverage | ≥92% | Forward-transform F1 check |
| Fiber trace mean (in WM) | ~0.3–0.5 | Unnormalized bedpostX fractions |
| Active domain components | 1 | Single sealed intracranial volume |
| Domain closure violations | 0 | D1 check |

### 8.2 Check Results

All CRITICAL checks should pass. Possible WARN outcomes:

- **V4 (dural volume):** At debug resolution (2 mm), dural membrane may be 0 voxels — this is expected and benign.
- **C2/C3 (membrane components):** At debug resolution, the falx may be discontinuous (fissure only 1–2 voxels wide). The dev profile should produce 1 connected component for each.
- **F1 (fiber coverage):** A small fraction (~5–8%) of WM voxels near the brain mask boundary may have zero M_0 due to the WM masking at bedpostX resolution (1.25 mm voxels that span WM/GM boundary). Coverage ≥90% is expected.

## 9. Design Rationale

**Why cross-cutting checks are necessary.** Per-step checks verify local invariants (e.g., Task 8 confirms zero vacuum inside the skull after CSF gap-fill). But Task 9 modifies the material map afterward — painting dural membrane over CSF voxels. The cross-cutting D1 check verifies the invariant still holds after all modifications. Similarly, the fiber forward-transform check (F1) validates the interaction between two independent outputs (material map + fiber texture) that no single step can verify.

**Why each visualization catches failures that statistics miss.** A brain parenchyma volume of 1,300 mL is normal — but it could comprise 1,300 mL of white matter with zero gray matter, or 1,300 mL correctly distributed. Only the triplanar overlay (Figure 1) reveals spatial accuracy. Similarly, the dural membrane volume might be correct but placed at the wrong anatomical location — only Figure 2 catches this. The SDF contour overlay (Figure 3) reveals whether the zero-contour dips into sulci, which the gradient magnitude statistic (a scalar) cannot localize.

**Why PNGs + JSON over HTML.** The outputs serve a paper-first workflow: PNGs go directly into the methods figure, and JSON feeds Table 1 (cross-subject aggregation of preprocessing metrics across N subjects). HTML would require a viewer and cannot be directly embedded in a paper. JSON is trivially parseable by the table-generation script.

**Why full forward-transform fiber check.** The runtime samples M_0 by transforming grid coordinates to physical space, then to diffusion voxel coordinates, then trilinear interpolation. The validation must test this exact path — not a proxy like nearest-neighbor lookup or checking at diffusion resolution. Any error in the affine composition (wrong matrix, transposed axes, off-by-one in the voxel-center convention) would only be caught by the forward-transform check. Reviewers of the paper will expect quantitative evidence that the fiber data reaches the solver correctly.

**Why single script.** The grid volumes total ~768 MB (mat + sdf + brain_mask) at 512³. Loading them in separate scripts would mean loading+freeing 768 MB per script, with significant I/O overhead for gzipped NIfTIs. A single script loads each volume once and shares it across all checks and figures. The phase-based memory management keeps peak usage at ~1.3 GB — well within the 5.7 GB system limit.

**Why phase-based memory management.** Loading everything upfront (mat 128 MB + sdf 512 MB + brain_mask 128 MB + fs_labels 256 MB + fiber 88 MB + T1w 512 MB) would total ~1,624 MB of data plus the int32 connected component array (512 MB) = ~2,136 MB. This is feasible on the 5.7 GB system but tight with Python/OS overhead. The phase-based approach peaks at ~1,280 MB (Phase 2) by never holding the fiber texture and grid volumes simultaneously, and by freeing the SDF after Figure 3 is generated.

**Why fs_labels_resampled is not loaded by default.** The FS labels (256 MB at int16) are only needed if the compartmentalization checks (C2/C3) require identifying CC landmarks for the dural region split. The material map alone (128 MB) suffices for most checks. The FS labels are loaded on demand during Phase 3 if needed for CC landmark identification, then freed.

**Why the T1w is resampled to 512³ in-memory.** The T1w underlay must be at the same resolution as the material map for accurate overlay. Resampling to the full grid using the existing `resample_to_grid` utility is the simplest approach and reuses proven code. The 512 MB output is transient — freed after figure generation. An alternative (per-slice resampling) would save memory but requires new code paths for extracting arbitrary 2D slices through the affine transform.

**Why --no-images for batch validation.** When validating N subjects, generating 5 high-resolution figures per subject is slow and disk-heavy. The `--no-images` flag skips all matplotlib rendering, reducing runtime from ~2 minutes to ~30 seconds per subject and producing only the JSON report. The figures can be generated selectively for subjects that fail or warrant inspection.
