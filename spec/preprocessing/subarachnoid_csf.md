# Preprocessing Step 4: Subarachnoid CSF Identification

This document specifies the identification and labeling of subarachnoid CSF voxels — the fluid-filled space between the brain surface and the inner skull. This step fills all remaining unlabeled (vacuum) voxels inside the skull with subarachnoid CSF (u8 = 8), ensuring the material map has no gaps.

## 1. Purpose

The subarachnoid CSF serves two roles in the simulation:

1. **Drainage pathway.** Fluid from a hemorrhage seeps through brain tissue (low permeability) and eventually reaches the subarachnoid space (high permeability), where it flows rapidly toward the skull boundary. The air halo at the skull surface (material ID 255, painted at runtime) acts as the Monro-Kellie Dirichlet boundary. Without subarachnoid CSF, there is no high-permeability path between the brain surface and the boundary — pressure would build unrealistically.

2. **Domain closure.** After this step, every voxel inside the skull (SDF < 0) has a nonzero material index. No vacuum remains inside the cranial cavity. This is the invariant that downstream steps (the Dural Membrane step, and eventually the runtime solver) rely on.

The subarachnoid CSF is material class u8 = 8 in the material map. It receives high K_iso in the permeability LUT (~10^-8 to 10^-7 m^2) — essentially free-flowing fluid. It is separated from ventricular CSF (u8 = 7) so the Monro-Kellie controller can track ventricular volume independently (ventricular expansion is a key clinical indicator).

## 2. The Two CSF Populations

The subarachnoid CSF is not a uniform shell. It consists of two distinct populations with different data sources:

### 2.1 Sulcal / Peri-cortical CSF

**What it is:** CSF filling the cortical sulci (folds), the interhemispheric fissure, cerebellar folia, and the thin space between the outer cortical surface and the edge of the brain mask.

**How we know it's there:** The brain mask (`brainmask_fs.nii.gz`) includes the full intracranial volume — brain tissue plus intracranial CSF spaces. The aparc+aseg parcellation only labels brain parenchyma. The difference is the sulcal/peri-cortical CSF.

**Measured for subject 157336 at source resolution (0.7 mm):**

| Property | Value |
|----------|-------|
| Voxel count | 569,972 |
| Volume | ~196 mL |
| Fraction of brain mask | 14.7% |
| Connected components | 1 large (98.6%) + 1,358 tiny fragments |
| Spatial distribution | 100% of outermost brain mask shell; extends up to ~20 mm deep into Sylvian/interhemispheric fissures |
| Neighbor labels | 60% border cortical parcels, 40% border other CSF voxels, <1% border white matter |

This population forms a contiguous envelope wrapping the cortical surface and penetrating every sulcus. It is the majority of the subarachnoid CSF volume.

### 2.2 Convexity Shell CSF

**What it is:** CSF in the thin layer between the outer brain surface (gyral crowns) and the inner skull (SDF = 0 isosurface). This corresponds anatomically to the subarachnoid space at the convexity plus the pia/arachnoid membranes (too thin to resolve individually).

**How we know it's there:** The skull SDF was built by morphologically closing and dilating the brain mask (see `skull_sdf.md`). The dilation extends R_dilate = 4 mm beyond the closed brain surface. Voxels in this shell are inside the skull (SDF < 0) but outside the brain mask.

**Estimated volume:** A naive estimate using the closed brain surface area (~700 cm^2) times R_dilate (4 mm) gives ~280 mL. The actual shell CSF is much less than this, because the skull SDF boundary was built from the *closed* brain mask, not the original brain mask. At cortical concavities (sulcal openings, fissures), the closing pushed the effective surface outward beyond the original brain mask boundary. The dilation then starts from this already-expanded surface. So the space between the original brain mask and the SDF = 0 surface is split between two populations: sulcal CSF (inside the brain mask, step 1) absorbs the interior portion, and shell CSF (outside the brain mask, step 2) absorbs only the outer portion. The true shell CSF (outside brain mask, inside SDF) is estimated at ~80-120 mL at source resolution.

### 2.3 Why FreeSurfer Label 24 is Insufficient

FreeSurfer's CSF label (FS label 24, mapped to u8 = 8 in the Label Remapping step) captures only 3,330 voxels (1.14 mL) in subject 157336. These are almost entirely deep midline CSF around the 3rd ventricle and thalami — not subarachnoid CSF. There is essentially zero coverage of the cortical convexity, sulcal spaces, or basal cisterns. Label 24 provides <1% of the actual subarachnoid CSF volume.

The label 24 voxels that were already mapped to u8 = 8 by the Label Remapping step are unaffected by this step (they have material_map != 0, so the gap-fill skips them).

## 3. Inputs

All inputs are at simulation grid resolution, produced by previous preprocessing steps:

| File | From | Content |
|------|------|---------|
| `material_map.nii.gz` | Label Remapping step | u8 material indices (0 = vacuum, 1-11 = tissue/fluid classes) |
| `skull_sdf.nii.gz` | Skull SDF step | Signed distance to inner skull surface (float32, mm) |
| `brain_mask.nii.gz` | Domain Geometry step | Binary brain mask (uint8, 0/1) |
| `grid_meta.json` | Domain Geometry step | Grid parameters (N, dx_mm) |

All files are in the same coordinate system (simulation grid, affine A_g_to_p) and have the same shape (N^3).

## 4. Algorithm

### 4.1 Overview

```
Load material_map, skull_sdf, brain_mask
  → identify sulcal CSF (inside brain mask, unassigned)
  → identify shell CSF (outside brain mask, inside skull, unassigned)
  → paint both as u8 = 8
  → validate: no vacuum inside skull
  → save updated material_map
```

### 4.2 Step 1: Sulcal CSF

```python
sulcal = (brain_mask == 1) & (material_map == 0)
n_sulcal = np.count_nonzero(sulcal)
material_map[sulcal] = 8
```

This captures voxels that FreeSurfer's brain mask considers intracranial but the aparc+aseg parcellation did not label. These are the sulcal spaces, interhemispheric fissure CSF, and peri-cortical fluid.

### 4.3 Step 2: Shell CSF

```python
shell = (sdf < 0) & (brain_mask == 0) & (material_map == 0)
n_shell = np.count_nonzero(shell)
material_map[shell] = 8
```

This captures voxels in the dilation shell between the brain mask boundary and the skull SDF = 0 surface. These represent the convexity subarachnoid space that lies beyond the brain mask.

**Why `material_map == 0` in both steps:** Some voxels outside the brain mask may have nonzero material IDs due to nearest-neighbor resampling at tissue boundaries — a cortical gray matter label (u8 = 2) can land at a simulation grid position where the independently-resampled brain mask is 0. The `material_map == 0` condition ensures these are not overwritten.

**Why `sdf < 0` exactly (not `sdf < -epsilon`):** The SDF transitions smoothly through zero at the skull surface. Voxels at SDF ~ 0 are boundary voxels where the cut-cell porosity phi_geo applies. Painting them as CSF is correct — they represent the innermost skull surface, which is occupied by CSF. Voxels at SDF > 0 become air halo at runtime.

### 4.4 Ordering with the Dural Membrane Step

The Dural Membrane step will overwrite some subarachnoid CSF voxels with u8 = 10 (dural membrane). The falx cerebri sits in the interhemispheric fissure, which this step fills with CSF. The tentorium cerebelli sits between the cerebrum and cerebellum, also in CSF-filled space. The Dural Membrane step carves thin sheets of dural membrane out of this CSF.

**This step must run before the Dural Membrane step.** The dural membrane is a refinement of the subarachnoid CSF region, not a separate domain.

## 5. Outputs

### 5.1 Updated Material Map

| Property | Value |
|----------|-------|
| File | `material_map.nii.gz` (overwritten in place) |
| Shape | N^3 |
| Dtype | `uint8` |
| Affine | A_g_to_p (unchanged) |

The material map is updated in place. After this step, voxels that were u8 = 0 (vacuum) inside the skull are now u8 = 8 (subarachnoid CSF). All other voxel values are unchanged.

### 5.2 No Additional Files

This step does not produce new output files. The only artifact is the updated material map. Volume counts and validation results are logged to stdout.

## 6. Implementation

### 6.1 Stack

Same as previous steps: Python 3, nibabel, numpy. The gap-fill itself uses only array indexing and boolean operations. The validation checks (Sections 7.4-7.5) additionally require scipy (`scipy.ndimage.label` for connected component analysis).

### 6.2 Algorithm Summary

```
Input:  subject_id, profile
Output: material_map.nii.gz (updated)

1. Load from data/processed/{subject_id}/{profile}/:
   material_map.nii.gz → mat[N,N,N] (uint8)
   skull_sdf.nii.gz    → sdf[N,N,N] (float32)
   brain_mask.nii.gz   → brain[N,N,N] (uint8)
   grid_meta.json      → dx_mm

2. Sulcal CSF:
   sulcal = (brain == 1) & (mat == 0)
   n_sulcal = count_nonzero(sulcal)
   mat[sulcal] = 8

3. Shell CSF:
   shell = (sdf < 0) & (brain == 0) & (mat == 0)
   n_shell = count_nonzero(shell)
   mat[shell] = 8

4. Validation (Section 7)

5. Save:
   material_map.nii.gz ← mat with original affine
```

### 6.3 Memory Analysis

**During gap-fill (steps 2-3):**

| Array | Size (512^3) | Size (256^3) |
|-------|:------------:|:------------:|
| material_map (uint8) | 128 MB | 16 MB |
| skull_sdf (float32) | 512 MB | 64 MB |
| brain_mask (uint8) | 128 MB | 16 MB |
| Boolean mask (sulcal or shell) | 128 MB | 16 MB |
| **Total** | **~896 MB** | **~112 MB** |

**During validation (Section 7):**

The domain closure check (Section 7.1) needs the SDF, material map, brain mask, and both boolean masks simultaneously — peak ~1,150 MB for 512^3. After the domain closure check passes, the SDF and brain mask can be freed. The topology checks (Sections 7.4-7.5) use `scipy.ndimage.label`, which allocates an int32 output array (512 MB for 512^3). Peak during topology checks (after freeing SDF): ~640 MB.

**Overall peak: ~1,150 MB** during the domain closure check. Comfortable on a 5.7 GB system.

### 6.4 Runtime

This step is trivially fast. The operations are element-wise boolean comparisons and assignments on N^3 arrays. Expected wall time: <5 seconds for 512^3, dominated by NIfTI I/O (the connected component checks in validation add a few seconds).

### 6.5 Idempotency

This step is safe to re-run. If the material map already contains u8 = 8 voxels from a previous run, the conditions `material_map == 0` exclude them, so no voxels are modified. The output is identical regardless of how many times the step is executed.

However, the per-population volume reports (Section 7.2) reflect only *newly painted* voxels. On a second run, n_sulcal and n_shell will both be zero (nothing new to paint), which is correct but potentially confusing. The total u8 = 8 count reported in Section 7.2 remains accurate on every run. To detect a re-run, the script can check whether u8 = 8 voxels already exist before the gap-fill:

```python
n_pre_existing = np.count_nonzero(material_map == 8)
if n_pre_existing > 5000:  # more than FS label 24 alone (~3,330 voxels)
    print(f"WARNING: {n_pre_existing:,} u8=8 voxels already present — "
          f"material map may have been modified by a previous run of this step. "
          f"Per-population counts below reflect only newly painted voxels.")
```

## 7. Validation

### 7.1 Domain Closure (Critical)

After this step, no vacuum should remain inside the skull:

```python
vacuum_inside = (sdf < 0) & (mat == 0)
n_violation = np.count_nonzero(vacuum_inside)
assert n_violation == 0, f"Domain closure violated: {n_violation} vacuum voxels inside skull"
```

This is the strongest invariant of the entire preprocessing pipeline. If it fails, there is a gap between the material map and the skull SDF that will cause the runtime solver to encounter undefined material at active voxels.

### 7.2 Population Volumes

Report the volume of each CSF population and the total:

```python
vol_voxel = dx_mm ** 3  # mm^3 per voxel
vol_sulcal = n_sulcal * vol_voxel / 1000  # mL
vol_shell  = n_shell  * vol_voxel / 1000  # mL
vol_total  = vol_sulcal + vol_shell

print(f"Sulcal CSF:  {n_sulcal:,} voxels  ({vol_sulcal:.1f} mL)")
print(f"Shell CSF:   {n_shell:,} voxels  ({vol_shell:.1f} mL)")
print(f"Total new:   {n_sulcal + n_shell:,} voxels  ({vol_total:.1f} mL)")
```

Also report the total u8 = 8 volume (including voxels already assigned by the Label Remapping step from FS label 24):

```python
n_total_sas = np.count_nonzero(mat == 8)
vol_total_sas = n_total_sas * vol_voxel / 1000
print(f"Total subarachnoid CSF (u8=8): {n_total_sas:,} voxels ({vol_total_sas:.1f} mL)")
```

**Expected volumes** (approximate, subject- and profile-dependent):

| Population | Expected range | Notes |
|------------|:--------------:|-------|
| Sulcal CSF | 100-200 mL | Depends on cortical folding, resampling resolution |
| Shell CSF | 60-150 mL | Scales with R_dilate and brain surface area |
| Total subarachnoid | 200-350 mL | Includes meninges and slight dilation overestimate |
| FS label 24 (pre-existing) | ~1 mL | Negligible contribution |

The total exceeds the physiological subarachnoid CSF volume (~100-150 mL) because: (a) the uniform R_dilate = 4 mm overestimates the convexity space in some regions, (b) the meninges (pia, arachnoid, dura — ~1-2 mm combined) are included as CSF since they cannot be resolved at simulation resolution, and (c) the morphological closing in the Skull SDF step may slightly expand the skull boundary into spaces that are anatomically extradural. This overestimate is acceptable — the Monro-Kellie controller adapts to whatever CSF volume exists, and permeability values can compensate for thickness errors.

### 7.3 Material Map Census

Report a complete voxel count by material class after the update:

```python
for uid in range(12):
    n = np.count_nonzero(mat == uid)
    vol = n * vol_voxel / 1000
    print(f"  u8={uid:3d}: {n:>10,} voxels  ({vol:>8.1f} mL)")
n_255 = np.count_nonzero(mat == 255)
print(f"  u8=255: {n_255:>10,} voxels  (air halo — should be 0 at this stage)")
```

Key checks:
- u8 = 0 (vacuum) should be the majority of voxels (the empty space outside the skull) and must not overlap with SDF < 0.
- u8 = 255 (air halo) should be 0 — it is only assigned at runtime.
- u8 = 7 (ventricular CSF) should be ~20-30 mL.
- u8 = 8 (subarachnoid CSF) should match the total from Section 7.2.
- u8 = 1-6, 9, 11 (tissue classes) should be unchanged from the Label Remapping step.

### 7.4 Sulcal CSF Topology

The sulcal CSF should form a connected envelope around the brain. Report the number of connected components:

```python
from scipy.ndimage import label as cc_label
sulcal_components, n_components = cc_label(sulcal)
print(f"Sulcal CSF connected components: {n_components}")
```

Expected: one large component containing >95% of the sulcal CSF voxels (the contiguous peri-cortical envelope), plus small fragments. Many small isolated components would indicate labeling artifacts or resampling problems.

### 7.5 Shell CSF Continuity

The shell CSF should form a continuous layer between the brain surface and the skull. Check that it does not have large gaps:

```python
shell_components, n_shell_comp = cc_label(shell)
if n_shell_comp > 0:
    sizes = np.bincount(shell_components.ravel())[1:]  # exclude background
    print(f"Shell CSF components: {n_shell_comp}, largest: {sizes.max():,}, "
          f"smallest: {sizes.min():,}")
```

Expected: one dominant component. If the shell is fragmented, the dilation radius R_dilate may be too small, or the brain mask has unexpected protrusions that push through the skull boundary.

## 8. Design Rationale

**Why two steps instead of one?** A single gap-fill (`(sdf < 0) & (material_map == 0)`) produces an identical material map. The two-step separation exists purely for observability. By splitting sulcal CSF (brain mask interior) from shell CSF (brain mask exterior), we can:
- Report volumes separately, making it easy to diagnose problems (e.g., if sulcal volume is zero, the Label Remapping step likely over-labeled; if shell volume is zero, the SDF is too tight)
- Trace each population back to its upstream step (sulcal → Label Remapping's label gaps, shell → Skull SDF's dilation radius)
- Confirm that the brain mask and SDF are geometrically consistent (the shell should be a continuous layer between the two boundaries)

**Why operate at simulation resolution, not source resolution?** The material map, SDF, and brain mask are all already resampled to the simulation grid by the preceding steps. Operating at simulation resolution avoids reaching back to source volumes, keeps the pipeline linear, and ensures the gap-fill is exact at the resolution the solver will use. The alternative (source-resolution gap-fill before resampling) would provide marginally better boundary precision but requires saving the Skull SDF step's intermediate binary mask and complicates the pipeline for negligible benefit.

**Why overwrite material_map.nii.gz in place?** The subarachnoid CSF identification is a refinement of the material map, not a separate data product. After this step, the material map is the canonical source of truth for all tissue/fluid assignments. Keeping the old and new versions as separate files invites confusion about which one downstream steps should read. The Label Remapping material map can be reconstructed from the resampled FS labels at any time.

**Why is the total subarachnoid volume larger than physiological values?** Physiological subarachnoid CSF is ~100-150 mL. Our estimate (200-350 mL) is higher because: (1) we cannot resolve the meninges (pia + arachnoid + dura ≈ 1-2 mm) at simulation resolution, so they are included as CSF; (2) the uniform R_dilate overestimates some regions while underestimating others (basal cisterns); (3) the morphological closing fills sulci slightly beyond their anatomical depth. This is acceptable for simulation purposes — the subarachnoid space functions as a high-permeability drainage pathway regardless of its exact volume, and the Monro-Kellie controller adjusts to the actual domain geometry.

**Why the SDF threshold is exactly zero.** The SDF = 0 isosurface is the skull inner surface. All CSF is inside the skull (SDF < 0). Using a negative threshold (SDF < -epsilon) would create a thin vacuum shell between CSF and air halo that the solver cannot handle — the air halo is painted at SDF > 0, so voxels at 0 < SDF < epsilon would be neither CSF nor halo. Using SDF < 0 ensures complete coverage.

**Why the 570k unlabeled voxels are CSF, not tissue.** The aparc+aseg parcellation labels every voxel it considers brain parenchyma. The 569,972 brain-mask-interior voxels with FS label 0 were deliberately left unlabeled — they are the CSF-filled spaces that the brain mask includes but the parcellation excludes. Evidence: they form one contiguous component wrapping the cortical surface (98.6% in a single connected component), 100% of the outermost brain mask shell is unlabeled, 60% border cortical parcellation labels, and their spatial distribution matches the expected pattern of sulcal and peri-cortical CSF. The 1,358 tiny isolated fragments (<0.24% of total) are either genuinely small CSF pockets between structures or minor labeling artifacts — too few to affect the simulation.

**Why FS label 24 doesn't solve this problem.** In subject 157336, label 24 provides only 3,330 voxels (1.14 mL) concentrated around the 3rd ventricle. It captures <1% of subarachnoid CSF. This is not a data quality issue — FreeSurfer's aparc+aseg is not designed to label the subarachnoid space. The gap-fill approach is the correct solution.
