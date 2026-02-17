# Preprocessing Step 5: Dural Membrane Reconstruction

This document specifies the reconstruction of the falx cerebri and tentorium cerebelli — near-impermeable fibrous sheets that divide the cranial cavity into semi-isolated pressure compartments. This step identifies voxels occupied by these membranes within the subarachnoid CSF and paints them as u8 = 10 (dural membrane).

## 1. Purpose

The dural membranes serve one role in the simulation: **pressure compartmentalization.**

Without internal membranes, the two cerebral hemispheres and the infratentorial space (cerebellum + brainstem) form a single connected fluid domain. Pressure from a hemorrhage would equalize across the entire brain almost instantly. In reality, these membranes create semi-isolated compartments where pressure builds locally, driving the lateralized pressure gradients responsible for midline shift and herniation — the primary clinical consequences of intracerebral hemorrhage.

The permeability contrast tells the story:

| Structure | Permeability (m²) |
|-----------|:-----------------:|
| Subarachnoid CSF | ~10⁻⁸ to 10⁻⁷ |
| Dural membrane | ~10⁻¹⁷ to 10⁻¹⁶ |

That is a **10⁹ to 10¹⁰ contrast ratio** across a 1–2 voxel boundary. The solver's entire precision architecture — the Hybrid Virtual-Double firewall at L2, the Double-Single diagonal storage, the Tri-Path Precision system — exists specifically to resolve pressure fields across these membranes.

The dural membrane is material class u8 = 10. It receives very low K_iso in the permeability LUT (~10⁻¹⁷ to 10⁻¹⁶ m²), with K_fiber = 0 (isotropic — no fiber structure). The three compartments communicate only through anatomical openings: below the falx (inferior to the corpus callosum) and through the tentorial notch (around the brainstem).

## 2. Anatomy

### 2.1 Falx Cerebri

A vertical, sickle-shaped sheet of dura mater in the longitudinal (interhemispheric) fissure between the left and right cerebral hemispheres. It runs from the crista galli anteriorly to the internal occipital protuberance posteriorly. Approximately 1–2 mm thick.

The falx does not reach all the way down. Its inferior free edge follows the superior surface of the corpus callosum. Below the CC, the two hemispheres communicate freely through the subarachnoid CSF. Posteriorly, the falx descends to meet the tentorium at the straight sinus.

**Measured for subject 157336 at source resolution (0.7 mm):**

| Property | Value |
|----------|-------|
| Interhemispheric fissure width | 2.8–5.6 mm (4–8 voxels) |
| CC AP extent | ~70 mm (Y=[88, 158] mm in ACPC) |
| CC vertical extent | ~31 mm (Z=[−3, 27] mm), arch-shaped |
| Fissure vertical extent above CC | 70–104 mm |
| Midsagittal plane | x ≈ 0.1 mm (essentially at ACPC midline) |

### 2.2 Tentorium Cerebelli

A tent-shaped, roughly horizontal sheet separating the cerebrum above from the cerebellum below. It attaches laterally to the petrous ridges and posteriorly to the transverse/straight sinuses. Its surface is not flat — it rises centrally toward the brainstem and slopes downward laterally.

The tentorium has a central opening called the **tentorial notch** (incisura) through which the brainstem passes. This is the only route for pressure to communicate between the supratentorial (cerebral) and infratentorial (cerebellar) compartments.

**Measured for subject 157336 at source resolution:**

| Property | Value |
|----------|-------|
| Tentorial plane z-level | 55.3 ± 7.0 mm in ACPC (tent-shaped, higher centrally) |
| Tentorial notch diameter | ~26 mm |
| Lateral span | ~110 mm |
| AP span | ~67 mm |
| Cerebrum–cerebellum gap | median 2.8 mm, mean 4.8 mm |
| Transition zone | z ≈ 52–55 mm (brainstem surround shifts from cerebellar to cerebral) |

### 2.3 Why FreeSurfer Cannot Help

These are thin fibrous membranes with minimal contrast on T1w MRI. FreeSurfer does not segment them. No standard neuroimaging pipeline does. The material map spec confirms:

> **u8 = 10: Dural Membrane** — No FreeSurfer labels. This class is assigned entirely during dural membrane reconstruction.

The membranes must be **reconstructed** from indirect geometric evidence: the spatial relationship between left and right hemisphere tissue (for the falx) and between cerebral and cerebellar tissue (for the tentorium).

## 3. Inputs

All inputs are at simulation grid resolution, produced by previous preprocessing steps:

| File | From | Content |
|------|------|---------|
| `material_map.nii.gz` | Subarachnoid CSF step | u8 material indices. After the Subarachnoid CSF step, every voxel inside the skull has a nonzero class. Subarachnoid CSF = u8 8. |
| `fs_labels_resampled.nii.gz` | Domain Geometry step | Original FreeSurfer aparc+aseg labels at simulation grid resolution (int16). Needed to distinguish left from right hemisphere and identify CC labels. |
| `grid_meta.json` | Domain Geometry step | Grid parameters (N, dx_mm) |

**Why fs_labels_resampled?** The material map collapses left/right into bilateral classes (e.g., u8 = 1 includes both Left- and Right-Cerebral-White-Matter). The falx reconstruction requires distinguishing hemispheres, which is only possible from the original FreeSurfer labels.

## 4. Algorithm

### 4.1 Overview

```
Load material_map, fs_labels_resampled, grid_meta
  → classify FS labels into left-cerebral, right-cerebral, cerebral-all, cerebellar
  → falx: EDT watershed between left and right cerebral tissue
       constrained to CSF, above corpus callosum
  → tentorium: EDT watershed between cerebral and cerebellar tissue
       constrained to CSF, excluding brainstem notch
  → merge falx + tentorium → paint u8 = 10
  → validate
  → save updated material_map
```

The core idea: each membrane sits at the **equidistant surface** between the tissue masses on either side of it. The falx is equidistant from left and right cerebral tissue. The tentorium is equidistant from cerebral and cerebellar tissue. Euclidean distance transforms find these surfaces; anatomical constraints refine them.

### 4.2 Label Classification

#### 4.2.1 Left Cerebral Tissue (for falx EDT)

| FS Label | Name |
|---------:|------|
| 2 | Left-Cerebral-White-Matter |
| 3 | Left-Cerebral-Cortex |
| 10 | Left-Thalamus |
| 11 | Left-Caudate |
| 12 | Left-Putamen |
| 13 | Left-Pallidum |
| 17 | Left-Hippocampus |
| 18 | Left-Amygdala |
| 19 | Left-Insula |
| 20 | Left-Operculum |
| 26 | Left-Accumbens-area |
| 27 | Left-Substancia-Nigra |
| 28 | Left-VentralDC |
| 78 | Left-WM-hypointensities |
| 81 | Left-non-WM-hypointensities |
| 1001–1035 | ctx-lh-* (left cortical parcels) |

#### 4.2.2 Right Cerebral Tissue (for falx EDT)

| FS Label | Name |
|---------:|------|
| 41 | Right-Cerebral-White-Matter |
| 42 | Right-Cerebral-Cortex |
| 49 | Right-Thalamus |
| 50 | Right-Caudate |
| 51 | Right-Putamen |
| 52 | Right-Pallidum |
| 53 | Right-Hippocampus |
| 54 | Right-Amygdala |
| 55 | Right-Insula |
| 56 | Right-Operculum |
| 58 | Right-Accumbens-area |
| 59 | Right-Substancia-Nigra |
| 60 | Right-VentralDC |
| 79 | Right-WM-hypointensities |
| 82 | Right-non-WM-hypointensities |
| 2001–2035 | ctx-rh-* (right cortical parcels) |

**Note on labels 19/20/55/56 (Insula/Operculum):** These labels are included for robustness but are not part of the standard FreeSurfer aparc+aseg lookup table. In HCP data, the insula is typically represented through cortical parcellation labels (e.g., ctx-lh-insula = 1035, captured by the 1001–1035 range). If labels 19/20/55/56 are absent from the data, their inclusion is harmless. Verify against the actual unique label set during implementation.

**Excluded from both sides:** Midline structures that bridge the hemispheres — corpus callosum (192, 250, 251–255), optic chiasm (85), bilateral hypointensities (77, 80). Also excluded: brainstem (16, 75, 76), cerebellum (7, 8, 46, 47), all ventricles (4, 5, 14, 15, 43, 44, 72), choroid plexus (31, 63), vessels (30, 62), CSF (24), vacuum (0).

**Why exclude midline structures?** The corpus callosum bridges the hemispheres. Including it in either side would pull the watershed off-center. Excluding it means the EDT distance passes through the CC, which is correct — the falx stops at the CC, it does not penetrate through it.

#### 4.2.3 Cerebral Tissue (for tentorium EDT)

For the tentorium, left/right distinction is irrelevant. All cerebral solid tissue forms the supratentorial mass:

```python
cerebral_mask = np.isin(mat, [1, 2, 3, 9])
```

This uses the material map (not FS labels) since hemisphere distinction is not needed:
- u8 = 1: Cerebral White Matter (includes CC, fornix, optic chiasm)
- u8 = 2: Cortical Gray Matter
- u8 = 3: Deep Gray Matter
- u8 = 9: Choroid Plexus (solid tissue in supratentorial ventricles)

**Excluded:** Ventricular CSF (u8 = 7), subarachnoid CSF (u8 = 8), vessels (u8 = 11), brainstem (u8 = 6), cerebellum (u8 = 4, 5), vacuum (u8 = 0). Only solid tissue contributes to the EDT — the membrane is the equidistant surface between tissue masses, not between fluid spaces.

#### 4.2.4 Cerebellar Tissue (for tentorium EDT)

```python
cerebellar_mask = np.isin(mat, [4, 5])
```

- u8 = 4: Cerebellar White Matter
- u8 = 5: Cerebellar Cortex

The brainstem (u8 = 6) is excluded from both cerebral and cerebellar masks. Since brainstem voxels are not in either seed mask, the EDT distance field passes through the brainstem without seeding from it. This is correct: the brainstem occupies the tentorial notch, and the EDT's passage through it ensures the watershed does not form there.

### 4.3 Falx Cerebri Reconstruction

#### 4.3.1 Hemisphere Distance Fields

Compute the Euclidean distance from every voxel to the nearest left-cerebral tissue and nearest right-cerebral tissue:

```python
fs = load(fs_labels_resampled)  # int16, N³

left_set = {2, 3, 10, 11, 12, 13, 17, 18, 19, 20, 26, 27, 28, 78, 81}
left_mask = np.isin(fs, list(left_set)) | ((fs >= 1001) & (fs <= 1035))

right_set = {41, 42, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 79, 82}
right_mask = np.isin(fs, list(right_set)) | ((fs >= 2001) & (fs <= 2035))

dist_left = ndimage.distance_transform_edt(~left_mask, sampling=dx_mm).astype(np.float32)
dist_right = ndimage.distance_transform_edt(~right_mask, sampling=dx_mm).astype(np.float32)
```

The `.astype(np.float32)` cast reduces memory from 1024 MB to 512 MB per array (at 512³). Float32 precision (~7 significant digits) is more than sufficient for distances in the range 0–200 mm.

#### 4.3.2 Watershed

The falx is where left and right distances are approximately equal:

```python
falx_watershed = np.abs(dist_left - dist_right) <= dx_mm
```

**Why threshold = dx_mm?** In a symmetric fissure, the distance difference changes at ~2 × dx_mm per voxel across the fissure (the EDT gradient is ~1.0 on each side, in opposite directions). A threshold of dx_mm selects voxels where |diff| ≤ dx_mm, which gives:
- Odd-width fissure (3, 5 voxels): 1 voxel thick (only the center voxel)
- Even-width fissure (4, 6 voxels): 2 voxels thick (both center voxels)

This produces a 1–2 voxel thick membrane, matching the 1–2 mm anatomical thickness of the falx at simulation resolution (dx = 1 mm for dev).

#### 4.3.3 CSF Constraint

The falx exists only in CSF-filled space (the interhemispheric fissure):

```python
falx_candidate = falx_watershed & (mat == 8)
```

This excludes brain tissue, ventricles, and vacuum from the falx. It also naturally limits the falx to the interhemispheric fissure — the watershed between left and right cerebral tissue only passes through CSF in the fissure; elsewhere it passes through solid tissue (the CC, deep gray, ventricles) which is excluded.

#### 4.3.4 Corpus Callosum Inferior Boundary

The falx's inferior free edge follows the superior surface of the corpus callosum. Below the CC, the hemispheres communicate freely — there is no falx.

Compute the CC boundary per coronal slice:

```python
cc_labels = {192, 251, 252, 253, 254, 255}
cc_mask = np.isin(fs, list(cc_labels))

# For each y-slice, find the maximum z-index of CC voxels
cc_superior_z = np.full(N, -1, dtype=np.int32)
for y in range(N):
    cc_slice = cc_mask[:, y, :]
    if cc_slice.any():
        cc_superior_z[y] = np.where(cc_slice.any(axis=0))[0].max()

# Exclude falx candidates at or below the CC
for y in range(N):
    if cc_superior_z[y] >= 0:
        falx_candidate[:, y, :cc_superior_z[y] + 1] = False
```

**Why per-slice rather than a single z-threshold?** The CC is arch-shaped: the splenium (posterior) and genu (anterior) dip lower than the central body. A single z-threshold would either cut the falx too high centrally or allow it too low at the ends. The per-slice approach follows the CC's actual shape.

**Anterior/posterior beyond CC extent:** At y-slices beyond the CC's AP range (anterior to the genu, posterior to the splenium), cc_superior_z[y] = −1 and no constraint is applied. The falx extends deeper at these locations, which is anatomically correct — the falx reaches the crista galli anteriorly and meets the tentorium posteriorly, both below the CC level.

#### 4.3.5 Combined Falx Mask

After the watershed, CSF constraint, and CC boundary:

```python
falx_mask = falx_candidate  # boolean, N³
n_falx = np.count_nonzero(falx_mask)
```

Free dist_left, dist_right, left_mask, right_mask, cc_mask (no longer needed).

### 4.4 Tentorium Cerebelli Reconstruction

#### 4.4.1 Compartment Distance Fields

Compute the distance from every voxel to the nearest cerebral tissue and nearest cerebellar tissue. These masks use the material map (not FS labels), since hemisphere distinction is not needed:

```python
cerebral_mask = np.isin(mat, [1, 2, 3, 9])
cerebellar_mask = np.isin(mat, [4, 5])

dist_cerebral = ndimage.distance_transform_edt(
    ~cerebral_mask, sampling=dx_mm).astype(np.float32)
dist_cerebellar = ndimage.distance_transform_edt(
    ~cerebellar_mask, sampling=dx_mm).astype(np.float32)
```

#### 4.4.2 Watershed

```python
tent_watershed = np.abs(dist_cerebral - dist_cerebellar) <= dx_mm
```

Same threshold logic as the falx: 1–2 voxel thickness at simulation resolution.

#### 4.4.3 CSF Constraint

```python
tent_candidate = tent_watershed & (mat == 8)
```

#### 4.4.4 Brainstem Notch Exclusion

The tentorial notch is the central opening through which the brainstem passes. The CSF constraint already excludes the brainstem itself (u8 = 6, not u8 = 8), but the thin CSF ring around the brainstem at the tentorial level (the perimesencephalic cisterns) would be incorrectly painted as tentorium without explicit exclusion.

Dilate the brainstem mask by R_notch to create the exclusion zone:

```python
brainstem = (mat == 6)

r_notch_vox = round(R_notch / dx_mm)  # 5 voxels at 1mm
r_step = min(3, r_notch_vox)
small_ball = build_ball(r_step)  # spherical structuring element
n_iter = math.ceil(r_notch_vox / r_step)

notch_exclusion = ndimage.binary_dilation(brainstem, small_ball, iterations=n_iter)
tent_candidate &= ~notch_exclusion
```

**Why R_notch = 5 mm?** The anatomical tentorial notch is ~26 mm in diameter. The brainstem nearly fills it (~25–32 mm wide at the transition level). The free edge of the tentorium is approximately 3–5 mm from the brainstem surface (the width of the perimesencephalic cisterns). A 5 mm exclusion margin beyond the brainstem surface preserves this CSF ring as open pathway.

#### 4.4.5 Combined Tentorium Mask

```python
tent_mask = tent_candidate
n_tent = np.count_nonzero(tent_mask)
```

Free dist_cerebral, dist_cerebellar, cerebral_mask, cerebellar_mask, notch_exclusion.

### 4.5 Merge and Paint

```python
dural = falx_mask | tent_mask
n_overlap = np.count_nonzero(falx_mask & tent_mask)
n_total = np.count_nonzero(dural)

material_map[dural] = 10
```

At the posterior junction where the falx descends to meet the tentorium (the straight sinus region), both masks may claim the same voxels. Since both assign u8 = 10, there is no conflict. The overlap count is reported for validation — it should be small (a few dozen voxels). However, the two independent watersheds may produce a thick or irregular junction. If the combined membrane exceeds ~3 voxels thick at the junction, it could artificially block CSF flow between the posterior fossa and the interhemispheric fissure. Section 8.2's thickness estimate should be inspected at the junction region specifically — see the junction check in Section 8.6.

### 4.6 Ordering with Previous Steps

**This step must run after the Subarachnoid CSF step.** The falx and tentorium occupy space that the Subarachnoid CSF step filled with subarachnoid CSF (u8 = 8). This step carves thin sheets of dural membrane out of that CSF, overwriting u8 = 8 with u8 = 10. Without the Subarachnoid CSF step running first, the interhemispheric fissure and cerebrum–cerebellum gap would still be vacuum (u8 = 0), and the CSF constraint (mat == 8) would find no candidates.

**This step runs before the Fiber Orientation step.** The Fiber Orientation step depends only on the raw FS labels (for WM masking), not on the material map or any other preprocessing output, so there is no ordering constraint between this step and the Fiber Orientation step. But logically, all material map modifications should be complete before final validation.

## 5. Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| T_watershed | dx_mm | EDT difference threshold. Gives 1–2 voxel membrane thickness at any resolution. |
| R_notch | 5 mm | Exclusion margin beyond brainstem surface for tentorial notch. Preserves ~5 mm CSF ring (perimesencephalic cisterns). |

Both parameters should be exposed as command-line arguments:

```
--watershed-threshold 1.0   # multiplier of dx_mm (default: 1.0)
--notch-radius 5.0          # mm (default: 5.0)
```

### 5.1 Sensitivity

**T_watershed:** At 0.5 × dx_mm, the membrane is strictly 1 voxel thick everywhere. Even-width fissures (4, 6 voxels) get 0 voxels at this threshold because the two center voxels have |dist_left − dist_right| = dx_mm, which exceeds 0.5 × dx_mm — no voxel center lands exactly on the equidistant plane. At 1.0 × dx_mm, 1–2 voxels. At 2.0 × dx_mm, 2–3 voxels — too thick, would block flow excessively. The default of 1.0 × dx_mm is the sweet spot.

**R_notch:** At 3 mm, the exclusion zone barely extends beyond the brainstem surface — some CSF voxels in the perimesencephalic cisterns may be painted as tentorium, partially closing the notch. At 5 mm, the cisterns are preserved. At 10 mm, too much tentorium is removed laterally. The 5 mm default is conservative. **Known limitation:** The spherical dilation creates a uniform exclusion zone, but the perimesencephalic cisterns are not uniform — they are wider laterally (ambient cisterns, ~5–10 mm) and narrower anteriorly (~3 mm). The spherical exclusion may over-exclude laterally or under-exclude anteriorly. This is acceptable for a first pass; a direction-dependent exclusion could be added if simulation results show flow artifacts at the notch.

### 5.2 Resolution Requirements

| Profile | dx_mm | Fissure width (voxels) | Falx thickness | Tentorium thickness |
|---------|:-----:|:----------------------:|:--------------:|:-------------------:|
| debug (256³) | 2.0 | 1–3 | 0–1 voxels | 0–1 voxels |
| dev (512³) | 1.0 | 3–6 | 1–2 voxels | 1–2 voxels |
| prod (512³) | 0.5 | 6–12 | 1–2 voxels | 1–2 voxels |

At debug resolution, the falx may be absent or discontinuous (the fissure may be only 1 voxel wide, leaving no room for a membrane between the two hemisphere voxels). This is acceptable for a debug profile — the membranes are not needed for basic pipeline testing. The dev and prod profiles produce well-defined membranes.

## 6. Outputs

### 6.1 Updated Material Map

| Property | Value |
|----------|-------|
| File | `material_map.nii.gz` (overwritten in place) |
| Shape | N³ |
| Dtype | `uint8` |
| Affine | A_g_to_p (unchanged) |

Voxels identified as falx or tentorium are overwritten from u8 = 8 (subarachnoid CSF) to u8 = 10 (dural membrane). All other voxel values are unchanged.

### 6.2 No Additional Files

Volume counts, per-structure breakdown, and validation results are logged to stdout.

## 7. Implementation

### 7.1 Stack

Python 3, nibabel, numpy, scipy (`scipy.ndimage` for EDT and morphological dilation). Same dependencies as the Skull SDF and Subarachnoid CSF steps.

### 7.2 Algorithm Summary

```
Input:  subject_id, profile
Output: material_map.nii.gz (updated)

Phase 0 — Label classification:
  1. Load fs_labels_resampled.nii.gz → fs[N,N,N] (int16)
     Load material_map.nii.gz → mat[N,N,N] (uint8)
     Load grid_meta.json → dx_mm

  2. Build left_mask, right_mask from FS labels (Section 4.2.1–4.2.2)
  3. Compute cc_superior_z[N] per coronal slice from CC labels 192, 251–255
  4. Free fs (no longer needed)

Phase 1 — Falx:
  5. dist_left  = edt(~left_mask).astype(float32)
  6. dist_right = edt(~right_mask).astype(float32)
  7. falx_watershed = |dist_left − dist_right| ≤ dx_mm
  8. falx_candidate = falx_watershed & (mat == 8)
  9. Apply CC inferior boundary constraint per y-slice
  10. falx_mask = falx_candidate
  11. Free dist_left, dist_right, left_mask, right_mask

Phase 2 — Tentorium:
  12. cerebral_mask  = mat ∈ {1,2,3,9}
  13. cerebellar_mask = mat ∈ {4,5}
  14. dist_cerebral   = edt(~cerebral_mask).astype(float32)
  15. dist_cerebellar  = edt(~cerebellar_mask).astype(float32)
  16. tent_watershed = |dist_cerebral − dist_cerebellar| ≤ dx_mm
  17. tent_candidate = tent_watershed & (mat == 8)
  18. Dilate brainstem (mat == 6) by R_notch → notch_exclusion
  19. tent_candidate &= ~notch_exclusion
  20. tent_mask = tent_candidate
  21. Free dist_cerebral, dist_cerebellar, cerebral_mask, cerebellar_mask

Phase 3 — Merge:
  22. dural = falx_mask | tent_mask
  23. mat[dural] = 10
  24. Validation (Section 8)
  25. Save material_map.nii.gz
```

### 7.3 Memory Analysis

All arrays are at simulation grid resolution (N³). Peak memory is tracked phase by phase for the dev profile (N = 512, dx = 1 mm).

**Phase 0 — Label classification:**

| Array | Size |
|-------|:----:|
| fs_labels (int16) | 256 MB |
| material_map (uint8) | 128 MB |
| left_mask (bool) | 128 MB |
| right_mask (bool) | 128 MB |
| cc_superior_z (int32) | negligible |
| **Total before free** | **640 MB** |
| After `del fs` | **384 MB** |

**Phase 1 — Falx EDT (sequential, peak during second EDT):**

| Array | Size |
|-------|:----:|
| left_mask + right_mask + mat | 384 MB |
| dist_left (float32, after cast) | 512 MB |
| dist_right (float64, during computation) | 1024 MB |
| **Peak** | **~1920 MB** |

The scipy EDT internally allocates float64 (1024 MB for 512³). After `.astype(np.float32)`, the float64 temporary is freed, leaving 512 MB. The peak occurs during the second EDT computation when the first float32 result and the second float64 temporary coexist.

After computing the watershed and freeing EDTs + hemisphere masks: ~256 MB (mat + falx_mask).

**Phase 2 — Tentorium EDT (sequential, peak during second EDT):**

| Array | Size |
|-------|:----:|
| mat + falx_mask | 256 MB |
| cerebral_mask + cerebellar_mask | 256 MB |
| dist_cerebral (float32) | 512 MB |
| dist_cerebellar (float64, during computation) | 1024 MB |
| **Peak** | **~2048 MB** |

After watershed and cleanup: ~384 MB (mat + falx_mask + tent_mask).

**Phase 2 — Notch dilation:**

Morphological dilation of brainstem (bool, 128 MB) with a small ball structuring element. Negligible additional memory compared to the EDT peak.

**Overall peak: ~2048 MB ≈ 2.0 GB** during the second tentorium EDT. With OS and Python overhead (~500 MB), total system usage ~2.5 GB of 5.7 GB available. Comfortable.

**At debug resolution (256³):** All arrays are 8× smaller. Peak ~256 MB. Trivial.

### 7.4 Runtime

The EDT is the dominant cost. `scipy.ndimage.distance_transform_edt` on a 512³ boolean array takes approximately 10–30 seconds per call (depending on the number of True voxels and cache behavior). Four EDT calls total (two for falx, two for tentorium), plus one morphological dilation (~2 seconds). Expected total wall time: **1–3 minutes** for the dev profile.

### 7.5 Idempotency

This step is safe to re-run. If the material map already contains u8 = 10 voxels from a previous run, those voxels have mat != 8, so the CSF constraint (mat == 8) excludes them from new candidates. The output is identical regardless of how many times the step executes.

However, the per-structure counts (n_falx, n_tent) report only newly painted voxels. On a re-run, both will be zero. To detect this:

```python
n_pre_existing = np.count_nonzero(mat == 10)
if n_pre_existing > 0:
    print(f"WARNING: {n_pre_existing:,} u8=10 voxels already present — "
          f"material map may have been modified by a previous run. "
          f"Per-structure counts below reflect only newly painted voxels.")
```

Note: u8 = 10 voxels also affect the tentorium EDT via the cerebral/cerebellar masks from the material map (which use u8 classes 1–5, 9 — not u8 = 10). On re-run, previously painted dural membrane voxels are neither cerebral nor cerebellar, so the EDT treats them as empty space. The watershed position is negligibly affected because these are thin (1–2 voxel) gaps in the tissue masses.

**Re-running with different parameters:** If the step is re-run with changed parameters (e.g., a different watershed threshold or notch radius), the old u8 = 10 voxels will not be reclaimed as CSF — they remain u8 = 10 and are excluded by the CSF constraint (mat == 8). The new watershed would form adjacent to the old membrane rather than replacing it. To re-run with different parameters, the script should first reset all u8 = 10 voxels back to u8 = 8:

```python
if n_pre_existing > 0:
    mat[mat == 10] = 8
    print(f"Reset {n_pre_existing:,} pre-existing u8=10 voxels to u8=8 for clean re-run.")
```

This ensures the reconstruction starts from a clean CSF field regardless of previous runs.

## 8. Validation

### 8.1 Membrane Continuity (Critical)

The dural membranes create **permeability barriers**, not topological barriers. The CSF remains connected through the sub-CC gap (below the falx) and the tentorial notch (through the tentorium). The CSF will likely remain 1–2 connected components after this step. The critical check is that the membranes themselves are continuous sheets without holes.

**Falx continuity:**

```python
from scipy.ndimage import label as cc_label

falx_components, n_falx_comp = cc_label(falx_mask)
falx_sizes = np.sort(np.bincount(falx_components.ravel())[1:])[::-1]
print(f"Falx connected components: {n_falx_comp}")
print(f"  Largest: {falx_sizes[0]:,} voxels ({falx_sizes[0] * vol_voxel:.1f} mL)")
if n_falx_comp > 1:
    print(f"  Second: {falx_sizes[1]:,} voxels")
```

**Expected:** 1 large connected component containing >90% of falx voxels. The falx is a single continuous sheet in the interhemispheric fissure. Small isolated fragments (< a few dozen voxels) are acceptable — they arise from EDT artifacts at fissure irregularities. Multiple large components would indicate the falx has a gap (e.g., from an overly aggressive CC constraint or insufficient fissure CSF).

**Tentorium continuity:**

```python
tent_components, n_tent_comp = cc_label(tent_mask)
tent_sizes = np.sort(np.bincount(tent_components.ravel())[1:])[::-1]
print(f"Tentorium connected components: {n_tent_comp}")
print(f"  Largest: {tent_sizes[0]:,} voxels ({tent_sizes[0] * vol_voxel:.1f} mL)")
```

**Expected:** 1 large connected component. The tentorium is a single sheet with a central hole (the notch). The hole does not fragment the sheet — it is an opening within the connected membrane, like a hole in a donut. Multiple large components would indicate the tentorium has a gap wider than the notch.

**CSF component report (informational):**

```python
csf_remaining = (mat == 8)
csf_components, n_csf_comp = cc_label(csf_remaining)
csf_sizes = np.sort(np.bincount(csf_components.ravel())[1:])[::-1]
print(f"CSF connected components after dural painting: {n_csf_comp}")
for i, s in enumerate(csf_sizes[:5]):
    print(f"  Component {i+1}: {s:,} voxels ({s * vol_voxel:.1f} mL)")
```

The CSF will likely remain 1 large connected component (the sub-CC gap and tentorial notch connect all compartments). This is not a failure — it reflects the anatomical reality that the compartments communicate through openings. The pressure isolation at runtime comes from the 10⁹ permeability contrast across the membrane, not from topological disconnection.

### 8.2 Volume and Thickness

Report the volume of each structure:

```python
vol_voxel = dx_mm ** 3 / 1000  # mL per voxel

vol_falx = n_falx * vol_voxel
vol_tent = n_tent * vol_voxel
vol_overlap = n_overlap * vol_voxel
vol_total = n_total * vol_voxel

print(f"Falx cerebri:   {n_falx:,} voxels  ({vol_falx:.1f} mL)")
print(f"Tentorium:      {n_tent:,} voxels  ({vol_tent:.1f} mL)")
print(f"Overlap:        {n_overlap:,} voxels  ({vol_overlap:.1f} mL)")
print(f"Total dural:    {n_total:,} voxels  ({vol_total:.1f} mL)")
```

**Expected ranges** (approximate, subject- and resolution-dependent):

| Structure | Expected volume | Notes |
|-----------|:--------------:|-------|
| Falx | 5–20 mL | Depends on fissure width and resolution |
| Tentorium | 3–15 mL | Depends on cerebrum–cerebellum gap |
| Overlap | < 1 mL | A few voxels at the posterior junction |

Literature reference: Schaller B, Chowdhury T, Engel D. "The falx cerebri and tentorium cerebelli: anatomical study with clinical implications." *World Neurosurgery* 2020;138:e700–e710. They measured falx area ~56.5 cm² and tentorium area ~57.6 cm² from CT. At 1–2 mm thickness, this gives ~5.7–11.3 mL per structure. Our voxel-based reconstruction should fall in a similar range.

**Thickness estimate:** Compute the mean thickness of each membrane along its normal direction:

```python
# Approximate: count dural voxels / (area estimate)
# Area from surface voxels (voxels with at least one non-dural face-neighbor)
from scipy.ndimage import binary_erosion
interior = binary_erosion(falx_mask)
surface_voxels = np.count_nonzero(falx_mask & ~interior)
# For a thin sheet, most voxels ARE surface voxels
# Thickness ≈ n_falx / (surface_voxels / 2) * dx_mm
# (divide by 2 because each surface voxel has two faces)
```

Expected: 1.0–2.0 mm (1–2 voxels at dev resolution).

### 8.3 Material Map Census

Report complete voxel counts after the update (same format as the Subarachnoid CSF step, Section 7.3). Key check: u8 = 10 should now have nonzero count, and u8 = 8 should have decreased by exactly n_total.

### 8.4 Medial Wall Proximity (Falx Quality Check)

The falx should sit between the left and right medial cortical surfaces. Validate using the pial surface medial wall vertices:

```python
import nibabel as nib
from scipy.spatial import cKDTree

# Load pial surfaces and medial wall ROIs (paths parameterized by subject_id)
base = f'data/raw/{subject_id}/T1w/Native'
l_pial = nib.load(f'{base}/{subject_id}.L.pial.native.surf.gii')
r_pial = nib.load(f'{base}/{subject_id}.R.pial.native.surf.gii')
mni_base = f'data/raw/{subject_id}/MNINonLinear/Native'
l_roi = nib.load(f'{mni_base}/{subject_id}.L.roi.native.shape.gii')
r_roi = nib.load(f'{mni_base}/{subject_id}.R.roi.native.shape.gii')

# Extract medial wall vertices (roi == 0)
l_coords = l_pial.darrays[0].data  # (N_vert, 3) in mm
r_coords = r_pial.darrays[0].data
l_medial = l_coords[l_roi.darrays[0].data == 0]  # ~5,700 vertices
r_medial = r_coords[r_roi.darrays[0].data == 0]  # ~6,000 vertices

# Build KD-trees
l_tree = cKDTree(l_medial)
r_tree = cKDTree(r_medial)

# For each falx voxel, compute distance to both medial walls
falx_ijk = np.argwhere(falx_mask)  # (n_falx, 3)
# Convert grid indices to physical ACPC coords (mm).
# This uses the grid-to-physical mapping: phys = (ijk - N/2) * dx_mm,
# which matches the A_g_to_p affine from domain_geometry.md.
# The surface .surf.gii coordinates are also in T1w ACPC space (mm).
falx_mm = (falx_ijk - N/2) * dx_mm

dist_to_left, _ = l_tree.query(falx_mm)
dist_to_right, _ = r_tree.query(falx_mm)

print(f"Falx distance to left medial wall:  "
      f"median={np.median(dist_to_left):.1f} mm, "
      f"95th={np.percentile(dist_to_left, 95):.1f} mm")
print(f"Falx distance to right medial wall: "
      f"median={np.median(dist_to_right):.1f} mm, "
      f"95th={np.percentile(dist_to_right, 95):.1f} mm")
```

**Expected:** Median distance 1–5 mm (the falx sits in the narrow fissure between the medial walls). The 95th percentile should be < 15 mm. Values much larger than this indicate the falx extends into regions far from the medial cortex — possibly an EDT artifact.

**Note:** This check requires surface files and is optional. The reconstruction algorithm does not depend on surfaces — they are used only for validation. If surface files are unavailable, skip this check.

### 8.5 Tentorial Notch Patency

Verify the notch is open (CSF remains around the brainstem at the tentorial level):

```python
brainstem = (mat == 6)

# Find the axial slice range of the brainstem
bs_slices = np.where(brainstem.any(axis=(0, 1)))[0]
bs_mid_z = bs_slices[len(bs_slices) * 2 // 3]  # upper third ≈ tentorial level

# At the tentorial level, check for CSF ring around brainstem
bs_slice = brainstem[:, :, bs_mid_z]
csf_slice = (mat[:, :, bs_mid_z] == 8)
dural_slice = (mat[:, :, bs_mid_z] == 10)

print(f"At z={bs_mid_z} (tentorial level):")
print(f"  Brainstem voxels: {bs_slice.sum()}")
print(f"  CSF voxels: {csf_slice.sum()}")
print(f"  Dural voxels: {dural_slice.sum()}")

# The CSF should form a ring around the brainstem (connected component
# containing brainstem-adjacent CSF voxels)
bs_dilated = ndimage.binary_dilation(bs_slice, iterations=1)
csf_adjacent = csf_slice & bs_dilated & ~bs_slice
n_adjacent_csf = np.count_nonzero(csf_adjacent)
print(f"  CSF voxels adjacent to brainstem: {n_adjacent_csf}")
assert n_adjacent_csf > 0, "Tentorial notch may be closed — no CSF adjacent to brainstem"
```

The check verifies that CSF exists immediately around the brainstem at the tentorial level. If n_adjacent_csf = 0, the tentorium (or notch exclusion) has sealed the notch, blocking supra-infratentorial communication.

### 8.6 Junction Thickness (Informational)

At the posterior falx-tentorium junction (straight sinus region), both masks independently produce membrane voxels. The combined thickness should not exceed ~3 voxels, which would artificially impede flow:

```python
# Find the overlap region and measure its extent
overlap = falx_mask & tent_mask
if overlap.any():
    overlap_ijk = np.argwhere(overlap)
    # Bounding box of the overlap
    bb_min = overlap_ijk.min(axis=0)
    bb_max = overlap_ijk.max(axis=0)
    bb_extent = (bb_max - bb_min + 1) * dx_mm
    print(f"Junction overlap: {n_overlap} voxels, "
          f"extent: {bb_extent[0]:.1f} x {bb_extent[1]:.1f} x {bb_extent[2]:.1f} mm")

    # Check thickness of combined dural mask at the junction
    # Sample a coronal slice through the overlap centroid
    junction_y = int(overlap_ijk[:, 1].mean())
    dural_slice = dural[:, junction_y, :]
    # Count contiguous dural voxels in z at the midline
    mid_x = N // 2
    z_col = dural_slice[mid_x, :]
    runs = np.diff(np.where(np.concatenate(([z_col[0]], z_col[:-1] != z_col[1:], [True])))[0])
    if z_col.any():
        max_run = runs[::2].max() if z_col[0] else (runs[1::2].max() if len(runs) > 1 else 0)
        print(f"  Max dural z-thickness at junction midline: {max_run} voxels ({max_run * dx_mm:.1f} mm)")
        if max_run > 3:
            print(f"  WARNING: Junction thickness > 3 voxels — may impede flow")
```

**Expected:** The overlap region should be small (a few dozen voxels) and no thicker than 2–3 voxels. Larger values indicate the two watersheds are producing redundant or diverging membrane sheets at the junction.

## 9. Design Rationale

**Why EDT watershed instead of surface-based construction?** The left and right pial surfaces could define the falx plane geometrically (the medial wall vertices trace its location). However: (a) the pial surfaces only cover the cerebral cortex, not the cerebellum, so a different approach is needed for the tentorium anyway; (b) surface-to-voxel distance queries for ~millions of CSF voxels against ~6k surface vertices require KD-trees and are more complex; (c) the EDT approach handles both structures with the same algorithm; (d) the published automated methods (Glaister et al. 2017) use the same fundamental idea (grow labels from each side, find the meeting boundary).

**Why not atlas-based warping?** The available nonlinear warp (`standard2acpc_dc.nii.gz`) is at 2 mm resolution. The structures we're placing are 1–2 mm thick. The warp doesn't have the spatial resolution to accurately deform something this thin. Additionally, warp quality is worst at tissue boundaries and in CSF-filled spaces (which have no contrast for the registration to align on). The literature (Schaller et al. 2020) documents large inter-subject anatomical variation in falx and tentorium geometry, making a template approach inherently limited.

**Why use the material map for tentorium EDT but FS labels for falx EDT?** The falx requires left/right hemisphere distinction, which only the original FS labels provide (the material map collapses left + right into bilateral classes). The tentorium only requires cerebral vs. cerebellar distinction, which the material map provides directly (u8 = 1,2,3 vs. u8 = 4,5). Using the material map for the tentorium avoids keeping the 256 MB FS labels array in memory during Phase 2, reducing peak memory.

**Why exclude midline structures from the falx hemisphere classification?** The corpus callosum (FS labels 192, 250–255) bridges the two hemispheres at the midline. Including it in either hemisphere would bias the EDT toward that side. Excluding it means the distance field passes through the CC, which correctly models the fact that the falx does not penetrate the CC — the CC is the boundary where the falx ends.

**Why exclude ventricles and CSF from tissue masks?** The EDT measures distance from solid tissue masses to find the natural boundary between them. Including fluid-filled ventricles would set the distance to zero inside the ventricles, distorting the watershed. The membranes sit between tissue masses, not between fluid compartments.

**Why per-slice CC constraint instead of a 3D CC mask dilation?** A 3D approach (e.g., dilate the CC mask and exclude the dilation from the falx) would be simpler code but anatomically imprecise. The CC is an arch — its superior surface varies by ~31 mm in z across its AP extent. The per-slice approach follows this arch exactly. The implementation cost is minimal (a loop over ~N/7 slices with a single array slice operation each).

**Why morphological dilation for the notch instead of EDT?** A brainstem EDT would give precise distance but costs 1024 MB (float64) for 512³. Morphological dilation with a small ball structuring element achieves the same result at a fraction of the memory cost (~128 MB for the boolean mask). The 5 mm margin does not require sub-voxel precision.

**Why the falx and tentorium use the same thickness threshold?** Both structures are anatomically 1–2 mm thick. At simulation resolution (1 mm), both should be 1–2 voxels thick. The same threshold (dx_mm) produces this for both structures. If future work requires different thicknesses, the parameters can be separated.

**What about the falx-tentorium junction?** Posteriorly, the falx descends to meet the tentorium at the straight sinus. Both algorithms independently identify membrane voxels in this region, and both paint u8 = 10. The union operation (Section 4.5) merges them without special treatment. The overlap is small (a few dozen voxels) and reported for monitoring.

**Why is this step likely to need iteration?** The dural membranes have the highest physical consequence of any preprocessing output — the 10⁹ contrast ratio means even small geometric errors can significantly affect pressure fields. The initial reconstruction should be validated by visualizing the membrane overlay on the T1w image and, later, by inspecting the pressure field behavior in early simulation runs. The parameters (threshold, notch radius) may need tuning based on these observations. The spec's problems.md notes: "A geometrically plausible first pass may be sufficient initially, refined later when you can visualize the pressure field behavior."

**Literature precedent.** The EDT watershed approach is equivalent to the method of Glaister J, Carass A, Prince JL. "Automatic falx cerebri and tentorium cerebelli segmentation from magnetic resonance images." *Proc SPIE 10133, Medical Imaging 2017: Image Processing*, 101330M. Their method — the only published automated approach for falx/tentorium segmentation from MRI — uses fast marching instead of EDT, but the principle is identical: grow hemisphere labels into the fissure space and find the boundary where they meet. Our approach adds explicit anatomical constraints (CC boundary, brainstem notch) that their method lacked, addressing their reported errors in the "frontal inferior region" of the falx.
