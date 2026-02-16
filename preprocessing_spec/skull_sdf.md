# Preprocessing Step 3: Skull SDF Construction

This document specifies the construction of the skull signed distance field (SDF) — a smooth 3D scalar volume encoding the distance from each voxel to the inner skull surface. The SDF defines the rigid boundary of the simulation domain.

## 1. Purpose

The runtime uses the skull SDF for two things:

1. **Halo injection** (Step 0, Action C): When an unallocated neighbor of an active fluid voxel has SDF > 0, it is activated and tagged as air halo (material ID 255). This dynamically paints the Monro-Kellie Dirichlet boundary.
2. **Cut-cell porosity** (φ_geo): Voxels near the SDF = 0 isosurface have partial volume occupancy. The SDF value determines the geometric porosity factor φ_geo ∈ [ε, 1.0], giving smooth boundary behavior instead of binary staircase edges.

The SDF is also used by the Subarachnoid CSF step to identify voxels that are inside the skull but not brain tissue.

**Sign convention:** SDF < 0 inside the skull cavity (brain/CSF domain), SDF > 0 outside (void), SDF = 0 at the skull inner surface.

## 2. What "Skull" Means

Not the bony skull. The SDF represents the **inner boundary of the cranial cavity** — approximately the inner table of the skull bone, which is the dural outer surface. The anatomical layering from inside to outside is:

```
brain tissue → pia mater → subarachnoid CSF → arachnoid → dura mater → skull bone → scalp
```

The SDF = 0 isosurface sits at the dura/skull bone interface. Everything inside (brain, CSF, meninges) has SDF < 0. Everything outside (bone, scalp, air) has SDF > 0.

FreeSurfer does not segment the skull or dura — these structures have poor T1w contrast. The SDF must be **constructed** from the brain mask by morphological expansion.

## 3. Inputs

All inputs are at source resolution (0.7 mm isotropic, 260 × 311 × 260) in T1w ACPC space.

| File | Content | Role |
|------|---------|------|
| `brainmask_fs.nii.gz` | Binary brain mask (uint8, 0/1) | Defines the brain interior |
| `Head.nii.gz` | Binary head mask (float32, 0/1) | Outer constraint — skull can't extend beyond the head |

Properties of the brain mask (verified):
- 3,883,660 nonzero voxels (~1,332 mL)
- Includes all ventricles, sulcal CSF (label 24), and choroid plexus
- Topologically solid: no internal holes, exactly one background connected component
- Fully contained within the head mask (no brain voxel is outside the head)

## 4. Algorithm

### 4.1 Overview

```
brain mask → inferior padding → morphological closing → outward dilation → (intersect with head mask) → signed EDT → resample to simulation grid
```

The morphological closing fills cortical sulci to produce a smooth outer envelope. The dilation expands this envelope outward to approximate the inner skull surface (accounting for the subarachnoid space between brain and skull). The signed Euclidean Distance Transform converts the binary mask to a smooth scalar field.

### 4.1a Inferior Padding

The brain mask extends to the very bottom of the acquisition volume (source voxel Z = 0, physical Z = -72 mm) — the brainstem/cerebellum is truncated by the MRI field of view. Without padding, the morphological dilation cannot extend below Z = 0 (the array boundary blocks it), so the skull surface would be flush with the brain at the inferior boundary instead of R_dilate mm outside it. This creates an SDF discontinuity: brain tissue at Z = 0 would have SDF ≈ 0 instead of SDF ≈ -R_dilate, and the resampling step would produce a sharp jump from negative SDF to cval = +100 at voxels below the source volume.

**Fix:** Pad both the brain mask and head mask with False slices below Z = 0 before any morphological operations. This gives the dilation room to extend inferiorly, creating a proper hemispherical cap below the brainstem. The result is a smooth sealed boundary at the foramen magnum with the same ~R_dilate margin as everywhere else.

```python
pad_z = r_close_vox + r_dilate_vox + 10  # 15 + 6 + 10 = 31 voxels (21.7 mm)
brain_mask = np.pad(brain_mask, ((0,0), (0,0), (pad_z, 0)), constant_values=False)
head_mask  = np.pad(head_mask,  ((0,0), (0,0), (pad_z, 0)), constant_values=False)

# Adjust source affine to reflect the new Z origin
A_padded = A_source.copy()
A_padded[2, 3] -= pad_z * voxel_size  # Z origin shifts from -72 to -93.7 mm
```

The padding adds 31 slices to axis 2 (the inferior/superior axis), increasing the volume from 260 × 311 × 260 to 260 × 311 × 291 (~12% larger). The padded slices are all False (background), so no phantom tissue is introduced. Subsequent morphological operations and the EDT operate on the padded volume using the adjusted affine `A_padded`. Only axis 2 needs padding — the other axes have ample margin (brain extent is well within the volume on X and Y).

### 4.2 Morphological Closing

A morphological closing is a dilation followed by an erosion with the same structuring element. It fills concavities while preserving convex contours.

**Why it's needed:** The brain mask follows every cortical fold. The skull does not — it's a smooth shell. Without closing, the SDF = 0 surface would dip into every sulcus, creating false "outside skull" regions in the subarachnoid space.

**Structuring element:** A spherical ball of radius R_close.

**Closing radius choice:** The Sylvian (lateral) fissure is the largest cortical concavity — 20-30 mm deep and 5-15 mm wide at its opening. It is funnel-shaped: narrow deep inside (~2-5 mm), wider at the surface. A closing radius of R_close = 10 mm fills the narrow deep portions of the fissure completely, and the subsequent outward dilation (Section 4.3) seals the wider opening. Together, the two operations close the Sylvian fissure and all smaller sulci. At 0.7 mm source resolution, R_close is ⌈10 / 0.7⌉ = 15 voxels.

**What closing preserves:** Locally convex surface regions are unchanged. The overall brain shape (frontal pole, occipital pole, temporal poles) is not shrunk or distorted. Only concavities are filled.

**Effect on the interhemispheric fissure:** The closing also fills the interhemispheric fissure (2-5 mm wide, well within the closing diameter), merging the two cerebral hemispheres into a single solid mask. This is correct — the skull encloses both hemispheres as a single cavity. The internal division between hemispheres (falx cerebri) is reconstructed separately in the Dural Membrane step (`dural_membrane.md`).

**Implementation — iterated small ball:**

A single closing with a 31^3 ball structuring element is correct but slow on a 260^3 volume (~14,000 True voxels per dilation check). An equivalent and much faster approach is to iterate with a smaller ball. Dilation by ball(r1) followed by dilation by ball(r2) equals dilation by ball(r1 + r2) — the Minkowski sum of two balls is a ball. So `n_iter` iterations with ball(r_step) gives an exact spherical dilation of radius `n_iter × r_step`.

```python
r_vox = round(R_close / voxel_size)  # 15 for R_close=10mm, vox=0.7mm

# Build a small ball structuring element (radius 3 voxels, 7x7x7)
r_step = 3
coords = np.arange(-r_step, r_step + 1)
x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
small_ball = (x**2 + y**2 + z**2) <= r_step**2
# small_ball.shape = (7, 7, 7), ~123 True voxels

n_iter = math.ceil(r_vox / r_step)   # 5 iterations for r_vox=15, r_step=3

closed = ndimage.binary_dilation(brain_mask, small_ball, iterations=n_iter)
closed = ndimage.binary_erosion(closed, small_ball, iterations=n_iter)
```

This produces an exact spherical closing of radius `n_iter × r_step` = 15 voxels (10.5 mm). Each iteration checks ~123 neighbors instead of ~14,000, making it roughly 40× faster than the single-pass approach (~30 seconds vs ~10-20 minutes).

**Important:** Do NOT use a non-spherical structuring element (e.g., 6-connected cross) for iteration. Iterating with a cross produces a diamond-shaped (L1-norm) dilation where the effective radius along space diagonals is only 58% of the axis radius. This would leave diagonally-oriented concavities unclosed.

### 4.3 Outward Dilation

After closing, the mask boundary sits approximately at the outer cortical surface (gyral crowns). The inner skull surface is further out — separated by the subarachnoid space (~2-5 mm in a healthy young adult) and the meninges (~1 mm).

Dilate the closed mask by R_dilate = 4 mm (⌈4 / 0.7⌉ = 6 voxels) using the same small ball structuring element as the closing.

```python
r_dilate_vox = round(R_dilate / voxel_size)  # 6
n_dilate_iter = math.ceil(r_dilate_vox / r_step)  # 2 iterations with ball(3)
skull_interior = ndimage.binary_dilation(closed, small_ball, iterations=n_dilate_iter)
```

The effective dilation radius is `n_dilate_iter × r_step` = 6 voxels (4.2 mm), closely matching the target R_dilate = 4 mm.

After this step, `skull_interior` is a binary mask representing the intracranial cavity bounded by the inner skull surface.

### 4.4 Head Mask Constraint

Intersect the dilated mask with the (padded) head mask to prevent the skull surface from extending beyond the actual head. This is a safety constraint — with R_dilate = 4 mm, the dilated brain mask is well within the head on all sides (the head extends 10+ mm beyond the brain everywhere). But it prevents artifacts in edge cases. The head mask was padded in Section 4.1a alongside the brain mask; the padded region is all False, so the intersection naturally prevents the skull surface from extending below the original acquisition volume.

```python
skull_interior = skull_interior & head_mask
```

### 4.5 Signed Euclidean Distance Transform

Compute the signed distance from each voxel to the boundary of `skull_interior`:

```python
# dt_outside: for each voxel OUTSIDE skull_interior, distance to nearest interior voxel
# For inside voxels, value is 0
dt_outside = ndimage.distance_transform_edt(~skull_interior, sampling=voxel_size)

# dt_inside: for each voxel INSIDE skull_interior, distance to nearest exterior voxel
# For outside voxels, value is 0
dt_inside = ndimage.distance_transform_edt(skull_interior, sampling=voxel_size)

# Combine: negative inside, positive outside
sdf = dt_outside - dt_inside
```

The resulting SDF has:
- Values in **mm** (physical units), because `sampling=voxel_size` is passed to the EDT
- Gradient magnitude ≈ 1.0 everywhere (it's a distance field)
- Smooth spatial variation suitable for trilinear interpolation

### 4.6 Resample to Simulation Grid

Resample the SDF from source resolution (padded) to the simulation grid using the `resample_to_grid` utility from the Domain Geometry step:

```python
sdf_sim = resample_to_grid(
    source_data=sdf,              # 260x311x291 float32 (padded)
    source_affine=A_padded,       # adjusted affine from Section 4.1a
    grid_affine=A_g_to_p,         # from grid_meta.json
    grid_shape=(N, N, N),
    order=1,                      # trilinear — SDF is continuous
    cval=100.0,                   # large positive = outside skull
    dtype=np.float32,
    slab_size=32
)
```

**cval = 100.0:** Simulation grid voxels that map outside the padded source volume extent receive SDF = +100 mm (firmly "outside skull"). This is a safe default — these voxels are far from the brain and will always be classified as exterior.

**Note on resample_to_grid:** The function signature from the Domain Geometry step takes a NIfTI file path. For the SDF, we already have the array in memory (from the EDT computation). The `resample_to_grid` function should be extended to accept either a file path or a `(data, affine)` pair as its first argument. When given a path, it loads the NIfTI as before; when given a tuple, it uses the provided array and affine directly. This is a minor change to the Domain Geometry utility (a type check on the first argument) that avoids unnecessary disk I/O for in-memory volumes like the SDF.

### 4.7 Foramen Magnum: Sealed Boundary

The foramen magnum is where the brainstem exits the skull inferiorly. In the source data, the brainstem extends to the very bottom of the acquisition volume (source voxel Z = 0, physical Z = -72 mm). The brain mask is truncated by the field of view here.

The inferior padding (Section 4.1a) gives the morphological dilation room to extend below the brainstem, creating a smooth hemispherical cap. The SDF transitions smoothly from negative (inside) to positive (outside) at this boundary, with the same ~R_dilate margin as everywhere else. The result is a sealed inferior boundary — no fluid escapes downward.

This is consistent with the spec's single-boundary Monro-Kellie model, where all volume exchange occurs through the air halo at the skull surface. The foramen magnum is not modeled as an open channel.

## 5. Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| R_close | 10 mm | Fills Sylvian fissure (widest concavity, up to 20 mm). Preserves convex brain contour. |
| R_dilate | 4 mm | Subarachnoid space (~2-5 mm) + meninges (~1 mm). Conservative estimate for healthy young HCP subject. |

Both parameters should be exposed as command-line arguments for tuning:

```
--close-radius 10.0   # mm
--dilate-radius 4.0   # mm
```

### 5.1 Sensitivity

**R_close:** Values from 7-15 mm all produce reasonable results. Below 7 mm, the Sylvian fissure may not fully close. Above 15 mm, the computation becomes slower with no benefit (there are no concavities wider than ~20 mm that need filling). The closed mask is insensitive to the exact value within this range because closing preserves convex contours — it only affects concavity filling.

**R_dilate:** This directly controls subarachnoid CSF volume (see `subarachnoid_csf.md`). At 3 mm, the subarachnoid space is thin — possibly too thin at the basal cisterns. At 5 mm, it provides generous margin but may overestimate convexity CSF. The impact is moderate — the Monro-Kellie controller adapts to whatever CSF volume exists, and permeability values can compensate for thickness errors.

### 5.2 Known Limitation: Basal Cisterns

The subarachnoid space varies in thickness from ~1 mm at the convexity to ~10-15 mm at the basal cisterns (prepontine, ambient, cisterna magna). A uniform 4 mm dilation underestimates the cisterns. This is acceptable for a first pass — the cisterns still exist as CSF voxels (just thinner than reality), and the simulation physics are not critically sensitive to their exact thickness. If refinement is needed, the head mask could be used to derive a spatially varying expansion (the gap between brain mask and eroded head mask is larger at the cisterns), but this adds complexity with modest benefit.

## 6. Outputs

### 6.1 Skull SDF Volume

| Property | Value |
|----------|-------|
| File | `skull_sdf.nii.gz` |
| Shape | N^3 (e.g., 512 × 512 × 512 for dev) |
| Dtype | `float32` |
| Affine | A_g→p (simulation grid coordinate system) |
| Units | mm (signed distance to skull inner surface) |
| Sign convention | Negative inside skull, positive outside |

Saved to `data/processed/{subject_id}/{profile}/` alongside the Domain Geometry outputs.

### 6.2 Size Estimate

A 512^3 float32 volume is 512 MB uncompressed. NIfTI gzip compression reduces this substantially — the SDF is smooth and spatially correlated, so compression ratios of 3-5x are typical. Expected file size: ~100-170 MB.

## 7. Implementation

### 7.1 Stack

Same as the Domain Geometry step: Python 3, nibabel, scipy (`scipy.ndimage` for morphological operations and EDT), numpy. No additional dependencies.

### 7.2 Algorithm Summary

```
Input:  subject_id, profile (→ Δx, N), R_close, R_dilate
Output: skull_sdf.nii.gz

1. Load brainmask_fs.nii.gz → brain_mask[260,311,260] (bool)
   Load Head.nii.gz → head_mask[260,311,260] (bool)
   Read A_source and voxel_size from NIfTI header (0.7 mm)

2. Inferior padding:
   r_close_vox = round(R_close / voxel_size)   # 15
   r_dilate_vox = round(R_dilate / voxel_size)  # 6
   pad_z = r_close_vox + r_dilate_vox + 10      # 31 voxels
   brain_mask = np.pad(brain_mask, ((0,0),(0,0),(pad_z,0)), constant_values=False)
   head_mask  = np.pad(head_mask,  ((0,0),(0,0),(pad_z,0)), constant_values=False)
   A_padded = A_source.copy()
   A_padded[2, 3] -= pad_z * voxel_size         # shift Z origin down

3. Morphological closing (iterated small ball):
   r_step = 3
   small_ball = build_ball(r_step)               # 7x7x7, ~123 True voxels
   n_close = ceil(r_close_vox / r_step)          # 5
   closed = binary_dilation(brain_mask, small_ball, iterations=n_close)
   closed = binary_erosion(closed, small_ball, iterations=n_close)

4. Outward dilation:
   n_dilate = ceil(r_dilate_vox / r_step)        # 2
   skull_interior = binary_dilation(closed, small_ball, iterations=n_dilate)

5. Head mask constraint:
   skull_interior = skull_interior & head_mask

6. Signed distance transform:
   dt_outside = distance_transform_edt(~skull_interior, sampling=(0.7, 0.7, 0.7))
   dt_inside  = distance_transform_edt(skull_interior,  sampling=(0.7, 0.7, 0.7))
   sdf_source = (dt_outside - dt_inside).astype(np.float32)

7. Resample to simulation grid:
   Load grid_meta.json → A_g→p, (N, N, N)
   sdf_sim = resample_to_grid((sdf_source, A_padded), A_g→p, (N,N,N),
                               order=1, cval=100.0, dtype=np.float32)

8. Save:
   skull_sdf.nii.gz ← sdf_sim with affine A_g→p
```

### 7.3 Memory Analysis

All morphological operations and EDT are performed at padded source resolution (260 × 311 × 291 = ~23.5M voxels, ~12% larger than the unpadded 21M). Peak memory during each phase:

| Phase | Arrays in memory | Size |
|-------|-----------------|------|
| Closing | brain_mask (bool) + closed (bool) + head_mask (bool) | 3 × 23.5 MB = 71 MB |
| EDT | skull_interior (bool) + dt_outside (f64) + dt_inside (f64) | 23.5 + 179 + 179 = 382 MB |
| SDF combine | sdf_source (f32) + one EDT (f64) | 90 + 179 = 269 MB |
| Resampling | sdf_source (f32) + sdf_sim (f32, 512^3) + slab coords (f64) | 90 + 512 + 192 = 794 MB |

Peak: ~794 MB during resampling. Comfortable on a 5.7 GB system.

**Note:** The EDT arrays are float64 (scipy default). The `dt_outside` and `dt_inside` arrays can be freed after computing `sdf_source`, reducing peak EDT memory. The resampling phase is the bottleneck, and it uses the same slab-based approach as the Domain Geometry step.

## 8. Validation

The script should report:

### 8.1 Intracranial Volume

Total volume where SDF < 0, computed from the simulation-resolution output:

$$V_{ICV} = n_{negative} \cdot \Delta x^3$$

Expected: ~1,500-1,750 mL. This is larger than the brain mask volume (~1,332 mL) because it includes the sulcal volume filled by morphological closing (~50-100 mL) and the dilation shell representing the subarachnoid space (~250-300 mL, from the outer envelope surface area ~700 cm² × R_dilate). For reference, physiological intracranial volume in healthy adults is ~1,400-1,700 mL, with the brain occupying ~80-85% of the ICV. The difference ($V_{ICV} - V_{brain\_mask}$) should be reported as the estimated subarachnoid + sulcal CSF volume.

### 8.2 Brain Containment

Every voxel of the resampled brain mask (from the Domain Geometry step) must have SDF < 0. Formally:

```python
brain_sim = load('brain_mask.nii.gz')  # Domain Geometry output
assert np.all(sdf_sim[brain_sim > 0] < 0), "Brain tissue found outside skull SDF"
```

Also report the minimum SDF value at brain mask voxels — this is the margin between the brain surface and the SDF = 0 isosurface. Expected: approximately -R_dilate (−4 mm) at the convexity, more negative deeper inside the brain.

### 8.3 Margin at Brain Surface

The SDF at the outermost brain voxels (the convex surface) should be approximately -R_dilate. If it's much less negative (e.g., -1 mm), the dilation was too small and the subarachnoid space is too thin. Report the 5th percentile of SDF values at brain boundary voxels:

```python
brain_surface = ndimage.binary_dilation(brain_sim > 0) & ~(brain_sim > 0)
sdf_at_surface = sdf_sim[brain_surface]
p5 = np.percentile(sdf_at_surface, 5)
print(f"SDF at brain surface: 5th percentile = {p5:.1f} mm (expected ≈ -{R_dilate})")
```

The 5th percentile (rather than minimum) is used because a few brain boundary voxels near deep sulci may have SDF values more negative than -R_dilate, which is normal — only the convexity surface should cluster near -R_dilate.

### 8.4 Gradient Magnitude Check

The SDF gradient magnitude should be approximately 1.0 everywhere (property of a distance field). Compute the 2D gradient magnitude on three orthogonal slices through the brain center and report the mean and std:

```python
for name, slc in [("axial",    sdf_sim[N//2, :, :]),
                   ("coronal",  sdf_sim[:, N//2, :]),
                   ("sagittal", sdf_sim[:, :, N//2])]:
    g0, g1 = np.gradient(slc, dx_mm)
    grad_mag = np.sqrt(g0**2 + g1**2)
    # Mask to region near the brain (exclude far-field where SDF is constant)
    near_brain = np.abs(slc) < 50.0  # within 50 mm of skull surface
    print(f"  {name}: mean |∇SDF| = {grad_mag[near_brain].mean():.3f}, "
          f"std = {grad_mag[near_brain].std():.3f}")
```

This is a 2D approximation — the gradient component perpendicular to each slice is missing, so the measured magnitude is ≤ 1.0 (it equals 1.0 only where the SDF gradient lies entirely within the slice plane). The check catches gross SDF artifacts (sign errors, discontinuities, non-monotonic regions) but not subtle 3D issues. Expected: mean ≈ 0.8-1.0, with values < 0.5 flagging problems.

### 8.5 Smoothness at Boundary

Report the standard deviation of SDF values in a thin shell around the SDF = 0 surface. In a smooth SDF, values transition linearly through zero, so the std should be proportional to the shell thickness. Large variance indicates a rough or jagged boundary.

## 9. Design Rationale

**Why morphological closing instead of pial surface hull?** The pial surfaces only cover the cerebral cortex — not the cerebellum or brainstem. They are also open surfaces (each hemisphere is an uncapped shell), requiring non-trivial mesh processing to create a closed surface for voxelization. The brain mask is already a clean solid volume covering all structures. Morphological closing is simpler, more robust, and produces equivalent results for our purposes.

**Why not extract the skull from T1w?** Skull bone has poor T1w contrast and is confounded by air-filled sinuses, vessels, and meninges. Robust skull extraction from T1w is a known hard problem in neuroimaging. The brain mask approach avoids this entirely.

**Why R_close = 10 mm?** The Sylvian fissure is funnel-shaped: narrow at depth (~2-5 mm), wider at the cortical surface (~10-15 mm). A 10 mm closing fills the narrow deep portions completely. Any remaining gap at the wider surface opening is then sealed by the 4 mm outward dilation (Section 4.3). The two operations work together — the closing doesn't need to seal the entire fissure alone. A smaller closing radius (< 7 mm) would leave too much of the fissure unclosed for the dilation to compensate. A larger radius provides no additional benefit.

**Why R_dilate = 4 mm?** The subarachnoid space in a healthy young adult is ~2-5 mm thick at the convexity. The meninges add ~1 mm. A 4 mm expansion places the SDF = 0 surface at approximately the inner skull table. This is deliberately conservative — a slightly thin subarachnoid space is better than an unrealistically thick one, and the Subarachnoid CSF step will paint whatever gap exists as CSF regardless.

**Why compute at source resolution (0.7 mm) and resample?** Morphological operations are exact on the native binary mask — no resampling staircase artifacts at the mask boundary. The source volume (260^3) is 15x smaller than the simulation grid (512^3), so all operations are faster. The SDF is a smooth field, so trilinear resampling to the simulation grid is clean and introduces no artifacts.

**Why sealed foramen magnum?** The spec's Monro-Kellie controller uses a single boundary condition (air halo pressure at the skull surface). Opening the foramen magnum would require a second boundary condition (spinal CSF drainage) that the controller is not designed to handle. Sealing it is consistent with the spec and simplifies the boundary topology. Physiologically, the foramen magnum flow rate (~0.5 mL/min) is negligible compared to the hemorrhage flow rates being simulated. The inferior padding (Section 4.1a) ensures the sealed boundary is smooth — the dilation extends below the brainstem into the padded region, creating a hemispherical cap with the same ~R_dilate margin as everywhere else.

**Why Head.nii.gz intersection?** Pure safety. The 4 mm dilation should never extend beyond the head mask (which is 10+ mm outside the brain everywhere). But the intersection prevents edge-case artifacts and costs nothing (single boolean AND on a 21M-element array).

**Why float32 output, not float16?** The SDF values range from approximately -130 mm (brain center) to +100 mm (grid corners). Float16 has ~3 digits of precision, giving ~0.1 mm resolution at these magnitudes — acceptable, but float32 provides comfortable margin for downstream computations (porosity calculation, gradient estimation). The runtime can downconvert to float16 when loading to VRAM if memory is tight. Preprocessing should preserve full fidelity.

**Why cval = 100.0 for resampling?** Grid voxels outside the source volume extent must have SDF > 0 (outside skull). A value of 100 mm is unambiguously exterior and won't create boundary artifacts. The exact value doesn't matter — any positive number larger than R_dilate works.
