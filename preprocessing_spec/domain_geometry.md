# Preprocessing Step 1: Domain Geometry and Coordinate Mapping

This document specifies the coordinate system mapping between HCP T1w ACPC space and the simulation grid, the resampling strategy, and the outputs produced. It is the foundation for all subsequent preprocessing steps.

## 1. Source Data

All source volumes for subject 157336 live under `data/raw/157336/T1w/` and share a common coordinate system:

| Property | Value |
|----------|-------|
| Voxel grid | 260 × 311 × 260 |
| Voxel size | 0.7 mm isotropic |
| NIfTI origin (qoffset) | (90.0, −126.0, −72.0) mm |
| Affine diagonal | (−0.7, +0.7, +0.7) — **negative X** |
| Physical coordinate system | RAS+ (NIfTI standard) |
| Voxel storage order | LAS (increasing voxel index X → Left, Y → Anterior, Z → Superior) |

The negative X scaling means voxel index 0 on the X axis corresponds to the rightmost physical coordinate (+90 mm), and voxel index 259 corresponds to the leftmost (−91.3 mm). The Y and Z axes have positive scaling and follow the standard anterior/superior directions.

The ACPC alignment places the anterior commissure (AC) at physical coordinate (0, 0, 0) mm, which falls at approximately voxel (128.6, 180.0, 102.9) in source space. All segmentation volumes (`aparc+aseg.nii.gz`, `wmparc.nii.gz`, `brainmask_fs.nii.gz`), structural images, and surface files share this space. Diffusion data (`Diffusion/`, `Diffusion.bedpostX/`) is registered to this space at 1.25 mm resolution (affine diagonal: −1.25, +1.25, +1.25, same origin).

## 2. Simulation Grid Definition

### 2.1 Grid Parameters by Profile

| Profile | Grid size $N^3$ | Spacing $\Delta x$ (mm) | Domain extent (mm) |
|---------|:---------------:|:-----------------------:|:-------------------:|
| debug | $256^3$ | 2.0 | 512 |
| dev | $512^3$ | 1.0 | 512 |
| prod | $512^3$ | 0.5 | 256 |

$N$ is the number of voxels per axis. The physical domain extent is $N \cdot \Delta x$ in each dimension.

**Prod profile margin note:** The prod domain is only ±128 mm per side. The tightest margin is on the **Y posterior** axis: the brain extends to −103.6 mm, giving only **24.4 mm** of margin before the grid edge at −128 mm. After accounting for subarachnoid space (~2–5 mm), skull representation (~5–10 mm), and SDF sampling (~5 mm), roughly 4–12 mm of slack remains. This is adequate but tight — the prod profile must be validated to confirm no tissue or SDF is clipped. Other axes are more comfortable (X: ~67 mm min, Z: ~55 mm min).

### 2.2 Coordinate Mapping

The simulation grid is **centered on the ACPC origin** (anterior commissure) and uses **RAS+ voxel ordering** — all three diagonal entries of $A_{g \to p}$ are positive. This means grid index (0, 0, 0) is the left-posterior-inferior corner and indices increase toward right, anterior, superior.

This differs from the source data's voxel ordering (LAS — negative X). The composite transform $M = A_{source}^{-1} \cdot A_{g \to p}$ (Section 3.1) absorbs this reflection automatically; no special handling is needed in the resampling code.

The affine transform from physical ACPC coordinates (mm) to simulation grid indices:

$$i = \frac{x}{\Delta x} + \frac{N}{2}, \quad j = \frac{y}{\Delta x} + \frac{N}{2}, \quad k = \frac{z}{\Delta x} + \frac{N}{2}$$

Equivalently, the physical coordinate of grid voxel $(i, j, k)$:

$$x = (i - \frac{N}{2}) \cdot \Delta x, \quad y = (j - \frac{N}{2}) \cdot \Delta x, \quad z = (k - \frac{N}{2}) \cdot \Delta x$$

This places the ACPC origin at grid center $(\frac{N}{2}, \frac{N}{2}, \frac{N}{2})$.

In matrix form, the **grid-to-physical** affine (4×4, for NIfTI sform):

$$A_{g \to p} = \begin{bmatrix} \Delta x & 0 & 0 & -\frac{N}{2} \cdot \Delta x \\ 0 & \Delta x & 0 & -\frac{N}{2} \cdot \Delta x \\ 0 & 0 & \Delta x & -\frac{N}{2} \cdot \Delta x \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

And the **physical-to-grid** inverse:

$$A_{p \to g} = \begin{bmatrix} \frac{1}{\Delta x} & 0 & 0 & \frac{N}{2} \\ 0 & \frac{1}{\Delta x} & 0 & \frac{N}{2} \\ 0 & 0 & \frac{1}{\Delta x} & \frac{N}{2} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 2.3 Padding Analysis

With ACPC centering, the brain extent in each axis (measured from `brainmask_fs` for subject 157336) and resulting margins. Because the brain is not centered on the AC, margins are asymmetric — the table shows both sides:

| Axis | Physical range (mm) | Extent | Dev grid side | Dev margin (−) | Dev margin (+) |
|------|:-------------------:|:------:|:---:|:-:|:-:|
| X (L↔R) | −60.5 to +62.7 | 123 mm | ±256 | 195 mm (L) | 193 mm (R) |
| Y (P↔A) | −103.6 to +70.0 | 174 mm | ±256 | **152 mm (P)** | 186 mm (A) |
| Z (I↔S) | −72.0 to +72.9 | 145 mm | ±256 | 184 mm (I) | 183 mm (S) |

The tightest margin (Y posterior, 152 mm) is far more than sufficient for the air halo (a few voxels) and skull SDF queries (which only need valid values within ~10–20 mm of the skull surface).

**Note on asymmetry:** The brain extends ~104 mm posterior to the AC but only ~70 mm anterior. This 1.5:1 ratio creates a 34 mm difference in Y-axis margins. The asymmetry is harmless for dev/debug profiles given their large total margins, but becomes relevant for the prod profile (see Section 2.1).

**Note on inferior extent:** The brain mask has nonzero voxels at source voxel Z = 0, the very bottom of the acquisition volume (physical Z = −72 mm). This means the most inferior cerebellum/brainstem may be slightly clipped by the MRI field of view. The simulation grid extends well beyond this (−256 mm for dev), so no additional clipping occurs during resampling, but the source data itself may be truncated inferiorly. This affects the skull SDF at the inferior boundary — the SDF cannot represent skull surface below the acquisition extent.

## 3. Resampling

### 3.1 Composing Transforms

To resample a source volume onto the simulation grid, we compose transforms:

```
source voxel indices ←[A_source]→ physical mm ←[A_p→g]→ simulation grid indices
```

For each simulation grid voxel $(i, j, k)$:
1. Compute physical coordinate via $A_{g \to p}$
2. Map to source voxel coordinate via $A_{source}^{-1}$ (the inverse of the source NIfTI's sform/qform affine)
3. Sample the source volume at that (possibly fractional) voxel coordinate

In practice, this is a single 4×4 matrix multiply: $M = A_{source}^{-1} \cdot A_{g \to p}$, mapping simulation grid indices directly to source voxel indices.

### 3.2 Interpolation Methods

| Data type | Method | Rationale |
|-----------|--------|-----------|
| Integer labels (`aparc+aseg`, `wmparc`, `brainmask_fs`, `ribbon`) | Nearest-neighbor | Labels are categorical; interpolation would produce invalid values |
| Continuous scalar volumes (T1w, T2w, SDF fields) | Trilinear | Smooth data, standard approach |
| Fiber orientations (bedpostX dyads) | See fiber texture spec | Requires sign-consistent interpolation + renormalization (separate preprocessing step) |

### 3.3 Out-of-Bounds Handling

Simulation grid voxels that map to coordinates outside the source volume extent receive:
- Label `0` (vacuum) for integer label volumes
- Value `0.0` for continuous volumes

This naturally fills the padding region with vacuum. **Implementation note:** `scipy.ndimage.map_coordinates` defaults to `mode='reflect'`, which mirrors out-of-bounds coordinates. To get constant-fill behavior, you must pass `mode='constant'` explicitly. Omitting this would silently produce reflected brain labels in the padding region instead of vacuum.

### 3.4 Downsampling Ratio

| Source | Source resolution | Dev grid (1 mm) | Ratio |
|--------|:-----------------:|:----------------:|:-----:|
| T1w / segmentation | 0.7 mm | 1.0 mm | 1.43× downsample |
| 3T diffusion / bedpostX | 1.25 mm | 1.0 mm | 0.80× upsample |
| 7T diffusion | 1.05 mm | 1.0 mm | 0.95× (near unity) |

The segmentation downsample (1.43×) is mild. Structures thinner than ~1.5 source voxels (~1 mm) could theoretically be missed by nearest-neighbor, but no brain structures at this scale are that thin. Nearest-neighbor is sufficient.

## 4. Outputs

The domain geometry step produces the following files, saved to a per-profile output directory (e.g., `data/processed/{subject_id}/dev/`):

### 4.1 Resampled FreeSurfer Labels

| Property | Value |
|----------|-------|
| File | `fs_labels_resampled.nii.gz` |
| Shape | $N^3$ (e.g., 512×512×512 for dev) |
| Dtype | `int16` |
| Affine | $A_{g \to p}$ (encodes the simulation grid coordinate system) |
| Content | FreeSurfer labels from `aparc+aseg.nii.gz`, resampled via nearest-neighbor. **Not yet remapped** to simulation u8 indices — that is Task 6. |

**Why int16, not uint8?** FreeSurfer labels range from 0 to 2035 (cortical parcels ctx-lh-\* = 1001–1035, ctx-rh-\* = 2001–2035). These exceed the uint8 range (0–255). The raw labels must be preserved at full fidelity so that the remapping step (Task 6) can distinguish every FreeSurfer label. The conversion to uint8 happens *after* remapping — the collapsed 11-class material index fits comfortably in uint8.

**Why output raw FreeSurfer labels first?** Separation of concerns. This step handles only geometry and resampling. Keeping the label remapping separate means you can open the resampled label volume in any NIfTI viewer (FSLeyes, freeview) and compare it against the original segmentation to catch resampling errors before the label collapse makes the volume harder to interpret.

### 4.2 Resampled Brain Mask

| Property | Value |
|----------|-------|
| File | `brain_mask.nii.gz` |
| Shape | $N^3$ |
| Dtype | `uint8` (0 or 1) |
| Content | `brainmask_fs.nii.gz` resampled via nearest-neighbor |

Used by later steps (skull SDF construction, subarachnoid space identification).

### 4.3 Grid Metadata

| Property | Value |
|----------|-------|
| File | `grid_meta.json` |

This sidecar provides all coordinate mapping parameters in a single machine-readable file, so downstream steps don't need to re-derive them from the NIfTI header. Schema:

```json
{
  "subject_id": "157336",
  "profile": "dev",
  "grid_size": 512,
  "dx_mm": 1.0,
  "domain_extent_mm": 512.0,
  "affine_grid_to_phys": [
    [1.0, 0.0, 0.0, -256.0],
    [0.0, 1.0, 0.0, -256.0],
    [0.0, 0.0, 1.0, -256.0],
    [0.0, 0.0, 0.0,    1.0]
  ],
  "affine_phys_to_grid": [
    [1.0, 0.0, 0.0, 256.0],
    [0.0, 1.0, 0.0, 256.0],
    [0.0, 0.0, 1.0, 256.0],
    [0.0, 0.0, 0.0,   1.0]
  ],
  "source_shape": [260, 311, 260],
  "source_voxel_mm": 0.7,
  "source_affine": [
    [-0.7, 0.0, 0.0,  90.0],
    [ 0.0, 0.7, 0.0,-126.0],
    [ 0.0, 0.0, 0.7, -72.0],
    [ 0.0, 0.0, 0.0,   1.0]
  ],
  "brain_bbox_grid": {
    "min": [195, 152, 184],
    "max": [319, 326, 329]
  },
  "brain_volume_ml": 1332.1,
  "brain_centroid_grid": [256.3, 234.2, 262.8]
}
```

### 4.4 Brain Bounding Box (in grid coordinates)

Computed from the resampled brain mask — the tightest axis-aligned box containing all nonzero voxels. Stored in `grid_meta.json` under `brain_bbox_grid`. Used by downstream steps to avoid iterating over the full $N^3$ grid when only the brain region matters.

### 4.5 Output Directory

All outputs are saved to `data/processed/{subject_id}/{profile}/` (e.g., `data/processed/157336/dev/`). The script creates this directory if it does not exist.

## 5. Implementation

### 5.1 Stack

- **Python 3** with **nibabel** (NIfTI I/O), **scipy** (`scipy.ndimage.map_coordinates` for resampling), **numpy** (array math)
- Single script parameterized by `--dx` (grid spacing) and `--subject` (subject ID)
- Grid size $N$ derived from profile: the script accepts `--dx` and `--grid-size` (or infers $N$ from a profile name)

### 5.2 Algorithm

```
Input:  subject_id, Δx, N
Output: fs_labels_resampled.nii.gz, brain_mask.nii.gz, grid_meta.json

1. Create output directory (data/processed/{subject_id}/{profile}/)
   if it does not exist.

2. Load source volumes:
   - aparc+aseg.nii.gz  → labels[260,311,260]
     On-disk dtype is float32 (integer-valued). Load via get_fdata() → float64,
     then cast: labels = np.round(img.get_fdata()).astype(np.int16)
   - brainmask_fs.nii.gz → mask[260,311,260]
     Cast similarly: mask = img.get_fdata().astype(np.uint8)

3. Read source affine A_source from NIfTI header (img.affine)
   Verify: diagonal should be (-0.7, +0.7, +0.7) for T1w space

4. Construct simulation grid affine (RAS+ orientation):
   A_g→p = diag(Δx, Δx, Δx, 1)
   A_g→p[0:3, 3] = [-N/2 * Δx, -N/2 * Δx, -N/2 * Δx]

5. Compute composite transform:
   M = A_source⁻¹ · A_g→p
   (M will have a negative [0,0] entry due to the source's negative X scaling)

6. For each slab of S=32 slices along axis 0:
   a. Build coordinate arrays of shape (3, S, N, N):
      For i in [slab_start .. slab_start+S), j in [0..N), k in [0..N):
        [sx, sy, sz, 1]ᵀ = M · [i, j, k, 1]ᵀ
      (Vectorized: use meshgrid + affine matrix multiply)

   b. Resample — NOTE: mode='constant' is required for cval to take effect:
      labels_slab = map_coordinates(labels, coords, order=0,
                                    mode='constant', cval=0)
      mask_slab   = map_coordinates(mask,   coords, order=0,
                                    mode='constant', cval=0)

   c. Write slab into preallocated output arrays

7. Cast outputs:
   - labels_sim = labels_sim.astype(np.int16)
   - mask_sim   = mask_sim.astype(np.uint8)

8. Compute brain bounding box from resampled mask

9. Save outputs:
   - fs_labels_resampled.nii.gz: labels_sim as int16, affine = A_g→p
   - brain_mask.nii.gz:          mask_sim as uint8,  affine = A_g→p
   - grid_meta.json:             all metadata (see Section 4.3 for schema)
```

### 5.3 Memory Considerations

A 512³ uint8 volume is 128 MB — fine. The coordinate arrays for `map_coordinates` are three float64 arrays of shape (512, 512, 512) = 3 × 1 GB = 3 GB. With 5.7 GB system RAM, this is tight.

**Mitigation:** Process the volume in slabs along one axis. For each slab of $S$ slices:
- Build coordinate arrays of shape (S, 512, 512) — manageable
- Resample that slab
- Write to the output array

A slab size of $S = 32$ uses 3 × 32 × 512 × 512 × 8 bytes = **192 MB** for the float64 coordinate arrays. Combined with the source volume (~84 MB as int16) and the output array (~256 MB as int16 at 512³), peak memory is ~530 MB — comfortable on a 5.7 GB system. 16 iterations to cover 512 slices.

### 5.4 Validation Checks

The script should report:
1. **Brain volume conservation** — compare total brain volume between source and resampled: $V_{source} = n_{source} \cdot 0.7^3$ vs $V_{resampled} = n_{resampled} \cdot \Delta x^3$, both in mm³. Should agree within ~3%. For subject 157336: $V_{source}$ = 3,883,660 × 0.343 = ~1,332 mL (consistent with a healthy adult brain, 1,200–1,500 mL).
2. **Label preservation** — set of unique labels in the resampled volume should be a subset of the source labels (no labels invented). Critical labels to verify present: 2, 41 (cerebral WM), 3, 42 (cortex), 4, 43 (lateral ventricles), 10, 49 (thalamus), 12, 51 (putamen), 16 (brainstem), 31, 63 (choroid plexus).
3. **Centering** — brain centroid in grid coordinates should be near $(N/2, N/2, N/2)$. For subject 157336, the physical centroid is at (+0.3, −21.8, +6.8) mm, so expect grid centroid near $(256.3, 234.2, 262.8)$ at dev resolution — shifted posteriorly and slightly superiorly from grid center.
4. **Bounding box margins** — distance from brain bbox to grid edges on each side. Report all 6 faces. Flag any margin < 30 mm (potential issue for skull SDF + air halo).

## 6. Reusable Resampling Utility

The coordinate mapping and slab-based resampling logic defined above is not specific to the segmentation volume. Later preprocessing steps need to resample other volumes onto the same simulation grid:

| Step | Source volume | Interpolation | Notes |
|------|--------------|:-------------:|-------|
| Task 7 (Skull SDF) | Derived from brain mask | Trilinear | SDF is continuous |
| Task 10 (Fiber texture) | `Diffusion.bedpostX/dyads{1,2,3}.nii.gz`, `mean_f{1,2,3}samples.nii.gz` | Special | Sign-consistent interpolation; see fiber texture spec |
| Visualization | `T1w_acpc_dc_restore_brain.nii.gz` | Trilinear | Optional, for overlay debugging |

All of these use the same $A_{g \to p}$ affine and the same composite-transform approach (Section 3.1). The implementation should expose a reusable function:

```python
def resample_to_grid(source_nifti_path, grid_affine, grid_shape,
                     order=0, cval=0, dtype=None, slab_size=32):
    """Resample a NIfTI volume onto the simulation grid.

    Args:
        source_nifti_path: Path to source .nii.gz
        grid_affine: 4x4 grid-to-physical affine (A_g→p)
        grid_shape: (N, N, N) output dimensions
        order: 0 = nearest-neighbor, 1 = trilinear
        cval: fill value for out-of-bounds (requires mode='constant' internally)
        dtype: output dtype (e.g. np.int16). If None, uses source dtype.
        slab_size: slices per slab for memory control

    Returns:
        numpy array of shape grid_shape with specified dtype
    """
```

This function is the shared workhorse. The domain geometry script calls it for `aparc+aseg` and `brainmask_fs`; later steps call it for their own inputs.

## 7. Design Rationale

**Why ACPC-centered rather than brain-centroid-centered?** The ACPC origin is a stable anatomical landmark that every HCP subject shares. Centering on it means the same grid coordinate always refers to the same anatomical neighborhood, which simplifies debugging and cross-subject comparison. The padding margins are so large (>150 mm minimum) that the slight asymmetry from ACPC vs. centroid centering is irrelevant.

**Why not resample to 0.7 mm (matching source)?** The simulation grid resolution is set by physics (the solver's Δx), not by the imaging resolution. Storing the material map at 0.7 mm would create a 731³ grid — larger than the 512³ the solver operates on, wasting memory and requiring a second resampling step at runtime.

**Why nearest-neighbor for labels?** Interpolating categorical data is undefined. Nearest-neighbor is the only valid method for integer labels. The 1.43× downsampling ratio is mild enough that no structures are lost.

**Why output unresampled FreeSurfer labels instead of remapped u8 indices?** Separation of concerns. This step handles only geometry and resampling. The label→u8 remapping (Task 6) is a pure lookup table operation on the already-resampled volume. Keeping them separate means you can inspect the resampled FreeSurfer labels directly in a viewer and verify spatial accuracy before the label collapse makes the volume harder to interpret.

**Why slab-based processing?** A 512³ grid with three float64 coordinate channels requires 3 GB. On a 5.7 GB system, this leaves insufficient room for the OS, Python, and the output arrays. Processing in 32-slice slabs reduces peak coordinate memory to ~192 MB (see Section 5.3) with negligible runtime cost.
