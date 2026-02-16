# Preprocessing Step 6: Fiber Orientation Texture

This document specifies the construction of the per-voxel structure tensor field M_0 from bedpostX diffusion data. M_0 encodes local white matter fiber architecture and is used by both the Darcy solver (anisotropic permeability) and the solid constitutive model (HGO anisotropic elasticity).

## 1. Purpose

The simulation needs directional information at every voxel to compute anisotropic permeability:

$$K_{tissue} = k_{iso} \cdot I + k_{fiber} \cdot M_0$$

For the axis-aligned Cartesian stencil, the face-normal projection reduces to:

$$K_{axis} = k_{iso} + k_{fiber} \cdot M_0[axis, axis]$$

where `k_iso` and `k_fiber` come from the Dual Global LUT (per material class), and `M_0` comes from this preprocessing step (per voxel).

The structure tensor is also used by the Holzapfel-Gasser-Ogden (HGO) constitutive model for anisotropic elastic response via the invariant $\bar{I}_{4\kappa}^* = \kappa \text{tr}(\bar{C}) + (1 - 3\kappa)(M_0 : \bar{C})$.

**Runtime lifecycle:** The preprocessed M_0 texture is sampled exactly once — at particle initialization (t=0). Each particle computes its M_0 by trilinear interpolation, stores it as a persistent attribute, and never reads the texture again. The texture can be freed after initialization. During simulation, particle-level M_0 tensors are rasterized to grid nodes during each P2G transfer. For the initial frame (before the first P2G), M_0 is rasterized directly from this texture to grid nodes.

## 2. Design Decision: Pre-Computed Tensor (Approach B)

The spec describes storing per-voxel fiber direction vectors {v1,v2,v3} and volume fractions {w1,w2,w3}, with M_0 computed per particle at runtime. We instead pre-compute M_0 during preprocessing and store the tensor directly. This is equivalent but solves several problems:

**Why pre-computed M_0 instead of raw vectors + weights:**

1. **Eliminates sign ambiguity.** bedpostX dyads are direction vectors, but fiber orientations are axial (v and −v are equivalent). Adjacent voxels may have arbitrary sign flips. Trilinear interpolation of sign-inconsistent vectors produces cancellation artifacts (e.g., interpolating [0,0,1] and [0,0,−1] yields [0,0,0]). Fixing this requires a non-trivial sign-consistency flood-fill. The outer product v⊗v is sign-invariant — the problem vanishes by construction.

2. **Eliminates the population correspondence problem.** bedpostX sorts fiber populations by volume fraction, not by spatial identity. "Population 2" in adjacent voxels may represent different physical tracts. Sign-fixing per population index can flip the wrong vector. M_0 = Σ w_n(v_n ⊗ v_n) folds all populations into a single tensor, making population ordering irrelevant.

3. **Correct interpolation.** Trilinear interpolation of symmetric positive semidefinite matrices is mathematically well-defined (linear combination of SPD matrices is SPD). Vector interpolation at tract boundaries produces artificial intermediate directions; tensor interpolation produces a physically correct fanned distribution.

4. **Smaller storage.** 6 floats per voxel (symmetric 3×3 upper triangle) vs 12 floats (3 vectors × 3 components + 3 weights).

5. **Simpler runtime.** Particles interpolate 6 values and store them directly. No per-particle vector renormalization, no tensor construction.

**What is lost:** Individual fiber population directions and weights cannot be recovered from M_0 without eigen-decomposition. This is acceptable — no runtime consumer needs individual populations. M_0 is the terminal representation in both the Darcy solver and the HGO model.

## 3. Inputs

All inputs are from the bedpostX directory at diffusion resolution (1.25 mm isotropic, 145 × 174 × 145).

| File | Shape | Content |
|------|-------|---------|
| `dyads1.nii.gz` | 145×174×145×3 | Primary fiber direction (unit vector) |
| `dyads2.nii.gz` | 145×174×145×3 | Second fiber direction (unit vector) |
| `dyads3.nii.gz` | 145×174×145×3 | Third fiber direction (unit vector) |
| `mean_f1samples.nii.gz` | 145×174×145 | Volume fraction of population 1 |
| `mean_f2samples.nii.gz` | 145×174×145 | Volume fraction of population 2 |
| `mean_f3samples.nii.gz` | 145×174×145 | Volume fraction of population 3 |
| `nodif_brain_mask.nii.gz` | 145×174×145 | Diffusion brain mask (binary) |

Additionally, from previous preprocessing steps:

| File | Source | Content |
|------|--------|---------|
| `aparc+aseg.nii.gz` | `data/raw/{subject_id}/T1w/` | FreeSurfer labels at 0.7 mm (for WM masking) |

**Properties of the bedpostX data (verified for subjects 157336 and 128935):**

- Diffusion affine: diagonal (−1.25, +1.25, +1.25), origin (90, −126, −72) — same physical space as T1w, same negative-X convention, different resolution
- All dyad vectors are exactly unit length (norm = 1.000000) within the brain mask
- All dyad vectors are exactly zero outside the brain mask
- No zero vectors exist inside the brain mask (bedpostX always assigns a direction; the volume fraction indicates confidence)
- Volume fractions do NOT sum to 1: mean(f1+f2+f3) = 0.316 within the brain mask. The remainder (1 − Σf) is the isotropic compartment (free water / non-directional tissue)
- f2 > 0.05 in 49.3% of brain voxels; f3 > 0.05 in 22.9%
- 32.6% of brain voxels map to anisotropic material classes (cerebral WM, cerebellar WM, brainstem)

## 4. Algorithm

### 4.1 Overview

```
bedpostX dyads + fractions → threshold → compute M_0 per voxel → WM mask → store upper triangle
```

### 4.2 Volume Fraction Thresholding

For each fiber population n ∈ {1,2,3}, if f_n < f_threshold, set f_n = 0 for that voxel. This suppresses noise in minor fiber populations.

```python
f_threshold = 0.05

f1 = nib.load(f'{bedpostx_dir}/mean_f1samples.nii.gz').get_fdata()
f2 = nib.load(f'{bedpostx_dir}/mean_f2samples.nii.gz').get_fdata()
f3 = nib.load(f'{bedpostx_dir}/mean_f3samples.nii.gz').get_fdata()

f1[f1 < f_threshold] = 0.0
f2[f2 < f_threshold] = 0.0
f3[f3 < f_threshold] = 0.0
```

After thresholding on subject 157336: ~100% of brain retains population 1 (f1 is always ≥ 0.05 in practice), ~49% retains population 2, ~23% retains population 3.

### 4.3 Structure Tensor Computation

For each voxel, compute:

$$M_0 = \sum_{n=1}^{3} f_n \cdot (v_n \otimes v_n)$$

where v_n is the unit dyad vector and f_n is the (thresholded) volume fraction.

**On normalization:** The spec defines M_0 with "normalized volume fractions (Σ w_n = 1)". We deliberately use the raw (unnormalized) bedpostX fractions instead. The reason: bedpostX fractions include an isotropic compartment (1 − Σf_n) representing free water and non-directional tissue. Normalizing would discard this information, inflating the anisotropic contribution in voxels with low fiber content. With raw fractions:

- A strongly anisotropic WM voxel (f1 = 0.7): tr(M_0) ≈ 0.7, strong directional effect
- A weakly anisotropic voxel (f1 = 0.1): tr(M_0) ≈ 0.1, weak directional effect
- An isotropic voxel (all f_n < threshold): M_0 = 0, no directional effect

This naturally couples M_0 magnitude to anisotropy strength. The permeability formula K_axis = k_iso + k_fiber · M_0[axis,axis] then correctly scales the fiber contribution by how much fiber is actually present.

The outer product v_n ⊗ v_n for a unit vector v = (a, b, c):

```
v ⊗ v = [[a², ab, ac],
          [ab, b², bc],
          [ac, bc, c²]]
```

Since this is symmetric, only 6 unique values need to be computed and stored: (M_00, M_11, M_22, M_01, M_02, M_12).

```python
d1 = nib.load(f'{bedpostx_dir}/dyads1.nii.gz').get_fdata()  # (145,174,145,3)
d2 = nib.load(f'{bedpostx_dir}/dyads2.nii.gz').get_fdata()
d3 = nib.load(f'{bedpostx_dir}/dyads3.nii.gz').get_fdata()

shape = d1.shape[:3]  # (145, 174, 145)
M0 = np.zeros(shape + (6,), dtype=np.float32)

# Indices into the 6-component upper triangle: [00, 11, 22, 01, 02, 12]
# For a vector v = (a,b,c): outer product upper triangle = [a², b², c², ab, ac, bc]
def add_outer(M0, dyad, frac):
    """Accumulate weighted outer product into M0 upper triangle.
    frac is shape (I,J,K), dyad is shape (I,J,K,3). All products broadcast naturally.
    """
    a, b, c = dyad[..., 0], dyad[..., 1], dyad[..., 2]
    M0[..., 0] += frac * a * a  # M_00
    M0[..., 1] += frac * b * b  # M_11
    M0[..., 2] += frac * c * c  # M_22
    M0[..., 3] += frac * a * b  # M_01
    M0[..., 4] += frac * a * c  # M_02
    M0[..., 5] += frac * b * c  # M_12

add_outer(M0, d1, f1)
add_outer(M0, d2, f2)
add_outer(M0, d3, f3)
```

### 4.4 White Matter Masking

Zero out M_0 for voxels that don't belong to anisotropic material classes. The anisotropic classes (those with K_fiber > 0 in the material map) are:

| u8 | Class | K_fiber |
|---:|-------|:-------:|
| 1 | Cerebral White Matter | > 0 |
| 4 | Cerebellar White Matter | > 0 |
| 6 | Brainstem | > 0 |

All other classes have K_fiber = 0, so M_0 has no effect on their permeability or elastic response at runtime. Zeroing them during preprocessing avoids storing noise and prevents the P2G rasterization from smearing fiber signal from WM into adjacent gray matter at tissue boundaries.

**Cross-resolution masking:** The FS labels are at 0.7 mm (260×311×260) and the bedpostX data is at 1.25 mm (145×174×145). Both share the same physical coordinate system (same origin, same axis convention). For each bedpostX voxel, find the nearest FS label voxel:

```python
fs_img = nib.load(f'{t1w_dir}/aparc+aseg.nii.gz')
fs_data = np.round(fs_img.get_fdata()).astype(np.int16)
fs_affine = fs_img.affine

diff_affine = nib.load(f'{bedpostx_dir}/dyads1.nii.gz').affine

# Composite transform: bedpostX voxel → physical → FS voxel
M_diff_to_fs = np.linalg.inv(fs_affine) @ diff_affine

# Build bedpostX voxel coordinate grid
di, dj, dk = np.meshgrid(
    np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
    indexing='ij'
)

# Transform to FS voxel coordinates
fi = M_diff_to_fs[0,0]*di + M_diff_to_fs[0,1]*dj + M_diff_to_fs[0,2]*dk + M_diff_to_fs[0,3]
fj = M_diff_to_fs[1,0]*di + M_diff_to_fs[1,1]*dj + M_diff_to_fs[1,2]*dk + M_diff_to_fs[1,3]
fk = M_diff_to_fs[2,0]*di + M_diff_to_fs[2,1]*dj + M_diff_to_fs[2,2]*dk + M_diff_to_fs[2,3]

# Nearest-neighbor lookup
fi = np.clip(np.round(fi).astype(int), 0, fs_data.shape[0] - 1)
fj = np.clip(np.round(fj).astype(int), 0, fs_data.shape[1] - 1)
fk = np.clip(np.round(fk).astype(int), 0, fs_data.shape[2] - 1)

fs_at_diff = fs_data[fi, fj, fk]
```

Then apply the mask:

```python
# FS labels that map to anisotropic material classes
aniso_labels = {
    # u8=1: Cerebral WM
    2, 41, 77, 78, 79, 85, 192, 250, 251, 252, 253, 254, 255,
    # u8=4: Cerebellar WM
    7, 46,
    # u8=6: Brainstem
    16, 75, 76,
}

is_aniso = np.isin(fs_at_diff, list(aniso_labels))
M0[~is_aniso] = 0.0
```

For subject 157336, this zeros out 67.4% of brain voxels (gray matter, CSF, deep gray), retaining M_0 in the 32.6% that are white matter or brainstem.

**Belt and suspenders:** This masking is technically redundant with the runtime LUT (non-WM classes have K_fiber = 0, so M_0 is multiplied by zero anyway). But it serves three purposes: (a) prevents P2G rasterization from smearing WM fiber signal into adjacent GM, (b) avoids storing noise in the texture, (c) makes the validation visualizations clean.

### 4.5 Storage Layout

The 6 components are stored in upper-triangle order: [M_00, M_11, M_22, M_01, M_02, M_12].

This order places the three diagonal elements first (indices 0, 1, 2). This is deliberate: the Darcy solver only reads M_0[axis,axis], so the diagonals should be in the first three channels for cache-friendly access when only the axis-projected permeability is needed.

## 5. Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| f_threshold | 0.05 | Suppresses noise in minor populations. bedpostX reports nonzero f2/f3 everywhere; values below 0.05 are below the detection floor. Standard threshold used in FSL's own visualization tools. |

Exposed as a command-line argument:

```
--f-threshold 0.05
```

### 5.1 Sensitivity

**f_threshold:** Values from 0.03 to 0.10 produce similar results. Below 0.03, noise from minor populations creates spurious off-diagonal M_0 entries in gray matter boundary voxels (though these are zeroed by the WM mask). Above 0.10, genuine secondary fiber populations in crossing-fiber regions (e.g., corona radiata / superior longitudinal fasciculus intersection) begin to be suppressed. The default of 0.05 is the conventional choice in the FSL ecosystem and matches the threshold used for `dyads2_thr0.05.nii.gz` already present in the bedpostX output directory.

## 6. Outputs

### 6.1 Structure Tensor Volume

| Property | Value |
|----------|-------|
| File | `fiber_M0.nii.gz` |
| Shape | 145 × 174 × 145 × 6 (matching bedpostX input dimensions) |
| Dtype | `float32` |
| Affine | Diffusion affine (−1.25, +1.25, +1.25, origin 90/−126/−72) |
| Channel order | [M_00, M_11, M_22, M_01, M_02, M_12] |
| Units | Dimensionless (elements of a structure tensor with tr(M_0) ≤ 1) |

Saved to `data/processed/{subject_id}/` (profile-independent — the texture is at native diffusion resolution and does not depend on the simulation grid spacing).

### 6.2 Size Estimate

145 × 174 × 145 × 6 × 4 bytes = **87.5 MB** uncompressed. After NIfTI gzip compression (the tensor field is smooth within WM and zero elsewhere), expected file size: **15–30 MB**.

### 6.3 Why Not Resample to Simulation Resolution

The texture is stored at native bedpostX resolution (1.25 mm) rather than at the simulation grid (e.g., 512³ at 1.0 mm). Reasons:

1. **No information gain.** Diffusion MRI fundamentally has no sub-1.25 mm directional information. Upsampling produces smoother interpolation of the same data.
2. **Profile-independent.** The same texture works for debug, dev, and prod profiles without regeneration.
3. **88 MB vs ~1.3 GB.** At 512³ × 6 × 4 bytes, the upsampled version would be 3.2 GB dense (~1.3 GB within brain mask). Storing the same information 15× is wasteful.
4. **Trilinear interpolation at runtime is trivial.** The texture is sampled once per particle at t=0 (a few seconds on GPU). The one-time interpolation cost is negligible.

The runtime loads this texture into a GPU 3D texture object and samples it using the diffusion affine to map simulation-space coordinates to texture coordinates.

## 7. Implementation

### 7.1 Stack

Python 3, nibabel (NIfTI I/O), numpy (array math). No scipy dependency. No additional packages beyond what the other preprocessing steps already use.

### 7.2 Algorithm Summary

```
Input:  subject_id, f_threshold (default 0.05)
Output: fiber_M0.nii.gz

1. Load bedpostX files:
   dyads{1,2,3}.nii.gz → d1, d2, d3   [each (145,174,145,3), float32]
   mean_f{1,2,3}samples.nii.gz → f1, f2, f3   [each (145,174,145), float32]
   Read diffusion affine from dyads1 header

2. Threshold volume fractions:
   f1[f1 < f_threshold] = 0.0
   f2[f2 < f_threshold] = 0.0
   f3[f3 < f_threshold] = 0.0

3. Compute M_0 upper triangle:
   M0 = zeros(145, 174, 145, 6)
   For each population n in {1,2,3}:
     M0[...,0] += f_n * d_n[...,0] * d_n[...,0]   # M_00
     M0[...,1] += f_n * d_n[...,1] * d_n[...,1]   # M_11
     M0[...,2] += f_n * d_n[...,2] * d_n[...,2]   # M_22
     M0[...,3] += f_n * d_n[...,0] * d_n[...,1]   # M_01
     M0[...,4] += f_n * d_n[...,0] * d_n[...,2]   # M_02
     M0[...,5] += f_n * d_n[...,1] * d_n[...,2]   # M_12

4. White matter masking:
   Load aparc+aseg.nii.gz, read FS affine
   Compute composite transform: bedpostX voxel → FS voxel
   For each bedpostX voxel, look up nearest FS label
   Zero M0 where FS label ∉ {anisotropic classes}

5. Save:
   fiber_M0.nii.gz ← M0 as float32 with diffusion affine
```

### 7.3 Memory Analysis

| Array | Shape | Dtype | Size |
|-------|-------|-------|------|
| d1, d2, d3 | (145,174,145,3) each | float32 | 3 × 43.7 MB = 131 MB |
| f1, f2, f3 | (145,174,145) each | float32 | 3 × 14.6 MB = 44 MB |
| M0 | (145,174,145,6) | float32 | 87.5 MB |
| fs_data | (260,311,260) | int16 | 42 MB |
| Coordinate arrays (fi,fj,fk) | (145,174,145) each | int64 | 3 × 29.1 MB = 87 MB |

Peak: ~392 MB. Comfortable on any modern system.

The dyad arrays can be freed after M_0 computation (before the masking step), reducing peak memory to ~305 MB. This optimization is optional given the small total.

### 7.4 Coordinate System Note

The bedpostX diffusion affine has the same axis convention as the T1w structural affine: negative X (LAS voxel storage), same origin (90, −126, −72). The composite transform M_diff_to_fs = inv(fs_affine) @ diff_affine absorbs the scaling difference (1.25 mm → 0.7 mm) automatically. No manual axis flipping is needed.

At runtime, the simulation grid uses RAS+ voxel ordering (positive X). The runtime must use the diffusion affine (not the simulation grid affine) when sampling this texture — the composite transform simulation_grid_voxel → physical → diffusion_voxel maps correctly regardless of the voxel ordering conventions, just as it does for all other cross-resolution lookups (Section 3.1 of the domain geometry spec).

**Vector coordinate system:** The bedpostX dyad vectors are stored in NIfTI physical coordinates (RAS+ mm-space), not in voxel-index space. The negative X scaling in the diffusion affine affects only the voxel-to-physical mapping, not the orientation of the stored vectors. Since the simulation also operates in RAS+ physical space, the M_0 tensor components computed from these vectors are directly compatible with the runtime deformation gradient F and derived tensors (C̄, b) — no rotation is needed. This is critical for the HGO model's double contraction M_0 : C̄, where both tensors must share the same coordinate frame. (The Darcy solver uses only diagonal elements M_0[axis,axis], which are invariant to axis sign flips regardless.)

## 8. Validation

### 8.1 Trace Statistics

The trace of M_0 equals the sum of the (thresholded) volume fractions: tr(M_0) = Σ f_n. Report statistics within the WM mask:

```python
trace = M0[..., 0] + M0[..., 1] + M0[..., 2]
wm_trace = trace[is_aniso & (trace > 0)]
print(f"M_0 trace in WM: mean={wm_trace.mean():.3f}, "
      f"median={np.median(wm_trace):.3f}, "
      f"p5={np.percentile(wm_trace, 5):.3f}, "
      f"p95={np.percentile(wm_trace, 95):.3f}")
```

Expected: mean ≈ 0.3–0.5 in cerebral WM (reflecting the partial volume of the anisotropic compartment). Values above 1.0 indicate a bug (impossible if dyads are unit vectors and fractions are ≤ 1). Values near 0 in WM indicate excessive thresholding or masking errors.

### 8.2 Positive Semi-Definiteness

M_0 = Σ f_n(v_n ⊗ v_n) is guaranteed PSD by construction (sum of PSD matrices with non-negative weights). Verify on a random sample:

```python
rng = np.random.default_rng(42)
sample_idx = np.argwhere(is_aniso & (trace > 0))
sample_idx = sample_idx[rng.choice(len(sample_idx), size=min(10000, len(sample_idx)), replace=False)]

n_negative = 0
for idx in sample_idx:
    i, j, k = idx
    m = M0[i, j, k]
    mat = np.array([[m[0], m[3], m[4]],
                    [m[3], m[1], m[5]],
                    [m[4], m[5], m[2]]])
    eigvals = np.linalg.eigvalsh(mat)
    if eigvals.min() < -1e-7:
        n_negative += 1

print(f"PSD check: {n_negative}/{len(sample_idx)} sampled voxels have negative eigenvalues")
```

Expected: 0 failures. Any failure indicates corrupted input data or a computation bug.

### 8.3 Spatial Smoothness

M_0 should vary smoothly within white matter tracts. Compute the voxel-to-voxel gradient magnitude of the dominant diagonal component along each axis:

```python
for ch, name in [(0, 'M_00 (X²)'), (1, 'M_11 (Y²)'), (2, 'M_22 (Z²)')]:
    grad = np.gradient(M0[..., ch])
    grad_mag = np.sqrt(sum(g**2 for g in grad))
    wm_grad = grad_mag[is_aniso & (trace > 0)]
    print(f"  {name}: mean |∇| = {wm_grad.mean():.4f}, "
          f"max |∇| = {wm_grad.max():.4f}")
```

Large gradient magnitudes (> 0.5 per voxel) at WM interiors (not boundaries) suggest artifacts. Some high gradients are expected at WM/GM boundaries due to the masking step.

### 8.4 Zero Check Outside WM

Verify that all non-anisotropic voxels have M_0 = 0:

```python
non_aniso_nonzero = np.count_nonzero(np.any(M0[~is_aniso] != 0, axis=-1))
print(f"Non-anisotropic voxels with nonzero M_0: {non_aniso_nonzero} (expected: 0)")
```

### 8.5 Volume Fraction Coverage

Report how many voxels retain each fiber population after thresholding:

```python
brain_mask = nib.load(f'{bedpostx_dir}/nodif_brain_mask.nii.gz').get_fdata() > 0
n_brain = brain_mask.sum()
print(f"Brain voxels: {n_brain:,}")
print(f"  f1 > threshold: {(f1[brain_mask] > 0).sum():,} ({(f1[brain_mask] > 0).sum()/n_brain*100:.1f}%)")
print(f"  f2 > threshold: {(f2[brain_mask] > 0).sum():,} ({(f2[brain_mask] > 0).sum()/n_brain*100:.1f}%)")
print(f"  f3 > threshold: {(f3[brain_mask] > 0).sum():,} ({(f3[brain_mask] > 0).sum()/n_brain*100:.1f}%)")
print(f"Anisotropic voxels (WM-masked): {is_aniso.sum():,} ({is_aniso.sum()/n_brain*100:.1f}%)")
```

### 8.6 Principal Direction Sanity

For a qualitative check, extract the principal eigenvector of M_0 at a few known anatomical landmarks and verify it matches expected fiber orientations:

| Location | Expected principal direction |
|----------|------------------------------|
| Corpus callosum (midsagittal) | Left-right (X-dominant) |
| Internal capsule | Superior-inferior (Z-dominant) |
| Cingulum bundle | Anterior-posterior (Y-dominant) |

```python
# Corpus callosum: approximately at bedpostX voxel (72, 100, 72)
# (midline, slightly anterior to center, at CC level)
cc_voxel = M0[72, 100, 72]
mat = np.array([[cc_voxel[0], cc_voxel[3], cc_voxel[4]],
                [cc_voxel[3], cc_voxel[1], cc_voxel[5]],
                [cc_voxel[4], cc_voxel[5], cc_voxel[2]]])
eigvals, eigvecs = np.linalg.eigh(mat)
principal = eigvecs[:, -1]  # eigenvector with largest eigenvalue
print(f"CC principal direction: {principal} (expect X-dominant)")
```

These checks are approximate — the exact voxel coordinates depend on the subject's anatomy. The test should verify that the dominant component matches expectation (e.g., |principal[0]| > 0.7 for the CC).

## 9. Design Rationale

**Why use bedpostX instead of running CSD from scratch?** The spec describes a CSD + peak extraction pipeline (Section 1.2). bedpostX has already performed a functionally equivalent analysis — Bayesian estimation of up to 3 crossing fiber populations with volume fractions (ball-and-stick model, distinct from CSD's spherical deconvolution, but producing the same output format). The bedpostX outputs (dyads + mean_f) are the same mathematical objects the spec's pipeline would produce: direction vectors and volume fractions for up to 3 fiber populations. Re-running CSD would duplicate work and require additional dependencies (MRtrix3 or equivalent) with no benefit.

**Why unnormalized volume fractions?** See Section 4.3. The bedpostX fractions encode both direction and anisotropy strength. Normalizing to Σw_n = 1 would discard the isotropic compartment information, making a nearly-isotropic voxel (f1 = 0.05) look identical to a strongly anisotropic one (f1 = 0.8) in terms of M_0 magnitude. The runtime formulas K_axis = k_iso + k_fiber · M_0[axis,axis] and the HGO invariant both scale correctly with unnormalized fractions. The spec's "normalized" wording describes the conceptual definition; the implementation adapts to the actual data source.

**Why mask at bedpostX resolution instead of simulation resolution?** The texture is stored at 1.25 mm. Masking at this resolution is simpler (single nearest-neighbor lookup per voxel) and produces clean output. The alternative — masking at runtime after trilinear interpolation — would require the runtime to know about FS labels, adding coupling. The slight resolution mismatch at WM/GM boundaries (a 1.25 mm voxel might span both WM and GM at 0.7 mm) is acceptable: at boundaries, the masking is conservative (a WM voxel might include some GM partial volume, giving a slightly lower M_0 than pure WM — this is physically reasonable for a transition zone).

**Why store as NIfTI instead of a raw binary?** Consistency with all other preprocessing outputs. NIfTI carries the affine transform, making coordinate mapping unambiguous. Any neuroimaging viewer (FSLeyes, freeview, 3D Slicer) can open the file for visual inspection. The runtime loads it into a GPU texture using the embedded affine.

**Why not store at simulation resolution?** See Section 6.3.

**Why profile-independent output?** Unlike the material map (which depends on Δx via resampling), the fiber texture is stored at native diffusion resolution and sampled at runtime via the coordinate transform. Changing the simulation profile (debug → dev → prod) changes Δx and N, but the fiber texture remains the same. This avoids regenerating a large file when switching profiles and ensures consistency.

## 10. Main Spec Amendments Required

Approach B (pre-computed M_0 tensor) changes the runtime behavior described in several passages of `spec.md`. These need updating once this preprocessing spec is accepted:

1. **spec.md Section 3.2 (line ~885–887):** Currently defines M_0 with "normalized volume fractions ($\sum w_n = 1$)". Should note that the preprocessing implementation uses unnormalized bedpostX fractions (see Section 4.3 of this document for rationale). The mathematical definition of M_0 is unchanged; only the weighting convention differs.

2. **spec.md Phase 2.1 (lines ~1092–1094):** Currently says particles retrieve "the set of dominant fiber vectors {v1, v2, v3} and their corresponding volume weights {w1, w2, w3}" and then "compress into a local structural tensor M_0." With Approach B, particles instead retrieve the pre-computed M_0 directly (6 components via trilinear interpolation of the fiber texture). No per-particle vector retrieval or tensor construction occurs at runtime.

3. **spec.md Phase 1.2 (lines ~1071–1080):** Describes a CSD + peak extraction + continuous field pipeline. The preprocessing implementation uses bedpostX outputs instead (functionally equivalent; see Section 9). The "Step C: Continuous Field Interpolation" description of storing vectors in a 3D texture should be updated to reflect that M_0 tensor components are stored instead.
