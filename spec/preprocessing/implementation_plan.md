# Preprocessing Pipeline Implementation Plan

## Context

The preprocessing specification (7 topic documents) and the main simulation spec (`spec/spec.md`) are located in `spec/`. No code exists yet. This plan implements the full pipeline: 7 scripts + shared utilities, targeting subject 157336 on the debug profile (256³, 2mm) first for fast iteration, then verifying on dev (512³, 1mm). Each step gets its own commit.

## Project Structure

```
preprocessing/
  utils.py                     # Shared: resample_to_grid, build_ball, affine construction, profile configs
  domain_geometry.py           # → fs_labels_resampled.nii.gz, brain_mask.nii.gz, grid_meta.json
  material_map.py              # → material_map.nii.gz
  skull_sdf.py                 # → skull_sdf.nii.gz
  subarachnoid_csf.py          # → material_map.nii.gz (updated)
  dural_membrane.py            # → material_map.nii.gz (updated)
  fiber_orientation.py         # → fiber_M0.nii.gz
  validation.py                # → validation_report.json, fig1-5.png
  run_all.py                   # Orchestrator: runs steps 1-7 in sequence
requirements.txt               # nibabel, scipy, numpy, matplotlib
```

Each script is independently runnable: `python preprocessing/domain_geometry.py --subject 157336 --profile debug`

Output directory: `data/processed/157336/debug/` (and `data/processed/157336/` for profile-independent fiber texture).

## Dependencies

`requirements.txt`: numpy, scipy, nibabel, matplotlib

## Implementation Order (8 tasks, sequential)

Each task follows the same workflow: **implement → run on debug → check output values against spec → fix if needed → commit → next step**.

### Task 1: Project scaffolding + shared utilities
**Files:** `requirements.txt`, `preprocessing/utils.py`

`utils.py` contains:
- **Profile configs:** dict mapping profile name → (N, dx_mm). debug=(256, 2.0), dev=(512, 1.0), prod=(512, 0.5)
- **`build_grid_affine(N, dx_mm)`:** Returns the 4×4 grid-to-physical affine A_g→p (RAS+, ACPC-centered)
- **`resample_to_grid(source, grid_affine, grid_shape, order, cval, dtype, slab_size)`:** The shared resampling workhorse. `source` accepts either a file path (str/Path) or a `(ndarray, affine)` tuple. Slab-based processing (default slab=32) for memory control. Uses `scipy.ndimage.map_coordinates` with `mode='constant'`.
- **`build_ball(radius_vox)`:** Builds a spherical boolean structuring element of given voxel radius
- **Path helpers:** `raw_dir(subject_id)`, `processed_dir(subject_id, profile)` returning Path objects

**Verify:** Import utils, construct debug affine, confirm grid center maps to physical (0,0,0).

### Task 2: Step 1 — Domain Geometry
**Spec:** `spec/domain_geometry.md`
**Files:** `preprocessing/domain_geometry.py`

Algorithm (from spec Section 5.2):
1. Load `aparc+aseg.nii.gz` → int16, `brainmask_fs.nii.gz` → uint8
2. Verify source affine diagonal ≈ (-0.7, +0.7, +0.7)
3. Construct grid affine via `build_grid_affine(N, dx_mm)`
4. Resample both volumes via `resample_to_grid` (order=0, nearest-neighbor, mode='constant', cval=0)
5. Compute brain bounding box from resampled mask
6. Save: `fs_labels_resampled.nii.gz` (int16), `brain_mask.nii.gz` (uint8), `grid_meta.json`
7. Print validation: brain volume conservation, label preservation, centering, bbox margins

CLI: `--subject`, `--profile` (or `--dx` + `--grid-size`)

**Verify:** Open `fs_labels_resampled.nii.gz` — labels should be in [0, 2035], brain volume ~1332 mL. Grid center should be near ACPC origin.

### Task 3: Step 2 — Material Map (Label Remapping)
**Spec:** `spec/material_map.md`
**Files:** `preprocessing/material_map.py`

Algorithm:
1. Load `fs_labels_resampled.nii.gz` (int16)
2. Build lookup array: int16 → uint8 (size 2036, covers all FS labels including ctx-lh/rh 1001-1035, 2001-2035)
3. Apply mapping vectorized: `material_map = lut[fs_labels]`
4. Log warnings for any fallback labels encountered
5. Save: `material_map.nii.gz` (uint8, same affine)
6. Print census: voxel count per u8 class

The full mapping table from the spec (Section "Complete Label Mapping") is hardcoded as a dict.

**Verify:** u8 values in {0..11}, no 255. Brain tissue classes present. Census volumes physiologically plausible.

### Task 4: Step 3 — Skull SDF
**Spec:** `spec/skull_sdf.md`
**Files:** `preprocessing/skull_sdf.py`

Algorithm (from spec Section 7.2):
1. Load `brainmask_fs.nii.gz` + `Head.nii.gz` at source resolution (0.7mm)
2. Inferior padding: pad_z = r_close_vox + r_dilate_vox + 10 on axis 2, adjust affine
3. Morphological closing: iterated small ball (r_step=3), n_iter=ceil(r_close_vox/r_step)
4. Outward dilation: same approach, n_iter=ceil(r_dilate_vox/r_step)
5. Head mask intersection
6. Signed EDT: dt_outside - dt_inside (sampling=voxel_size for mm units)
7. Resample to simulation grid via `resample_to_grid((sdf, A_padded), ...)` with order=1, cval=100.0
8. Save: `skull_sdf.nii.gz` (float32)
9. Print validation: ICV volume, brain containment, margin at surface, gradient magnitude

CLI: `--subject`, `--profile`, `--close-radius` (default 10.0), `--dilate-radius` (default 4.0)

**Verify:** SDF < 0 everywhere brain mask is 1. ICV ~1500-1750 mL. Gradient magnitude ≈ 1.0.

### Task 5: Step 4 — Subarachnoid CSF
**Spec:** `spec/subarachnoid_csf.md`
**Files:** `preprocessing/subarachnoid_csf.py`

Algorithm (trivially simple — the spec is detailed but the code is ~30 lines of logic):
1. Load `material_map.nii.gz`, `skull_sdf.nii.gz`, `brain_mask.nii.gz`, `grid_meta.json`
2. Sulcal CSF: `(brain == 1) & (mat == 0)` → paint u8=8
3. Shell CSF: `(sdf < 0) & (brain == 0) & (mat == 0)` → paint u8=8
4. Validate: domain closure `(sdf < 0) & (mat == 0)` must be empty
5. Save updated `material_map.nii.gz`
6. Print: population volumes, material census, topology checks

**Verify:** Zero vacuum inside skull (D1 invariant). Sulcal CSF ~100-200 mL, shell ~60-150 mL.

### Task 6: Step 5 — Dural Membrane
**Spec:** `spec/dural_membrane.md`
**Files:** `preprocessing/dural_membrane.py`

Algorithm (most complex step, ~200 lines):

Phase 1 — Falx:
1. Load `fs_labels_resampled.nii.gz` → classify left/right hemisphere labels
2. Compute CC superior boundary per coronal slice (FS labels 192, 251-255)
3. Free fs_labels
4. EDT left, EDT right (cast to float32 after each)
5. Watershed: `|dist_left - dist_right| <= dx_mm`
6. CSF constraint: `& (mat == 8)`
7. CC inferior boundary: exclude falx at/below CC per y-slice
8. Free EDTs + hemisphere masks

Phase 2 — Tentorium:
9. cerebral_mask = mat ∈ {1,2,3,9}, cerebellar_mask = mat ∈ {4,5}
10. EDT cerebral, EDT cerebellar
11. Watershed + CSF constraint
12. Brainstem notch exclusion: dilate (mat==6) by R_notch=5mm
13. Free EDTs + tissue masks

Phase 3 — Merge:
14. `mat[falx | tent] = 10`
15. Save updated `material_map.nii.gz`
16. Print: volumes, connectivity, notch patency

CLI: `--subject`, `--profile`, `--watershed-threshold` (default 1.0), `--notch-radius` (default 5.0)

**Note (debug profile):** At 2mm resolution, the falx may be absent or discontinuous (fissure only 1-3 voxels wide). The spec acknowledges this. Don't treat zero dural voxels at debug resolution as a failure.

**Verify:** Falx and tentorium each form ~1 connected component (at dev resolution). Tentorial notch open (CSF adjacent to brainstem).

### Task 7: Step 6 — Fiber Orientation
**Spec:** `spec/fiber_orientation.md`
**Files:** `preprocessing/fiber_orientation.py`

Algorithm:
1. Load bedpostX: dyads{1,2,3}, mean_f{1,2,3}samples
2. Threshold: f_n < 0.05 → 0.0
3. Compute M_0 upper triangle [M_00, M_11, M_22, M_01, M_02, M_12] = Σ f_n * (v_n ⊗ v_n)
4. WM masking: composite transform bedpostX voxel → FS voxel, nearest-neighbor lookup, zero non-anisotropic voxels
5. Save: `fiber_M0.nii.gz` (float32, shape 145×174×145×6, diffusion affine)
6. Print: trace stats, PSD check, coverage, principal direction landmarks

Output is **profile-independent** → saved to `data/processed/{subject_id}/` (no profile subdirectory).

CLI: `--subject`, `--f-threshold` (default 0.05)

**Verify:** tr(M_0) mean ~0.3-0.5 in WM. Zero outside WM mask. CC principal direction is X-dominant.

### Task 8: Step 7 — Validation + Orchestrator
**Spec:** `spec/validation.md`
**Files:** `preprocessing/validation.py`, `preprocessing/run_all.py`

The validation script runs all cross-cutting checks (35 checks across 6 modules) and generates 5 diagnostic figures. Implementation follows the spec's phase-based memory management.

The orchestrator `run_all.py` simply calls steps 1-7 in sequence with consistent CLI args:
```
python preprocessing/run_all.py --subject 157336 --profile debug
```

**Verify:** All CRITICAL checks pass. Figures show correct anatomy. validation_report.json is well-formed.

## Development Strategy

1. **Debug profile first** (256³, 2mm) for all steps — each step runs in seconds
2. After full pipeline works on debug, run on **dev profile** (512³, 1mm) — each step takes minutes
3. Compare dev output values against spec's expected ranges (documented for subject 157336)
4. Commit after each working step

## Known Edge Cases

- **Debug resolution dural membranes:** May be absent/discontinuous at 2mm. Not a bug — spec documents this.
- **`resample_to_grid` dual interface:** Must handle both file path and (array, affine) tuple as first arg (needed by skull SDF step).
- **FS label 255 vs u8 255:** FreeSurfer label 255 (CC_Anterior) maps to u8=1 (cerebral WM). No conflict — u8=255 (air halo) is runtime-only.
- **`map_coordinates` mode:** Must use `mode='constant'` explicitly — scipy defaults to `mode='reflect'` which would silently mirror brain labels into padding.
