# Dural Membrane Holes — Diagnostic Report

**Date:** 2026-02-17
**Subject:** 157336, dev profile (512³, 1.0mm)
**Affected step:** `preprocessing/dural_membrane.py`

## Summary

The falx cerebri has ~30% holes by fissure area. Left and right subarachnoid CSF are connected through these holes above the corpus callosum, undermining the membrane's purpose as a 10⁹ permeability barrier. The tentorium is similarly sparse at the cerebrum–cerebellum boundary.

## Evidence

### Barrier integrity test

Supratentorial CSF above the CC was split into left (x < 256) and right (x ≥ 256). Connected component analysis found **3 bridging components** crossing the midline through falx holes:

| Component | Size (voxels) | y-range | z-range |
|-----------|:------------:|---------|---------|
| #1 | 65,089 | 162–315 | 288–330 |
| #122 | 74,613 | 172–300 | 288–331 |
| #59 | 671 | 214–259 | 298–311 |

Essentially all left and right CSF is connected. The falx does not compartmentalize.

### Fissure coverage

Interhemispheric fissure defined as (y,z) positions with CSF or dural in x=[251:262]:

- **Total fissure positions:** 12,822
- **Sealed by dural:** 8,983 (70.1%)
- **CSF-only holes:** 3,839 (29.9%)

MIP across x-slices (does the falx exist at ANY nearby x?):

| x-band | Slices | Dural | Coverage |
|--------|:------:|:-----:|:--------:|
| [256:257] | 1 | 3,526 | 40.3% |
| [255:258] | 3 | 7,396 | 65.6% |
| [254:259] | 5 | 8,706 | 72.4% |
| [253:260] | 7 | 8,868 | 71.9% |

Even across 7 x-slices, 28% of fissure positions have no dural voxel at all. The holes are real 3D gaps, not single-slice artifacts.

### Hole sizes

87 distinct holes. Largest: 2,485 mm². Top 5:

| Hole | Area (mm²) |
|------|:----------:|
| #1 | 2,485 |
| #2 | 534 |
| #3 | 98 |
| #4 | 97 |
| #5 | 78 |

### Coronal gap profile

Gap rate varies by coronal slice. Mid-to-posterior fissure (y=223–283) has 40–45% gaps; anterior (y < 193) and far posterior (y > 293) are 3–10%:

| y-slice | Fissure z-cols | Sealed | Gaps | Gap % |
|---------|:--------------:|:------:|:----:|:-----:|
| 173 | 93 | 60 | 33 | 35% |
| 213 | 93 | 73 | 20 | 22% |
| 233 | 61 | 36 | 25 | 41% |
| 263 | 69 | 38 | 31 | 45% |
| 293 | 79 | 77 | 2 | 3% |

### Tentorium

At the tentorial level (z=228–240), dural voxel counts per axial slice are 250–950 against 4,000–5,500 CSF voxels. The notch zoom at z=231 shows tentorium only on the far lateral edges with no coverage medially.

## Root Causes

### 1. `_FALX_MAX_HEMI_DIST = 15mm` rejects valid deep-fissure voxels

35% of gap positions fail this guard. The 15mm limit was intended to filter spurious watershed matches in open posterior fossa CSF, but it also rejects the deep inferior interhemispheric fissure where the brain curves away from the midline.

Smoking gun at y=231, z=204–211:

```
z=204: diff=0.05mm (perfect equidistance), dist_L=27.9, dist_R=28.0 → REJECTED
z=205: diff=0.02mm (perfect equidistance), dist_L=27.5, dist_R=27.5 → REJECTED
z=206: diff=0.02mm (perfect equidistance), dist_L=27.0, dist_R=27.0 → REJECTED
```

These voxels sit exactly on the equidistant plane but get killed because hemisphere distances are 25–28mm in the deep fissure.

### 2. `_cap_runs(axis=0, max_thick=2)` over-thins the membrane

51.5% of gap positions (1,977 of 3,839) had at least one CSF voxel with diff ≤ 1mm — the watershed should have selected them. They were removed by either `_cap_runs` or the CC boundary constraint.

`_cap_runs` runs along the x-axis at each (y,z) and keeps at most 2 contiguous voxels. When the watershed selects 3+ x-positions (common — the fissure is 3–6 voxels wide and the equidistant surface has sub-voxel wobble), it trims to 2. The surviving 2 voxels may not include x=256 at some (y,z) positions, creating apparent holes in single-slice views. More importantly, at positions where the fissure-width guard or CC constraint already reduced candidates to 1–2, `_cap_runs` can eliminate the last survivors.

### 3. Watershed threshold is not the bottleneck

Increasing threshold yields diminishing returns:

| Threshold | Coverage |
|:---------:|:--------:|
| 1.0mm | 74.5% |
| 1.5mm | 75.1% |
| 2.0mm | 76.2% |
| 3.0mm | 78.6% |
| 5.0mm | 83.2% |

The equidistant surface at x=256 has median diff = 1.92mm — almost double the 1mm threshold. But even at 5mm threshold, coverage saturates at 83% because the hemi guard and cap_runs are the binding constraints.

### 4. Asymmetric cortical folding shifts the watershed off-midline

The watershed surface is not a flat plane. Asymmetric sulcal patterns between left and right medial walls cause the equidistant surface to wobble in x. At x=256, only 16% of fissure CSF passes the watershed; at x=257, 19%. The falx is spread across x=254–258 rather than concentrated at the midline.

### Material at gap locations (x=256)

| Material | Count | % |
|----------|:-----:|:-:|
| Subarachnoid CSF (u8=8) | 2,489 | 64.8% |
| Vacuum (u8=0) | 627 | 16.3% |
| Cortical GM (u8=2) | 250 | 6.5% |
| Cerebellar Cortex (u8=5) | 189 | 4.9% |
| Brainstem (u8=6) | 143 | 3.7% |
| Cerebral WM (u8=1) | 71 | 1.8% |
| Ventricular CSF (u8=7) | 61 | 1.6% |
| Deep GM (u8=3) | 9 | 0.2% |

65% of gaps are pure CSF — genuine pressure leaks, not tissue-sealed gaps.

## Validation Checks That Missed This

The validation report shows PASS for all checks:

- **C2 (falx largest component > 90%):** 17 components, largest = 99.6%. **This checks connectivity, not hole coverage.** A Swiss-cheese sheet is one connected component but doesn't seal the compartment.
- **C3 (full dural largest component > 90%):** 81 components, largest = 98.5%. Same issue.
- **V4 (dural volume 2–50 mL):** 22.0 mL — within bounds but the volume doesn't reveal that 30% of the fissure is unsealed.

**Missing check:** No validation tests whether the falx actually separates left from right CSF. The CSF bridging test (connected component analysis of supratentorial CSF above the CC) should be added.

## Diagnostic Scripts

Two diagnostic scripts were written in the project root:

- `diagnose_falx.py` — barrier integrity, hole analysis, CSF bridging, material breakdown
- `diagnose_falx2.py` — recomputes EDTs, analyzes watershed diff at gap locations, threshold sensitivity

Both load `data/processed/157336/dev/material_map.nii.gz` and `fs_labels_resampled.nii.gz`.

## Suggested Fixes

1. **Remove or raise `_FALX_MAX_HEMI_DIST`** from 15mm to ≥30mm (or remove entirely). The 15mm cutoff kills the entire deep inferior fissure. The CSF constraint (`mat == 8`) already limits the falx to the fissure — the hemi guard is redundant and harmful.

2. **Reconsider `_cap_runs`**. The spec calls for 1–2 voxel thickness, but aggressive per-column trimming creates holes. Options:
   - Remove `_cap_runs` entirely and rely on the watershed threshold for thickness control
   - Apply morphological closing after `_cap_runs` to seal 1-voxel holes
   - Increase `max_thick` to 3

3. **Add a morphological closing step** after the watershed to fill small holes in the 2D (y,z) falx projection. A closing with a small structuring element (radius 1–2) would seal most gaps without thickening the membrane in x.

4. **Add a barrier integrity validation** that tests whether left and right supratentorial CSF are separated by the falx (the CSF bridging test from `diagnose_falx.py`).
