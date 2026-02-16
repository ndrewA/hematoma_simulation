# Material Class Map: FreeSurfer Labels → Simulation u8 Index

This document defines the complete mapping from FreeSurfer aparc+aseg (and wmparc) integer labels to the simulation's u8 material index. The material class definitions and their physical roles are specified in the main simulation spec (Section 3.1, Material Class Definitions).

## Summary

| u8 | Class | $K_{fiber}$ |
|---:|-------|:-----------:|
| 0 | Vacuum | — |
| 1 | Cerebral White Matter | > 0 |
| 2 | Cortical Gray Matter | 0 |
| 3 | Deep Gray Matter | 0 |
| 4 | Cerebellar White Matter | > 0 |
| 5 | Cerebellar Cortex | 0 |
| 6 | Brainstem | > 0 |
| 7 | Ventricular CSF | 0 |
| 8 | Subarachnoid CSF | 0 |
| 9 | Choroid Plexus | 0 |
| 10 | Dural Membrane | 0 |
| 11 | Vessel / Venous Sinus | 0 |
| 255 | Air Halo | — |

---

## Complete Label Mapping

### u8 = 0: Vacuum

| FS Label | Name |
|---------:|------|
| 0 | Unknown (background) |

### u8 = 1: Cerebral White Matter

| FS Label | Name | Notes |
|---------:|------|-------|
| 2 | Left-Cerebral-White-Matter | |
| 41 | Right-Cerebral-White-Matter | |
| 77 | WM-hypointensities | T1 signal anomaly in WM; structurally still WM |
| 78 | Left-WM-hypointensities | |
| 79 | Right-WM-hypointensities | |
| 85 | Optic-Chiasm | Crossing optic nerve fibers |
| 192 | Corpus_Callosum | Midline commissure |
| 250 | Fornix | Limbic WM tract |
| 251 | CC_Posterior | Splenium |
| 252 | CC_Mid_Posterior | |
| 253 | CC_Central | Body of CC |
| 254 | CC_Mid_Anterior | |
| 255 | CC_Anterior | Genu. FS label 255 ≠ simulation ID 255 |

Note: CC labels 251-255 are also used as geometric constraints during dural membrane reconstruction (the corpus callosum forms the inferior boundary of the falx cerebri).

### u8 = 2: Cortical Gray Matter

| FS Label | Name | Notes |
|---------:|------|-------|
| 3 | Left-Cerebral-Cortex | Generic cortex label |
| 42 | Right-Cerebral-Cortex | |
| 19 | Left-Insula | Cortical structure |
| 55 | Right-Insula | |
| 20 | Left-Operculum | Cortical structure |
| 56 | Right-Operculum | |
| 80 | non-WM-hypointensities | Signal anomaly in GM; rare in HCP |
| 81 | Left-non-WM-hypointensities | |
| 82 | Right-non-WM-hypointensities | |
| 1001–1035 | ctx-lh-* | Left cortical parcels (Desikan-Killiany) |
| 2001–2035 | ctx-rh-* | Right cortical parcels |

In HCP aparc+aseg, the cortical ribbon is labeled with parcellation IDs (1001-1035, 2001-2035) rather than the generic 3/42. Both are included for robustness.

### u8 = 3: Deep Gray Matter

| FS Label | Name | Notes |
|---------:|------|-------|
| 10 | Left-Thalamus | |
| 49 | Right-Thalamus | |
| 11 | Left-Caudate | Basal ganglia |
| 50 | Right-Caudate | |
| 12 | Left-Putamen | Most common ICH site (~35%) |
| 51 | Right-Putamen | |
| 13 | Left-Pallidum | Basal ganglia |
| 52 | Right-Pallidum | |
| 17 | Left-Hippocampus | |
| 53 | Right-Hippocampus | |
| 18 | Left-Amygdala | |
| 54 | Right-Amygdala | |
| 26 | Left-Accumbens-area | Ventral striatum |
| 58 | Right-Accumbens-area | |
| 27 | Left-Substancia-Nigra | Midbrain nucleus |
| 59 | Right-Substancia-Nigra | |
| 28 | Left-VentralDC | Ventral diencephalon |
| 60 | Right-VentralDC | |

### u8 = 4: Cerebellar White Matter

| FS Label | Name |
|---------:|------|
| 7 | Left-Cerebellum-White-Matter |
| 46 | Right-Cerebellum-White-Matter |

### u8 = 5: Cerebellar Cortex

| FS Label | Name |
|---------:|------|
| 8 | Left-Cerebellum-Cortex |
| 47 | Right-Cerebellum-Cortex |

### u8 = 6: Brainstem

| FS Label | Name | Notes |
|---------:|------|-------|
| 16 | Brain-Stem | Medulla, pons, midbrain |
| 75 | Left-Locus-Coeruleus | Brainstem nucleus |
| 76 | Right-Locus-Coeruleus | |

### u8 = 7: Ventricular CSF

| FS Label | Name | Notes |
|---------:|------|-------|
| 4 | Left-Lateral-Ventricle | Largest CSF compartment |
| 43 | Right-Lateral-Ventricle | |
| 5 | Left-Inf-Lat-Vent | Temporal horn |
| 44 | Right-Inf-Lat-Vent | |
| 14 | 3rd-Ventricle | Connects laterals via foramina of Monro |
| 15 | 4th-Ventricle | Between brainstem and cerebellum |
| 72 | 5th-Ventricle | Rare normal variant (cavum septi pellucidi) |

### u8 = 8: Subarachnoid CSF

| FS Label | Name | Notes |
|---------:|------|-------|
| 24 | CSF | FreeSurfer's sulcal/cisternal CSF label |

Additional subarachnoid voxels are identified during preprocessing: voxels that are (a) inside the skull SDF and (b) not brain tissue or ventricular CSF are painted with this class. This captures the subarachnoid space and cisterns that FreeSurfer's label 24 does not fully cover.

### u8 = 9: Choroid Plexus

| FS Label | Name |
|---------:|------|
| 31 | Left-choroid-plexus |
| 63 | Right-choroid-plexus |

### u8 = 10: Dural Membrane

No FreeSurfer labels. This class is assigned entirely during dural membrane reconstruction (preprocessing Problem 3). Voxels identified as falx cerebri or tentorium cerebelli are overwritten with this material ID.

### u8 = 11: Vessel / Venous Sinus

| FS Label | Name |
|---------:|------|
| 30 | Left-vessel |
| 62 | Right-vessel |

### u8 = 255: Air Halo

Not assigned during preprocessing. This ID is painted at runtime by the topology update kernel (Step 0, Action C: Halo Injection).

---

## Fallback Rules for Rare / Unexpected Labels

Any FreeSurfer label not explicitly listed above is handled by the following fallback rules. The preprocessing script must log a warning for every voxel that hits a fallback path.

| FS Label | Name | Fallback | Rationale |
|---------:|------|:--------:|-----------|
| 1, 40 | Cerebral Exterior | → 2 | Surface label; if in volume, treat as cortex |
| 6, 45 | Cerebellum Exterior | → 5 | Surface label; treat as cerebellar cortex |
| 9, 48 | Thalamus-unused | → 3 | Deprecated thalamus label |
| 21, 22, 23 | Line-1/2/3 | → 0 | Internal placeholders; not tissue |
| 25, 57 | Lesion | → 2 | Absent in healthy HCP; treat as tissue if present |
| 29, 61 | Undetermined | → 2 | Unclassified tissue; conservative default |
| 32–39 | Left gyral labels | → 2 | Cortical structures |
| 64–71 | Right gyral labels | → 2 | Cortical structures |
| 73, 74 | Left/Right-Interior | → 2 | Rare; treat as tissue |
| 83, 84 | Left/Right-F1 | → 2 | Frontal subregion |
| Any other 1–999 | — | → 2 + **warn** | Assume tissue; don't create voids |
| Any other ≥ 1000 | — | → 2 + **warn** | Likely cortical parcellation label |

---

## Design Rationale

**Why 11 classes and not fewer?** Each class represents a physically distinct permeability regime. Merging classes (e.g., cortical + deep gray) would prevent independent tuning of LUT values for tissues that are known to differ mechanically and are distinct hemorrhage sites clinically. With 254 available IDs, there is no cost to maintaining the distinction.

**Why separate ventricular from subarachnoid CSF?** Both are pure fluid with identical physics. The separation enables (a) independent volume tracking for the Monro-Kellie controller (ventricular expansion is a key clinical indicator), and (b) the option to assign lower effective permeability to subarachnoid CSF to represent arachnoid trabeculae. If unnecessary, identical LUT values make them equivalent at runtime.

**Why is brainstem anisotropic ($K_{fiber} > 0$)?** The brainstem contains major fiber tracts (corticospinal, medial lemniscus, cerebellar peduncles) that bedpostX captures. Setting $K_{fiber} > 0$ lets the $\mathbf{M}_0$ tensor direct hemorrhage spread along these tracts, which is important for modeling pontine hemorrhage.

**Why keep choroid plexus separate?** It is solid vascularized tissue that sits inside the fluid-filled ventricles. Mapping it to ventricular CSF (pure fluid) overestimates local flow. Mapping it to gray matter blocks ventricular CSF circulation. It needs intermediate permeability — high enough not to obstruct the ventricle, low enough to reflect that it is tissue.

**FreeSurfer label 255 vs simulation ID 255.** FreeSurfer uses label 255 for CC_Anterior (genu of the corpus callosum). The remapping converts this to simulation u8 = 1 (cerebral white matter). No conflict with simulation ID 255 (air halo), which is only assigned at runtime.

**Corpus callosum as cerebral WM (not separate class).** The per-voxel $\mathbf{M}_0$ tensor already captures the CC's coherent fiber geometry. All cerebral WM shares the same myelinated-axon microstructure, so there is no basis for different $(K_{iso}, K_{fiber})$ values. The CC labels are still identifiable during preprocessing for geometric purposes (falx reconstruction), even though they share a material class with other WM.

**WM hypointensities (labels 77-82).** In healthy HCP subjects these are minimal. They represent regions where T1 signal is slightly abnormal but the tissue is structurally white matter. Mapping to WM is appropriate. If pathological data were used, a separate class might be warranted.

**Default fallback is cortical gray matter (u8 = 2).** This is the most conservative choice for unexpected labels: it creates tissue rather than vacuum (avoiding holes in the domain), with low permeability that won't create spurious high-flow channels. Any label hitting the fallback triggers a preprocessing warning for manual review.
