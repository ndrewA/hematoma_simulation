  Preprocessing Problems: Complete Inventory

  I traced every input the runtime consumes back to the raw HCP data. There are 10 distinct problems, organized from foundational (must solve first) to dependent.

  ---
  1. Domain Geometry: Physical Extent and Grid Spacing

  The spec says 512^3 virtual resolution but never defines Δx (the grid spacing). This is the first decision because everything else depends on it.

  - Your T1w data is 260x311x260 at 0.7mm. The brain is roughly 170mm x 200mm x 140mm.
  - At Δx = 0.7mm (matching T1w): 512 * 0.7 = 358mm domain — way too large, wastes 60%+ of the address space on void.
  - At Δx = 0.35mm: 179mm domain — tight fit for a head, ~2x supersampling of the T1w data.
  - At Δx = 0.4mm: 205mm domain — reasonable margin, still ~1.75x supersampling.

  This also determines how the HCP ACPC coordinate system maps to simulation grid indices (i,j,k). You need to define:
  - The physical origin of grid index (0,0,0)
  - The affine transform from physical mm → grid index
  - How much padding surrounds the brain (enough for the air halo + skull SDF queries)

  ---
  2. Segmentation Remapping: FreeSurfer Labels → u8 Material Index

  FreeSurfer's aparc+aseg.nii.gz contains ~100+ integer labels. The simulation needs a small number of material classes, each with a u8 index pointing into the permeability LUT. You need to design and implement
  a label collapse table.

  Decisions required:
  - How many distinct material classes? The spec's architecture supports 254 (IDs 1-254), but the physics only distinguish a handful of permeability regimes.
  - Which FreeSurfer labels map to each class? Key groupings:
    - Cerebral white matter (labels 2, 41, plus wmparc subdivisions) — anisotropic permeability, fiber-dependent
    - Cerebral cortex / gray matter (labels 3, 42) — isotropic, lower permeability
    - Deep gray matter (thalamus 10/49, caudate 11/50, putamen 12/51, pallidum 13/52, hippocampus 17/53, amygdala 18/54) — isotropic, possibly different stiffness
    - CSF / ventricles (lateral 4/43, inferior lateral 5/44, 3rd ventricle 14, 4th ventricle 15) — pure fluid domain, φ^f = 1
    - Choroid plexus (31/63) — sits inside ventricles but is solid tissue, not fluid
    - Cerebellum (WM 7/46, cortex 8/47) — distinct mechanical properties
    - Brainstem (16) — mixed fiber/gray, different mechanics
    - CSF (non-ventricular) (label 24 = CSF) — subarachnoid/sulcal CSF
    - Vessel / sinus (label 30 = vessel) — vascular structures
    - Optic chiasm (85), CC (251-255 in wmparc) — specialized structures
  - What about label 0 (unknown/background) — becomes vacuum (material ID 0)?

  The choroid plexus problem: Labels 31/63 are inside the ventricle volume but are solid tissue that produces CSF. Treating them as fluid would be wrong; treating them as solid wall would block ventricular CSF
  flow. Likely needs to be a permeable tissue class.

  ---
  3. Dural Membrane Reconstruction (Falx Cerebri + Tentorium Cerebelli)

  This is the hardest problem and the most consequential for physics. The spec's entire precision architecture — the 10^9 contrast ratio, the Hybrid Virtual-Double firewall at L2, the Double-Single diagonal
  storage — exists specifically to resolve flow across these membranes. Without them, the two hemispheres are a single connected fluid domain, and pressure would equalize far too quickly.

  What these structures are:
  - Falx cerebri: A vertical sickle-shaped sheet of dura mater in the longitudinal fissure between the cerebral hemispheres. ~1-2mm thick. Runs from the crista galli anteriorly to the internal occipital
  protuberance posteriorly.
  - Tentorium cerebelli: A tent-shaped horizontal sheet separating the cerebrum from the cerebellum. Attaches to the petrous ridges laterally and the straight/transverse sinuses posteriorly. Has a central
  opening (tentorial notch/incisura) through which the brainstem passes.

  Why FreeSurfer can't help: These are thin fibrous membranes with minimal contrast on T1w MRI. FreeSurfer doesn't attempt to segment them. No standard neuroimaging pipeline does.

  Possible approaches:

  (a) Surface-based geometric construction:
  - The falx sits in the interhemispheric fissure. You have left/right pial surfaces in T1w/Native/. The space between the hemispheres at the midline defines the falx location.
  - Algorithm: for each voxel near the midsagittal plane, check if it lies between the left and right medial pial surfaces. If yes, and it's not brain tissue (it's in the fissure), mark it as falx.
  - The tentorium is harder geometrically. It roughly follows the line where the inferior temporal/occipital cortex meets the superior cerebellar surface. You could use the cerebellar surface + the inferior
  cerebral surface to define it.
  - Challenge: The falx doesn't extend all the way down — it stops above the corpus callosum. The tentorium has a notch. These anatomical details matter for flow topology.

  (b) Atlas-based warping:
  - Take a dural membrane atlas defined in MNI space (from a template or manually defined) and warp it to native space using the available standard2acpc_dc.nii.gz nonlinear warp.
  - Pros: Captures correct anatomical topology (notch, extent).
  - Cons: Requires finding or creating such an atlas. Warp quality at thin structures is uncertain.

  (c) Rule-based geometric heuristics:
  - Falx: a thin slab around x = midline, masked to the interhemispheric fissure region (between the hemispheres, above the corpus callosum).
  - Tentorium: a curved sheet at approximately y = -20mm to -60mm (posterior), z = roughly the level of the tentorium, with a hole for the brainstem.
  - Pros: Simple, fast.
  - Cons: Anatomically imprecise, brittle across subjects.

  (d) Hybrid — surface geometry + anatomical constraints:
  - Use pial surfaces to find the interhemispheric fissure
  - Use FreeSurfer's corpus callosum labels (251-255 in wmparc) to bound the inferior extent
  - Use cerebellar segmentation labels to define the tentorial boundary
  - Impose known anatomical constraints (tentorial notch diameter ~25-30mm, falx doesn't penetrate below the CC)

  This is the problem most likely to need iteration. A geometrically plausible first pass may be sufficient initially, refined later when you can visualize the pressure field behavior.

  ---
  4. Skull SDF Construction

  The runtime uses the Skull SDF for two purposes:
  1. Halo injection (Step 0, Action C): neighbors outside the skull (SDF > 0) become air halo voxels (material ID 255)
  2. Porosity factor (φ_geo): sub-voxel volume fraction for cut-cell boundaries at the skull surface

  What "skull" means here: Not the bony skull itself, but the inner boundary of the cranial cavity — the inner table of the skull, which is approximately the dural surface. For simulation purposes, this is the
  rigid container that the brain sits in.

  Construction approach:
  - Start with brainmask_fs.nii.gz (FreeSurfer brain mask). This represents the intracranial volume.
  - Dilate slightly (~1-2 voxels) to include the subarachnoid CSF space that lies between the brain surface and the dura.
  - Alternatively, use the pial surfaces: everything inside the outer pial convex hull + some margin = "inside skull."
  - Compute the Euclidean Distance Transform of the binary mask. Sign it: negative inside, positive outside.
  - Resample to simulation resolution.

  Subtleties:
  - The SDF must be smooth enough for trilinear interpolation to produce valid porosity factors.
  - At the foramen magnum (where the brainstem exits the skull), the SDF defines whether the inferior boundary is open or closed. This affects CSF drainage and the Monro-Kellie volume balance.
  - The SDF should be computed from the convex-ish outer boundary, not from the folded cortical surface (which would create SDF = 0 surfaces deep inside the sulci).

  ---
  5. Subarachnoid Space / External CSF

  Between the brain surface (pia mater) and the skull (dura/arachnoid), there's a layer of CSF — the subarachnoid space. This is a critical fluid compartment:
  - It cushions the brain
  - CSF circulates through it
  - In the simulation, it provides the drainage pathway that the Monro-Kellie controller manages

  FreeSurfer's label 24 (CSF) captures some of this, but it's incomplete — it mainly labels the sulcal CSF visible on T1w. The cisterns (major CSF pools like the ambient cistern, prepontine cistern) may not be
  well labeled.

  Approach:
  - The subarachnoid space can be approximated as the voxels that are:
    - Inside the skull SDF (SDF < 0)
    - NOT inside the brain mask (brain tissue labels)
    - NOT white matter, gray matter, or deep structures
  - Alternatively: voxels between the pial surface and the skull boundary
  - These get assigned a CSF material index (high permeability, fluid domain)

  Complication: In a healthy brain, the subarachnoid space is thin (~1-3mm) and varies across the surface. In atrophied brains or near the Sylvian fissure, it can be much wider. The HCP subject is young and
  healthy, so this space will be relatively thin.

  ---
  6. Fiber Orientation Texture

  The spec needs a 3D texture storing per-voxel fiber orientations {v1,v2,v3} and volume fractions {w1,w2,w3}, sampled at particle initialization time via trilinear interpolation.

  Source: bedpostX outputs (dyads{1,2,3}.nii.gz + mean_f{1,2,3}samples.nii.gz), already in T1w space. These provide exactly what the spec needs.

  Problems to solve:

  (a) Resolution mismatch: bedpostX is at 1.25mm, simulation is at ~0.35-0.4mm. The texture needs to be queryable at simulation resolution via trilinear interpolation. Two options:
  - Store at native 1.25mm resolution and let the runtime trilinear interpolation handle it (since the spec says this is only sampled at t=0, once per particle). This is simpler and doesn't introduce
  interpolation artifacts.
  - Upsample to simulation resolution. Overkill since the diffusion data fundamentally doesn't contain sub-1.25mm information.

  (b) Direction vector interpolation: The dyads are direction vectors (unit vectors). Trilinear interpolation of unit vectors doesn't produce unit vectors. You need to either:
  - Interpolate and renormalize (fast, usually fine)
  - Use dyadic tensor interpolation (store v⊗v instead of v, interpolate the tensor, extract the principal direction). This is more correct for crossing fibers but heavier.
  - Since the spec compresses to the structural tensor M_0 = Σ w_n (v_n ⊗ v_n) immediately at initialization, interpolating the dyads + weights and computing M_0 per particle is probably sufficient.

  (c) Sign ambiguity: bedpostX dyads are direction vectors, but fiber directions are axial (v and -v are equivalent). The stored vectors may have arbitrary sign flips between adjacent voxels, causing
  interpolation artifacts. This needs to be resolved by ensuring local sign consistency (flipping vectors so they point "consistently" in the same general direction as their neighbors).

  (d) Volume fraction thresholding: mean_f2samples and mean_f3samples may contain noise (small but nonzero values everywhere). The spec mentions filtering peaks below a noise threshold. Need to threshold these
  (e.g., f < 0.05 → treated as no fiber population).

  (e) Masking: bedpostX computes fibers everywhere within the diffusion brain mask, including CSF and deep gray matter where fiber orientations are meaningless. These voxels should have their fiber data zeroed
  out. Only white matter voxels should carry anisotropic fiber information.

  ---
  7. Permeability LUT Design

  The runtime needs a 256-entry f64 lookup table mapping u8 material indices to scalar permeability values. The spec mentions ranges of [10^-11, 10^-2] m² and a 10^9 contrast ratio, but no concrete numbers.

  Values needed from literature:

  ┌─────────────────────────────┬────────────────────┬──────────────────────────────────────┐
  │          Material           │ Permeability (m²)  │                Notes                 │
  ├─────────────────────────────┼────────────────────┼──────────────────────────────────────┤
  │ CSF / bulk fluid            │ ~10^-8 to 10^-7    │ Very high, essentially free flow     │
  ├─────────────────────────────┼────────────────────┼──────────────────────────────────────┤
  │ Damaged tissue / blood      │ ~10^-9 to 10^-8    │ High, approaches fluid               │
  ├─────────────────────────────┼────────────────────┼──────────────────────────────────────┤
  │ Gray matter                 │ ~10^-13 to 10^-12  │ Low, isotropic                       │
  ├─────────────────────────────┼────────────────────┼──────────────────────────────────────┤
  │ White matter (along fiber)  │ ~10^-12 to 10^-11  │ Higher along tracts                  │
  ├─────────────────────────────┼────────────────────┼──────────────────────────────────────┤
  │ White matter (across fiber) │ ~10^-14 to 10^-13  │ Very low perpendicular to fibers     │
  ├─────────────────────────────┼────────────────────┼──────────────────────────────────────┤
  │ Dural membrane              │ ~10^-17 to 10^-16  │ Near-impermeable (the 10^9 contrast) │
  ├─────────────────────────────┼────────────────────┼──────────────────────────────────────┤
  │ Skull/bone                  │ 0.0 (Neumann wall) │ Perfectly sealed                     │
  └─────────────────────────────┴────────────────────┴──────────────────────────────────────┘

  Complications:
  - The spec uses a scalar u8 → scalar K lookup, but the physics actually needs an anisotropic permeability tensor K_tissue = k_iso I + k_fiber M_0. So the LUT likely provides a base permeability that gets
  modulated by the fiber tensor at runtime.
  - Need to understand exactly how the u8 index relates to the tensor construction. Does white matter get a single u8 index, with the anisotropy coming from M_0? Or do different WM regions get different base
  permeabilities?
  - The spec's harmonic mean transmissibility at faces uses the u8 indices of the two adjacent voxels. This only works correctly for scalar permeability. The anisotropic part must be handled differently (likely
  through the M_0 tensor at P2G time, not through the LUT).

  ---
  8. Brainstem Boundary Condition

  The brain connects to the spinal cord through the foramen magnum. The spec never addresses what happens at this boundary.

  Options:
  - Sealed (Neumann): Treat the inferior brainstem surface as a wall. No fluid escapes. The Monro-Kellie controller handles all volume exchange through the skull surface air halo.
  - Open (Dirichlet): Allow CSF drainage downward. This is more physiologically correct (CSF flows into the spinal canal), but adds a second boundary condition that the volume controller must account for.
  - Partially open: Model the foramen magnum as a narrow aperture with its own permeability.

  For a first pass, sealed is simpler and consistent with the spec's single-boundary Monro-Kellie model.

  ---
  9. Hemorrhage Source Definition

  The spec describes hemorrhage mechanics in detail (damage → liquefaction → bleeding) but never specifies how the initial bleed is set up during preprocessing. This is ultimately a runtime configuration, but
  preprocessing should provide:

  - Anatomical coordinates for common hemorrhage locations (basal ganglia, lobar, thalamic, pontine, cerebellar — the standard clinical types)
  - A way to specify the source: probably a region of voxels where particles are injected with fluid phase properties and a damage value of D=1

  This is less of a preprocessing problem and more of a configuration interface. But it depends on having the coordinate mapping (Problem 1) and material map (Problem 2) in place first.

  ---
  10. Validation and Sanity Checking

  After all preprocessing outputs are generated, you need to verify:

  - Topological closure: The material map + skull SDF together form a closed domain. No voxels labeled as tissue exist outside the skull. No unlabeled gaps exist between tissue and skull (everything should be
  either tissue, CSF, or dural membrane).
  - Membrane continuity: The falx and tentorium actually separate the compartments they should. Check that there's no single connected component of CSF/tissue spanning both hemispheres (except through the
  tentorial notch and inferior to the falx).
  - Fiber orientation quality: The M_0 tensors should be smoothly varying within white matter and zero in gray matter/CSF. Visualize a slice to confirm.
  - SDF smoothness: The skull SDF should be a smooth signed distance field without sharp artifacts or incorrect sign regions.
  - Volume sanity: Total intracranial volume should be ~1400-1500 mL for a typical adult. Total ventricular CSF ~20-30 mL. These can be computed from the material map as a basic sanity check.
  - Resolution consistency: All outputs should be in the same coordinate system and queryable at the simulation grid spacing.

  ---
  Dependency Order

  Problem 1 (Domain geometry)
      ↓
  Problem 2 (Label remapping)  ←→  Problem 4 (Skull SDF)
      ↓                                ↓
  Problem 5 (Subarachnoid CSF)    Problem 3 (Dural membranes)
      ↓                                ↓
      └──────── merge into final material map ────────┘
                          ↓
  Problem 6 (Fiber texture)    Problem 7 (Permeability LUT)
                          ↓
  Problem 8 (Brainstem BC)     Problem 9 (Hemorrhage config)
                          ↓
                  Problem 10 (Validation)

  Problems 1 and 2 are where I'd start. What are your thoughts?
