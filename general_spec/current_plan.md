● The right order is: resolve a handful of spec decisions first (quick), then go straight into preprocessing. Most of the spec issues are either editorial fixes or runtime concerns that don't block preprocessing
   at all.

  Here's why. Preprocessing needs to produce a material map, an SDF, and a fiber texture. To design those, you need exactly three decisions from the spec issues:

  1. Δx — what physical resolution does the simulation grid have? Without this, you can't set up coordinate mapping or resample anything.
  2. Anisotropic permeability — does the u8 index encode a scalar base permeability per tissue class, with M_0 providing directional modulation at runtime? Or does it need to encode directional variants? This
  determines how you design the material classes.
  3. Material class table — which FreeSurfer labels collapse to which u8 index?

  Everything else can wait.

  Recommended order:

  Phase 0: Unblock preprocessing (~1 session)

  #: 1
  Task: Decide Δx — ~0.4mm fits the brain in 512^3 with reasonable margin. Quick decision.
  Effort: 5 min
  ────────────────────────────────────────
  #: 2
  Task: Decide anisotropy handling — almost certainly: u8 → scalar K_base from LUT, M_0 tensor provides the directional component at runtime, face transmissibility = n·K·n projected along the normal. This is
  what
     the spec intends despite the inconsistency.
  Effort: 15 min
  ────────────────────────────────────────
  #: 3
  Task: Define material class table — list the classes (WM, GM, CSF, deep gray, cerebellum, brainstem, dural membrane, etc.), assign u8 indices.
  Effort: 30 min
  ────────────────────────────────────────
  #: 4
  Task: Fix the spec typos — path naming, L2 resolution, dead LUT scaling text, P2G dimensions. Editorial.
  Effort: 15 min

  Phase 1: Preprocessing (the real work)

  ┌─────┬────────────────────────────────────────────────────────────────────┬───────────────────────────────┐
  │  #  │                                Task                                │          Depends on           │
  ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤
  │ 5   │ Domain geometry + coordinate mapping (origin, affine, crop bounds) │ Δx decision                   │
  ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤
  │ 6   │ FreeSurfer label remapping → u8 material map                       │ material class table          │
  ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤
  │ 7   │ Skull SDF construction                                             │ domain geometry               │
  ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤
  │ 8   │ Subarachnoid CSF identification                                    │ SDF + material map            │
  ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤
  │ 9   │ Dural membrane reconstruction (hardest step)                       │ material map + FS labels      │
  ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤
  │ 10  │ Fiber orientation texture (bedpostX → M_0-ready format)            │ domain geometry               │
  ├─────┼────────────────────────────────────────────────────────────────────┼───────────────────────────────┤
  │ 11  │ Validation + visualization                                         │ everything above              │
  └─────┴────────────────────────────────────────────────────────────────────┴───────────────────────────────┘

  Phase 2: Remaining spec decisions (overlap with Phase 1)

  These don't block preprocessing but need resolving before you write solver code:

  ┌─────┬───────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  #  │                                                 Task                                                  │
  ├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 12  │ Material property table — permeability values, elastic moduli, densities, viscosities from literature │
  ├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 13  │ Hemorrhage source mechanism — how the bleed initiates                                                 │
  ├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 14  │ Regime I→II transition criterion                                                                      │
  ├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 15  │ Anisotropic transmissibility stencil — write the actual face projection math into the spec            │
  └─────┴───────────────────────────────────────────────────────────────────────────────────────────────────────┘

  The logic is: preprocessing gives you something visible. Once you can render a slice of the material map, the SDF, and the fiber orientations overlaid on anatomy, the remaining decisions (permeability values,
  hemorrhage source location, regime transitions) become much easier to reason about because you have concrete geometry to point at.

  Want to start with Phase 0? We can knock out all four decisions and spec fixes right now, then move into preprocessing with a clean foundation.
