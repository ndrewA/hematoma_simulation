# General Pipeline Specification

### Phase 1: Anatomical Acquisition & Continuous Field Reconstruction

The pipeline begins by ingesting discrete clinical imaging data and converting it into continuous functional fields. This step decouples the simulation resolution from the input scanner resolution.

**1.1 Structural Segmentation**
Raw T1-weighted and T2-weighted MRI volumes are ingested.
*   **Skull Stripping:** The cranium is segmented to define the rigid kinematic boundary condition $\Gamma_{skull}$ where $\mathbf{v} = 0$.
*   **Ventricular Parcellation:** The lateral and third ventricles are identified. These zones are flagged as purely fluid domains ($\phi^f = 1, \phi^s = 0$) and assigned a bulk modulus representing Cerebrospinal Fluid (CSF).
*   **Parenchymal Masking:** The brain tissue volume is isolated as the domain for solid particle generation.

**1.2 High-Throughput Microstructure Reconstruction (HARDI)**
To resolve complex crossing-fiber geometries and ensure continuous evaluations across the domain, the system processes raw High Angular Resolution Diffusion Imaging (HARDI) data into a queryable vector field.

*   **Step A: Constrained Spherical Deconvolution (CSD)**
    The system applies **Multi-Shell Multi-Tissue CSD** to the raw dMRI signal to compute the Fiber Orientation Distribution (FOD). This process deconvolves the single-fiber response function from the measured signal to identify multiple distinct fiber populations within each voxel, separating white matter anisotropy from isotropic CSF compartments.

*   **Step B: Offline Peak Extraction**
    The system identifies significant fiber peaks during the pre-processing phase using a Newton-optimization algorithm that locates the local maxima of the FOD.
    *   **Data Structure:** For every voxel, the system stores the direction vectors $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$ and their corresponding normalized volume fractions $\{w_1, w_2, w_3\}$.
    *   **Filtering:** Peaks with volume fractions below a defined noise threshold are discarded.

*   **Step C: Continuous Field Interpolation**
    The resulting vector data is stored in a high-density **3D Texture**. During the simulation, the system utilizes hardware-accelerated **Trilinear Interpolation** to sample this texture. This generates a continuous orientation field $\Psi(\mathbf{x})$, enabling particles to retrieve fiber directions and weights at arbitrary sub-voxel coordinates.

---

### Phase 2: Domain Initialization & Hierarchical Allocation

The system instantiates the "Sparse Pyramid" memory structure required for the Multigrid solver.

### 2.1 Dense-to-Sparse Particle Seeding
Lagrangian Material Points (particles) are generated within the parenchymal mask to discretize the continuum.

* **Rejection Sampling:** The system iterates through the domain bounding box. Particles are instantiated only at coordinates where the tissue probability map is non-zero.
* **Texture-Based Binding:** At instantiation ($t=0$), each particle samples the continuous Orientation Texture (generated in Phase 1.2) at its specific floating-point coordinate $\mathbf{x}_p$.
* **Retrieval:** The particle retrieves the set of dominant fiber vectors $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$ and their corresponding volume weights $\{w_1, w_2, w_3\}$.
* **Tensor Initialization:** These components are immediately compressed into a local structural tensor $\mathbf{M}_0$. This tensor is stored as a persistent attribute on the particle, decoupling the simulation from the static texture lookups for all subsequent time steps.
* **Grid Rasterization:** During each P2G transfer, the particle-level $\mathbf{M}_0$ tensors are rasterized to grid nodes using the same B-spline weights as mass and momentum. The resulting grid-level $\mathbf{M}_0$ field is stored in Tier 2 and used by the Darcy solver to compute axis-projected anisotropic permeability. For the initial frame (before the first P2G transfer), $\mathbf{M}_0$ is rasterized directly from the preprocessed fiber texture to grid nodes.
* **Reference Volume Capture:** Immediately following instantiation, the system calculates the global target volume sum ($V_{target} = \sum_p V_p^0$). This scalar is stored in constant global memory and serves as the fixed set-point for the Monro-Kellie feedback loop.

**2.2 Level 0 Virtual Mapping**
Particle positions $\mathbf{x}_p$ are quantized into **$N^3$** grid coordinates.
*   **Morton Hashing:** Coordinates are interleaved into Z-order curve indices.
*   **Block Activation:** These indices are masked to identify occupied **$8 \times 8 \times 8$** Micro-Blocks. Pointers to these blocks are allocated in the Level 0 hash table.

**2.3 Pyramid Construction (Levels $1 \dots D$)**
To support the Geometric Multigrid solver, coarser grid levels are allocated recursively.
*   **Topology Kernel:** A compute kernel scans the active blocks of Level $L$. For every active block, it calculates the parent block index in Level $L+1$.
*   **Allocation:** If the parent block is unallocated, it is instantiated. This guarantees that the coarse solver grids strictly cover the domain of the fine physics grid.

---

### Phase 3: The Partitioned Multi-Scale Integration Loop

The core simulation loop advances the system state using a **Partitioned Sub-Cycling Architecture**. To reconcile the divergent stability requirements of the quasi-static solid mechanics and the diffusion-dominated fluid mechanics, the execution logic decouples the expensive Pressure Poisson solver from the inexpensive explicit solid integration.

**3.1 Adaptive Sub-Cycling Protocol**
The system executes a fixed ratio of Solid Sub-Steps ($N_{sub}$) for every single Fluid Macro-Step.
* **Regime I (Inertial):** 1:1 Ratio. The phases are tightly coupled to resolve acoustic interactions.
* **Regime II (Quasi-Static):** High Ratio (e.g., 10:1). The fluid pressure field is solved once per macro-step ($\Delta t_{fluid}$), creating a static drag field that persists through multiple explicit solid integration steps ($\Delta t_{solid}$).

#### Step 0: Regime Arbitration, Topology & Memory Maintenance
*Executed once per Fluid Macro-Step.*

At the initiation of the macro-frame, the system synchronizes the Lagrangian material state with the Eulerian sparse grid. This phase manages the memory locality of particle data to maximize cache efficiency and arbitrates the validity of the cached linear operators.

* **Action A: Atomic Spatial Hashing & Volume Integration**
*Frequency: Every Frame.*
To enable block-local indirect dispatch, particles are logically mapped to active grid blocks. Simultaneously, the system monitors global volume conservation to detect numerical drift.
1. **Bin Reset:** The `particle_count` atomic counters for all potentially active grid blocks are reset to zero.
2. **Atomic Hashing & Reduction:** A compute kernel iterates over all particles.
* **Mapping:** The kernel computes the parent block index $B$ and performs an atomic increment on `particle_count[B]`.
* **Volume Sum:** The kernel atomically accumulates the current particle volume ($V_p = J_p V_p^0$) into a global scalar `total_active_volume`.
3. **Registration:** The returned slot index $k$ is used to write the Particle ID into the block-local `pid_buffer[B][k]`. If $k$ exceeds the block capacity $N_{cap}$, the particle is registered in a global overflow buffer.

* **Action B: Lazy Memory Defragmentation (Adaptive Kinetic Trigger)**
*Frequency: Adaptive (Drift-Gated).*
To reconcile memory locality with simulation velocity, the defragmentation schedule is governed by a **Lagrangian Displacement Budget** rather than a fixed clock.
* **The Drift Metric:** A global accumulator $\delta_{drift}$ integrates the maximum particle velocity per frame (tracked during the G2P Gather):
$$ \delta_{drift} \leftarrow \delta_{drift} + (\|\mathbf{v}\|_{max} \times \Delta t) $$
* **Trigger Condition:** The reorganization pipeline is activated only when $\delta_{drift}$ exceeds the **Coherence Threshold** ($\tau_{cohere} \approx 0.5 \times \text{BlockWidth}$). This strictly ensures that bandwidth-heavy sorting occurs only when particles have physically migrated far enough to degrade cache coherency.
* **Execution Pipeline:**
1. **Prefix Sum:** Global write offsets are calculated based on the populated `particle_count` of each active block.
2. **Stream Compaction:** A kernel iterates through grid blocks in strict **Morton (Z-Curve) Order**. It copies particle attributes ($\mathbf{x}, \mathbf{v}, \mathbf{F}$) from the Active Bank to the Target Bank to restore spatial contiguity.
3. **Bank Swap & Reset:** The global array pointers are swapped, the `pid_buffer` is updated, and the $\delta_{drift}$ accumulator is reset to $0.0$.

* **Action C: L0 Topology Activation**
The Sparse SNode tree is updated to reflect the new particle distribution.
1. **Block Activation:** The `ti.activate` command is executed for all blocks containing active particles (`particle_count > 0`).
2. **Halo Injection:** A kernel queries the Global Skull SDF in the neighborhood of active fluid voxels. Neighboring voxels that return `Outside` (`SDF > 0`) are explicitly activated and assigned the Air material index (`255`).

* **Action D: Coarse Operator Maintenance (RSGM Protocol)**
The system determines the validity of the dense linear operators ($L_2 \dots L_D$) based on the simulation regime and spectral health.
* **Regime I:** The dense operator rebuild is skipped; the solver defaults to the Matrix-Free Diagonal path.
* **Regime II:** The spectral convergence rate $\rho$ from the previous frame is evaluated. If $\rho \ge 0.6$ (indicating topological drift), the `is_topology_dirty` flag is raised to trigger the "Split-Stage Block-Local" reconstruction pipeline. Otherwise, the existing cached `f32` operators are retained.

* **Action E: The Global Volume Compensator (Alpha-Coupled PD Protocol)**
*Frequency: Every Frame.*
To enforce the fixed-volume doctrine of the cranial vault while mitigating low-frequency oscillations ("Breathing") induced by the high inertia of Regime II Mass Scaling ($\alpha \approx 2000$), the system utilizes a **Proportional-Derivative (PD)** controller with regime-aware gain scheduling and slew rate limiting.

1.  **State Tracking:**
    The system maintains the global volume from the previous frame to compute the instantaneous volumetric velocity:
    $$ \dot{V}_{current} = \frac{total\_active\_volume^{(n)} - total\_active\_volume^{(n-1)}}{\Delta t_{macro}} $$

2.  **The Alpha-Coupled Control Law:**
    The target boundary pressure $P_{target}$ is calculated by balancing elastic restoration against inertial damping:
    $$ P_{target} = \underbrace{\left( \frac{K_{p}}{\alpha} \right) \cdot (V_{target} - V_{total}^{(n)})}_{\text{Scaled Elasticity}} - \underbrace{(K_{d} \cdot \alpha) \cdot \dot{V}_{current}}_{\text{Inertial Damping}} $$
    *   **Stiffness Scaling ($K_p / \alpha$):** The proportional gain is inversely scaled by the Mass Scaling factor $\alpha$. Since the tissue response time is dilated, the restoring force is weakened to prevent "Spring-Mass" resonance.
    *   **Damping Scaling ($K_d \cdot \alpha$):** The derivative gain is proportionally scaled. Since the effective nodal mass is scaled by $\alpha^2$, the system momentum is massive. The damping term is amplified to dissipate this kinetic energy and prevent target overshoot.

3.  **Slew Rate Saturation (Anti-Shock):**
    To prevent "Water Hammer" effects where instantaneous pressure jumps destabilize the Pressure Poisson solver's preconditioner, the change in boundary pressure per frame is clamped:
    $$ \Delta P = \text{clamp}(P_{target} - P_{halo}^{prev}, -\delta_{max}, +\delta_{max}) $$
    $$ P_{halo}^{new} = P_{halo}^{prev} + \Delta P $$
    *   **$\delta_{max}$:** The maximum allowable pressure change per macro-step (e.g., $50 \text{ Pa}$). This physically represents the finite flow rate limits of physiological drainage.

4.  **Boundary Injection:**
    The resulting scalar $P_{halo}^{new}$ is written to the global constant buffer ($P_{halo}^{prev}$ is updated for the next frame). This value replaces `0.0` as the Dirichlet condition for Air voxels.

* **Action F: Kinematic Priming (Grid Population)**
The Eulerian fields are populated to establish the mass, momentum, chemical, and rheological source terms for the solver.
* **Execution:** The system executes the **Shared-Memory P2G Scatter**.
* **Mechanism:** This kernel utilizes the updated `pid_buffer` and bank-aware shared memory padding to accumulate the following attributes from particles to grid nodes with optimal bandwidth utilization:
1. **Mass & Momentum:** $m_i$ and $(m\mathbf{v})_i$ for the velocity update.
2. **Osmotic Concentration ($C_{osm}$):** Derived from tracer iron levels for the Starling source term.
3. **Coagulation Factor ($\alpha_{clot}$):** The kernel computes the sigmoid age function $S(\tau)$ for each particle and performs a mass-weighted average onto the node. This rasterizes the Lagrangian age data into an Eulerian field, enabling the solver to identify and "freeze" old blood regions.

* **Regime Parameters:**
* **Regime I (Inertial):**
    *   Mass Scaling: **$\alpha = 1$** (Real-Time).
    *   Permeability: **$K = K_{phys}$**.
    *   Local Damping: **$\beta = 0.0$**.
    *   Global Control: **$K_d \approx 0$** (Inertial damping is physical).
    *   Default Solver: **Path A**.
* **Regime II (Swelling):**
    * Mass Scaling: **$\alpha \approx 2000$** (Stability Only).
    * Permeability: **$K = K_{phys}$** (Real Physics).
    * Local Damping: **Kinetic Switch** (Energy Minimization).
    * Global Control: **Proportional Only** (Derivative term disabled).
    *   Default Solver: **Path B**.
* **Regime III (Seepage):**
    *   Status: **Particles Locked / Solid Fixed**.
    *   Permeability: **$K = K_{phys}$** (Restored).
    *   Default Solver: **Path B**.

#### Step 1: Fluid Phase Pressure Projection (The Macro-Step)
*Executed once per Fluid Macro-Step.*

The system solves the elliptic Pressure Poisson Equation (PPE) to update the pressure field $P$ and reconstruct the resulting hydraulic forces. This phase utilizes the source terms ($\nabla \cdot \mathbf{v}^*$ and $\Psi_{osmotic}$) populated during the Kinematic Priming (Step 0, Action F) and respects the global boundary pressure ($P_{halo}$) set by the Volume Compensator.

**1.1 Solver Execution (Hybrid Fallback Protocol)**
The system resolves the linear system $\mathbf{A}x = b$ to enforce mass conservation.

* **Path Arbitration:**
* **Regime I (Inertial):** Defaults to **Path A (Matrix-Free Diagonal)** for low latency.
* **Regime II & III (Quasi-Static / Seepage):** Defaults to **Path B (Geometric Multigrid)**. These regimes involve high-contrast hydrostatic equilibrium (where $T \approx 10^{-9}$) and large time-step flux integration, requiring the robust global error propagation of the V-Cycle.

* **Source Term Logic (Regime II Split):**
During the Regime II Macro-Step ($\Delta t \approx 4.0s$), the source term $b$ is constructed assuming a **Fixed Skeleton**. The kinematic divergence term ($\nabla \cdot \mathbf{v}^*$) is calculated with $\mathbf{v}^*_{solid} = 0$, ensuring the pressure solution is driven exclusively by the Osmotic/Chemical potentials and the global boundary conditions.

* **Fallback Logic:** If Path A fails to converge within the iteration budget ($K_{fast}$), the system triggers an immediate intra-frame switch to Path B, forcing a topology rebuild if necessary.

* **Precision:** The convergence criterion is strictly governed by the **Precise Residual** kernel ($L_0$ / Transient DS), ensuring mass conservation regardless of the preconditioner path.

**1.2 Topological Flux Reconstruction (Drag Field Generation)**
Upon convergence, the pressure gradient is reconstructed to generate the fluid drag force field $\mathbf{f}_{drag}$ used by the solid phase.
* **Interface Gating:** Gradients are computed at face centers ($\mathbf{I} \pm 0.5$).
* **Solid/Vacuum (Neumann):** If a face connects to a Solid or Vacuum voxel, the gradient is forced to $0.0$. This ensures no ghost forces are applied to particles near sealed walls.
* **Air/Fluid (Dirichlet):** If a face connects to an Active Fluid or Air voxel, the gradient is computed normally using the converged pressure values (where $P_{Air} = P_{halo}$).
* **Field Freeze:** The resulting nodal force vector $\mathbf{f}_{drag} = -\phi^f \nabla P_{reconstructed}$ is stored in the grid state and remains constant for the duration of the subsequent solid sub-cycles.

#### Step 2: The Solid Sweep (The Sub-Cycles)
*Executed $N_{sub}$ times (Regime I) OR until Equilibrium (Regime II).*

The solid phase is advanced explicitly. Depending on the regime, this phase either performs time integration (Regime I) or iterative geometric relaxation (Regime II).

### **2.1 P2G Scatter (Shared-Memory Block Fusion)**
To eliminate the bandwidth bottleneck of global atomics, the scatter operation utilizes a **Block-Local Cache Protocol** executing entirely within L1 Shared Memory. This kernel is strictly optimized for occupancy by limiting register usage via precision tiering.

* **Execution Topology:**
The kernel employs **Indirect Dispatch**, launching exactly one GPU Thread Block for every Active $L_0$ Grid Block. This guarantees a 1:1 mapping between the compute unit and the spatial data volume.

* **Shared Memory Allocation (Precision-Gated):**
Each thread block allocates a `ti.simt.block.shared_array` to function as a local scratchpad.
* **Accumulator Precision:** The container is allocated strictly in **Single Precision (`f32`)**. The use of Transient Double-Single (`vec2f`) arithmetic is **prohibited** in this phase to prevent register spilling.
* **Physical Layout:** To prevent serialization, the array is physically allocated with **X-axis padding** to create an odd-numbered stride (e.g., a $6 \times 6 \times 6$ logical tile is padded to **$7 \times 6 \times 6$** physical stride).
* **Mechanism:** The odd-numbered stride desynchronizes the memory bank mapping of vertical grid neighbors, ensuring that $3 \times 3 \times 3$ atomic stencils issued by a warp distribute writes across distinct memory banks.

* **The Register-Scoped Fusion Pipeline:**
To further mitigate register pressure, the kernel execution utilizes **Variable Scoping** to enforce compiler-level register reuse between kinematic and thermodynamic operations:
1. **Collaborative Load:** Threads cooperatively load the existing grid state from Global Memory into the padded `f32` Shared Memory tile.
2. **Scope A (Kinematics):** Threads iterate over block-local particles to load Mass and Velocity. Weights are computed, and momentum is atomically accumulated in Shared Memory. **Crucially, Velocity registers must be invalidated immediately after this step** to free register file space.
3. **Scope B (Thermodynamics):** Threads load Particle Pressure and the Affine Matrix ($C$). Weights are reused (or cheaply recomputed). Pressure history is accumulated into the Shared Memory tile using the recycled registers from Scope A.
4. **Barrier Synchronization:** `ti.simt.block.sync()` is executed to ensure all particles in the block have completed their contributions.
5. **Coalesced Flush:** The aggregated `f32` results in the Shared Memory tile are written back to the Global $L_0$ SNode.

### **2.2 Grid Momentum Update (Dual-Mode)**
The grid velocity update logic is determined by the active regime.

*   **Force Aggregation:**
    All regimes first compute the net nodal force, combining the reconstructed hydraulic drag (from Step 1) with internal elastic/gravity forces:
    $$ \mathbf{F}_{net} = \mathbf{F}_{elastic} + \mathbf{f}_{drag} + m_i \mathbf{g} $$

*   **Regime I: Symplectic Time Integration**
    The system advances physical time using a standard symplectic step with local damping to suppress high-frequency noise.
    $$ \mathbf{v}^{n+1} = \mathbf{v}^n + \Delta t_{sub} \cdot m_i^{-1} (\mathbf{F}_{net} + \mathbf{F}_{damp}) $$

*   **Regime II: Pseudo-Time Relaxation**
    The update functions as an energy minimization iterator to accommodate the fluid volume added during the Macro-Step.
    *   **Mass Scaling:** The update uses the augmented mass $m_i(\alpha)$ to ensure stability against the large pressure gradients.
    *   **Kinetic Damping (Energy Switch):** The system monitors the work done by the net force. If the velocity opposes the acceleration (indicating the node has passed the equilibrium point), the velocity is zeroed to dissipate energy immediately.
        $$ \text{if } (\mathbf{v}^n \cdot \mathbf{F}_{net} < 0) \quad \mathbf{v}^{n+1} = 0 $$
    *   **Convergence Check:** The loop tracks the maximum nodal velocity $\|\mathbf{v}\|_{max}$. If this value falls below the tolerance threshold $\epsilon_{tol}$, the Solid Sweep terminates early, concluding the macro-frame.

* **2.3 G2P Gather (Topology-Aware Indirect Dispatch):**
Particles sample the updated grid state to advance their position and capture the Eulerian fields. To maintain cache coherency during high-velocity regimes, this kernel utilizes **Indirect Dispatch** instead of linear iteration.
* **Execution Topology:** The kernel launches one thread block per Active Grid Block (utilizing the `pid_buffer` built in Step 0). Threads iterate strictly over the particles registered to that local spatial block. This ensures that all threads in a Warp access the same $3 \times 3 \times 3$ grid neighborhood, maximizing L2 Grid Cache hit rates regardless of physical memory fragmentation.
* **Kinematics:** Particles sample $\mathbf{v}^{n+1}$ and $\nabla \mathbf{v}$ to update position and deformation.
$$ \mathbf{x}_p^{n+1} = \mathbf{x}_p^n + \Delta t_{sub} \mathbf{v}_p^{n+1} $$
* **State Capture:** Particles sample the converged **Grid Pressure ($P$)** from the previous Step 1 at the new position $\mathbf{x}_p^{n+1}$. This effectively advects the pressure field with the material deformation, updating the persistent attribute $P_p$ for the next frame's P2G scatter.
* **Drift Metric Integration:** During the update, the kernel tracks the block-local maximum particle velocity magnitude ($\|\mathbf{v}\|_p$). The global maximum is accumulated into the $\delta_{drift}$ metric to drive the **Adaptive Kinetic Trigger** (Action B) for the subsequent frame.

* **2.4 Constitutive Evolution:**
The Deformation Gradient $\mathbf{F}_p$ is updated.
* **Plasticity:** The Cowper-Symonds yield surface $\sigma_y(\dot{\varepsilon})$ evaluates the effective strain rate. Due to the time-dilation effect of Mass Scaling in Regime II, $\dot{\varepsilon}$ is low, effectively defaulting the material to its static yield stress $\sigma_0$.
* **Damage:** Local damage variables $D$ are evolved based on the new stress state, updating the permeability tensor $\mathbf{K}_{eff}$ for the next macro-frame.

#### Step 3: Regime III Seepage Bypass
When the system is in **Regime III** (Particles Locked), the Solid Sweep (Step 2) is strictly **suspended**.

* **Macro-Stepping:** The system advances with a macro time-step ($\Delta t \approx 1.0s$).
* **Execution:** Only **Step 0** (Topology) and **Step 1** (Fluid Solve) are executed per frame.
* **Tracer Advection:** Passive tracer particles are advected through the fixed domain using the resulting Darcy velocity field.

---

### Phase 4: Volumetric Signal Synthesis (Bloch-Torrey)

Once the mechanical simulation completes a frame, the physical state is rasterized into Magnetic Resonance signal parameters.

**4.1 Tracer Advection (Hemoglobin Chronology)**
Passive tracer particles, seeded at the hemorrhage source, track the age of the blood.
*   **Iron Mapping:** The age of the tracer determines the chemical state (Oxy $\to$ Deoxy $\to$ Methemoglobin). This is mapped to a local paramagnetic concentration $[Fe](\mathbf{x})$.

**4.2 Relaxation Parameter Mapping**
A compute shader iterates over every voxel in the target resolution.
*   **Fraction Calculation:** The local fluid saturation $\phi^f$ and solid fraction $\phi^s$ are retrieved.
*   **$T_1 / T_2$ Synthesis:** The longitudinal ($R_1$) and transverse ($R_2$) relaxation rates are computed as weighted averages of the tissue/fluid fractions and the local iron concentration $[Fe]$.

**4.3 The "Perfect Physics" Volume**
The Bloch equations are solved analytically for each voxel using the scanner settings (Echo Time $TE$, Repetition Time $TR$). The result is a volumetric scalar field $\mathcal{I}_{phy}$ representing the ideal, noise-free NMR signal intensity.

---

### Phase 5: Output Serialization (DICOM)

The generated analytical signal volume is formatted for clinical ingest.

**5.1 Intensity Normalization**
The raw Bloch-Torrey scalar field $\mathcal{I}_{phy}$ is normalized to the 12-bit integer range $[0, 4095]$ consistent with standard Hounsfield-like MRI units.

**5.2 Partial Volume Downsampling**
To match the requested clinical slice thickness (e.g., 5mm), the isotropic simulation grid ($\Delta x^3$ voxels) is downsampled via weighted slab averaging. This physically simulates the "Slice Select" gradient of the MRI scanner.

**5.3 Metadata Injection**
The volume is wrapped in a DICOM container. Simulation parameters (Pressure gradients, Strain invariants) are embedded in private DICOM tags for traceability.
