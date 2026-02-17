# Voxel System Specification

### I. Spatial Domain & Hierarchical Memory Layout

**1.1 Virtual Address Space**
The computational domain is defined as a bounded cubic volume with a virtual resolution of **$N^3$** voxels (see Section 1.1 for the definition of $N$ and $\Delta x$). This creates a global Cartesian coordinate system serving as the reference frame for both Eulerian (grid) and Lagrangian (particle) data.
* **Virtualization:** The domain is sparsely virtualized. Memory is not pre-allocated for the full $N^3$ volume. Instead, the system allocates physical memory pages only for regions containing active material (Solid or Fluid), reducing the memory footprint from potential Gigabytes to Megabytes.
* **Coordinate Unification:** The coordinate system is normalized such that grid indices $(i, j, k)$ map directly to the SNode accessor keys, ensuring seamless coupling between particle positions and grid nodes.

**1.2 Sparse SNode Topology**
Instead of manual address hashing, the system leverages the Taichi SNode (Sparse Node) system to define a multi-level pointer hierarchy. This structure automatically handles the mapping of 3D coordinates to linear memory.

* **Tree Architecture:** The memory layout is defined as a two-level sparse tree optimized for the $N^3$ domain:
1. **Level 1 (Coarse Pointers):** A dense pointer array acting as the Page Directory. It subdivides the domain into a grid of **$(N/8)^3$** pointer blocks.
2. **Level 0 (Leaf Blocks):** Bitmasked dense blocks containing **$8 \times 8 \times 8$** voxels. These hold the actual physical data fields ($P, \mathbf{v}, \mathbf{K}$) and utilize hardware bit-counting for efficient iteration over active cells.
**1.3 Hardware-Managed Memory Locality (Implicit Morton)**
The SNode compiler backend automatically enforces spatial locality when mapping the tree to physical RAM.
* **Z-Order Traversal:** The compiler maps the $(i, j, k)$ keys to physical addresses using a Z-order curve (Morton code) or similar space-filling curve. This ensures that voxels that are spatially adjacent in 3D are stored in adjacent or near-adjacent cache lines.
* **Stencil Optimization:** This layout minimizes Translation Lookaside Buffer (TLB) thrashing during 7-point or 27-point stencil operations (Laplacians, Divergence), as fetching a voxel's neighbors is statistically likely to hit the same L1/L2 cache line.

**1.4 Coordinate Quantization**
To interface Lagrangian particles with the Eulerian SNode tree, continuous positions are quantized implicitly:
* **Mapping:** A particle at $\mathbf{x}_p \in \mathbb{R}^3$ contributes to the grid node indices via floor projection relative to grid spacing $\Delta x$:
$$ \mathbf{I} = \lfloor \mathbf{x}_p / \Delta x \rfloor $$
* **Base Offset:** Kernels access the grid using the integer vector $\mathbf{I}$ directly as the SNode key. The bit-level offsets and page lookups are handled by the Taichi runtime's memory accessors.

**1.5 Particle Memory Topology (Dual-Layer Indirect Indexing)**
To reconcile the high frequency of particle motion with the high cost of memory movement, the system utilizes a decoupled architecture where logical residence is separated from physical storage.

*   **Layer A: The Atomic Index Buffer (Logical Topology)**
    Grid blocks manage particle ownership via lightweight integer lookups rather than physical data sorting.
    *   **Block-Local Tables:** Each active SNode block owns a fixed-capacity integer array `pid_buffer[block_id][N_cap]` (e.g., $N_{cap}=512$) and an atomic scalar `particle_count[block_id]`.
    *   **Atomic Hashing:** Every frame, particles are mapped to blocks via atomic increments on `particle_count`. The resulting slot index is used to store the Particle ID in `pid_buffer`.
    *   **Role:** This provides the P2G kernels with a direct iterator for all particles residing in Block $B$ without requiring a global search or full array sort.

*   **Layer B: Double-Buffered SoA (Physical Topology)**
    Physical attributes ($\mathbf{x}, \mathbf{v}, \mathbf{F}$) are stored in Structure-of-Arrays (SoA) format to maximize SIMD lane alignment. To support non-blocking defragmentation, these arrays are **Double-Buffered**.
    *   **Bank Switching:** Two distinct memory banks are allocated: `Bank_0` (Active) and `Bank_1` (Target).
    *   **Lazy Defragmentation:** In standard frames, data is read/written strictly to the Active bank via indirect indexing. Periodically, a "Shuffle" operation compacts data from Active to Target in strict Morton (Z-Curve) order, restoring spatial locality.
    *   **Pointer Swap:** Post-shuffle, the bank pointers are swapped, ensuring that subsequent frames access physically contiguous memory for improved cache line utilization.

*   **Access Pattern Constraints:**
    *   **Indirect Load:** Compute kernels executing on a Grid Block access physics data via the indirection chain: `Block` $\to$ `pid_buffer` $\to$ `Global_Attribute_Array`.
    *   **Collision Handling:** If `particle_count` exceeds `N_cap`, excess particles are shunted to a global overflow buffer to prevent data corruption, where they are processed by a fallback safety kernel.

---

### II. Hybrid Hierarchical Architecture (The Solver Pyramid)

The system employs a specialized hybrid memory architecture designed to bridge the gap between high-resolution sparse physics and high-bandwidth dense numerical solving. This structure, termed the "Hybrid Pyramid," transitions from a spatially sparse backend at fine resolutions to a contiguous dense backend at coarse resolutions.

**2.1 Hybrid Memory Backend**
To optimize the Geometric Multigrid (GMG) solver on GPU architectures, the grid hierarchy spans $D+1$ discrete resolution levels ($L_0$ to $L_D$). The memory layout changes dynamically based on the resolution regime:
*   **Sparse Backend ($L_0, L_1$):** At the finest resolutions (**$N^3$ and $(N/2)^3$**), the domain is sparsely virtualized using Bitmasked SNodes. This ensures that memory is allocated only for active physical regions, accommodating the fractal nature of biological tissue without allocating vacuum.
*   **Dense Backend ($L_2 \dots L_D$):** As the grid coarsens (**$(N/4)^3$ down to $8^3$**), the sparsity benefit diminishes while the overhead of pointer chasing increases. Consequently, these levels are allocated as **Dense Padded Arrays** in linear memory. This layout maximizes cache coherency and memory bandwidth for the coarse-grid correction kernels.

**2.2 Level Definitions & Allocation Strategy**
The hierarchy is strictly inclusive; the allocation of a fine-grid block implies the existence of valid memory in the corresponding coarse-grid region.

*   **Level 0 ($N^3$ - Primary Physics):** The ground-truth simulation layer. It uses a **Bitmasked SNode** backend to store the explicit physical state ($P, \mathbf{v}, \mathbf{K}$).
*   **Level 1 ($(N/2)^3$ - Matrix-Free Correction):** An intermediate sparse layer used to aggregate residuals before bridging to the dense solver. It retains the **Bitmasked SNode** structure to handle complex boundary geometries efficiently.
*   **Level 2 ($(N/4)^3$ - The Bridge):** The transition layer. This is the first **Dense** level, serving as the interface where sparse data is scattered into contiguous memory.
*   **Levels $3 \dots D{-}1$:** Standard coarse grids stored as flat, C-contiguous arrays. Used for rapid error smoothing.
*   **Level $D$ ($8^3$ - Base Solver):** The coarsest level. The entire domain fits within a single GPU thread block's Shared Memory, allowing for exact solving via Red-Black Gauss-Seidel without global memory latency.

**2.3 Dense Padding Protocol ($L_2 \dots L_D$)**
To execute high-throughput stencil operations (Laplacians, Restrictions) on the dense levels without branching logic (e.g., checking array bounds), all dense allocations utilize a **"Sealed Vacuum"** padding strategy.
*   **Padding Dimensions:** A level with nominal resolution $N_\ell^3$ is allocated physically as $(N_\ell+2)^3$ (e.g., $L_2$ at $(N/4)^3$ is allocated as $(N/4+2)^3$).
*   **Active vs. Ghost:**
    *   **Active Domain:** Indices range from $1$ to $N_\ell$. These voxels contain valid solver data.
    *   **Ghost/Vacuum:** Indices $0$ and $N_\ell+1$ on all axes represent the inactive void.
*   **Topological Isolation:** These ghost voxels are initialized to a safe identity state ($A_{ii}=1, b=0$). This allows kernels to blindly read neighbors; if a kernel reads a ghost voxel, the mathematical result naturally resolves to zero flux, enforcing a Homogeneous Neumann condition unconditionally.

**2.4 Coordinate Mapping & Translation**
A uniform coordinate system governs the interaction between the Sparse Global domain and the Dense Local domain. An explicit mapping function handles the index translation, accounting for the dense padding offset.

*   **Sparse $\to$ Dense Projection:** A sparse voxel at global index $\mathbf{I}_{sparse}$ maps to a dense index $\mathbf{I}_{dense}$ at level $k$ via:
    $$ \mathbf{I}_{dense} = (\mathbf{I}_{sparse} // 2^k) + \mathbf{1} $$
*   **Dense $\to$ Sparse Reconstruction:**
    $$ \mathbf{P}_{sparse} = (\mathbf{P}_{dense} - \mathbf{1}) \cdot 2^k $$
*   **The Offset:** The addition of $\mathbf{1}$ ensures that the geometric origin $(0,0,0)$ of the sparse world maps strictly to memory address $(1,1,1)$ in the dense arrays, preserving the integrity of the boundary padding at index 0.

**2.5 Bitmasked Leaf Topology ($L_0, L_1$)**
For the sparse levels, topology is managed via 64-bit integer masks within Micro-Blocks ($4 \times 4 \times 4$).
*   **Intrinsic Iteration:** Compute kernels utilize hardware-intrinsic population counts and leading-zero counts to iterate strictly over active bits. This allows SIMD lanes to process dense packets of work within the sparse structure, skipping empty space without the warp divergence penalties typically associated with sparse matrices.

---


### III. Memory Layout & Precision Strategy

The system utilizes a specialized **Dual-Path Precision Architecture** to resolve the conflict between the high dynamic range required for physical accuracy ($10^9$ permeability contrast) and the high memory bandwidth required for solver performance. Rather than enforcing a uniform precision across the entire domain, the memory layout is stratified based on the numerical role of each data field.

**3.1 Quantized Indexing Protocol**

The system utilizes a quantized storage strategy to reconcile memory bandwidth constraints with the high dynamic range of biological material properties.

*   **Dual Global Lookup Table (LUT):**
    Two global constant arrays store 256 discrete permeability coefficient pairs. Both arrays are allocated in **Double Precision (`f64`)** to accurately represent the logarithmic physiological range $[10^{-17}, 10^{-2}] \text{m}^2$:
    *   **$K_{iso}$[256]:** The isotropic base permeability for each material class.
    *   **$K_{fiber}$[256]:** The additional along-fiber permeability. For isotropic materials (gray matter, CSF, dural membranes, damaged tissue), $K_{fiber} = 0$. For white matter, $K_{fiber} > 0$ encodes the directional preference along fiber tracts.
    Both LUT arrays store the unscaled physical permeability values ($K_{LUT} = K_{phys}$) across all simulation regimes. The permeability coefficients are regime-invariant; regime-dependent behavior is achieved through the time-stepping strategy (Regime I: explicit coupling, Regime II: operator splitting with macro-steps, Regime III: solid phase locked) rather than through scaling of material properties.

*   **Voxel-Level Indexing:**
    Individual active voxels store an 8-bit unsigned integer (`u8`) acting as a direct pointer into both LUT arrays.

*   **On-the-Fly Reconstruction:**
    During the **Precise Path** (Path C residual evaluation), the compute kernel retrieves the physical permeability pair $(k_{iso}, k_{fiber})$ via a direct lookup using the stored `u8` index. The axis-projected permeability is then computed as $K_{axis} = k_{iso} + k_{fiber} \cdot \mathbf{M}_0[axis, axis]$, where $\mathbf{M}_0$ is the Structure Tensor read from the Tier 2 grid field. The retrieved `f64` values are immediately cast into the **Transient Double-Single (`vec2f`)** format. This allows the flux integration logic to operate on the full-precision coefficient, bypassing single-precision registers entirely to preserve barrier integrity.
    During the **Lightweight Path** (Path A preconditioner), the kernel retrieves only $k_{iso}$, omitting the fiber contribution to preserve the zero-storage character of the path.

*   **Material Class Definitions:**
    Each `u8` index corresponds to one of the following material classes. IDs 0 and 255 are reserved control states; IDs 1–254 are active degrees of freedom ($\chi_{\Omega} = 1.0$). The complete FreeSurfer label remapping is defined in the preprocessing specification (`spec/material_map.md`).

    | u8 ID | Class | $K_{fiber}$ | Role |
    |------:|-------|:-----------:|------|
    | 0 | Vacuum | — | Outside cranium. Zero flux (Neumann wall). |
    | 1 | Cerebral White Matter | $> 0$ | Anisotropic. Fiber-directed flow via $\mathbf{M}_0$. Includes corpus callosum, fornix, optic chiasm. |
    | 2 | Cortical Gray Matter | $0$ | Cerebral cortex. Isotropic. |
    | 3 | Deep Gray Matter | $0$ | Basal ganglia, thalamus, hippocampus, amygdala, accumbens, substantia nigra, ventral DC. Isotropic. Primary hemorrhage sites. |
    | 4 | Cerebellar White Matter | $> 0$ | Anisotropic. Arbor vitae; separate base permeability from cerebral WM. |
    | 5 | Cerebellar Cortex | $0$ | Isotropic. Denser neuronal packing than cerebral cortex. |
    | 6 | Brainstem | $> 0$ | Mixed nuclei and tracts. Anisotropic (corticospinal, medial lemniscus, peduncles). |
    | 7 | Ventricular CSF | $0$ | Pure fluid ($\phi^f = 1$). Lateral, 3rd, 4th ventricles. Highest $K_{iso}$. |
    | 8 | Subarachnoid CSF | $0$ | Pure fluid. Sulcal and cisternal spaces between pia and skull. |
    | 9 | Choroid Plexus | $0$ | Permeable vascularized tissue within ventricles. Intermediate $K_{iso}$. |
    | 10 | Dural Membrane | $0$ | Falx cerebri and tentorium cerebelli. Near-impermeable ($10^9$ contrast). |
    | 11 | Vessel / Venous Sinus | $0$ | Large-caliber vascular structures. High $K_{iso}$. |
    | 255 | Air Halo | — | Dynamic Dirichlet boundary ($P = P_{halo}$). |

### 3.2 The Tri-Path Precision Summary

The solver architecture utilizes three distinct paths (defined in V-Cycle Specification Sections 3.1–3.3). This section summarizes how they interact with the memory hierarchy defined above.

*   **Path A (Lightweight Diagonal Preconditioner):**
    *   **Role:** Default preconditioner for Regime I. Reconstructs the diagonal scaling factor on-the-fly from `u8` indices using only $k_{iso}$ from the Dual Global LUT.
    *   **Storage:** Zero. Does not access the cached $L_2 \dots L_D$ operators.

*   **Path B (Geometric Multigrid Preconditioner):**
    *   **Role:** Default preconditioner for Regimes II/III. Generates the error correction vector $z \approx \mathbf{A}^{-1}r$ via the V-Cycle. Prioritizes SIMD throughput and memory bandwidth over bit-exact precision.
    *   **Storage:** The linear operators are pre-computed and stored in **Dense Padded Arrays** on levels $L_2 \dots L_D$.
    *   **Cascaded Layout:**
        *   **The Precision Firewall ($L_2$):** Utilizes a **Hybrid Virtual-Double** layout. Transmissibilities are stored in `f32`, while the matrix diagonal is stored in `vec2f` (Double-Single). The solver utilizes **Shared-Memory Staged Arithmetic**, loading the hybrid operator into a local L1 cache tile and performing the critical pivot subtraction ($D' = A_{ii} - T \cdot C_{prev}$) using mixed-precision intrinsics. This preserves the microscopic barrier residue ($10^{-9}$) against bulk fluid terms ($1.0$) while maximizing DRAM throughput.
        *   **The Accelerator ($L_3 \dots L_D$):** Standard **Single Precision (`f32`)**. These levels utilize the **Epsilon Floor** topology, ensuring that all physical connections are represented by at least a minimum conductance ($\epsilon$) to maintain graph connectivity without extended precision storage.

*   **Path C (Exact Residual):**
    *   **Role:** Calculating the exact physical error $r = b - \mathbf{A}x$ to determine convergence, arbitrate between Path A and Path B, and generate the high-precision defect vector.
    *   **Storage:** **Matrix-Free**. The operator is never stored; the stencil is reconstructed on-the-fly using `u8` indices and the full anisotropic permeability pair $(k_{iso}, k_{fiber})$.
    *   **Execution:** The flux divergence accumulation is performed using **Transient Double-Single (`vec2f`)** arithmetic within the GPU registers. This resolves minute net flux imbalances ($r \approx 10^{-15}$) across sealed boundaries, preventing "Hydraulic Tunneling" while storing the resulting defect vector in bandwidth-efficient `f32`.

### 3.3 Hierarchical Field Organization

Data fields are grouped into three distinct tiers, strictly aligned with the mixed-precision requirements of the physical operators. All fields within a leaf block are arranged in an AOSOA (Array-of-Structures-of-Arrays) layout to ensure coalesced memory access during stencil traversal.

* **Tier 1: Low-Bit Geometry (`u8` / `f16`)**
* **Permeability Index (`u8`):** The 8-bit pointer into the Dual Global `f64` Lookup Tables ($K_{iso}$, $K_{fiber}$).
* **Porosity Factor (`f16`):** A geometric scalar $\phi_{geo} \in [\epsilon, 1.0]$ derived from the Skull SDF. This field encodes sub-voxel boundary information, allowing the solver to compute weighted "Cut-Cell" fluxes rather than binary staircase boundaries.

* **Tier 2: Standard State (`f32`)**
* **Pressure Solution ($P$):** The primary state variable ($x$) for the linear system. It is stored in `f32` to facilitate rapid IO and visualization.
* **Coarse Linear Operator ($A_{ii}, T_{axis}$):** Explicit stencil coefficients stored only on Levels $L_2 \dots L_D$.
* **Structure Tensor ($\mathbf{M}_0$):** The local fiber orientation tensor ($\mathbf{a}_0 \otimes \mathbf{a}_0$), stored in single-precision to maintain stability when evaluating exponential anisotropic invariants.
* **Osmotic Scalar ($C_{osm}$):** The local concentration of osmotically active solutes, rasterized from tracer particles.
* **Coagulation Factor ($\alpha_{clot}$):** A normalized scalar $\in [0, 1]$ derived from the local average particle age.
* **Role:** This field acts as the master coupling variable for rheological aging. It modulates the effective permeability $\mathbf{K}_{eff}$ (forcing it to zero as $\alpha_{clot} \to 1$) and scales the elastic retraction forces during the solid update.
* **Kinematics ($\mathbf{v}$):** The grid velocity field used for advection and momentum transfer.
* **Deformation Invariant ($\lambda_{strain}$):** The Von Mises strain invariant computed during the solid update, used to condition the Neural Renderer.

* **Tier 3: The Solver Shadow (Mixed-Precision)**
To minimize VRAM footprint while preserving barrier integrity, the PCG auxiliary vectors utilize a **Mixed-Precision Shadow Layout**. These fields are co-located in the same $L_0$ leaf block as the Pressure field to maximize cache coherency during the iterative solve.
* **The Judge ($r$ - `f32`):** The Residual vector. Stored in **Single Precision** after being computed via transient Double-Single arithmetic to detect microscopic flux leakage ($< 10^{-11}$) that would otherwise underflow in single-precision.
* **The Guide ($d$ - `f32`):** The Search Direction. Stored in single-precision. Since this vector serves only to direct the descent steps, minor floating-point quantization is acceptable and is naturally corrected by the exact residual check in subsequent iterations.
* **The Scratchpad ($z/q$ - `f32`):** A temporal buffer utilizing **Memory Aliasing** to reduce allocation overhead.
* *Phase A:* Stores the Preconditioned Residual $z = \mathcal{M}^{-1}r$ immediately following the V-Cycle.
* *Phase B:* Overwritten to store the Matrix-Vector product $q = \mathbf{A}d$ during the descent step.
* *Constraint:* This field is transient and does not persist data between solver iterations.

---

### IV. Adaptive Multi-Regime Time Integration

To efficiently simulate physical phenomena spanning orders of magnitude in time ($10^{-4}s$ to $10^5s$), the system utilizes a **Triple-Regime** execution model. The main simulation loop acts as a state machine, selecting specific compute kernels and time-step strategies based on the current kinetic energy and simulation clock.

### 4.1 The Dynamic Fluid Mask
To optimize the linear solver, the fluid domain is treated as a dynamic sparse subset of the global domain.

*   **Active Topology:** A voxel is flagged as `active_fluid`—and thus allocated in the solver's sparse matrix—if it meets one of two conditions:
    1.  It contains particles explicitly labeled as **Fluid**.
    2.  It contains **Solid** particles that have sustained structural damage ($D > \epsilon$, typically $0.1$).

*   **Rationale:** While intact tissue is effectively impermeable, damaged tissue develops micro-cracks and increased permeability. By including damaged solid voxels in the fluid solve, the system allows the pressure field to propagate into and through the injury site, physically simulating the mechanics of internal bleeding or edema expansion.

*   **SNode Optimization:** The Pressure Poisson solver iterates *only* over these active voxels using Taichi's `struct_for`. Intact tissue regions (where $D \approx 0$ and no fluid exists) are skipped. Since these skipped voxels define the domain boundary, they are implicitly treated as **Homogeneous Neumann** surfaces ($\partial P / \partial n = 0$). This strategy effectively removes millions of degrees of freedom from the linear system, focusing computational power solely on the evolving hydraulic network.

**4.2 Regime I: Dynamic Inertial (Impact Physics)**
*Time Scale: $0s < t < 1s$*
Used for rapid kinematic events (projectile impact, skull fracture, wave propagation).
* **Integration Scheme:** Explicit Symplectic Euler.
* **CFL Condition:** The time step $\Delta t_{dyn}$ is strictly limited by the acoustic wave speed in the solid:
$$ \Delta t_{dyn} \le C \frac{\Delta x}{c_{sound}}, \quad \text{typically } \Delta t \approx 10^{-4}s $$
* **Momentum Coupling:** Fluid and Solid phases exchange momentum explicitly on the grid nodes. To ensure stability, both phases advance with the same synchronized $\Delta t_{dyn}$. (Note: Sub-cycling is disabled to prevent symplectic energy drift during high-velocity impacts).

### 4.3 Regime II: Accelerated Poroelasticity (Sequential Operator Splitting)
*Time Scale: $1s < t < 3h$*
To resolve long-duration swelling events within the runtime budget, the system utilizes a **Fixed-Stress Split** architecture. This method decouples the fluid diffusion timescale from the solid inertial timescale, alternating between a physical flux calculation and a geometric relaxation step.

*   **Phase A: Fluid Macro-Step (Physical Driver)**
    The Pressure Poisson solver executes using a large physical time step of **$\Delta t_{macro} \approx 4.0s$**.
    *   **Physical Permeability:** The solver utilizes the true physical permeability tensor ($\mathbf{K}_{active} = \mathbf{K}_{physical}$) without scaling.
    *   **Flux Integration:** The implicit V-Cycle calculates the net fluid mass flux into the tissue over the 4-second interval. This results in an updated pressure field $P^{n+1}$ and a provisional increase in nodal fluid mass, creating an "over-pressurized" state relative to the current mesh volume.

*   **Phase B: Solid Relaxation (Pseudo-Time Equilibration)**
    The solid phase executes an **Explicit Relaxation Loop** to deform the mesh until it accommodates the new fluid volume computed in Phase A. This process operates in pseudo-time, iterating until mechanical equilibrium is reached.
    *   **Mass Scaling for Stability:** Nodal mass is augmented by factor **$\alpha \approx 2,000$**. Unlike Regime I, this is not used for time dilation, but solely to increase inertial resistance, preventing numerical instability when the solid is subjected to the large pressure gradients generated by the macro-step.
    *   **Kinetic Damping:** To accelerate convergence, the system monitors the total kinetic energy of the grid. If the dot product of velocity and acceleration is negative (indicating an overshoot), velocity is zeroed.
    *   **Termination Criterion:** The loop continues until the maximum nodal velocity falls below a tolerance threshold ($\|\mathbf{v}\|_{max} < \epsilon$), ensuring that internal elastic forces have balanced the hydraulic pressure ($\sum \mathbf{F} \approx 0$).

*   **Global Volume Control:**
    The global boundary pressure ($P_{halo}$) is updated once per macro-step based on the total volume error. The high-frequency derivative control used in Regime I is replaced by a proportional stiffness term, as the damping is now handled intrinsically by the solid relaxation loop.

### 4.4 Regime III: Quasi-Static Seepage (Long-Term Drainage)
*Time Scale: $t > 3h$*
Used for diffusion-dominated flow where tissue deformation has ceased and the system has reached mechanical equilibrium.

* **Trigger Condition:** Automatically activated when the maximum unbalanced force on the grid falls below a tolerance threshold for $>100$ macro-frames:
$$ \max(\|\mathbf{F}_{total}\|) < \epsilon_{tol} $$
* **Algorithm:**
1. **Particle Lock:** The Solid Phase kernels (Step 2: P2G, Update, G2P) are suspended. Particles act as static permeability maps.
2. **Fluid Diffusion (Re-Coupled):** The system solves the transient Darcy diffusion equation for fluid saturation using a Macro-Step $\Delta t_{macro} \approx 1.0s$. The Permeability LUT retains physical values ($K = K_{phys}$), and with the solid phase locked, the full diffusion throughput drives drainage towards equilibrium.
3. **Passive Advection:** Tracer particles (blood contrast) are advected through the fixed domain using the Darcy velocity field $\mathbf{v} = -\frac{\mathbf{K}}{\mu} \nabla P$.
* **Symplectic Re-entry (Wake-Up Protocol):**
If an external interaction (e.g., surgical tool SDF or new hemorrhage source) is detected:
1. **Mass Reset:** The Mass Scaling factor is strictly reset to $\alpha=1$ (Real Physics).
2. **Damping Reset:** The Local Damping coefficient is reset to $\beta=0.0$ (Inertial Physics).
3. **Unlock:** The system instantly reverts to **Regime I**, enabling high-resolution acoustic wave propagation to model the tool interaction.
