# V Cycle specification

## 1. Architectural Overview

The system is a specialized Geometric Multigrid (GMG) solver designed to resolve the elliptic Pressure Poisson Equation (PPE) on a computational domain with a virtual resolution of **$N^3$** voxels, where $N$ is a power of two determined by the active simulation profile. The solver is optimized for GPU architectures via Taichi Lang, specifically addressing the challenge of resolving hydraulic pressure fields across material contrasts exceeding $10^9$ (e.g., fluid vs. dural membranes).

The architecture is defined by three core principles:
1. **Hybrid Memory Backend:** The grid hierarchy transitions from a Sparse (Bitmasked) backend at fine resolutions to a Dense (Padded) backend at coarse resolutions to optimize bandwidth.
2. **Tri-Path Precision:** The system utilizes a tiered precision model (Robust Geometric MG, Lightweight Matrix-Free Diagonal, and Exact Residual) to dynamically arbitrate between solver throughput and physical fidelity based on the active inertial regime.
3. **Topological Isolation:** The solver mathematically isolates the active physical domain from the inactive void space via zero-flux barriers, ensuring unconditional Neumann stability without branching logic in the inner loops.

### 1.1 Simulation Profiles & Configuration Parameters

The system is parameterized by two primary configuration values that determine the resolution, memory footprint, and multigrid depth:

*   **$N$ (Virtual Grid Size):** The number of voxels per axis. Must be a power of two. The physical domain spans $N \times \Delta x$ millimeters per axis.
*   **$\Delta x$ (Grid Spacing):** The physical side length of one voxel in millimeters.

From these, the following quantities are derived:

*   **Multigrid Depth $D$:** $D = \log_2(N / 8)$. The solver spans $D+1$ discrete levels ($L_0$ to $L_D$), with the coarsest level always being $8^3$.
*   **Physical Domain Extent:** $N \cdot \Delta x$ mm per axis.

Three standard profiles are defined:

| Profile | $N$ | $\Delta x$ | Domain | Levels | Backend | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **debug** | $128$ | $2.0\text{mm}$ | $256\text{mm}$ | $L_0 \dots L_4$ (5) | GPU (Vulkan) | Fast iteration, solver tuning |
| **dev** | $256$ | $1.0\text{mm}$ | $256\text{mm}$ | $L_0 \dots L_5$ (6) | CPU | Anatomy validation, correctness |
| **prod** | $512$ | $0.5\text{mm}$ | $256\text{mm}$ | $L_0 \dots L_6$ (7) | Cloud GPU | Full resolution results |

The remainder of this specification uses the symbolic parameters $N$, $\Delta x$, and $D$ to describe all resolution-dependent quantities.

---

## 2. Grid Hierarchy & Memory Layout

The solver spans $D+1$ discrete resolution levels ($L_0$ to $L_D$), utilizing a strict power-of-two descent.

### 2.1 Level Definitions

| Level | Resolution | Backend Type | Precision | Role |
| :--- | :--- | :--- | :--- | :--- |
| **$L_0$** | **$N^3$** | **Sparse** | **Standard f32 (P2G) / Transient DS (Solver)** | Primary Physics State ($P$) |
| **$L_1$** | **$(N/2)^3$** | **Sparse** | **Transient DS / Store f32** | Matrix-Free Correction |
| **$L_2$** | **$(N/4)^3$** | **Dense** | **Hybrid Virtual-Double** | **The Precision Firewall** (Resolves $10^9$ Contrast) |
| **$L_3 \dots L_{D-1}$** | **$(N/8)^3 \dots 16^3$** | **Dense** | **Explicit f32** | High-Throughput Coarse |
| **$L_D$** | **$8^3$** | **Dense** | **Explicit f32** | Base Solver (Shared Memory) |

### 2.2 Dense Padding & Layout Strategy ($L_2 \dots L_D$)

*   **Padding Protocol:**
    All dense grid levels utilize a uniform **1-voxel padding** on all six faces. For a level with nominal resolution $N_\ell^3$ (e.g., $N_\ell = N/4$ at $L_2$), this expands the physical memory allocation to $(N_\ell+2)^3$.
    *   **Active Domain:** Indices $[1, N_\ell]$ contain valid physical state data ($P, v$) evolved by the solver.
    *   **Ghost Zone:** Indices $0$ and $N_\ell+1$ represent the inactive vacuum or boundary condition. These voxels are pre-initialized to neutral identity values (e.g., $P=0$ or $A_{ii}=1$), allowing stencil kernels to read neighbor values unconditionally without boundary-check branching.

*   **Isomorphic Tiling Constraint:**
    To enforce deterministic parallelism and maximize GPU occupancy during the Sparse $\to$ Dense transition, the active leaf block size of the $L_1$ Sparse SNode level is strictly constrained to **$8 \times 8 \times 8$**.
    *   **Warp Alignment:** During restriction ($L_1 \to L_2$), an $8^3$ fine block maps to a $4^3$ (64-voxel) coarse block. This workload aligns exactly with two full GPU Warps (64 threads), preventing execution units from stalling due to under-filled thread groups (a risk with smaller block sizes).
    *   **Mapping Guarantee:** This configuration ensures that every $2 \times 2 \times 2$ fine-voxel cluster required to compute a single coarse voxel resides strictly within the same memory block. No coarse voxel dependency ever straddles a sparse block boundary, eliminating race conditions between thread blocks.

*   **Memory Layout:**
    Data is stored in flattened **Linear C-Contiguous** arrays. This layout couples with the tiling strategy to optimize memory bandwidth.
    *   **Coalescing:** Bridge kernels transfer data in localized spatial tiles ($4 \times 4 \times 4$ coarse voxels). Because these tiles map to sequential segments in the flattened linear index space, the resulting memory traffic manifests as **Block-Coalesced** transactions rather than scattered random access.

*   **Indexing:**
    $$ I_{flat} = (i \cdot Width + j) \cdot Depth + k $$

### 2.3 Coordinate Mapping
Explicit mapping handles the transition between the Sparse Global domain and the Dense Local domain using deterministic bitwise operations.

*   **Sparse $\to$ Dense:** $\mathbf{I}_{dense} = (\mathbf{I}_{sparse} \gg k) + \mathbf{1}$.
*   **Dense $\to$ Sparse:** $\mathbf{P}_{sparse} = (\mathbf{P}_{dense} - \mathbf{1}) \ll k$.
*   **The Offset:** The addition of $\mathbf{1}$ ensures that a sparse voxel at the geometric origin $(0,0,0)$ maps to valid memory at $(1,1,1)$ in the padded dense array. The use of bitwise shifts ($\gg, \ll$) guarantees deterministic integer mapping without the latency of floating-point division.

---


### 3. Precision Strategy: The Tri-Path Model

To reconcile the conflicting requirements of high-velocity inertial dynamics (Regime I) and high-contrast hydrostatic equilibrium (Regime II), the system utilizes a three-tiered representation of the linear operator $\mathbf{A}$. This architecture decouples the **Solver Search Direction** (which prioritizes throughput and low latency) from the **Residual Evaluation** (which prioritizes exact physical conservation).

### 3.1 Path A: The Lightweight Preconditioner (Matrix-Free Diagonal)
* **Role:** The "Lightweight" path. It serves as the default preconditioner for **Regime I (Dynamic Inertial)**, where the physics is dominated by local acoustic wave propagation rather than global diffusion.
* **Storage Strategy (Matrix-Free):** This path utilizes **Zero Storage**. It does not access the cached $L_2$ arrays.
* **Execution:** The diagonal scaling factor $M^{-1}_{ii}$ is reconstructed on-the-fly during the PCG descent step. The kernel reads the local `u8` material index, retrieves the **isotropic base permeability** $k_{iso}$ from the Dual Global LUT, and sums the harmonic means of the six neighbors to compute the diagonal entry $A_{ii}$. The anisotropic fiber contribution ($k_{fiber} \cdot \mathbf{M}_0$) is intentionally omitted to preserve the zero-storage, lightweight character of this path; the exact anisotropic physics are enforced by the Path C residual evaluation.
$$ z_i = \frac{r_i}{\sum_{faces} T_{face}(k_{iso}[\mathbf{u8}])} $$
* **Latency:** By bypassing the memory clearing, upscaling, and geometric reconstruction phases required by Path B, this path allows the solver to begin iterating immediately at the start of a frame. It effectively normalizes the $10^9$ contrast ratio without the overhead of a grid hierarchy.

### 3.2 Path B: The Robust Preconditioner (Geometric Multigrid)

*   **Role:** The "Heavyweight" path. It acts as the primary preconditioner for **Regime II (Quasi-Static)** and **Regime III (Seepage)**, and serves as the rigorous fail-safe fallback when the lightweight Path A fails to converge due to complex hydrostatic bottlenecks.

*   **Storage Strategy (Cascaded Precision):** The linear operator storage utilizes a **Hybrid Virtual-Double** architecture to reconcile physical fidelity with memory bandwidth limitations.
    *   **The Firewall ($L_2$):** The dense operator at Level 2 ($(N/4)^3$) utilizes a split-precision layout to minimize memory footprint while preventing "Hydraulic Tunneling" or artificial blockage due to underflow.
        *   **Transmissibility ($T$):** Stored in **Single Precision (`f32`)**. Since hydraulic conductance values represent physical scalars within the range $[10^{-12}, 1.0]$, they fit within the standard floating-point dynamic range without quantization loss.
        *   **Diagonal ($A_{ii}$):** Stored in **Double-Single Precision (`vec2f`)**. As the diagonal represents the summation of all neighbor connections ($\sum T$), it mixes disparate magnitudes (e.g., bulk fluid $1.0$ vs. membrane $10^{-9}$). Storing the High and Low words ensures the contribution of high-resistance barriers is not lost to rounding errors during operator construction.
    *   **The Accelerator ($L_3 \dots L_D$):** Once the grid coarsens past $L_2$, geometric averaging dominates local contrasts. These levels store operators strictly in **Single Precision (`f32`)** to maximize SIMD throughput and minimize VRAM footprint during the deep V-Cycle smoothing steps.

*   **Amortization:** Because this path relies on explicit storage, it incurs a setup cost (the **"Split-Stage Block-Local"** rebuild). To maximize efficiency, these operators are **Cached**. They are not rebuilt every frame but are maintained over time using the Reactive Spectral-Guided Maintenance (RSGM) protocol.

*   **Execution & Topology:** The V-Cycle executes using **Alternating Direction Line Relaxation (ADLR)** to resolve anisotropic error modes.
    *   **At $L_2$ (Transient Double-Single / Exact Topology):** The solver utilizes **Transient Precision** arithmetic to invert the linear system along axial lines. The Thomas Algorithm (TDMA) kernel loads the `vec2f` Diagonal and `f32` Transmissibility, performing the critical pivot subtraction ($D' = A_{ii} - T \cdot C_{prev}$) using mixed-precision intrinsics. This resolves the weak coupling across semi-permeable membranes ($T \approx 10^{-9}$) by preserving the non-zero residue in the denominator, effectively forcing fluid flux through high-contrast barriers.
* **At $L_3 \dots L_D$ (`f32` / Safe-Floor Topology):** The solver operates on a clamped topology where micro-barriers are treated as **Minimum Conductance Channels ($T=\epsilon$)**. In this regime, the tridiagonal matrix remains globally coupled even across high-resistance zones. The solver resolves the global pressure distribution by enforcing a minimum viable flux through constricted channels, ensuring that pressure gradients successfully propagate across the domain to guide the fine-scale solver, while strictly maintaining topological isolation at true zero-flux boundaries.


### 3.3 Path C: The Precise Judge (The Residual)

*   **Role:** The "Ground Truth." This path calculates the exact physical error $r = b - \mathbf{A}x$ to drive global convergence and govern the transition between Path A and Path B.
*   **Storage Strategy (Matrix-Free):** Like Path A, this path does not access pre-computed stored operators. The operator is reconstructed on-the-fly from the instantaneous Lagrangian configuration of the particles.
*   **Precision (Transient Double-Single):** To prevent "Hydraulic Tunneling" (where faint fluxes are lost to catastrophic cancellation), the kernel utilizes **Transient Precision** for flux integration. While the final residual vector is stored in **Single Precision (`f32`)**, all intermediate divergence calculations are performed using **Double-Single (`vec2f`)** accumulators within the GPU registers.
*   **Reconstruction:** The kernel reconstructs the flux for each face using the `u8` material indices. These fluxes are aggregated into a local `vec2f` register using mixed-precision addition intrinsics. This technique resolves the minute net flux imbalance ($r \approx 10^{-15}$) across sealed boundaries, even when the gross flux magnitudes are large.
*   **Usage:**
    1.  **Convergence Check:** It determines when the solver has finished.
    2.  **Path Arbitration:** If the residual fails to decrease efficiently while using Path A (Diagonal), this judge triggers the "Fallback Protocol," forcing a switch to Path B (Multigrid).
    3.  **Correction:** It generates the high-precision defect vector used by both preconditioners, ensuring that even if Path A or B uses a simplified topology, the final solution strictly respects the exact boundaries.

---

### 4. Operator Topology & Initialization

This phase manages the construction of the explicit `f32` linear operators for the coarse dense levels ($L_2 \dots L_D$). To minimize computational overhead, this process is governed by a strict **Conditional Execution Protocol**.

### 4.1 Reactive Spectral-Guided Maintenance (RSGM)

The validity of the cached linear operators ($L_2 \dots L_D$) is governed by a **Reactive Protocol** rather than a predictive geometric metric. Instead of attempting to predict when the operator has degraded based on particle velocity or displacement, the system monitors the actual convergence efficiency of the linear solver. This decouples the cost of topological reconstruction from the physical kinematics, allowing the system to tolerate significant geometric drift so long as the cached operator continues to effectively guide the solver toward the solution.

**4.1.1 The Spectral Convergence Metric ($\rho$)**
The decision to refresh the preconditioner is derived strictly from the algebraic behavior of the solver. The system calculates the scalar convergence rate $\rho$ at the conclusion of every Multigrid (Path B) execution:
$$ \rho = \frac{\|r_{final}\|}{\|r_{initial}\|} $$
This metric serves as a direct proxy for the spectral quality of the preconditioner:
*   **Healthy State ($\rho < 0.4$):** Indicates that the cached operator effectively approximates the low-frequency error modes of the current physical state. The geometric mismatch is negligible regarding solver performance.
*   **Degraded State ($\rho \ge 0.6$):** Indicates that the topological divergence between the cached operator and the instantaneous physics (e.g., a channel opening that the operator thinks is closed) is retarding convergence.

### 4.1.2 Maintenance Logic & Execution States

The system acts as a state machine that persists the dense operator data across frames until a specific failure criterion is met. This persistence model decouples the cost of topological reconstruction from the simulation frame rate.

*   **State A: Cached Operator (Clean)**
    *   **Condition:** The previous frame's convergence was Healthy ($\rho < 0.4$) AND no "Fallback Trigger" occurred.
    *   **Action:** The `is_topology_dirty` flag remains **False**. The system bypasses the reconstruction pipeline entirely. The V-Cycle executes using the existing, pre-computed `f32` stencil coefficients ($A, T$), amortizing the setup cost over multiple simulation steps.

*   **State B: Active Rebuild (Dirty)**
    *   **Condition:** The previous frame's convergence was Degraded ($\rho \ge 0.6$) OR a "Fallback Trigger" was activated by the Path A solver (see Section 4.1.3).
    *   **Action:** The `is_topology_dirty` flag is set to **True**, initiating the Split-Stage maintenance sequence:
        1.  **Global Dense Clear:** The system explicitly resets the $L_2$ dense arrays to zero via a high-bandwidth memset.
        2.  **Block-Local Flux Integration:** The shared-memory reduction kernel executes over active $L_1$ blocks to rigorously populate the $L_2$ firewall.
        3.  **Recursive Propagation:** The operators are downsampled from $L_2$ through $L_D$, updating the coarse-grid accelerators.
        4.  **Reset:** The spectral monitoring state is reset for the new frame.

### 4.1.3 Regime Arbitration & The Hybrid Fallback

This protocol automatically adapts the maintenance frequency to the simulation regime by arbitrating between the solver paths (see Sections 3.1–3.3 above).

*   **Dynamic Inertial (Regime I):**
    *   **Path Selection:** The system defaults to **Path A (Matrix-Free Diagonal)**.
    *   **Maintenance:** Since Path A relies on on-the-fly stencil reconstruction and does not utilize the cached $L_2$ operators, the RSGM protocol is **Suspended**. No operator rebuilds occur, ensuring zero setup latency per frame.
    *   **Fallback Trigger:** If Path A fails to converge within a fixed iteration budget (indicating a complex hydrostatic bottleneck or "locking"), the system aborts the step, sets `is_topology_dirty = True`, and forces an immediate intra-frame switch to **Path B**. This guarantees robust convergence without paying the reconstruction cost during successful frames.

*   **Quasi-Static & Seepage (Regime II/III):**
    *   **Path Selection:** The system defaults to **Path B (Geometric Multigrid)**.
    *   **Maintenance:** The RSGM protocol is **Active**. In these regimes, particle motion is slow or zero. Consequently, the spectral health $\rho$ remains stable for long periods (tens to hundreds of frames), and the system naturally maintains State A (Clean), minimizing overhead during long-term equilibrium solves.

### 4.2 Storage Format (Persistent Cache)
The linear operators for the coarse grid levels ($L_2 \dots L_D$) are stored in persistent global memory arrays. These arrays facilitate the caching mechanism by preserving the geometric conductance of the domain between simulation steps. To optimize memory bandwidth while maintaining numerical stability at the "Precision Firewall," the storage format varies by level.

* **Format:** Symmetric 7-point Laplacian stencil.
* **Components:** The operator is decomposed into a Diagonal component ($A_{ii}$) and three axial Transmissibility components ($T_{x}, T_{y}, T_{z}$).

**4.2.1 The Hybrid Layout ($L_2$)**
At Level 2, the dense operator utilizes a mixed-precision storage strategy to accommodate the high dynamic range of the boundary conditions without saturating memory bandwidth.

* **Transmissibility ($T_{axis}$):** Stored in **Single Precision (`f32`)**.
* Since the hydraulic conductance of a face is a direct physical scalar typically ranging from $10^{-12}$ (membrane) to $1.0$ (bulk), it fits within the standard floating-point dynamic range without quantization loss.
* **Diagonal ($A_{ii}$):** Stored in **Double-Single Precision (`vec2f`)**.
* The diagonal represents the summation of all six neighbor connections ($\sum T$). This summation mixes disparate magnitudes (e.g., summing a bulk fluid connection of $1.0$ with a membrane connection of $10^{-9}$). The storage utilizes a two-component vector (High Word, Low Word) to ensure that the contribution of high-resistance barriers is preserved in the lower bits and is not erased by rounding errors against the bulk flow.

**4.2.2 The Standard Layout ($L_3 \dots L_D$)**
At the coarser levels, where geometric averaging dominates local contrasts, both the Transmissibility and Diagonal components are stored in standard **Single Precision (`f32`)**.

**4.2.3 Topological Data Mapping**
The storage indices correspond directly to the integer grid nodes $\mathbf{I} = (i,j,k)$, but the physical meaning of the data depends on the component type:

* **Transmissibility ($T_{axis}$):**
The value stored at index $\mathbf{I}$ in the $T_{axis}$ array represents the hydraulic conductance of the **Virtual Face** located at the half-integer offset $\mathbf{I} + 0.5 \cdot \mathbf{e}_{axis}$.
* Specifically, $T_{x}[\mathbf{I}]$ governs the flux connectivity between Node $\mathbf{I}$ and Node $\mathbf{I} + (1, 0, 0)$.
* This value is spatially situated at the interface between the two control volumes.

* **Diagonal ($A_{ii}$):**
The value stored at index $\mathbf{I}$ in the $A_{ii}$ array represents the total flux capacity of Node $\mathbf{I}$. It is the row sum of the matrix row corresponding to that node, calculated by summing the transmissibility values of all six faces (North, South, East, West, Top, Bottom) connected to the Voronoi cell.

* **Symmetry & Stencil Retrieval:**
To minimize memory footprint, the system stores only the positive-axis coefficients ($i \to i+1$). The full 7-point stencil for a node at $\mathbf{I}$ is reconstructed during traversal using the symmetry of the Laplacian:
* **Outgoing Flux (Positive):** Read directly from $T_{axis}[\mathbf{I}]$.
* **Incoming Flux (Negative):** Read from the neighbor's storage at $T_{axis}[\mathbf{I} - \mathbf{e}_{axis}]$.
* **Rationale:** Since the conductance of the face between $i$ and $i+1$ is identical to the conductance between $i+1$ and $i$, the single stored value at index $\mathbf{I}$ serves both nodes.

### 4.3 Split-Stage Block-Local Upscaling (Rebuild Step)

When `is_topology_dirty` is **True**, the system executes a decoupled initialization sequence to populate the cascaded precision operators ($L_2 \dots L_D$). This protocol replaces monolithic kernels with a split-stage pipeline designed to isolate memory bandwidth operations from high-precision arithmetic, maximizing GPU occupancy and ensuring compatibility with Graph API execution models.

**Stage 1: Global Dense Clear ($L_2$ Reset)**
* **Operational Scope:** A high-bandwidth kernel iterates linearly over the entire **Dense** $L_2$ memory range.
* **Action:** All Transmissibility ($T$) arrays are reset to `0.0` (Single Precision), and all Diagonal ($A_{ii}$) arrays are reset to `[0.0, 0.0]` (Double-Single Precision).
* **Rationale:** Utilizing a raw memset approach is bandwidth-limited but computationally instantaneous. It removes all conditional initialization and branching logic from the subsequent complex builder kernel, allowing the integration phase to perform purely additive operations without checking for uninitialized memory.

**Stage 2: Block-Local Flux Integration ($L_0/L_1 \to L_2$)**
* **Target:** Level 2 ($L_2$) Dense Transmissibility Arrays (`f32`).
* **Architecture:** Sparse-Driven Shared-Memory Reduction.
* **Execution Topology:** The kernel utilizes **Indirect Dispatch**, iterating strictly over the **Active Pointer Blocks** of the $L_1$ Sparse Hierarchy.
* **Thread Mapping:**
* **One Block per Tile:** One GPU Thread Block is assigned to exactly one Active $L_1$ Sparse Leaf Block ($8 \times 8 \times 8$ voxels), which maps to a $4 \times 4 \times 4$ destination region at $L_2$ (64 threads).
* **Locality:** This 1:1 mapping guarantees that all fine-scale geometric data required to compute the coarse voxel flux resides within the thread group, enabling efficient L1 caching.
* **Execution Protocol:**
1. **Collaborative Load (L1 Cache):** Threads cooperatively load the fine-scale `u8` Material Indices, LUT IDs, and the diagonal elements of the **Structure Tensor ($\mathbf{M}_0$)** from Global Memory into a `ti.simt.block.shared_array`. This consolidates memory traffic into coalesced bursts. The per-face transmissibility is computed using the axis-projected anisotropic permeability $K_{axis} = k_{iso} + k_{fiber} \cdot \mathbf{M}_0[axis, axis]$ before harmonic averaging.
2. **In-Block Reduction:** The thread block performs a parallel reduction (tree-sum) within Shared Memory to aggregate the four fine sub-face fluxes into a single coarse face flux.
3. **Coalesced Write:** The final aggregated result is written directly to the **Dense $L_2$ Transmissibility Array** in `f32`. Since spatially adjacent fine-scale fluxes typically possess similar magnitudes (e.g., all fluid or all solid), the summation into single-precision retains sufficient fidelity for the storage of the conductance term.

**Stage 3: Recursive Coarsening & The Topological Floor**

* **Target:** Levels $L_3 \dots L_D$ Dense Arrays (Single Precision).
* **Execution:** The system recursively builds the coarser operators by summing the transmissibilities of the preceding finer level (e.g., $L_2 \to L_3$, then $L_3 \to L_4$).
* **The Connectivity-Preserving Clamp ($L_2 \to L_3$):**
When downsampling from the Hybrid Firewall ($L_2$) to the Single-Precision Accelerator ($L_3$), the system applies a three-state mapping to the aggregated transmissibility ($T_{sum}$) to ensure the coarse graph remains topologically isomorphic to the fine grid.
* **Condition 1 (Physical Wall):** If $T_{sum} \equiv 0.0$, the coarse transmissibility is set to **0.0**. This preserves the integrity of true Neumann boundaries, ensuring that solid voxels and vacuum regions remain perfectly insulating.
* **Condition 2 (The Epsilon Floor):** If $T_{sum}$ is non-zero but falls below the single-precision epsilon floor (typically $T_{sum} < 10^{-7}$), the value is clamped to **$10^{-7}$**. This enforces a minimum viable conductance for narrow channels that would otherwise underflow or be absorbed during coarse-grid arithmetic.
* **Condition 3 (Bulk Flow):** If $T_{sum} \ge 10^{-7}$, the value is stored directly as **$T_{sum}$**.
* **Rationale:** This preserves the binary connectivity of the domain across the entire hierarchy. By ensuring that every valid physical path on the fine grid is represented by at least a minimal conductance on the coarse grid, the system prevents the "Zero-Correction" failure mode. This allows the coarse solver to successfully propagate global pressure gradients through constricted passages, while the high-precision smoother at $L_2$ subsequently resolves the exact flux magnitudes against the un-clamped coefficients.


**Stage 4: Diagonal Reconstruction ($L_2$ Hybrid)**
*   **Execution:** A final kernel iterates over all dense nodes to construct the matrix diagonal $A_{ii}$.
*   **Row Sum ($L_2$):** The diagonal is computed by summing the *stored* transmissibility values of the six neighbor faces. To prevent the erasure of high-resistance barrier coefficients ($10^{-9}$) by bulk fluid coefficients ($1.0$), this summation utilizes **Transient Double-Single Arithmetic**.
    *   **Logic:** The kernel reads the six `f32` neighbor transmissibilities and aggregates them into a local `vec2f` register. This aggregation utilizes **Error-Free Transformation (EFT)** logic to capture the rounding error of each addition in the lower-order word of the vector, creating an exact accumulation of the disparate magnitudes.
    *   **Storage:** The result is stored in the `vec2f` Diagonal array. This ensures that the implicitly defined barrier residue ($A_{ii} - \sum T_{fluid}$) remains non-zero and bit-exact.
*   **Row Sum ($L_3 \dots L_D$):** The sum is performed in standard `f32`, consistent with the Safe-Floor topology.
*   **Sentinel Identity:** If $A_{ii} == 0.0$ (Vacuum), the diagonal is forced to $1.0$ to maintain matrix regularity ($1 \cdot P = 0$) in void regions.

### 4.4 Geometric Upscaling Theory (Conservative Transmissibility)
To generate coarse-level operators ($L_2 \dots L_D$) that rigorously respect microscopic barriers, the system utilizes **Conservative Transmissibility Upscaling (CTU)**. This method relies on the principle that the flux capacity of a large interface is the sum of the flux capacities of its constituent sub-interfaces (the Parallel Resistor Model).

**4.4.1 Flux Conservation Principle**
Rather than averaging material properties (permeability), the system upscales the structural conductance values directly. For a coarse face $F$ composed of a set of fine faces $\{f_i\}$, the coarse transmissibility $T_F$ is defined as:
$$ T_F = \sum_{i \in F} t_{f_i} $$
This summation preserves the integral of the flux. If a set of fine faces are all blocked (transmissibility zero), their sum is zero. This prevents the "dilution" of high-resistance barriers that occurs when averaging permeability values.

**4.4.2 Recursive Hierarchy**
The upscaling process propagates from the finest dense level up to the coarsest base solver:
1.  **$L_{fine} \to L_2$:** Handled via the Fused CTU kernel (Atomic Summation of $L_0/L_1$ fluxes).
2.  **$L_k \to L_{k+1}$:** Handled via recursive summation. The transmissibility of a face at Level $k+1$ is the arithmetic sum of the four corresponding spatially-aligned sub-faces at Level $k$.

**4.4.3 Implicit Topology & Connectivity Rules**
In the coarse-grid solver ($L_2 \dots L_D$), the explicit tracking of material states (Fluid, Solid, Air) is obsoleted. Instead, the physics are governed entirely by the **Implicit Topology** of the transmissibility field ($T_{axis}$). The solver kernel blindly executes flux operations, relying on the value of $T$ to enforce complex boundary conditions and geometric constraints.

* **Active Hydraulic Connectivity ($T > 0$):**
A non-zero transmissibility value between two nodes implies the existence of a valid, face-aligned physical path on the fine grid. Because the upscaling process strictly integrates surface fluxes, a positive $T_{coarse}$ guarantees that the connected volume is not just geometrically adjacent, but topologically amenable to flow.

* **Implicit Neumann Barriers ($T = 0$):**
A transmissibility of zero represents a perfectly insulating wall. This state arises from two distinct physical scenarios, which the solver treats identically:
1. **Solid Interfaces:** A boundary between a Fluid voxel and a Solid/Vacuum voxel.
2. **Pinched Channels (Diagonal Separation):** A boundary between two Fluid voxels that connect only diagonally. Since the surface integration step in the CTU protocol rejects internal diagonal fluxes, the face transmissibility between these diagonally adjacent regions remains exactly $0.0$.
* *Effect:* In both cases, the zero-flux condition ($\partial P / \partial n = 0$) is enforced naturally. The fluid pressure reflects off the interface, strictly preserving the isolation of diagonally separated fluid pockets.

* **Implicit Dirichlet Sinks (High $T$):**
Open boundaries (Air) are encoded not as fixed nodes, but as standard voxels with high-conductance connections to a virtual sink. The transmissibility $T_{drain}$ incorporates the high permeability of the atmosphere. During the matrix solve, this high conductance dominates the flux balance, effectively pinning the local pressure to the boundary value ($P \to 0$) without requiring conditional "is_boundary" logic.

* **Topological Isolation (Vacuum Stabilization):**
Voxels that are geometrically allocated (due to padding or block granularity) but hydraulically disconnected from the domain are identified by their total flux capacity.
* **Condition:** If the sum of all incident transmissibilities is zero ($ \sum T_{faces} = 0 $), the voxel is effectively in a vacuum state.
* **Stabilization:** To prevent matrix singularity, these nodes are assigned a dummy diagonal $A_{ii} = 1.0$. This results in the trivial equation $1 \cdot P = 0$, ensuring that isolated ghost voxels do not generate floating-point errors or influence the convergence of the active physical domain.

---

### 5. Runtime Execution (The V-Cycle)

The solver executes a Multigrid V-Cycle to compute the error correction vector $z$. This process acts as the Preconditioner for the outer PCG loop.

### 5.1 Downsweep (Restriction)

The residual $r$ is projected from fine to coarse levels to capture global low-frequency error modes. The restriction operator is defined strictly as a volume integral (summation) rather than interpolation. This treats the residual as an **Extensive** physical quantity (Net Mass Flux), preserving the total error magnitude across the precision hierarchy.

**1. $L_0 \to L_1$ (Sparse $\to$ Sparse):**
*   **Matrix-Free Aggregation:** A compute kernel iterates over active $L_0$ voxels, aggregating their residual values into the corresponding active $L_1$ parent blocks.
*   **Transient Accumulation:** While the source residuals are stored in memory as `f32`, the aggregation uses **Transient Double-Single (`vec2f`)** accumulators within the kernel. This ensures that the summation of thousands of microscopic flux imbalances preserves the exact global defect magnitude before it is written to the $L_1$ array.

**2. $L_1 \to L_2$ (The Sparse $\to$ Dense Bridge)**
To guarantee race-free writes, maximize memory bandwidth, and handle vacuum topology without branching, this step is executed via the **Pre-Cleared Deterministic Gather Protocol** with **Shared-Memory Staging**.

*   **Phase A: The Vacuum Reset (Global Memset):**
    Prior to kernel launch, the target **Dense $L_2$ Residual Array** is explicitly reset to `0.0` via a high-bandwidth `memset`. Since the subsequent compute kernel iterates *only* over Active Sparse Blocks, this pre-clear guarantees that any dense voxel corresponding to unallocated sparse void correctly retains the value `0.0` (zero error). This handles the "Vacuum" topology implicitly.

*   **Phase B: The Owner-Computes Gather Kernel:**
    *   **Dispatch Topology:** The kernel utilizes **Indirect Dispatch**, launching one thread block for every Active $L_1$ Pointer Block.
    *   **Thread Mapping:** Threads are mapped 1:1 to the **Destination Coarse Voxels** in the $L_2$ block (64 threads per block).
    *   **Execution Protocol:**
        1.  **Bitwise Mapping:** The thread calculates its logical target coordinate using deterministic bitwise operations: $\mathbf{I}_{dense} = (\mathbf{I}_{fine\_base} \gg 1) + \mathbf{1}$.
        2.  **Register-Level Gather:** The thread explicitly loads the eight constituent fine-scale residuals ($2 \times 2 \times 2$) from the SNode directly into private registers.
        3.  **Transient Reduction:** The summation is performed within the register file using **Transient Double-Single (`vec2f`)** arithmetic to capture precision across the scale change.
        4.  **Shared-Memory Staging:** The reduced scalar result is cast to `f32` and written to a **Block-Local Shared Memory** buffer. This step is critical for decoupling the sparse-grid thread indexing (often Morton/Z-curve) from the dense-grid linear layout.
        5.  **Block Synchronization:** A hardware barrier (`ti.simt.block.sync()`) ensures all 64 threads have populated the staging buffer.
        6.  **Coalesced Global Write:** The threads cooperatively flush the data from Shared Memory to the **Dense $L_2$ Array**. By reading from the reordered shared buffer, the warp generates strictly contiguous, aligned write transactions to Global Memory, achieving maximum bus utilization.

**3. $L_2 \to L_3$ (The Precision Transition):**
*   **Volume Aggregation:** The kernel executes a fixed-stride reduction to sum the residuals of the eight spatially corresponding $L_2$ child voxels into the $L_3$ parent voxel.
*   **Precision Casting:** This step handles the transition from the "Firewall" to the "Accelerator." The aggregated **Double-Single** residual sum is cast to **Single Precision (`f32`)** for storage in the $L_3$ array.
*   **Validity:** Since the residual represents a macroscopic mass deficit—and local high-frequency errors have already been resolved by the $L_2$ smoother—this casting is numerically safe for the coarse-grid correction.

**4. $L_3 \to L_D$ (Dense $\to$ Dense):**
*   **Recursive Summation:** For the remaining levels, the kernel performs standard `f32` summation. For every coarse voxel, the residuals of the eight child voxels are added together:
    $$ r_{coarse}^{(i,j,k)} = \sum_{x=0}^1 \sum_{y=0}^1 \sum_{z=0}^1 r_{fine}^{(2i+x, 2j+y, 2k+z)} $$
*   **Conservation of Defect:** This summation maintains the integral of the residual. The global mass error calculated at the coarsest level ($L_D$) is algebraically equivalent to the global mass error at the finest level ($L_0$), ensuring the base solver addresses the true net defect of the system.
*   **Topological Consistency:** Because ghost voxels (padding) are pre-initialized to $0.0$, summation operations at boundaries naturally contribute zero flux to coarser vacuum regions, maintaining correct domain topology without conditional branching.

---

### 5.2 The Bottom Solver ($L_D$)
The coarsest level ($8^3$) represents the global equilibrium and is solved exactly using a single GPU thread block. This kernel is highly optimized for register throughput and L1 cache latency, bypassing Global Memory during the iterative solution.

* **Shared Memory Allocation Strategy:**
* **Container:** The kernel utilizes `ti.simt.block.shared_array` to explicitly manage the solution state in L1 Shared Memory.
* **Bank-Aware Padding:** While the logical grid dimensions are $10 \times 10 \times 10$ (Active + Ghost), the shared memory array is allocated with **X-axis padding** (e.g., physically $12 \times 10 \times 10$). This padding stride ensures that threads within the same Warp access distinct shared memory banks when reading column-wise neighbors, thereby preventing serialization due to bank conflicts.

* **Thread Topology (Bitwise Unprojection):**
* **Configuration:** The kernel is launched with a single block of **512 threads**, providing a 1:1 mapping between threads and active voxels.
* **Index Mapping:** To eliminate the latency of integer division and modulo arithmetic within the inner loop, the kernel maps the linear thread ID ($tid \in [0, 511]$) to logical coordinates $(x,y,z)$ via single-cycle bit manipulation:
$$ x = tid \ \& \ 7; \quad y = (tid \gg 3) \ \& \ 7; \quad z = tid \gg 6 $$
* **Offset:** These logical coordinates are offset by $+1$ to align with the active region of the padded shared memory container ($1 \dots 8$).

* **Algorithm: Synchronized Red-Black Gauss-Seidel:**
The solver executes for a fixed number of iterations ($N=64$) to ensure convergence to machine precision. Each iteration consists of two strictly ordered phases to guarantee data consistency:
1. **Parity Evaluation:** Threads determine their update phase via bitwise parity: $P = (x + y + z) \ \& \ 1$.
2. **Red Phase:** Threads with $P=0$ read neighbors from Shared Memory, compute the stencil, and update their local value.
3. **Barrier I:** `ti.simt.block.sync()` is executed. This hardware barrier enforces that all Red updates are committed to memory before any Black thread reads data.
4. **Black Phase:** Threads with $P=1$ read the updated Red neighbors and perform their update.
5. **Barrier II:** `ti.simt.block.sync()` is executed to ensure block-wide consistency before the start of the next iteration.

* **Coalesced IO:**
* **Load:** At kernel start, threads cooperatively load the operator $\mathbf{A}$, RHS $b$, and initial guess $z$ from Global Memory into the padded Shared Memory structure. Boundary voxels (Ghosts) are strictly initialized to zero.
* **Writeback:** Upon completion of the iteration loop, threads write the converged solution from Shared Memory back to Global Memory using the computed linear index.

### 5.3 Upsweep (Prolongation)

The correction vector $z$ is interpolated back up the grid hierarchy to update the fine-scale solution. To prevent "Correction Bleeding"—where high-magnitude pressure corrections in fluid channels leak into neighboring solids due to geometric averaging—the system utilizes a **Matrix-Dependent Prolongation** strategy combined with robust line smoothing.

**1. Recursive Prolongation ($L_D \to \dots \to L_2$):**
The system iteratively upscales the correction from the coarsest level ($L_D$) down to the precision firewall ($L_2$) using physics-aware operators.

* **Operator-Induced Interpolation:** The interpolation weights for a child voxel are not fixed geometric constants but are derived dynamically from the **Transmissibility ($T$)** coefficients of the **Child (Destination) Level**.
* **The Physical Throttle ($L_3 \to L_2$):** Crucially, when prolonging from the Accelerator ($L_3$) to the Firewall ($L_2$), the weights are calculated using the **Exact Hybrid Transmissibility** stored at $L_2$ (e.g., $10^{-9}$), *not* the clamped Epsilon value used at $L_3$ ($10^{-7}$). This ensures that while the coarse solver was allowed to pass flux easily to maintain connectivity, the resulting correction magnitude is mathematically restricted by the true physical resistance before entering the fine-grid state.
* **Solid/Vacuum Isolation:** If the destination face represents a physical wall ($T \equiv 0.0$), the interpolation weight resolves strictly to zero. This ensures the correction vector is confined to the active hydraulic volume and never leaks into solid matter.
* **Channel Continuity:** If the destination face has non-zero conductance (even if microscopic), the interpolation generates a non-zero weight, allowing the pressure update to traverse the channel.
* **Precision Hand-off:** The correction values remain in **Single Precision (`f32`)**. The high-fidelity physics are enforced by the *Operator's* Hybrid precision ($A_{ii}$ as `vec2f`) during the subsequent smoothing step, not by the storage precision of the correction vector itself.
*   **Post-Smoothing (ADLR):** Immediately following interpolation, a single pass of **Alternating Direction Line Relaxation (ADLR)** is applied using the **Pencil-Staged Hybrid Kernel**. By utilizing the Shared-Memory staging protocol to load the $L_2$ operator, this step performs high-precision **Transient Double-Single** smoothing without incurring strided memory penalties. This relaxes the interpolated correction against the local operator, removing high-frequency interpolation errors and resolving anisotropic discontinuities introduced by the resolution change.

**2. $L_2 \to L_1$ (The Dense $\to$ Sparse Bridge):**
* **Gather Kernel:** The kernel iterates over the **active** blocks of the destination sparse level ($L_1$).
* **Coordinate Mapping:** The kernel calculates the sampling coordinate in the continuous dense index space:
$$ \mathbf{P}_{sample} = (\mathbf{I}_{sparse} \times 0.5) + \mathbf{1.0} $$
* **High-Fidelity Sampling:** The kernel performs **Trilinear Interpolation** directly against the $L_2$ **Single-Precision** array. Because the $L_2$ field has been rigorously constrained by the Operator-Induced Prolongation and Hybrid ADLR smoother, the values at the boundaries are sharp and physically valid despite the lower storage precision.
* **Boundary Safety:** Implicit Neumann conditions are preserved during sampling. When $\mathbf{P}_{sample}$ falls near the edge of the active domain, the filter blends the valid physical value with the ghost value (guaranteed to be $0.0$), correctly interpolating the correction to zero at the boundary without conditional branching.

**3. $L_1 \to L_0$ (Sparse $\to$ Sparse):**
* **Final Interpolation:** The correction is projected from the intermediate sparse level to the finest physics grid ($L_0$) using standard linear weighting.
* **Smoothing:** A final smoothing pass is applied using the matrix-free operator. This step utilizes **Transient Double-Single Arithmetic** for the stencil accumulation to eliminate any remaining grid-aligned artifacts, ensuring the final update vector is perfectly smooth relative to the instantaneous particle configuration.

### 5.4 Asynchronous Consistency & Stagnation Failsafe

To reconcile the computational efficiency of the Amortized Geometric Multigrid with the strict accuracy requirements of the physical simulation, the solver employs an asynchronous consistency protocol. This mechanism ensures that the physical validity of the solution is never compromised by the temporal latency of the coarse-grid operators.

**5.4.1 Decoupled Precision Roles**

The solver architecture distinctly separates the definition of the physical error from the approximation of the search direction.

*   **The Physical Judge (The Residual):**
    The convergence criterion is governed exclusively by the residual vector $r = b - \mathbf{A}_{exact}x$. This calculation is performed on the $L_0$ grid using the **current frame's** particle distribution via the `u8` lookup. Because the residual kernel always evaluates the instantaneous geometry of the material points, the solver enforces the current boundary conditions regardless of the state of the coarse operators.

*   **The Heuristic Guide (The Preconditioner):**
    The V-Cycle serves to generate an approximate correction vector $z \approx \mathbf{A}^{-1}r$. By utilizing the cached `f32` operators ($L_2 \dots L_D$), the preconditioner directs the solver based on the cached topological state, which may contain slight temporal latencies regarding barrier states.

*   **Self-Correction:**
    If the cached preconditioner generates a search direction that violates the current physics (e.g., suggesting flux through a recently closed barrier), the exact residual calculation in the subsequent PCG iteration detects the resulting non-zero error. The Conjugate Gradient algorithm naturally orthogonalizes against this invalid component, generating a corrective search vector that strictly respects the true boundary.

### 5.4.2 The Spectral Watchdog (Consistency Protocol)

To reconcile the computational efficiency of cached operators with the potential for "Topological Snapping" (e.g., instantaneous tissue rupture or valve closure), the system employs an active spectral monitoring protocol. This mechanism detects when the cached preconditioner ($L_2 \dots L_D$) has become spectrally orthogonal to the physical reality ($L_0$), preventing solver cycles from being wasted on a divergent system.

**5.4.2.1 Convergence Rate Metric**
At every iteration $k$ of the PCG loop, the solver computes the scalar convergence rate $\rho_k$, defined as the ratio of the current residual norm to the previous residual norm:
$$ \rho_k = \frac{\|r_k\|}{\|r_{k-1}\|} $$
In a healthy Multigrid V-Cycle, this ratio is typically low ($0.1 < \rho < 0.3$), indicating that the preconditioner is effectively suppressing error modes. A high ratio ($\rho \to 1.0$) indicates that the search direction suggested by the cached operator is ineffective reducing the physical error.

**5.4.2.2 Dual-Stage Trigger Logic**
The Watchdog evaluates two distinct failure modes based on the iteration phase:

* **Trigger A: The Early Abort (Spectral Divergence)**
* **Scope:** Active during the initial iterations ($k < 3$).
* **Condition:** $\rho_k > 0.7$.
* **Rationale:** If the residual fails to drop significantly in the first few steps, it implies a critical topological mismatch (e.g., the preconditioner assumes an open channel where the physics defines a solid wall). The solver immediately infers that the cached operator is invalid.
**Trigger B: The Late Abort (Stagnation)**
* **Scope:** Active after the nominal convergence budget ($k > K_{limit} \approx 15$).
* **Condition:** $\|r\| > \epsilon_{tol}$.
* **Rationale:** If the solver fails to converge within the expected budget despite a healthy start, it implies accumulated geometric drift or numerical conditioning issues that require a fresh operator.
* **Regime II Exception:** During the initial V-Cycles of a Fluid Macro-Step, the absolute tolerance $\epsilon_{tol}$ is relaxed. Because the solver integrates fluxes over a large physical interval ($4.0s$), high initial residual magnitudes are physically valid. In this context, the watchdog suspends the absolute check and relies strictly on the relative convergence rate ($\rho$) to detect stagnation.

**5.4.2.3 Emergency Recovery Routine**
If either trigger condition is met, the solver executes an interrupt routine:
1. **Interrupt:** The PCG iteration loop is terminated immediately.
2. **Force Dirty State:** The `is_topology_dirty` flag is forcibly set to **True**.
3. **Immediate Reconstruction:** The execution pipeline suspends the solver logic and runs the "Fused Upscaling" kernel to rebuild the dense operators ($L_2 \dots L_D$) based on the current instantaneous $L_0$ configuration.
4. **Solver Restart:** The PCG state is reset. The current pressure estimate $x$ is retained as the initial guess to preserve progress, but the search directions $d$ and residuals $r$ are re-initialized against the newly reconstructed operators.

This protocol ensures that "Water Hammer" events—where topology changes abruptly—are detected and resolved with a latency of only 2-3 iterations, preserving real-time performance.

### 5.5 PCG Outer Loop Arithmetic: The Masked Reduction Protocol

The Conjugate Gradient solver computes global scalar coefficients using a topology-aware inner product defined strictly over the active physical domain $\Omega$. This protocol formally decouples the memory allocation granularity from the algebraic degrees of freedom.

#### 5.5.1 The Masked Inner Product
For any two scalar vector fields $\mathbf{x}$ and $\mathbf{y}$ defined on the $L_0$ sparse grid, the inner product $\langle \mathbf{x}, \mathbf{y} \rangle_{\Omega}$ is defined as the summation of the element-wise product weighted by a binary characteristic function $\chi_{\Omega}$:

$$ \langle \mathbf{x}, \mathbf{y} \rangle_{\Omega} = \sum_{\mathbf{I} \in \mathcal{L}_0} \chi_{\Omega}(\mathbf{I}) \cdot (x_{\mathbf{I}} \cdot y_{\mathbf{I}}) $$

This operation accumulates contributions exclusively from voxels that represent valid, evolving degrees of freedom in the linear system, mathematically nullifying values residing in padding or fixed-boundary regions.

#### 5.5.2 The Characteristic Function ($\chi_{\Omega}$)
The weighting function $\chi_{\Omega}(\mathbf{I})$ is derived directly from the Tier 1 Material Index (`u8`) stored at voxel $\mathbf{I}$. It acts as a coefficient filter during the reduction kernel execution.

* **Active Fluid/Tissue (Material ID `1` $\dots$ `254`):**
* **Value:** $\chi_{\Omega} = 1.0$.
* **Role:** These voxels are the primary unknowns (Degrees of Freedom). Their vector components are fully accumulated into the global sums.

* **Vacuum/Solid (Material ID `0`):**
* **Value:** $\chi_{\Omega} = 0.0$.
* **Role:** While these voxels may exist within allocated $4 \times 4 \times 4$ SNode blocks due to the sparsity pattern, they carry trivial identity equations ($1 \cdot P = 0$). The mask forces their contribution to zero.

* **Air Halo (Material ID `255`):**
* **Value:** $\chi_{\Omega} = 0.0$.
* **Role:** These voxels act as **Dynamic Dirichlet Boundaries** ($P=P_{halo}$). Although allocated to maintain topological connectivity, they are mathematically fixed for the duration of the solve. Excluding them from the reduction ensures the search direction remains orthogonal to the boundary constraints (i.e., the solver does not attempt to "correct" the boundary value set by the Monro-Kellie controller).

### 5.5.3 Kernel Execution Strategy

The reduction operations are implemented using a topology-aware execution model that maximizes memory bandwidth efficiency while ensuring strictly branchless instruction pipelines within the compute units.

* **Indirect Sparse Traversal:**
The reduction kernel utilizes **Indirect Dispatch**, iterating strictly over the **Active Block List** generated during the Topology Maintenance phase (Step 0). By reusing the same active index buffer employed by the Operator Builder, the solver eliminates the overhead of re-traversing the SNode tree structure. This ensures that unallocated regions of the global domain are skipped entirely at the hardware dispatch level, providing a static resource binding compatible with Graph API capture.

* **Branchless Arithmetic Gating:**
Within an active block, the logic retrieves the `u8` Material ID and computes the characteristic function $\chi_{\Omega}$ as a scalar float ($0.0$ or $1.0$). The accumulation is performed via arithmetic masking (`sum += mask * value`) rather than conditional control flow. This strategy preserves SIMD coherency within the warp, ensuring that threads corresponding to vacuum padding (ghost voxels) execute no-op instructions without causing execution divergence or serialization of the active fluid threads.

#### 5.5.4 Integration into Descent Steps
This masked arithmetic governs the two critical scalar updates in the Conjugate Gradient loop:

1.  **The Step Size ($\alpha$):**
    Calculated using the masked dot product of the search direction $d$ and the matrix-vector product $q = \mathbf{A}d$:
    $$ \alpha = \frac{\langle r, z \rangle_{\Omega}}{\langle d, q \rangle_{\Omega}} $$
    This ensures the step length is derived solely from the energy minimization within the fluid volume.

2.  **The Orthogonalization Scalar ($\beta$):**
    Calculated using the masked dot product of the residual $r$ and the preconditioned residual $z$:
    $$ \beta = \frac{\langle r_{new}, z_{new} \rangle_{\Omega}}{\langle r_{old}, z_{old} \rangle_{\Omega}} $$
    This ensures the new search direction remains conjugate to the previous direction relative to the active fluid operator, independent of the values in the ghost or vacuum padding.

---

## 6. Numerical Implementation Details

### 6.1 Alternating Direction Line Relaxation (ADLR)

To resolve anisotropic error modes, the system utilizes a **Tridiagonal Line Solver** (Line Gauss-Seidel). The smoothing operation decomposes into three sequential axial sweeps (X, Y, Z). In each pass, GPU threads map to the cross-sectional plane orthogonal to the active axis, solving coupled 1D Tridiagonal systems ($A\mathbf{x}=\mathbf{b}$) using the **Thomas Algorithm** (TDMA):
$$ -T_{i-1/2} z_{i-1} + A_{ii} z_i - T_{i+1/2} z_{i+1} = \hat{r}_i $$

Due to the linear C-Contiguous memory layout (where $k$/Z is the fast axis), simple iteration along the $i$/X and $j$/Y axes results in strided memory access. To reconcile this, the solver utilizes a **Dual-Kernel Memory Architecture**.

**1. The Native-Axis Kernel (Z-Sweep)**
*   **Execution:** Solves lines along the contiguous $k$-axis.
*   **Mapping:** Threads map directly to physical lines ($1 \text{ Thread} \leftrightarrow 1 \text{ Line}$).
*   **Memory Access:** Since adjacent threads process adjacent memory addresses ($k, k+1, \dots$), memory coalescing is intrinsic. The kernel executes directly against Global Memory buffers.

**2. The Transposed-Axis Kernel (X/Y-Sweeps)**
*   **Execution:** Solves lines along the strided $i$ and $j$ axes using the **Pencil-Staging Protocol**. This strategy decouples the **Memory Topology** (Load/Store) from the **Solver Topology** (Compute) using L1 Shared Memory as a high-bandwidth transposition buffer.
*   **Phase A: Coalesced Load (The Tile):** A Thread Block (e.g., Warp of 32) cooperatively loads a spatial tile of 32 parallel lines from Global Memory. Crucially, threads read using **Z-Major Indexing** regardless of the active solver axis. This ensures strictly coalesced, 128-byte aligned transactions that saturate the VRAM bus.
*   **Phase B: In-Place Transpose:** The tile is stored in a `ti.simt.block.shared_array` utilizing **Bank-Aware Padding** (Stride $+1$) to prevent bank conflicts. A hardware barrier (`sync`) aligns the block.
*   **Phase C: Shared-Memory Solve:** Threads logically re-map their IDs to select a specific "Pencil" (Row/Column) from the Shared Memory buffer. The Thomas Algorithm executes entirely within the high-bandwidth L1 cache and registers, independent of Global Memory latency.
*   **Phase D: Coalesced Writeback:** Threads revert to Z-Major indexing to flush the results from Shared Memory back to Global Memory in coalesced bursts.

**3. Precision Variants & Constraints ($L_2$ vs. Coarse)**
The line solver is compiled in two distinct variants to support the cascaded precision model within the constraints of the Shared Memory architecture.

*   **The Hybrid Firewall Kernel ($L_2$):**
    This kernel implements a **Transient Double-Single** variant of the Thomas Algorithm.
    *   **Capacity:** At $L_2$ ($(N/4)^3$), a single grid line contains $N/4$ voxels. The storage required per line—Diagonal (`vec2f`), Transmissibility (`f32`), and RHS (`f32`)—fits within the Shared Memory capacity of a single Streaming Multiprocessor (SM), enabling the Pencil-Staging protocol without register spilling.
    *   **Hybrid Load:** The Diagonal is loaded into the shared tile as `vec2f` (High/Low words), while Transmissibility is loaded as `f32`.
    *   **Transient Pivot:** During the forward elimination phase within Shared Memory, the denominator calculation ($D' = A_{ii} - T \cdot C_{prev}$) is performed using mixed-precision intrinsics. This utilizes **Error-Free Transformation (EFT)** logic to capture the subtraction error in the lower-order word of the `vec2f` register. This guarantees that microscopic barrier residues ($10^{-9}$) are preserved even when subtracted from bulk fluid coefficients ($1.0$).

*   **The Accelerator Kernel ($L_3 \dots L_D$):**
    This kernel is instantiated with **Single Precision (`f32`)**.
    *   **Topology:** It operates on the "Safe-Floor" topology ($T \ge \epsilon$).
    *   **Execution:** Since grid dimensions are small ($(N/8)^3$ and below), the entire pencil allows for full register residency during the sweep. The solver equilibrates pressure across high-contrast restrictions by treating them as minimum conductance channels, ensuring valid search directions for the global system.

**Boundary & Vacuum Logic:**
Explicit padding voxels (Indices $0$ and $N+1$) serve as fixed boundary conditions. During the **Coalesced Load** phase, these ghost values are pulled into the Shared Memory tile along with the active data. This enables branchless execution of the Thomas Algorithm within the tile. For Vacuum voxels residing within the active range ($A_{ii}=1, T=0$), the system simplifies to the identity equation ($1 \cdot z = 0$), ensuring the correction vector remains stable at $0.0$ in the void.


### 6.2 Residual Calculation (Precise Path)

The residual vector $r$ represents the local imbalance between the target source terms and the computed mass flux ($r_i = b_i - (\nabla \cdot \mathbf{J})_i$). It acts as the strict physical judge of convergence. To resolve microscopic net flux imbalances ($r \approx 10^{-15}$) arising from the summation of large opposing flows ($Q_{in} \approx Q_{out} \approx 1.0$), the kernel utilizes **Transient Double-Single Arithmetic**. While the final residual is stored in memory as `f32` to save bandwidth, all intermediate flux reconstruction and divergence accumulations occur within high-precision registers.

**1. Accumulator Initialization & Masking**
The kernel initializes a local high-precision accumulator (`vec2f`) to zero. Material masking is applied immediately:
* **Air/Ghost:** If the node represents an open boundary or padding, the residual is forced to $0.0$. (The error at a Dirichlet node is definitionally zero as the value is enforced explicitly by the boundary condition).
* **Fluid/Tissue:** If the node is active, the kernel proceeds to flux integration.

**2. High-Fidelity Interface Flux Reconstruction**
The divergence is computed by summing the signed fluxes $Q$ entering or leaving the node through its six faces. Crucially, the calculation must respect the dynamic boundary value set by the Volume Compensator.

* **Anisotropic Permeability Retrieval (`f64`):** The kernel reads the `u8` material index and retrieves the physical permeability pair $(k_{iso}, k_{fiber})$ from the **Dual Global LUT** in **Double Precision (`f64`)**. The kernel then reads the Structure Tensor diagonal element $\mathbf{M}_0[axis, axis]$ from the Tier 2 grid field to compute the axis-projected permeability:
$$ K_{axis} = k_{iso} + k_{fiber} \cdot \mathbf{M}_0[axis, axis] $$
This ensures that the directional permeability for high-contrast barriers (e.g., $10^{-11}$) and fiber-aligned flow paths is loaded with full bit-wise fidelity. For isotropic materials ($k_{fiber} = 0$), this reduces to the scalar lookup.
* **Transient Flux Calculation (`vec2f`):** The transmissibility $T_{face}$ (harmonic mean of the axis-projected $K_{axis}$ values of the two adjacent nodes) and the pressure difference $\Delta P$ are computed.
$$ Q_{face} = T_{face} \cdot (P_{neighbor} - P_{center}) $$
* **Boundary Lookup Protocol:** To enforce the Monro-Kellie logic, the kernel conditionally retrieves $P_{neighbor}$ based on the neighbor's material state:
* **Active Fluid (1..254):** Read $P$ directly from the grid state.
* **Air Halo (255):** Read the global scalar $P_{halo}$ (Volume Compensator). This allows the boundary to exert active pressure (positive or negative) rather than just passive drainage.
* **Vacuum (0):** The flux term $Q_{face}$ is implicitly masked to $0.0$ (Neumann condition).
* **Compensated Accumulation:** The computed `vec2f` flux is added to the local accumulator. The High-Word/Low-Word structure explicitly captures the rounding error of each addition, ensuring that the algebraic sum of fluxes across a sealed membrane sums to exactly zero.

**3. Divergence, Osmosis & Storage**
The final residual is the difference between the composite target source ($b$) and the computed net flux ($Ax$).
* **Augmented Source Construction ($b$):** The kernel constructs the effective Right-Hand Side (RHS) by combining mechanical and chemical driving forces.
* **Kinematic Term:** The divergence of the intermediate velocity field ($\nabla \cdot \mathbf{v}^*$) represents the hydrostatic compression or expansion required by mass conservation.
* **Osmotic Term:** The kernel retrieves the **Osmotic Scalar ($C_{osm}$)** from Tier 2 memory. This value is converted into a Starling potential $\Psi$, representing the oncotic suction of local solutes. This term is added to the kinematic divergence, effectively defining a "virtual sink" that drives fluid influx in regions of high concentration.
* **Computation:** The composite source term $b_{total}$ is subtracted from the accumulated flux total using double-single subtraction.
* **Downcast & Storage:** The resulting scalar represents the net defect of the system (Hydrostatic + Osmotic). Since the magnitude of the error is significantly smaller than the magnitude of the constituent fluxes (typically $< 10^{-6}$), the high-precision result is safe to cast to **Single Precision (`f32`)** for storage in the global residual array. This maintains the necessary physical accuracy for the convergence check while minimizing memory bandwidth.

### 6.3 Boundary Logic: The "Explicit Halo" Topology

The solver utilizes a **Tri-State Topology** to distinguish between sealed internal voids (Vacuum) and open drainage surfaces (Air). This logic is implemented via an **Explicit Sparse Halo** strategy, which decouples the definition of the physical boundary (determined by geometry during the topology update) from the enforcement of the boundary condition (determined by memory state during the solver loop). By explicitly allocating the boundary layer in the sparse grid, the inner solver kernels remain branchless regarding geometric calculations.

* **State 1: The Active Domain (Fluid/Tissue)**
* **Definition:** Voxels containing active particles ($N_p > 0$) or damaged solid material points. These are assigned Material IDs `1..254`.
* **Solver Behavior:** These voxels are actively solved for the Pressure field $P$ (Degrees of Freedom).
* **Interface Physics:** Connections between two active voxels represent standard internal flow. The transmissibility $T$ is calculated as the harmonic mean of the **axis-projected permeability** $K_{axis}$ of the two adjacent nodes, where $K_{axis} = k_{iso} + k_{fiber} \cdot \mathbf{M}_0[axis, axis]$ for the face normal direction. This Two-Point Flux Approximation (TPFA) governs mass-conserving anisotropic Darcy flow.

* **State 2: The Air Halo (Dynamic Monro-Kellie Boundary)**
* **Definition:** A sparsely allocated boundary layer (one voxel thick) representing the interface between the fluid domain and the skull/atmosphere. This state is assigned the reserved Material ID `255`.
* **Allocation Protocol:** This layer is not static; it is dynamically "painted" during the topology update phase (Step 0, Action C). Any unallocated neighbor of an Active Fluid voxel is queried against the global Skull SDF. If the neighbor lies outside the skull domain (`SDF > 0`), it is explicitly activated in the sparse tree and tagged as Air.
* **Solver Behavior:** The Air node acts as a **Dynamic Dirichlet Boundary** driven by the global volume error. It is not solved for pressure; instead, it enforces a time-variant boundary value on its active neighbors:
$$ P_{boundary} \equiv P_{halo} $$
Here, $P_{halo}$ is the global correction scalar computed by the Alpha-Coupled PD Controller in Step 0.
* **Interface Physics:** Flux exchange between an Active voxel and an Air voxel is driven by the pressure differential $\Delta P = P_{halo} - P_{internal}$.
* **Standard Drainage:** If global volume is stable ($\dot{V} \approx 0$) and near target, the boundary acts as a standard atmospheric sink.
* **Damped Restoration:** If the volume deviates, the boundary acts as a **Viscoelastic Membrane**. The $P_{halo}$ value applies a restoring force proportional to the error, but subtracts a damping force proportional to the expansion rate. This effectively "catches" the expanding brain tissue, slowing it down before it hits the skull limit to prevent recoil.

* **State 3: The Vacuum (Sealed Void)**
* **Definition:** Any voxel that remains **Unallocated** in the Sparse Hierarchy ($L_0$) or is explicitly flagged as a solid wall (Material ID `0`).
* **Implicit Logic:** By definition, the solver kernels iterate only over active SNodes. Unallocated space is skipped.
* **Solver Behavior:** This state enforces a **Homogeneous Neumann Condition** ($\partial P / \partial n = 0$).
* **Interface Physics:** If an Active voxel neighbors a Vacuum voxel (which occurs when the neighbor's SDF query indicated it was "Inside" the skull/bone), the connection is treated as a solid wall. The transmissibility is strictly forced to $0.0$, causing fluid pressure to reflect off the boundary with zero mass flux leaving the domain.

* **Coarse-Scale Consistency ($L_2 \dots L_D$):**
* **Upscaling:** The "Fused CTU" kernel respects this tri-state logic when aggregating fluxes for the coarse dense arrays.
* **Air Flux:** Fluxes across Active-Air interfaces are summed into the coarse transmissibility. This ensures that the dynamic boundary pressure ($P_{halo}$) propagates correctly from the coarse grid down to the fine grid during the V-Cycle.
* **Vacuum Flux:** Fluxes across Active-Vacuum interfaces sum to zero, ensuring that sealed boundaries remain sealed even at the coarsest level of the Multigrid hierarchy.
