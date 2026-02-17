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
| **debug** | $256$ | $2.0\text{mm}$ | $512\text{mm}$ | $L_0 \dots L_5$ (6) | GPU (Vulkan) | Fast iteration, solver tuning |
| **dev** | $512$ | $1.0\text{mm}$ | $512\text{mm}$ | $L_0 \dots L_6$ (7) | CPU | Anatomy validation, correctness |
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

===========================================================================

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


=========================================================================
# Mathematical Specification

### I. Continuum Mixture Framework

**1.1 Domain Definition**
The computational domain $\Omega \subset \mathbb{R}^3$ is modeled as a saturated porous continuum composed of two immiscible phases: a solid skeleton ($\alpha = s$) and a fluid phase ($\alpha = f$).
*   **Volume Fractions:** Let $\phi^\alpha(\mathbf{x}, t)$ denote the volume fraction of phase $\alpha$. The saturation constraint is:
    $$ \phi^s + \phi^f = 1 $$
*   **Mixture Density:** The apparent density $\rho_{mix}$ at any point is:
    $$ \rho_{mix} = \phi^s \rho_s + \phi^f \rho_f $$
    where $\rho_s$ and $\rho_f$ are the intrinsic densities of the solid (tissue) and fluid (blood/CSF).

**1.2 Governing Conservation Laws**
Assuming the phases are intrinsically incompressible (though the mixture is compressible due to drainage), the conservation of momentum for the mixture is:
$$ \rho_{mix} \dot{\mathbf{v}} = \nabla \cdot \boldsymbol{\sigma}_{eff} + \rho_{mix} \mathbf{g} $$
where $\dot{\mathbf{v}}$ is the material time derivative of velocity, $\mathbf{g}$ is gravitational acceleration, and $\boldsymbol{\sigma}_{eff}$ is the effective stress tensor combining solid elasticity and fluid pressure:
$$ \boldsymbol{\sigma}_{eff} = \boldsymbol{\sigma}_{s} - \phi^f P \mathbf{I} $$

---

### II. Spatial Discretization (MLS-MPM & Finite Volume)

### 2.1 Particle State (Lagrangian)
The solid phase is discretized into $N_p$ material points which function as quadrature points for the continuum equations. Each particle $p$ acts as a persistent container for physical state data, carrying the following attributes throughout the simulation:

* **Kinematics:**
* **Position ($\mathbf{x}_p \in \mathbb{R}^3$):** The evolving spatial coordinate in the global frame.
* **Velocity ($\mathbf{v}_p \in \mathbb{R}^3$):** The linear velocity vector.
* **Deformation:**
* **Deformation Gradient ($\mathbf{F}_p \in \mathbb{R}^{3\times3}$):** A tensor tracking the mapping from the reference configuration to the current configuration, used to compute strain and stress.
* **Affine State (APIC):**
* **Affine Matrix ($\mathbf{C}_p \in \mathbb{R}^{3\times3}$):** Stores the local velocity gradient approximation. This tensor preserves angular momentum during the Particle-to-Grid (P2G) transfer, reducing energy dissipation compared to standard PIC methods.
* **Thermodynamics:**
* **Persistent Pressure ($P_p \in \mathbb{R}$):** Stores the scalar fluid pressure value sampled from the grid at the end of the previous frame. This attribute is used to reconstruct the initial solver guess ($x_0$) after particle advection, ensuring the pressure field follows the material topology.
* **Inertia:**
* **Mass ($m_p$):** The constant mass of the material point.
* **Initial Volume ($V_p^0$):** The reference volume, used for integration weights.
* **Chronology:**
* **Birth Stamp ($t_0$):** The simulation time at which the particle was instantiated or injected.
* **Derived Age ($\tau$):** The elapsed lifespan ($\tau = t_{current} - t_0$), used to drive time-dependent rheological transitions such as coagulation (permeability decay) and syneresis (clot retraction).

**2.2 Grid Basis Functions**
The Eulerian domain is discretized by a uniform Cartesian grid with spacing $\Delta x$. Particle-Grid interactions are mediated by Quadratic B-Spline shape functions $N(\mathbf{x})$, which provide $C^1$ continuity.
For a grid node $i$ located at $\mathbf{x}_i$ and a particle at $\mathbf{x}_p$, the interpolation weight $w_{ip}$ is the tensor product of 1D basis functions:
$$ w_{ip} = N\left( \frac{x_p - x_i}{\Delta x} \right) N\left( \frac{y_p - y_i}{\Delta x} \right) N\left( \frac{z_p - z_i}{\Delta x} \right) $$
where the 1D basis function $N(u)$ is defined as:
$$ N(u) = \begin{cases} \frac{3}{4} - |u|^2 & : 0 \leq |u| < \frac{1}{2} \\ \frac{1}{2}(\frac{3}{2} - |u|)^2 & : \frac{1}{2} \leq |u| < \frac{3}{2} \\ 0 & : |u| \ge \frac{3}{2} \end{cases} $$

**2.3 Collocated Vertex-Centered Topology**
The system utilizes a **Collocated Discretization** scheme where the MPM kinematic grid and the Finite Volume thermodynamic grid share a unified topological definition. This alignment ensures that mass, momentum, and pressure are defined at identical spatial coordinates, eliminating interpolation errors between the advection and projection steps.

* **Nodal Definition (The Anchor):**
All primary state variables are stored at the integer grid indices $\mathbf{I} = (i, j, k)$. This includes:
* **Kinematic State:** Grid velocity $\mathbf{v}_i$ and Mass $m_i$ (accumulated via P2G).
* **Thermodynamic State:** Pressure $P_i$ and Material Indices.
* **Geometric State:** The Level Set / SDF values used for boundary enforcement.

* **Control Volume (The Voxel):**
For the purposes of the Finite Volume solver, the "Voxel" associated with node $\mathbf{I}$ is defined as the cubic Voronoi cell centered at $\mathbf{I}$. The spatial extent of this volume ranges from $\mathbf{I} - 0.5$ to $\mathbf{I} + 0.5$. Mass $m_i$ residing at the node is interpreted as being distributed throughout this control volume.

* **Virtual Interfaces (The Faces):**
While state variables are nodal, flux exchanges and transmissibility coefficients are defined on the **Faces** of the control volumes. These faces are located at half-integer offsets relative to the nodes.
* **Face X:** Located at $\mathbf{I} + (0.5, 0, 0)$. Mediates flux between Node $(i,j,k)$ and Node $(i+1,j,k)$.
* **Face Y:** Located at $\mathbf{I} + (0, 0.5, 0)$. Mediates flux between Node $(i,j,k)$ and Node $(i,j+1,k)$.
* **Face Z:** Located at $\mathbf{I} + (0, 0, 0.5)$. Mediates flux between Node $(i,j,k)$ and Node $(i,j,k+1)$.

This strictly defines the derivative stencil: gradients and divergences are calculated by evaluating differences across these specific interfaces, gated by the transmissibility of the face.

---

### III. Solid Kinematics & Constitutive Dynamics

**3.1 Finite Strain Kinematics**
The local deformation is tracked via the deformation gradient $\mathbf{F}_p = \partial \mathbf{x} / \partial \mathbf{X}$.
*   **Right Cauchy-Green Tensor:** $\mathbf{C} = \mathbf{F}^T \mathbf{F}$
*   **Jacobian:** $J = \det(\mathbf{F})$
*   **Isochoric Deformation:** $\bar{\mathbf{F}} = J^{-1/3}\mathbf{F}, \quad \bar{\mathbf{C}} = J^{-2/3}\mathbf{C}$

### 3.2 Microstructural Anisotropy
The tissue architecture is encoded in the structural tensor $\mathbf{M}_0$. To represent crossing fiber geometries, $\mathbf{M}_0$ is defined as the weighted sum of the constituent fiber populations identified during initialization:

$$ \mathbf{M}_0 = \sum_{n=1}^{N} w_n (\mathbf{v}_n \otimes \mathbf{v}_n) $$

where $\mathbf{v}_n$ are the primary eigenvectors of the fiber peaks, $w_n$ are the normalized volume fractions ($\sum w_n = 1$), and $N$ is the number of detected peaks (up to 3). This formulation allows the material to exhibit high stiffness along multiple independent axes simultaneously.

* **Dispersion:** The generalized structure invariant $\bar{I}_{4\kappa}^*$ is evaluated against this composite tensor to account for microscopic fiber splaying:
$$ \bar{I}_{4\kappa}^* = \kappa \text{tr}(\bar{\mathbf{C}}) + (1 - 3\kappa) (\mathbf{M}_0 : \bar{\mathbf{C}}) $$
Here, $\bar{\mathbf{C}}$ is the isochoric Right Cauchy-Green deformation tensor, and the double dot product $(\mathbf{M}_0 : \bar{\mathbf{C}})$ computes the weighted strain projection along all fiber directions.

**3.3 Hyperelastic Energy Density**
The Helmholtz free energy $\Psi$ is decoupled into three terms:
$$ \Psi = \Psi_{vol}(J) + \Psi_{iso}(\bar{I}_1) + \Psi_{aniso}(\bar{I}_{4\kappa}^*) $$

1.  **Volumetric (Penalty):** Controls compressibility.
    $$ \Psi_{vol}(J) = \frac{K_{bulk}}{2}(J-1)^2 $$
2.  **Isotropic (Neo-Hookean):** Matrix shear response.
    $$ \Psi_{iso}(\bar{I}_1) = \frac{\mu}{2}(\bar{I}_1 - 3), \quad \bar{I}_1 = \text{tr}(\bar{\mathbf{C}}) $$
3.  **Anisotropic (Holzapfel-Gasser-Ogden):** Exponential stiffening of fibers.
    $$ \Psi_{aniso} = \frac{k_1}{2k_2} \left[ \exp(k_2 \langle \bar{I}_{4\kappa}^* - 1 \rangle^2) - 1 \right] $$
    *   $\langle \cdot \rangle$: Macaulay brackets (fibers only resist tension).

**3.4 Kirchhoff Stress Tensor**
The stress $\boldsymbol{\tau} = \mathbf{P}\mathbf{F}^T$ required for the momentum update is derived via:
$$ \boldsymbol{\tau} = 2 \frac{\partial \Psi}{\partial \mathbf{b}} \mathbf{b} \quad (\text{where } \mathbf{b} = \mathbf{F}\mathbf{F}^T) $$
Explicitly for implementation:
$$ \boldsymbol{\tau} = J \frac{\partial \Psi_{vol}}{\partial J} \mathbf{I} + \text{dev}\left( 2 \frac{\partial \bar{\Psi}}{\partial \bar{\mathbf{C}}} \bar{\mathbf{C}} \right) $$

### 3.5 Rate-Dependent Plasticity, Damage & Chrono-Rheology

**3.5.1 Dynamic Yield Surface (Cowper-Symonds)**
The material yield threshold is modeled as a function of deformation speed to capture the strain-rate hardening characteristic of biological tissue (soft under quasi-static loads, brittle under ballistic impact). The dynamic yield stress $\sigma_y$ is defined by the Cowper-Symonds relation:

$$ \sigma_y(\dot{\varepsilon}) = \sigma_{0} \left[ 1 + \left( \frac{\dot{\varepsilon}_{eff}}{C} \right)^{1/P} \right] $$

*   $\sigma_{0}$: The static yield stress (approx. 1 kPa for brain tissue, 80 MPa for bone).
*   $\dot{\varepsilon}_{eff}$: The effective strain rate, calculated as $\sqrt{\frac{2}{3} \mathbf{D}_{rate}:\mathbf{D}_{rate}}$, where $\mathbf{D}_{rate}$ is the symmetric part of the velocity gradient.
*   $C, P$: Viscoplastic material constants governing the rate sensitivity.

**3.5.2 Damage Evolution**
A scalar damage field $D \in [0, 1]$ tracks the irreversible loss of microstructural integrity. Damage accumulates only when the local Von Mises stress $q$ exceeds the dynamic yield surface $\sigma_y$:

$$ D^{n+1} = \min \left( 1, D^n + \frac{\langle q - \sigma_y \rangle}{\eta_{visc}} \Delta t \right) $$

Here, $\langle \cdot \rangle$ denotes the Macaulay bracket (ramp function), ensuring damage only grows when the yield condition is violated. $\eta_{visc}$ is a viscosity regularization parameter defining the temporal scale of the fracture process.

**3.5.3 Constitutive Transmutation (Liquefaction)**
To model the physical transition from intact solid to liquefied hematoma without violating mass conservation, the stress response is split into volumetric and deviatoric components which degrade independently.

1.  **Deviatoric Collapse (Shear Strength):** The deviatoric part of the Kirchhoff stress tensor $\boldsymbol{\tau}$, which governs resistance to shape change, is scaled by the structural integrity $(1-D)$. As $D \to 1$, the effective shear modulus vanishes, and the material loses the ability to sustain shear stress, effectively behaving as a fluid.
    $$ \text{dev}(\boldsymbol{\tau}_{final}) = (1 - D) \cdot \text{dev}(\boldsymbol{\tau}_{elastic}) $$

2.  **Volumetric Preservation (Bulk Strength):** The hydrostatic component of the stress (pressure) is left undegraded. The bulk modulus $K$ remains active regardless of the damage state. This ensures that even fully failed particles ($D=1$) resist compression, preventing volumetric implosion and ensuring they continue to displace fluid correctly in the pressure solver.
    $$ \boldsymbol{\tau}_{final} = \text{dev}(\boldsymbol{\tau}_{final}) + J \frac{\partial \Psi_{vol}}{\partial J} \mathbf{I} $$

---

**3.5.4 Chrono-Rheological Hardening (The Clot Clamp)**
To simulate the macroscopic phase transition of hematoma from liquid sol to semi-solid gel, the constitutive model utilizes a time-dependent hardening law driven by particle age.

* **The Sigmoid Switch:** A master transition variable $S(\tau) \in [0, 1]$ is defined based on the particle age $\tau$.
$$ S(\tau) = \frac{1}{1 + e^{-k(\tau - T_{clot})}} $$
Where $T_{clot}$ represents the characteristic coagulation time (typically 6 hours) and $k$ governs the sharpness of the phase transition.

* **Viscosity Ramp:** As the fluid ages, its resistance to flow increases. The particle viscosity $\eta$ used in the P2G momentum transfer is interpolated between liquid blood viscosity and clot gel viscosity:
$$ \eta_{eff} = \eta_{fluid} + S(\tau) \cdot (\eta_{gel} - \eta_{fluid}) $$
This artificial viscosity dampens velocity in older regions, stabilizing the interface and preventing "sloshing" artifacts in Regime II.

* **Elastic Retraction (Syneresis Stress):** Mature clots actively contract, exerting a pulling force on the surrounding parenchyma. An isotropic contractile stress is added to the total Kirchhoff stress tensor:
$$ \boldsymbol{\tau}_{total} = \boldsymbol{\tau}_{elastic} + \boldsymbol{\tau}_{retract} $$
$$ \boldsymbol{\tau}_{retract} = - \sigma_{contract} \cdot S(\tau) \cdot \mathbf{I} $$
Where $\sigma_{contract}$ is the maximum retraction pressure. This term ensures that as the hematoma ages, it naturally pulls the midline structures back toward the injury site, mimicking biological clot retraction.

### IV. Fluid Dynamics (Porous Media)

### 4.1 Damage-Coupled & Chrono-Rheological Permeability

The local hydraulic conductivity tensor $\mathbf{K}$ is governed by a dual-dependency model: it is first defined spatially by the microstructural architecture and damage state, and subsequently modulated temporally by the blood age (coagulation state).

* **Stage 1: Spatial Permeability Mixing ($\mathbf{K}_{base}$)**
The baseline permeability is derived from the material phase and orientation:
* **Intact Tissue ($D=0$):** Permeability is anisotropic, governed by the porous architecture. The tensor integrates contributions from fiber bundles ($\mathbf{M}_0$) to permit flow along valid anatomical tracts while maintaining high resistivity perpendicular to the fiber plane.
$$ \mathbf{K}_{tissue} = k_{iso}\mathbf{I} + k_{fiber} \mathbf{M}_0 $$
For the axis-aligned Cartesian stencil, the face-normal projection $\mathbf{n} \cdot \mathbf{K}_{tissue} \cdot \mathbf{n}$ reduces to reading a single diagonal element: $K_{axis} = k_{iso} + k_{fiber} \cdot \mathbf{M}_0[axis, axis]$. The scalars $k_{iso}$ and $k_{fiber}$ are stored per material class in the Dual Global LUT (Voxel System Spec, Section III.3.1: Quantized Indexing Protocol), while $\mathbf{M}_0$ is stored per voxel in the Tier 2 grid field.
* **Liquefied Hematoma ($D=1$):** In damaged regions, the solid matrix is mechanically disrupted. The permeability approaches that of bulk fluid, modeled as a high-magnitude isotropic tensor.
$$ \mathbf{K}_{blood} = k_{max}\mathbf{I} $$
* **Geometric Transition:** The baseline tensor is computed via logarithmic interpolation to handle the scale difference between tissue and blood:
$$ \mathbf{K}_{base} = \exp\left( (1-D)\ln(\mathbf{K}_{tissue}) + D\ln(\mathbf{K}_{blood}) \right) $$

* **Stage 2: The Chronological Clamp ($\mathbf{K}_{final}$)**
To enforce the fluid-to-solid phase transition of clotting blood, the baseline permeability is modulated by the local **Coagulation Factor ($\alpha_{clot}$)** retrieved from the Tier 2 grid memory.
$$ \mathbf{K}_{final} = \mathbf{K}_{base} \cdot (1.0 - \alpha_{clot}) + \epsilon $$
* **Mechanism:** In regions of fresh hemorrhage ($\alpha_{clot} \approx 0$), conductivity remains high, allowing rapid expansion. As the blood ages and coagulates ($\alpha_{clot} \to 1$), the effective permeability drops toward zero ($\epsilon$). This effectively seals off older parts of the hematoma, forcing new bleeding to expand the boundary rather than diffusing through the existing core.

### 4.2 The Osmotic-Hydrostatic Poisson Equation

The solver governs the pressure field $P$ via the Starling equation for trans-endothelial flux, coupling mechanical hydrostatics with chemical osmosis. The standard divergence constraint is augmented by an **Osmotic Source Term** $\Psi$, representing the chemical potential of blood breakdown products.

$$ \nabla \cdot \left( \frac{\mathbf{K}_{eff}}{\mu_f} \nabla P \right) = \underbrace{\frac{1}{\Delta t} \nabla \cdot \mathbf{v}^*}_{\text{Hydrostatic}} + \underbrace{\Psi_{osmotic}([Fe])}_{\text{Oncotic}} $$

* **Hydrostatic Term:** Represents the mechanical compression or expansion of the control volume derived from the intermediate velocity field divergence.
* **Oncotic Term ($\Psi$):** A scalar source term derived from the local solute concentration $[Fe]$. This term acts as a negative divergence source (virtual sink), mathematically forcing the solver to generate pressure gradients that drive fluid influx into regions of high hematoma concentration, even against hydrostatic resistance.
---

### V. Solver Acceleration: Temporal Predictor Protocol

The MG-PCG solver is mathematically robust enough to converge from an arbitrary initial state. However, to minimize the computational cost per frame, the system utilizes a **Temporal Predictor Protocol**. By initializing the solver close to the solution manifold, the initial residual norm $\|r\|_0$ is minimized, drastically reducing the number of V-Cycles required to reach convergence.

**7.1 Temporal Initial Guess (Zero-Order)**
For the majority of simulation steps, the pressure field $P$ exhibits high temporal continuity; the internal stress distribution of the tissue does not change instantaneously. The system leverages this coherence by carrying the solution forward.
*   **The Initialization:** The initial guess $x_0$ for the linear system at frame $n$ is defined simply as the finalized pressure field from the previous frame:
    $$ x_0^{(n)} = P^{(n-1)} $$
*   **Spectral Benefit:** While the Multigrid preconditioner is excellent at resolving global (low-frequency) errors, it is computationally expensive. By initializing with $P^{(n-1)}$, the low-frequency components—representing the bulk pressure gradients across the organ—are already largely resolved. The solver therefore only needs to resolve the high-frequency local perturbations introduced by the new increment of deformation.

### 7.2 Lagrangian Pressure Advection (Particle-Carried Warm Start)
To maintain solver efficiency within a dynamic sparse topology—where grid blocks are frequently allocated or deallocated as fluid moves—the system utilizes a **Particle-Carried Initial Guess**. This method leverages the Lagrangian persistence of the Material Points to naturally transport the pressure solution alongside the fluid material.

**7.2.1 State Capture (G2P Protocol)**
At the conclusion of the Pressure Projection phase (Phase 3, Step 1), the converged grid pressure field $P_{grid}^{(n)}$ is interpolated onto the particles using the basis functions. This stores the hydrostatic state of the fluid as a persistent attribute on the material points:
$$ P_p^{(n)} = \sum_{\mathbf{I}} N(\mathbf{x}_p - \mathbf{x}_{\mathbf{I}}) \cdot P_{\mathbf{I}}^{(n)} $$

### **7.2.2 Predictor Reconstruction (P2G Protocol)**
At the initialization of the subsequent frame's linear solve, the initial guess vector $x_0^{(n+1)}$ is reconstructed by scattering the stored particle pressures back onto the new grid configuration during the P2G phase. To account for variations in particle density, the scatter is normalized via mass-weighting:

$$ x_0^{(\mathbf{I})} = \frac{\sum_{p} w_{ip} m_p P_p^{(n)}}{\sum_{p} w_{ip} m_p} $$

* **Precision Constraint (Single Precision Strategy):** Unlike the Residual Evaluation (Path C), this reconstruction step utilizes **Standard Single Precision (`f32`)** arithmetic. Since this vector serves only as the search initiation point ($x_0$) for the Krylov subspace, microscopic floating-point errors introduced by `f32` accumulation are mathematically negligible and naturally corrected by the first V-Cycle iteration. This reduction allows the kernel to minimize register pressure during the bandwidth-heavy scatter.
* **Normalization:** The term $\sum w_{ip} m_p$ is the nodal mass. This weighting ensures that the projected pressure is intensive and independent of the number of particles per cell.
* **Vacuum Handling:** If a node has negligible mass (no active particles), $x_0$ defaults to $0$.

**Rationale:** By attaching the pressure guess to the moving particles, the initial solution naturally advects with the wavefront. Even if fluid enters a voxel that did not exist in the previous frame, the particles carrying the fluid bring the correct pressure history with them, allowing the solver to iterate from a physically valid state rather than a cold start.

**7.3 Cold-Start Dynamics & Robustness**
In scenarios where temporal history is invalid or unavailable—specifically at simulation initialization ($t=0$) or immediately following a topological cut (surgical incision)—the Temporal Predictor is disabled.
*   **The Identity Reset:** The initial guess is reset to zero ($x_0 = 0$).
*   **MG Resilience:** Unlike standard PCG solvers, which may require thousands of iterations to resolve global pressure from a cold start, the MG-PCG architecture is naturally resilient to initialization errors. The first V-Cycle immediately propagates boundary information from the coarse levels ($L_D$) to the fine levels ($L_0$).
*   **Elimination of Warm-Up Kernels:** Consequently, the explicit "Hierarchical Warm-Start" pass used in previous architectures is obsolete. The V-Cycle Preconditioner itself acts as the warm-up mechanism, allowing the system to converge from a cold state in typically $< 10$ cycles without specialized initialization logic.

**7.4 Numerical Efficiency Targets**
The integration of Temporal Prediction with the MG-PCG architecture shifts the performance profile from "Many Cheap Iterations" to "Few Expensive Cycles." The system targets the following convergence bounds:

*   **Steady-State / Quasi-Static:** **1 to 2 V-Cycles.**
    With a valid temporal guess, the initial residual is often already near tolerance. The V-Cycle serves primarily as a verification step.
*   **Transient Fluctuations:** **3 to 6 V-Cycles.**
    During standard deformation, the MG-PCG rapidly suppresses the error modes introduced by particle movement.
*   **Cold Start / Topology Change:** **8 to 12 V-Cycles.**
    Even with zero initialization, the spectral efficiency of the Geometric Multigrid backend ensures global convergence in fewer than a dozen outer iterations.
*   **Comparison:** This represents a roughly **100x reduction in iteration count** compared to a standard Diagonally Preconditioned PCG, rendering the solver performance largely independent of the grid resolution $N$.

---

### VI. Signal Synthesis (Bloch-Torrey Physics)

**8.1 Voxel Mixture**
For a voxel $\mathbf{x}$, the local composition dictates the Nuclear Magnetic Resonance parameters.
$$ W(\mathbf{x}) = \phi^s W_{tissue} + \phi^f W_{fluid} $$

**8.2 Relaxation Rate Equations**
The longitudinal ($R_1$) and transverse ($R_2$) relaxation rates are computed based on tissue fractions and hemorrhage aging:
$$ R_1(\mathbf{x}) = (1-\phi^f)R_{1,t} + \phi^f R_{1,f} + r_1 [Fe] $$
$$ R_2(\mathbf{x}) = (1-\phi^f)R_{2,t} + \phi^f R_{2,f} + r_2 [Fe] $$
* $[Fe] \propto \int_0^t \mathcal{S}_{bleed} dt$: Local iron concentration derived from tracer particle age.
* $r_{1,2}$: Paramagnetic relaxivity constants of Methemoglobin.

**8.3 Signal Intensity**
The "Perfect Physics" volume is generated analytically for a Spin-Echo sequence:
$$ \mathcal{I}_{phy}(\mathbf{x}) = \rho_{PD}(\mathbf{x}) (1 - e^{-TR \cdot R_1}) e^{-TE \cdot R_2} $$

---

===============================================================


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
