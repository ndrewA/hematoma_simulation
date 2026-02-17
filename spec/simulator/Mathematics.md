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
