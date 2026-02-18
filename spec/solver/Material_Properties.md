# Material Properties Specification

This document defines all material property values consumed by the solver. Values are drawn from experimental measurements and established computational models in the brain biomechanics literature. Where direct measurements do not exist, values are extrapolated from structurally analogous tissues.

---

## I. Permeability Lookup Table (Dual Global LUT, `f64`)

The Darcy solver retrieves per-axis permeability as $K_{axis} = k_{iso} + k_{fiber} \cdot \mathbf{M}_0[axis, axis]$. Both arrays store intrinsic permeability in SI units (m$^2$).

| u8 | Class | $k_{iso}$ \[m$^2$\] | $k_{fiber}$ \[m$^2$\] | Along-fiber total | Notes |
|---:|---|---|---|---|---|
| 0 | Vacuum | 0 | 0 | — | Zero flux (Neumann wall) |
| 1 | Cerebral White Matter | $1.0 \times 10^{-15}$ | $1.4 \times 10^{-15}$ | $2.4 \times 10^{-15}$ | Cheng & Bilston 2007; 2:1 anisotropy ratio from Vidotto et al. 2021 |
| 2 | Cortical Gray Matter | $5.0 \times 10^{-16}$ | 0 | $5.0 \times 10^{-16}$ | CED / MPET literature consensus |
| 3 | Deep Gray Matter | $3.0 \times 10^{-16}$ | 0 | $3.0 \times 10^{-16}$ | Denser nuclear packing than cortex; extrapolated |
| 4 | Cerebellar White Matter | $8.0 \times 10^{-16}$ | $8.0 \times 10^{-16}$ | $1.6 \times 10^{-15}$ | Branching arbor vitae geometry; reduced anisotropy vs cerebral WM |
| 5 | Cerebellar Cortex | $4.0 \times 10^{-16}$ | 0 | $4.0 \times 10^{-16}$ | Dense granular layer |
| 6 | Brainstem | $6.0 \times 10^{-16}$ | $6.0 \times 10^{-16}$ | $1.2 \times 10^{-15}$ | Mixed nuclei + longitudinal tracts (~50/50 WM/GM) |
| 7 | Ventricular CSF | $1.0 \times 10^{-8}$ | 0 | $1.0 \times 10^{-8}$ | Near-free fluid; open cavities with ependymal lining |
| 8 | Subarachnoid CSF | $1.0 \times 10^{-9}$ | 0 | $1.0 \times 10^{-9}$ | Trabecular resistance from arachnoid pillars |
| 9 | Choroid Plexus | $1.0 \times 10^{-11}$ | 0 | $1.0 \times 10^{-11}$ | Fenestrated capillaries; AQP1-mediated water transport |
| 10 | Dural Membrane | $1.0 \times 10^{-18}$ | 0 | $1.0 \times 10^{-18}$ | Dense collagen barrier; comparable to articular cartilage |
| 11 | Vessel / Venous Sinus | $1.0 \times 10^{-8}$ | 0 | $1.0 \times 10^{-8}$ | Open endothelialized lumen |
| 255 | Air Halo | — | — | — | Dynamic Dirichlet boundary ($P = P_{halo}$) |

The parenchymal range spans $\sim$3 orders of magnitude ($3 \times 10^{-16}$ to $2.4 \times 10^{-15}$). The full domain contrast is $\sim$$10^{10}$ ($10^{-18}$ dura to $10^{-8}$ CSF).

---

## II. Hyperelastic Constitutive Properties

### 2.1 Neo-Hookean Ground Substance

| u8 | Class | $\mu$ \[Pa\] | $K_{bulk}$ \[Pa\] | Source |
|---:|---|---:|---:|---|
| 1 | Cerebral White Matter | 1,000 | $5.0 \times 10^7$ | Budday et al. 2017 (CC ~490, CR ~660 Pa; compromise) |
| 2 | Cortical Gray Matter | 1,400 | $5.0 \times 10^7$ | Budday et al. 2017 (stiffest region tested) |
| 3 | Deep Gray Matter | 700 | $5.0 \times 10^7$ | Budday et al. 2017 (basal ganglia) |
| 4 | Cerebellar White Matter | 900 | $5.0 \times 10^7$ | ~15–25% softer than cerebral WM (Prange & Margulies 2002) |
| 5 | Cerebellar Cortex | 900 | $5.0 \times 10^7$ | Cerebellum softer than cerebrum across all testing methods |
| 6 | Brainstem | 900 | $5.0 \times 10^7$ | Mixed nuclei + tracts; lower range (Prange & Margulies 2002) |
| 7 | Ventricular CSF | 1 | $5.0 \times 10^7$ | Numerical stabilization; pure fluid ($\phi^f = 1$) |
| 8 | Subarachnoid CSF | 1 | $5.0 \times 10^7$ | Numerical stabilization; pure fluid ($\phi^f = 1$) |
| 9 | Choroid Plexus | 600 | $5.0 \times 10^7$ | Soft vascularized epithelium; no direct measurements exist |
| 10 | Dural Membrane | $1.09 \times 10^7$ | $1.05 \times 10^8$ | E = 31.5 MPa, $\nu$ = 0.45 (Kleiven 2007; Zwirner et al. 2019) |
| 11 | Vessel / Venous Sinus | $3.0 \times 10^6$ | $8.0 \times 10^7$ | Compromise between bridging veins (3–7 MPa) and sinus walls (33–58 MPa) |

$K_{bulk} = 50$ MPa for all brain parenchyma classes enforces near-incompressibility ($\nu \approx 0.4999$), following Giordano & Kleiven 2014. Dura and vessel bulk moduli are derived from their respective Young's moduli and Poisson ratios.

### 2.2 Holzapfel–Gasser–Ogden Fiber Reinforcement

Only tissues with organized fiber architecture receive non-zero HGO parameters. The anisotropic energy is $\Psi_{aniso} = \frac{k_1}{2 k_2}\left[\exp(k_2 \langle \bar{I}_{4\kappa}^* - 1 \rangle^2) - 1\right]$.

| u8 | Class | $k_1$ \[Pa\] | $k_2$ \[–\] | $\kappa$ \[–\] | Notes |
|---:|---|---:|---:|---|---|
| 1 | Cerebral White Matter | 11,590 | 0 | per-voxel | Giordano & Kleiven 2014; $k_2 = 0$ gives linear reinforcement |
| 4 | Cerebellar White Matter | 9,000 | 0 | per-voxel | Reduced vs cerebral WM; branching arbor vitae |
| 6 | Brainstem | 10,000 | 0 | per-voxel | Corticospinal tract, medial lemniscus, peduncles |
| All others | — | 0 | 0 | $\frac{1}{3}$ | Isotropic; $\Psi_{aniso}$ term disabled |

**$k_2 = 0$ rationale:** Brain white matter does not exhibit the exponential stiffening of arterial walls. Setting $k_2 = 0$ reduces the HGO exponential to linear transversely-isotropic reinforcement, which matches available brain tissue data.

**$\kappa$ per-voxel:** For anisotropic classes, the fiber dispersion parameter is derivable from DTI fractional anisotropy. High FA $\to$ low $\kappa$ (well-aligned), low FA $\to$ $\kappa \to \frac{1}{3}$ (isotropic). Typical ranges: corpus callosum $\kappa \approx 0.0$–$0.1$, corona radiata $\kappa \approx 0.1$–$0.2$, crossing regions $\kappa \approx 0.2$–$0.3$.

---

## III. Density and Porosity

| u8 | Class | $\rho_s$ \[kg/m$^3$\] | $\rho_f$ \[kg/m$^3$\] | $\phi^f$ \[–\] | Source |
|---:|---|---:|---:|---:|---|
| 1 | Cerebral White Matter | 1,100 | 1,000 | 0.70 | WM water content ~68–72% (Fatouros et al. 1991; Abbas et al. 2015) |
| 2 | Cortical Gray Matter | 1,080 | 1,000 | 0.82 | GM water content ~80–84% (Abbas et al. 2015) |
| 3 | Deep Gray Matter | 1,080 | 1,000 | 0.80 | Similar to cortical GM |
| 4 | Cerebellar White Matter | 1,100 | 1,000 | 0.70 | Same as cerebral WM |
| 5 | Cerebellar Cortex | 1,080 | 1,000 | 0.80 | Same as cortical GM |
| 6 | Brainstem | 1,090 | 1,000 | 0.75 | Intermediate (mixed WM/GM) |
| 7 | Ventricular CSF | — | 1,004 | 1.00 | Horlocker & Wedel 1993; Bloomfield et al. 1998 |
| 8 | Subarachnoid CSF | — | 1,004 | 1.00 | Same as ventricular CSF |
| 9 | Choroid Plexus | 1,060 | 1,000 | 0.88 | Highly vascularized; estimated |
| 10 | Dural Membrane | 1,130 | 1,000 | 0.08 | Dense collagen (Kleiven 2007; Lipphaus & Witzel 2021) |
| 11 | Vessel / Venous Sinus | 1,100 | 1,060 | 0.20 | $\rho_f$ = blood density (Barber et al. 1970) |

Intrinsic solid densities are back-calculated from bulk tissue density and porosity via $\rho_{bulk} = \phi^s \rho_s + \phi^f \rho_f$. The key physical constraint: **white matter has significantly lower water content than gray matter** (~70% vs ~82%), which couples directly to consolidation timescales in the biphasic model.

---

## IV. Plasticity and Damage (Cowper–Symonds)

### 4.1 Static Yield Stress

| u8 | Class | $\sigma_0$ \[Pa\] | Source |
|---:|---|---:|---|
| 1–6, 9 | Brain parenchyma | 1,000 | Franceschini et al. 2006 (tensile rupture ~4.7 kPa); Rashid et al. 2012 (shear failure 1.15–2.52 kPa at 30–120/s) |
| 10 | Dural Membrane | $7.0 \times 10^6$ | Zwirner et al. 2019 (tensile strength 7 $\pm$ 4 MPa) |
| 11 | Vessel / Venous Sinus | $3.0 \times 10^6$ | Monson et al. 2003 (bridging vein failure) |

### 4.2 Rate Sensitivity and Damage Viscosity

No published Cowper–Symonds coefficients exist specifically for brain tissue. The values below are estimated from regression of Rashid et al. (2012, 2013) dynamic tensile/shear data at strain rates 30–90/s.

| Parameter | Brain (u8 1–6, 9) | Dura (u8 10) | Vessel (u8 11) |
|---|---:|---:|---:|
| $C$ \[s$^{-1}$\] | 30 | 100 | 50 |
| $P$ \[–\] | 2.5 | 3.0 | 3.0 |
| $\eta_{visc}$ \[Pa$\cdot$s\] | 200 | 10,000 | 5,000 |

**Calibration note:** These parameters require simulation-based calibration. The recommended procedure is to simulate uniaxial tension at 30/s and 90/s strain rates and adjust $C$ and $P$ until the dynamic yield stress matches Rashid et al. experimental values (3.1 kPa and 6.5 kPa respectively at 30% strain). $\eta_{visc}$ controls the temporal scale of damage evolution — too low causes mesh-dependent catastrophic failure, too high prevents damage entirely.

---

## V. Fluid Properties

| Parameter | Symbol | Value | Unit | Source |
|---|---|---|---|---|
| Blood dynamic viscosity | $\mu_f$ | $3.5 \times 10^{-3}$ | Pa$\cdot$s | Baskurt & Meiselman 2003 (Newtonian plateau at >100 s$^{-1}$, 37$\degree$C) |
| CSF dynamic viscosity | $\mu_{CSF}$ | $7.0 \times 10^{-4}$ | Pa$\cdot$s | Bloomfield et al. 1998 ($\approx$ water at 37$\degree$C) |
| Blood density | $\rho_{blood}$ | 1,060 | kg/m$^3$ | Standard hematology (whole blood, Hct ~40%) |
| CSF density | $\rho_{CSF}$ | 1,004 | kg/m$^3$ | Horlocker & Wedel 1993 |

---

## VI. Chrono-Rheological Parameters

These parameters drive the sigmoid switch $S(\tau) = \frac{1}{1 + e^{-k(\tau - T_{clot})}}$ and its downstream effects on viscosity, permeability, and retraction stress (Mathematics Spec, Section III.5.4).

### 6.1 Coagulation Sigmoid

| Parameter | Symbol | Value | Unit | Source |
|---|---|---|---|---|
| Gelation midpoint | $T_{clot}$ | 360 | s | Ranucci et al. 2014 (TEG gel point 60–120 s; R-time 5–10 min) |
| Sigmoid steepness | $k$ | 0.01 | s$^{-1}$ | Derived: 90% transition width $\approx$ 440 s (spans ~1–8 min) |

The sigmoid center at 6 minutes captures the liquid-to-gel rheological transition. Full mechanical maturation (fibrin cross-linking, platelet retraction) occurs over 1–2 hours and is captured by the retraction stress term ramping in after the initial gelation.

### 6.2 Viscosity Ramp

| Parameter | Symbol | Value | Unit | Source |
|---|---|---|---|---|
| Fresh blood viscosity | $\eta_{fluid}$ | $3.5 \times 10^{-3}$ | Pa$\cdot$s | Same as $\mu_f$ |
| Clot gel effective viscosity | $\eta_{gel}$ | 1.0 | Pa$\cdot$s | ~300$\times$ increase; Ranucci et al. 2014, Hajjarian et al. 2015 |

$\eta_{gel}$ is an effective damping parameter, not a true viscosity (mature clot is a viscoelastic solid with $G' \approx$ 200–600 Pa). The value is chosen so that the P2G damping timescale $\tau_{damp} \sim \rho \Delta x^2 / \eta_{gel}$ is shorter than $\Delta t$, arresting all flow within clotted regions.

### 6.3 Clot Retraction

| Parameter | Symbol | Value | Unit | Source |
|---|---|---|---|---|
| Retraction pressure | $\sigma_{contract}$ | 200 | Pa | Lam et al. 2011; Tutwiler et al. 2016 |

Sufficient to compress brain parenchyma ($\mu \approx$ 1 kPa) by several percent strain, consistent with the ~40% volume reduction observed in vitro (Jeong et al. 2021). Must not exceed fibrin network rupture stress (~5 kPa).

### 6.4 Hematoma Permeability

| Parameter | Symbol | Value | Unit | Source |
|---|---|---|---|---|
| Liquefied hematoma permeability | $k_{max}$ | $1.0 \times 10^{-12}$ | m$^2$ | Fresh unclotted blood; Wufsus et al. 2013 |
| Sealed clot permeability floor | $\epsilon$ | $1.0 \times 10^{-17}$ | m$^2$ | Dense retracted clot; Wufsus et al. 2013 |

The chrono-rheological clamp $\mathbf{K}_{final} = \mathbf{K}_{base} \cdot (1 - \alpha_{clot}) + \epsilon$ drives permeability through the following progression:

| Stage | Time | $\alpha_{clot}$ | Effective $k$ \[m$^2$\] | Physical state |
|---|---|---|---|---|
| Fresh hemorrhage | 0–2 min | ~0 | $\sim 10^{-12}$ | Bulk liquid blood |
| Early gelation | 2–10 min | 0.1–0.3 | $\sim 10^{-13}$ | Fibrin network forming |
| Gel point | 5–15 min | ~0.5 | $\sim 10^{-14}$ | Crosslinked fibrin; flow dramatically reduced |
| Mature clot | 30 min–2 hr | ~0.9 | $\sim 10^{-16}$ | Dense fibrin + platelets, retracted |
| Organized hematoma | >24 hr | ~1.0 | $\sim 10^{-17}$ | Sealed; approaching tissue permeability |

---

## VII. MRI Relaxation Parameters (Bloch–Torrey Signal Synthesis)

All values at 3T unless noted.

### 7.1 Tissue Relaxation Rates

| Tissue | $R_1$ \[s$^{-1}$\] | $R_2$ \[s$^{-1}$\] | $T_1$ \[ms\] | $T_2$ \[ms\] | Source |
|---|---:|---:|---:|---:|---|
| White matter | 1.11 | 14.3 | 900 | 70 | Stanisz et al. 2005 |
| Gray matter | 0.71 | 11.1 | 1,400 | 90 | Stanisz et al. 2005 |
| CSF | 0.23 | 0.50 | 4,300 | 2,000 | Bojorquez et al. 2017 |

### 7.2 Paramagnetic Relaxivity (Methemoglobin)

| Parameter | Value | Unit | Source |
|---|---|---|---|
| $r_1$ (MetHb) | 0.4 | s$^{-1}$ mM$^{-1}$ | Gomori et al. 1987 (per mM heme iron) |
| $r_2$ (MetHb, effective) | 8.0 | s$^{-1}$ mM$^{-1}$ | Gomori et al. 1987 (compartment-averaged) |
| \[Hb\] physiological | 9.2 | mM (heme) | Standard hematology (~2.3 mM tetramer) |

**Cross-check:** Full MetHb conversion gives $\Delta R_1 \approx 0.4 \times 9.2 = 3.7$ s$^{-1}$, yielding $T_1 \approx 210$ ms (consistent with bright T1 signal in subacute hematomas).

### 7.3 Hematoma Signal Evolution

| Stage | Time | Dominant species | T1W | T2W |
|---|---|---|---|---|
| Hyperacute | 0–12 h | Oxy-Hb (diamagnetic) | Iso | Iso/bright |
| Acute | 12 h–3 d | Deoxy-Hb (paramagnetic, intracellular) | Iso | **Dark** |
| Early subacute | 3–7 d | MetHb (paramagnetic, intracellular) | **Bright** | **Dark** |
| Late subacute | 7–28 d | MetHb (paramagnetic, extracellular) | **Bright** | **Bright** |
| Chronic | >28 d | Hemosiderin (superparamagnetic) | Iso/dark | **Dark rim** |

---

## VIII. References

1. Abbas Z, Hattingen E, Geisel O, Deichmann R (2015). Quantitative water content mapping at clinically relevant field strengths. *NeuroImage* 112:390–399.
2. Barber TW, Brockway JA, Higgins LS (1970). The density of tissues in and about the head. *Acta Neurol Scand* 46(1):85–92.
3. Baskurt OK, Meiselman HJ (2003). Blood rheology and hemodynamics. *Semin Thromb Hemost* 29(5):435–450.
4. Bloomfield IG, Johnston IH, Bilston LE (1998). Effects of proteins, blood cells and glucose on the viscosity of cerebrospinal fluid. *Pediatr Neurosurg* 28(5):246–251.
5. Bojorquez JZ et al. (2017). What are normal relaxation times of tissues at 3 T? *Magn Reson Imaging* 35:69–80.
6. Bradley WG Jr (1993). MR appearance of hemorrhage in the brain. *Radiology* 189(1):15–26.
7. Budday S, Sommer G, Birkl C et al. (2017). Mechanical characterization of human brain tissue. *Acta Biomaterialia* 48:319–340.
8. Budday S, Nay R, de Rooij R et al. (2015). Mechanical properties of gray and white matter brain tissue by indentation. *J Mech Behav Biomed Mater* 46:318–330.
9. Cheng S, Bilston LE (2007). Unconfined compression of white matter. *J Biomech* 40(1):117–124.
10. Fatouros PP et al. (1991). In vivo brain water determination by T1 measurements. *Magn Reson Med* 17(2):402–413.
11. Franceschini G, Bigoni D, Regitnig P, Holzapfel GA (2006). Brain tissue deforms similarly to filled elastomers and follows consolidation theory. *J Mech Phys Solids* 54(12):2592–2620.
12. Giordano C, Kleiven S (2014). Connecting fractional anisotropy from medical images with mechanical anisotropy of a hyperviscoelastic fibre-reinforced constitutive model for brain tissue. *J R Soc Interface* 11(91):20130914.
13. Gomori JM, Grossman RI, Yu-Ip C, Asakura T (1987). NMR relaxation times of blood: dependence on field strength, oxidation state, and cell integrity. *J Comput Assist Tomogr* 11(4):684–690.
14. Hajjarian Z et al. (2015). Optical thromboelastography to evaluate whole blood coagulation. *J Biophotonics* 8(5):372–381.
15. Horlocker TT, Wedel DJ (1993). Density, specific gravity, and baricity of spinal anesthetic solutions at body temperature. *Anesth Analg* 76(5):1015–1018.
16. Jeong HG et al. (2021). Hematoma Hounsfield units and expansion of intracerebral hemorrhage. *Int J Stroke* 16(2):191–198.
17. Kaczmarek M, Subramaniam RP, Neff SR (1997). The hydromechanics of hydrocephalus. *Bull Math Biol* 59(2):295–323.
18. Kleiven S (2007). Predictors for traumatic brain injuries evaluated through accident reconstructions. *Stapp Car Crash J* 51:81–114.
19. Lam WA et al. (2011). Mechanics and contraction dynamics of single platelets and implications for clot stiffening. *Nat Mater* 10(1):61–66.
20. Lipphaus A, Witzel U (2021). Three-dimensional finite element analysis of the dural folds and the human skull under head acceleration. *Anat Rec* 304(10):2301–2312.
21. Monson KL (2003). Mechanical and failure properties of human cerebral blood vessels. PhD Dissertation, UC Berkeley.
22. Prange MT, Margulies SS (2002). Regional, directional, and age-dependent properties of the brain undergoing large deformation. *J Biomech Eng* 124(2):244–252.
23. Ranucci M et al. (2014). Blood viscosity during coagulation at different shear rates. *Physiol Rep* 2(7):e12065.
24. Rashid B, Destrade M, Gilchrist MD (2012). Mechanical characterization of brain tissue in tension at dynamic strain rates. *J Mech Behav Biomed Mater* 14:477–487.
25. Rashid B, Destrade M, Gilchrist MD (2013). Mechanical characterization of brain tissue in simple shear at dynamic strain rates. *J Mech Behav Biomed Mater* 28:71–85.
26. Shahim K et al. (2010). Finite element analysis of normal pressure hydrocephalus: influence of CSF content and anisotropy in permeability. *Appl Bionics Biomech* 7(3):187–197.
27. Stanisz GJ et al. (2005). T1, T2 relaxation and magnetization transfer in tissue at 3T. *Magn Reson Med* 54(3):507–512.
28. Tutwiler V et al. (2016). Kinetics and mechanics of clot contraction are governed by the molecular and cellular composition of the blood. *Blood* 127(1):149–159.
29. Vidotto M et al. (2021). On the microstructural origin of brain white matter hydraulic permeability. *PNAS* 118(36):e2105328118.
30. Wufsus AR, Macera NE, Neeves KB (2013). The hydraulic permeability of blood clots as a function of fibrin and platelet density. *Biophys J* 104(8):1812–1823.
31. Zwirner J, Ondruschka B, Scholze M et al. (2019). Mechanical properties of human dura mater in tension — an analysis at an age range of 2 to 94 years. *Sci Rep* 9:16655.
