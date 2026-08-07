# koral-zig — Physics & Numerics Reference

This document is the physicist-facing reference for **koral-zig**, a Zig 0.16
reimplementation of the KORAL general-relativistic radiation-MHD code
(`../koral_lite`). It states, as precisely as the code allows, *which equations
the solver integrates and how*. Every equation is cross-referenced to the module
and function that evaluates it, so the science can be validated or extended
against the source of truth.

The default science target is **PUFFY**, a radiation-supported, geometrically
thick ("puffy") accretion torus around a black hole, evolved in Modified
Kerr–Schild (MKS2) coordinates. The validated reference run is the 2D
axisymmetric 10 M$_\odot$ Schwarzschild torus; the mass, spin, grid, and most
microphysics choices are runtime-retargetable (§10, and the Sgr A* / AGN presets
in `koral/problems/puffy/`). See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for how the modules fit together and
[`USER_GUIDE.md`](USER_GUIDE.md) for how to configure and run problems.

Units: unless a quantity is explicitly labelled CGS or Kelvin, all equations are
in **geometrized units** (GU): $G = c = 1$, with the black-hole mass setting the
length scale. Conversions live in [`koral/units.zig`](../koral/units.zig) and the
constant bundle [`koral/physics/thermo.zig`](../koral/physics/thermo.zig)
(`Consts`). Temperatures are in Kelvin in both systems.

---

## Table of contents

1. [Governing system: GR conservation laws](#1-governing-system-gr-conservation-laws)
2. [Coordinates and metric](#2-coordinates-and-metric)
3. [Ideal MHD: stress-energy, magnetic 4-vector, equation of state](#3-ideal-mhd-stress-energy-magnetic-4-vector-equation-of-state)
4. [Radiation: M1 closure and stress tensor](#4-radiation-m1-closure-and-stress-tensor)
5. [The radiation four-force $G^\mu$: opacities and Comptonization](#5-the-radiation-four-force-g-mu-opacities-and-comptonization)
6. [Radiative shear viscosity](#6-radiative-shear-viscosity)
7. [Mean-field ($\alpha$–$\Omega$) dynamo and scale height](#7-mean-field-alphaomega-dynamo-and-scale-height)
8. [Numerical scheme](#8-numerical-scheme)
9. [Floors and ceilings](#9-floors-and-ceilings)
10. [The PUFFY problem](#10-the-puffy-problem)
11. [Appendix: constants and unit conversions](#11-appendix-constants-and-unit-conversions)

---

## 1. Governing system: GR conservation laws

koral-zig integrates the equations of general-relativistic radiation
magnetohydrodynamics (GRRMHD) as a finite-volume system in conservation form.
The continuum equations are:

**Baryon-number (mass) conservation:**
$$\nabla_\mu (\rho u^\mu) = 0,$$
where $\rho$ is the rest-mass density and $u^\mu$ the gas four-velocity.

**Gas energy–momentum with radiation coupling:**
$$\nabla_\mu T^{\mu\nu} = G^\nu,$$
where $T^{\mu\nu}$ is the ideal-MHD stress-energy tensor (§3) and $G^\nu$ is the
radiation four-force (§5).

**Radiation moments (M1):**
$$\nabla_\mu R^{\mu\nu} = -G^\nu,$$
with $R^{\mu\nu}$ the radiation stress tensor (§4). The **sign convention** is
that $G^\nu$ is *added* to the gas and *subtracted* from the radiation: energy
and momentum removed from radiation appear in the gas, so the combined
$T + R$ system is conserved up to geometric source terms. This sign is visible
directly in the residuals of the implicit source solver
([`koral/solve/implicit.zig`](../koral/solve/implicit.zig), `residual`): the gas
energy residual carries $-\,dt\,\sqrt{-g}\,G$ while the radiation energy residual
carries $+\,dt\,\sqrt{-g}\,G$.

**Induction / Faraday (ideal MHD):** the magnetic field is evolved through the
spatial part of the homogeneous Maxwell equations, expressed as the antisymmetric
induction flux of the field 4-vector (§3, §8 constrained transport):
$$\partial_t(\sqrt{-g}\,B^i) + \partial_j\!\left[\sqrt{-g}\,(b^j u^i - b^i u^j) \right] = 0,$$
i.e. the conserved flux row $F[B^k] = \sqrt{-g}\,(b^k u^{d} - b^{d} u^k)$ in flux
direction $d$ (the code carries the field 4-vector $b^\mu$ directly, so no
$1/u^t$ factor appears — that factor belongs only to the lab-field form
$\sqrt{-g}\,(B^k u^d - B^d u^k)/u^t$) ([`koral/physics/flux.zig`](../koral/physics/flux.zig),
`fFluxPrime`). Divergence-freeness $\nabla_i B^i = 0$ is maintained discretely by
flux-CT (§8).

### 1.1 Conserved / primitive split (NV = 13 for PUFFY)

The state vector carries **primitive** variables `pp[]`; the finite-volume update
advances **conserved** variables `uu[]`, and a nonlinear inversion recovers the
primitives each substep. The variable layout is generated at compile time from the
active module set ([`koral/layout.zig`](../koral/layout.zig), `VarLayout`). For the
PUFFY configuration `{hydro, mhd, radiation}` the width is $N_V = 13$ with the
fixed C-compatible ordering:

| index | tag | primitive meaning | conserved meaning |
|------:|-----|-------------------|-------------------|
| 0 | `rho` | rest-mass density $\rho$ | $\sqrt{-g}\,\rho u^t$ |
| 1 | `uu`  | internal energy $u$ | $\sqrt{-g}\,(T^t{}_t + \rho u^t)$ (energy minus rest mass) |
| 2–4 | `vx,vy,vz` | relative velocity $\tilde u^i$ (**VELR**) | $\sqrt{-g}\,T^t{}_i$ |
| 5 | `entr` | entropy $S(\rho,u)$ | $\sqrt{-g}\,S u^t$ |
| 6–8 | `b1,b2,b3` | lab field $B^i$ | $\sqrt{-g}\,B^i$ |
| 9 | `ee` | radiation-frame energy $\hat E$ | $\sqrt{-g}\,R^t{}_t$ |
| 10–12 | `fx,fy,fz` | radiation-frame velocity $\tilde u_r^i$ (**VELR**) | $\sqrt{-g}\,R^t{}_i$ |

The forward map (primitives → conserved) is
[`koral/p2u.zig`](../koral/p2u.zig) (`p2u`, `p2uMhd`, `p2uRad`); the inverse
(conserved → primitives) is [`koral/solve/invert.zig`](../koral/solve/invert.zig)
(`u2pMhd`) for the MHD block and
[`koral/solve/invert_rad.zig`](../koral/solve/invert_rad.zig) (`u2pRad`) for the
M1 radiation block.

Velocities are stored as the **relative 4-velocity** $\tilde u^i$ measured by the
normal (Eulerian) observer — `VELR` in
[`koral/relele.zig`](../koral/relele.zig) — which is robust in the ergoregion and
at high Lorentz factor. Conversions among the four-velocity $u^\mu$ (VEL4), the
coordinate three-velocity $u^i/u^t$ (VEL3) and $\tilde u^i$ (VELR) are all in
`relele.zig` (`convVelsCore`).

The two extra module combinations that change $N_V$ (not used by PUFFY): adding
two-temperature electrons gives $N_V = 15$; adding photon-number transport gives
$N_V = 14$.

---

## 2. Coordinates and metric

The **internal computational grid is uniform** by construction; all curvature and
grid concentration live in the metric ([`koral/grid.zig`](../koral/grid.zig)). The
metric $g_{\mu\nu}$, its inverse $g^{\mu\nu}$, $\sqrt{-g}$, the Christoffel symbols
$\Gamma^i{}_{jk}$, and coordinate Jacobians are assembled per cell by evaluating
the analytic covariant metric in **forward-mode dual arithmetic**
([`koral/math/dual.zig`](../koral/math/dual.zig),
[`koral/metric/metric.zig`](../koral/metric/metric.zig), `compute`), so derivatives
are exact to machine precision. The metric is **stationary** ($\partial_t
g_{\mu\nu} = 0$): only the three spatial slots are seeded as dual variables.

Four coordinate systems are supported (`config.Coords`):
`mink`, `bl`, `ks`, `mks2`. PUFFY uses `mks2`. Forms live in
[`koral/metric/forms.zig`](../koral/metric/forms.zig).

### 2.1 Minkowski

Flat Cartesian, $g_{\mu\nu} = \mathrm{diag}(-1,1,1,1)$ (`gcovMink`). Used for the
radiative shock-tube tests.

### 2.2 Kerr in Boyer–Lindquist (BL)

With $G=c=M=1$ and spin $a$, defining $\Sigma = r^2 + a^2\cos^2\theta$ and
$\Delta = r^2 - 2r + a^2$ (`gcovBl`):

$$g_{tt} = \frac{2r}{\Sigma} - 1, \qquad
g_{t\phi} = -\frac{2 a r \sin^2\theta}{\Sigma}, \qquad
g_{rr} = \frac{\Sigma}{\Delta}, \qquad g_{\theta\theta} = \Sigma,$$
$$g_{\phi\phi} = \sin^2\theta\left(r^2 + a^2 + \frac{2 a^2 r \sin^2\theta}{\Sigma}\right).$$

The quantity $2r/\Sigma$ is the perturbed part of $g_{tt}$ (`gttpert`), stored
separately for cancellation-free energy assembly. The horizon roots are
$r_\pm = 1 \pm \sqrt{1-a^2}$ (`metric.rHorizonBL`, alongside `rIscoBL`); for
Schwarzschild ($a=0$) $r_+ = 2$.

### 2.3 Kerr–Schild (KS)

Horizon-penetrating: the extra $g_{tr}$, $g_{r\phi}$ terms make the metric
regular across the horizon (`gcovKs`). With $\Sigma$ as above and
$1 + 2r/\Sigma$ appearing throughout:

$$g_{tt} = \frac{2r}{\Sigma}-1,\quad g_{tr} = \frac{2r}{\Sigma},\quad
g_{rr} = 1 + \frac{2r}{\Sigma},\quad g_{t\phi} = -\frac{2ar\sin^2\theta}{\Sigma},$$
$$g_{r\phi} = -a\left(1+\frac{2r}{\Sigma}\right)\sin^2\theta,\quad
g_{\phi\phi} = \sin^2\theta\left[\Sigma + a^2\left(1+\frac{2r}{\Sigma}\right)\sin^2\theta\right].$$

### 2.4 Modified Kerr–Schild 2 (MKS2)

MKS2 is the exact diagonal pushforward of KS through the maps
$$r = R_0 + e^{x^1}, \qquad \theta = \theta(x^2),$$
where $R_0 = $ `MKSR0` and the polar map, controlled by $H_0 = $ `MKSH0`, is
(`mks2ThetaMetric`)
$$\theta(x^2) = \frac{\pi}{2}\left[1 + \cot\!\left(\frac{\pi H_0}{2}\right)\tan\!\big(H_0 \pi (x^2 - \tfrac12)\big)\right].$$
The metric is
$$g'_{ij} = g^{\mathrm{KS}}_{ij}\,\frac{\partial x^{\mathrm{KS}}_i}{\partial x'^i}\,\frac{\partial x^{\mathrm{KS}}_j}{\partial x'^j},$$
with the diagonal Jacobian factors $(1,\; dr/dx^1 = e^{x^1},\; d\theta/dx^2,\; 1)$
(`gcovMks2`). $R_0$ shifts the inner radial boundary off the singularity;
$H_0 < 1$ concentrates cells toward the midplane. The reference PUFFY run uses
`MKSR0 = 0.1`, `MKSH0 = 0.9`; both are params-file-overridable, and $R_0$ may be
negative (the AGN preset uses `MKSR0 = -1.5`, pushing the innermost cells deeper
toward the horizon rather than away from it).

> **Deliberate C quirk (bit-comparability):** MKS2 uses *two different values of
> $\pi$*. The metric/Christoffel expressions use the truncated
> $\pi_c = 3.141592654$ (KORAL's `#define Pi`), while the coordinate point
> transforms and `gttpert` use the exact `std.math.pi`. The two "theta" flavors
> therefore disagree at $\sim10^{-9}$. This is intentional and must not be
> "fixed." See [`ARCHITECTURE.md`](ARCHITECTURE.md) on the oracle strategy.

### 2.5 Determinant, log-derivative, Christoffels, lapse

From the dual determinant $\det$ (`metric.compute`):
$$\sqrt{-g} = \sqrt{-\det}, \qquad
\partial_i \ln\sqrt{-g} = \tfrac12\,\frac{(\det)_{,i}}{\det}.$$
Christoffel symbols use the standard half-inverse-metric formula, with all
$\partial_t$ terms dropped by stationarity:
$$\Gamma^i{}_{jk} = \tfrac12\,g^{il}\big(\partial_j g_{lk} + \partial_k g_{lj} - \partial_l g_{jk}\big).$$
The ADM lapse is
$$\alpha = \sqrt{-1/g^{tt}}$$
(`precompute.geometryAt`). At cell centers the Christoffel trace is replaced by a
finite-difference $\sqrt{-g}$-gradient (`applyKrisCorrection`, KORAL's `GDETIN=1`
/ `MODYFIKUJKRZYSIE`) so that geometric source terms telescope against discrete
flux divergence and uniform states stay uniform.

### 2.6 Coordinate transforms

Point transforms and analytic Jacobians route through KS as a hub
([`koral/metric/coco.zig`](../koral/metric/coco.zig), `cocoN`, `dxdx`), exactly as
in KORAL. The BL$\leftrightarrow$KS time shift is
$$\Delta t = \frac{2}{\sqrt{1-a^2}}\,\mathrm{atanh}\!\frac{\sqrt{1-a^2}}{1-r} + \ln\Delta$$
(`blKsTimeShift`); it is NaN between the horizons where $\Delta<0$, faithfully
matching C. Tensors and whole primitive states are transformed by
[`koral/frames.zig`](../koral/frames.zig) (`trans2Coco`, `trans22Coco`,
`transPallCoco`).

---

## 3. Ideal MHD: stress-energy, magnetic 4-vector, equation of state

### 3.1 Gamma-law equation of state

The gas is an ideal fluid with adiabatic index $\Gamma$ (`gam`; PUFFY uses
$\Gamma = 5/3$). Pressure ([`koral/physics/hydro.zig`](../koral/physics/hydro.zig),
`calcTij`):
$$p = (\Gamma-1)\,u.$$
The relativistic gas enthalpy density is $w = \rho + u + p$.

The **entropy** variable is the logarithmic form (`sFromU`, KORAL
`calc_Sfromu`), with polytropic index $n = 1/(\Gamma-1)$:
$$S(\rho,u) = \rho\,\ln\!\left(\frac{p^{\,n}}{\rho^{\,n+1}}\right)
= \rho\,\ln\!\left(\frac{[(\Gamma-1)u]^{\,n}}{\rho^{\,n+1}}\right),$$
with the exact analytic inverse (`uFromS`)
$$u(S,\rho) = \frac{1}{\Gamma-1}\left(\rho^{\,n+1}\,e^{S/\rho}\right)^{\Gamma-1}.$$

### 3.2 Magnetic 4-vector $b^\mu$

The evolved primitive field is the lab-frame "curly" $B^i$ (KORAL `pp[B1..B3]`).
The fluid-frame magnetic 4-vector (satisfying $b^\mu u_\mu = 0$) is
([`koral/physics/bfield.zig`](../koral/physics/bfield.zig), exported as
`koral.physics.mhd`; `bconFrom4vel`):
$$b^0 = B^i u_i, \qquad b^j = \frac{B^j + b^0 u^j}{u^t}\;(j=1,2,3),$$
with the magnetic energy density $b^2 = b^\mu b_\mu$
(`bconBcovBsqFrom4vel`). The inverse map recovering $B^i$ from $b^\mu$ is
$B^i = b^i u^t - b^0 u^i$ (`bigBconFrom4vel`).

### 3.3 Ideal-MHD stress-energy tensor

The full stress-energy tensor is assembled (both indices up) in
`hydro.calcTij`:
$$T^{\mu\nu} = (\rho + u + p + b^2)\,u^\mu u^\nu + \left(p + \tfrac12 b^2\right)g^{\mu\nu} - b^\mu b^\nu.$$
In code: $\eta = \rho + u + p + b^2$ (total enthalpy density), $p_{\rm tot} = p +
b^2/2$ (total pressure), and $T^{ij} = \eta\,u^i u^j + p_{\rm tot}\,g^{ij} - b^i
b^j$. In the pure-hydro limit $b^\mu = 0$, $b^2 = 0$. The magnetic contribution is
gated at compile time by whether the layout carries `b1`.

The conserved energy row is written **cancellation-free** in `p2u.p2uMhd` and
`flux.fFluxPrime` using $1 + u_t$ (`calcUtp1`, KORAL `calc_utp1`) so the
non-relativistic energy $\approx T^t{}_t + \rho u^t$ retains precision when
$\rho u^t$ nearly cancels the rest mass.

### 3.4 Sound and magnetosonic speeds

Wave speeds live in
[`koral/physics/wavespeeds.zig`](../koral/physics/wavespeeds.zig), **not** in
`hydro.zig`. The fluid-frame relativistic magnetosonic speed squared combines the
sound speed $c_s^2$ and Alfvén speed $v_A^2$ (`gasWspeed2`):
$$v_{\rm tot}^2 = c_s^2 + v_A^2 - c_s^2 v_A^2,$$
$$c_s^2 = \frac{\Gamma(\Gamma-1)u}{\rho + u + (\Gamma-1)u} = \frac{\Gamma p}{\rho h}\ \ (\text{ceiling } 0.95),\qquad
v_A^2 = \frac{b^2}{b^2 + \rho + u + p}\ \ (\text{floor } 0),$$
where $\rho h = \rho + u + p$ is the enthalpy. The lab-frame characteristic
speeds $dx^i/dt$ come from the HARM quadratic in `lrCore`, boosting the isotropic
fluid-frame $v^2$ through the inverse metric; co-going speeds are clamped so the
left speed $\le 0 \le$ right speed (`gasWavespeedsLr`).

---

## 4. Radiation: M1 closure and stress tensor

Radiation is treated in the grey **M1** two-moment approximation
([`koral/physics/radiation.zig`](../koral/physics/radiation.zig)). The radiative
state is the radiation-rest-frame energy density $\hat E$ (`pp[EE]`) and the
relative (VELR) spatial components of the radiation-frame 4-velocity $u_r^i$
(`pp[FX..FZ]`).

### 4.1 M1 stress tensor and Eddington relation

The M1 closure posits an isotropic radiation field in the radiation rest frame,
giving the lab-frame stress tensor (`calcRij`, KORAL `calc_Rij_M1`):
$$R^{\mu\nu} = \frac{4}{3}\,\hat E\, u_r^\mu u_r^\nu + \frac{1}{3}\,\hat E\, g^{\mu\nu}.$$
The $\tfrac43$/$\tfrac13$ split is the Eddington relation in the comoving frame
($P = \tfrac13 \hat E$). The radiation-frame 4-velocity is reconstructed from the
VELR primitives by solving the normalization $u_r^\mu u_{r\mu} = -1$
(`urfCon`, via `relele.convVels` VELR→VEL4).

### 4.2 Fluid-frame radiation energy $\hat E_{\rm ff}$

Projecting the (mixed) radiation stress onto the **gas** 4-velocity gives the
radiation energy density measured in the fluid frame (`calcFfRtt`,
`calcFfEhat`):
$$\hat E_{\rm ff} = -R^t{}_t\big|_{\rm ff} = -\,R^\mu{}_\nu\,u^\nu u_\mu.$$
Note the distinction: `pp[EE]` $= \hat E$ is the *radiation-rest-frame* energy (a
primitive), while $\hat E_{\rm ff}$ is the radiation energy seen by the *gas* — it
uses the gas velocity `pp[VX..VZ]`, not the radiation velocity.

### 4.3 M1 conserved → primitive inversion

Given conserved $R^t{}_\mu$, the closed-form inversion recovers $(\hat E, u_r^i)$
([`koral/solve/invert_rad.zig`](../koral/solve/invert_rad.zig), `u2pRad`). The
relative Lorentz factor squared follows from a closed-form (JCM) solution of the
M1 quadratic (`m1GammaRel2`); the energy is
$$\hat E = \frac{3\,R^{tt}\,\alpha^2}{4\,\gamma_{\rm rel}^2 - 1}\qquad(\text{`m1Erf`}),$$
and the velocity from
$$\tilde u_r^i = \frac{\alpha\left(R^{t\,i} + \tfrac13 \hat E\, g^{ti}(4\gamma_{\rm rel}^2 - 1)\right)}{\tfrac43 \hat E\,\gamma_{\rm rel}}.$$
When the generic solution fails (unphysical $\gamma_{\rm rel}^2$, $\hat E$ below
floor, or $\gamma_{\rm rel}$ above the ceiling), a two-branch **cold fallback**
recomputes at fixed $\gamma^2 \approx 1$ and $\gamma^2 = \gamma_{\max}^2$, keeping
whichever reproduces the original $R^t{}_t$ more closely (`m1Cold`).

### 4.4 Optical-depth-limited radiation wave speeds

In the optically thin limit the M1 characteristic speed is
$c/\sqrt3$, i.e. $v^2 = 1/3$, used isotropically for the timestep
(`calcRadWavespeeds`). In optically thick cells the effective diffusion speed is
reduced (Sądowski et al.):
$$v_{\rm rad}^2 = \min\!\left(\frac13,\; \frac{(4/3)^2}{\tau_{\rm tot}^2}\right),$$
where the per-direction optical depth is $\tau_{\rm tot} = \chi\,\Delta x$ with the
total extinction $\chi = \kappa + \kappa_{\rm es}$ (`calcChi`). The unlimited
speeds set the CFL timestep; the $\tau$-limited speeds are used in the flux
combination.

> **Deliberate C quirk:** the $z$-direction limiter uses the *$y$* optical depth
> (`rv2z = rv2dim[1]`), a preserved KORAL bug pinned by a theory test.

An optional axis guard, `DAMPRADWAVESPEEDNEARAXIS` (params key
`dampradwavespeednearaxis`, off by default, 2 in the AGN preset), zeroes the
optical depths within N cells of either $\theta$ pole so the radiation speeds
there stay at the undamped $v^2 = 1/3$ — preventing the $\tau$ limiter from
starving the timestep in the low-density polar funnel.

---

## 5. The radiation four-force $G^\mu$: opacities and Comptonization

The radiation–gas coupling $G^\mu$ carries all the microphysics
([`koral/physics/radforce.zig`](../koral/physics/radforce.zig),
[`koral/physics/opacities.zig`](../koral/physics/opacities.zig),
[`koral/physics/thermo.zig`](../koral/physics/thermo.zig)).

### 5.1 Thermodynamics and temperatures

The ideal-gas temperature from internal energy (`thermo.tFromUrho`):
$$T_{\rm gas} = \frac{\mu_{\rm gas} m_p}{k_B}\,\frac{p}{\rho}, \qquad p = (\Gamma-1)u,$$
with mean molecular weights $\mu_{\rm gas}, \mu_i, \mu_e$ from the composition
(`thermo.Composition`; the reference PUFFY run overrides $\mu_{\rm gas}=1$,
$\mu_i=\mu_e=2$ while keeping $X = 1$ for the opacity formulas). Setting the
params keys `hfrac`/`hefrac` instead activates the full C composition formulas:
$\mu_{\rm gas} = 1/(0.5 + 1.5X + 0.25Y + \langle 1/A\rangle Z)$,
$\mu_i = 1/(X + 0.25Y + \langle 1/A\rangle Z)$, $\mu_e = 2/(1+X)$, with
$Z = 1 - X - Y$, $\langle 1/A\rangle = 0.0570490404519$,
$\langle Z^2/A\rangle = 5.14445150955$. In the single-temperature build
$T_e = T_i = T_{\rm gas}$, floored at $10^2$ K ($T_{\rm gas}$ itself is not
floored) (`tempsFromUrho`). Thermal electron density
$n_e = \rho/(\mu_e m_p)$ (`thermalNe`).

**LTE blackbody relation** (`Consts.lteEfromT`/`lteTfromE`, and
`units.lteEfromT`):
$$E_{\rm LTE} = 4\,\sigma_{\rm rad}\,T^4, \qquad T = \left(\frac{E}{4\sigma_{\rm rad}}\right)^{1/4},$$
i.e. the radiation constant $a = 4\sigma_{\rm rad}$ in GU. The frequency-integrated
blackbody source function used in the four-force is $B = \sigma_{\rm rad}T_e^4/\pi$,
entering as $\kappa_{\rm gas,abs}\cdot 4\pi B = \kappa_{\rm gas,abs}\cdot 4\sigma_{\rm rad}T_e^4$.

### 5.2 The lab- and fluid-frame four-force

The thermal radiation four-force is (`calcGiFromState`, KORAL
`calc_all_Gi_with_state`):
$$G^i_{\rm lab} = -(\kappa_{\rm rad,ross}+\kappa_{\rm es})\,R^i{}_j u^j
- \left[(\kappa_{\rm rad,ross}+\kappa_{\rm es}-\kappa_{\rm rad,abs})\,\hat E_{\rm ff} + \kappa_{\rm gas,abs}\,4\pi B\right]u^i.$$
The spatial fluid-frame components are obtained by Lorentz-boosting $G_{\rm lab}$
to the fluid frame (`frames.boost2Lab2Ff`); the **time component is overwritten
with the exact expression**
$$G^0_{\rm ff} = -\kappa_{\rm gas,abs}\,4\pi B + \kappa_{\rm rad,abs}\,\hat E_{\rm ff},$$
which is the emission/absorption energy balance in the comoving frame.

After the Comptonization term is added, **both frames of $G^\mu$ are scaled by
`par.opdamp`** (C `OPDAMPINIMPLICIT`). The default is 1.0 (exact no-op); the
implicit solver's opacity-damping ladder (§8.5) sets
$\mathrm{opdamp} = f^{-\ell}$ when retrying failed cells at reduced coupling.

### 5.3 Absorption/emission opacities

`opacities.calcOpacitiesFromState` computes six opacity flavors (Planck-mean gas
emission `gas_abs`, Planck-mean radiation absorption `rad_abs`, photon-number
`gas_num`/`rad_num`, and Rosseland `gas_ross`/`rad_ross`) summing the enabled
channels:

**Thermal bremsstrahlung (free-free):**
$$\epsilon_{\rm ff} = 1.4\times10^{-27}\,n_e\,\frac{X + Y + \langle Z^2/A\rangle Z_{\rm frac}}{m_p}\,\sqrt{T_e}\;\cdot 1.2\,(1 + 4.4\times10^{-10}T_e),$$
$$\kappa_{\rm gas,ff} = \left(\frac{\epsilon_{\rm ff}}{B_{\rm bb}}\right)\rho \quad\text{(CGS→GU)},\qquad B_{\rm bb} = 4\sigma_{\rm rad}T_e^4,$$
with $1.2$ the Gaunt factor and $(1+4.4\times10^{-10}T_e)$ the relativistic
correction.

The free-free **Rosseland** means are scaled from the Planck mean: the gas
channel is $\kappa_{\rm gas,ross} = 0.0330\,\kappa_{\rm gas,ff}$ and the
radiation channel is $\kappa_{\rm rad,ross} =
\kappa_{\rm gas,ff}\,s(\zeta)\,\zeta^{-3}$ with $\zeta = T_{\rm rad}/T_e$ and the
fit $s(\zeta) = 14.12/(432.7 - 106.8\,\zeta^{-3/5} + 43.17\,\zeta^{-4/5} +
57.88\,\zeta^{-1})$; the radiation Planck absorption uses
$\kappa_{\rm rad,ff} = \kappa_{\rm gas,ff}\,\log(1+1.6\zeta)/\log(2.6)\,\zeta^{-3}$.

**Synchrotron:** self-absorption fitting functions in the magnetic field
$B_{\rm mag}$ (from $b^2$) and characteristic frequency $\nu_{\rm mb} =
1.19\times10^{-13}T_e^2$, e.g. the emissivity
$\epsilon_{\rm syn} = 3.61\times10^{-34}(n_e/\rho)T_e^2 B_{\rm mag}^2$; all five
synchrotron opacities are multiplied by the relativistic suppression
$$T_{\rm rel,fac} = \frac{T_{\rm rel}^2}{1 + T_{\rm rel}^2},\qquad T_{\rm rel} = T_e/(m_e c^2/k_B),$$
suppressing synchrotron at non-relativistic $T_e$.

An alternative NR treatment, the **synchrotron bridge** (C
`USE_SYNCHROTRON_BRIDGE_FUNCTIONS`; params key `synchrotron_bridge`, off in the
validated build), *replaces* the $T_{\rm rel,fac}$ suppression with: a clamp
$T_{\rm rad,syn} = \max(T_{\rm rad},\,T_{\rm rad,bb}^{4/3}/T_e^{1/3})$ feeding
every $T_{\rm rad}$-dependent channel; additive non-relativistic components
$\propto 2.35869\times10^{-21}(n_e/\rho)\,B_{\rm mag}^2/T^3$ on both synchrotron
absorption channels ($T = T_{\rm rad,syn}$ for `rad_abs`, $T = T_e$ for
`gas_abs`); and a number-opacity crossover factor
$(T_e/T_c)/(1 + T_e/T_c)$ with $T_c = 5.07783\times10^9$ K.

### 5.3b MESA Rosseland opacity tables

For AGN-mass runs the analytic free-free Rosseland fits are inadequate (bound-free
and line opacities dominate at $10^4$–$10^6$ K), so both free-free **Rosseland**
channels can instead be looked up from a tabulated MESA opacity
([`koral/physics/mesa.zig`](../koral/physics/mesa.zig), `MesaTable`; data in
`data/mesa_tables/`, e.g. `a09_z0.02_x0.7.data`; params key `mesa_table`). The
table stores $\log_{10}\kappa_{\rm Ross}$ [cm²/g] on a $(\log T,\,\log R)$ grid
with $\log R \equiv \log\rho - 3\log T + 18$; lookup is bilinear in log–log space
with both axes clamped to the grid edges (no extrapolation). The wired-in channels
become
$$\kappa_{\rm rad,ross} = \max\big(\kappa_{\rm MESA}(T_{\rm rad}, \rho) - 0.2(1+X),\,0\big)\,\rho,\qquad
\kappa_{\rm gas,ross} = \max\big(\kappa_{\rm MESA}(T_e, \rho) - 0.2(1+X),\,0\big)\,\rho,$$
(CGS→GU), i.e. the electron-scattering part is subtracted since it is carried
separately as $\kappa_{\rm es}$. The free-free **Planck** channels keep the
bremsstrahlung formula (and are forced on whenever a MESA table is loaded,
regardless of the `bremsstrahlung` toggle, matching C). The table file must match
the configured composition (`hfrac` → X, metallicity → Z) — koral-zig does *not*
auto-select the file the way C does.

### 5.4 Electron scattering (Klein–Nishina-corrected Thomson)

The scattering opacity (`opacities.kappaEsPuffy`, PUFFY hook):
$$\kappa_{\rm es} = \kappa_{\rm CGS\to GU}\!\left[0.2\,(1+X)\,f_{\rm KN}\right]\rho,\qquad
f_{\rm KN} = \frac{1}{1 + (T_{\rm kn}/4.5\times10^{8})^{0.86}},$$
where $0.2(1+X)$ cm$^2$/g is the Thomson opacity ($=0.4$ for $X=1$) and $T_{\rm kn}
= T_{\rm rad}$. There are two $\kappa_{\rm es}$ flavors that differ only in which
temperature is used for $T_{\rm rad}$: the four-force path uses $T_{\rm rad} =
T_{\rm rad,bb}$; the wavespeed/$\chi$ path (`calcKappaes`) uses $T_{\rm rad} =
T_e$.

Scattering can also be switched off entirely (`radforce.KappaesMode = .none`,
params key `scattering`; C leaves `PR_KAPPAES` undefined in the AGN problem so
`calc_kappaes` returns 0). Because the Compton coefficient is
$\propto \kappa_{\rm es}$ (§5.5), turning scattering off also zeroes the
Comptonization four-force.

### 5.5 Thermal Comptonization

The Compton energy-exchange term (`comptComptonCoeff`, KORAL
`calc_Compt_Gi_with_state`), with $\theta_e = (k_B/m_e c^2)T_e$:
$$G^\mu_{\rm Compt} = C\,u^\mu,\qquad G^0_{\rm ff,Compt} = C,$$
$$C = \kappa_{\rm es}\,\hat E_{\rm ff}\,4\,\frac{k_B}{m_e c^2}(T_{\rm rad}-T_e)\,\frac{1 + 3.683\,\theta_e + 4\,\theta_e^2}{1 + 4\,\theta_e}.$$
The sign follows $(T_{\rm rad}-T_e)$: hot electrons up-scatter (heat the
radiation), cold electrons down-scatter. Comptonization is a per-problem toggle
(on in PUFFY, off in grey tube tests).

---

## 6. Radiative shear viscosity

PUFFY includes an M1 radiative shear-viscosity closure that adds an anisotropic
stress to the radiation flux
([`koral/physics/radvisc.zig`](../koral/physics/radvisc.zig)).

### 6.1 Lab-frame shear tensor

From the radiation-frame velocity field (slots `FX,FY,FZ`, VELR), the lab-frame
shear tensor and expansion are computed with centered covariant derivatives
(`calcShearLab`, KORAL `calc_shear_lab`):
$$\sigma_{ij} = \tfrac12\left(\mathcal{D}_{ik}P^k{}_j + \mathcal{D}_{jk}P^k{}_i\right) - \tfrac13\,\theta\,P_{ij},\quad i,j=1,2,3,$$
where $\mathcal{D}_{ij} = u_{i,j} - \Gamma^k{}_{ij}u_k$ is the covariant velocity
gradient, $P_{ij} = g_{ij} + u_i u_j$ and $P^i{}_j = \delta^i{}_j + u^i u_j$ are the
projection tensors orthogonal to $u^\mu$, and the expansion is
$$\theta = u^\mu{}_{;\mu} = \sum_i\left(\partial_i u^i + \Gamma^i{}_{ik}u^k\right).$$
The time components follow from $u^\mu\sigma_{\mu\nu} = 0$.

### 6.2 Viscosity coefficient

The kinematic viscosity is the shear-$\alpha$ prescription with the photon mean
free path (`calcRadVisccoeff`, KORAL `calc_rad_visccoeff`):
$$\nu = \alpha_{\rm visc}\,\lambda_{\rm mfp},\qquad \lambda_{\rm mfp} = 1/\chi,$$
with $\alpha_{\rm visc} = 0.1$ (now a runtime knob, `radvisc.Params.alpha` on
`Sim.Options`). The mean free path is capped by the BL radius, smoothly killed
inside $r = 1.2\,r_{\rm h}$ via a step function, and set to zero at/inside the
horizon. The coefficient is further limited for explicit stability
(`RADVISCNUDAMP`): $\nu \le \Delta x_{\min}^2/(2\cdot 2\,dt)$, where
$\Delta x_{\min}$ is the **metric-scaled** cell size
$\min_d \Delta x_d\sqrt{g_{dd}}$ over the active dimensions.

### 6.3 Viscous radiation stress

The viscous stress is Navier–Stokes-like (`calcRijVisc`, KORAL `calc_Rij_visc`):
$$R^{ij}_{\rm visc} = -2\,\nu\,\hat E\,\sigma^{ij},$$
with $\hat E = $ `pp[EE]` and $\sigma^{ij}$ from raising both indices of the shear
tensor. It is precomputed once per step over the domain plus a ghost ring
(corners excluded, `ifOutsideGc` — the C `if_outsidegc` rule)
(`calcRijViscTotal`) and added at faces to the M1 radiation flux, damped by a
characteristic-velocity cap `MAXRADVISCVEL` $= 0.1$ (runtime knob
`radvisc.Params.maxvel`) to keep the diffusive flux causal (`addRadViscFlux`).

The shear algebra itself is a pure function `shearFromGradients(du, du2, ucon,
ucov, gg, kr)`, split out of `calcShearLab` so the invariants
$\sigma_{\mu\nu} = \sigma_{\nu\mu}$ and $\sigma_{\mu\nu}u^\nu = 0$ can be gated
on analytic velocity fields (`koral/radvisc_tests.zig`); `calcShearLab` retains
only the finite-difference gather with C's corner-avoidance one-sided-derivative
rules.

---

## 7. Mean-field ($\alpha$–$\Omega$) dynamo and scale height

PUFFY uses the KORAL "mimic dynamo" — a sub-grid mean-field prescription that
regenerates poloidal field from toroidal field to sustain magnetization in a 2D
axisymmetric run ([`koral/magn/dynamo.zig`](../koral/magn/dynamo.zig)). It runs
after each explicit RK substep (`applyDynamo`).

### 7.1 Density-weighted scale height

The angular disk scale height at each radius (`calcScaleHeight`, KORAL
`calc_avgs_throughout`):
$$\left(\frac{H}{R}\right)(r) = \sqrt{\frac{\sum_{\theta,\phi}\rho\sqrt{-g}\,(\pi/2 - \theta_{\rm BL})^2}{\sum_{\theta,\phi}\rho\sqrt{-g}}},$$
the density-weighted RMS angular displacement from the midplane. (Innermost radial
index stored unnormalized — a preserved C quirk.)

### 7.2 Dynamo vector potential

Per cell, a toroidal vector-potential increment is generated from the existing
$B^\phi$ (`mimicDynamo`, KORAL `mimic_dynamo`):
$$\Delta A_\phi = \alpha_{\rm eff}\,\frac{H_{\rm d\theta}/(\pi/2)}{0.4}\,\frac{dt}{P_K}\,r\,g_{\phi\phi}\,B^\phi\,f_{\rm radius}\,f_{\rm mag}\,f_{\rm angle},$$
where $P_K = 2\pi/\Omega_K$ is the Keplerian period with $\Omega_K = 1/(a +
\sqrt{r^3})$. The gates $f_{\rm angle}$ (disables the dynamo where the field pitch
already exceeds `THETAANGLE` $=0.25$), $f_{\rm radius}$ (smoothly off inside ISCO),
and the parabolic height weight $f_{\rm mag} = f_{z H} = \max(0, 1 - z_H^2)$
localize it to the disk body. Cells with $r < 1.0001\,r_{\rm horizon}$ contribute
nothing. The pitch angle is $-b^r b^\phi\sqrt{g_{rr}g_{\phi\phi}}/b^2$ evaluated
in BL (`fieldAngle`, C `calc_angle_brbphibsq`), clamped at $\ge -1$ and treated as
gate-closed when non-finite; the scale height is clamped at
$H_{\rm d\theta} \le 0.9\,\pi/2$ before use. The tunables live in `dynamo.Params`:
`ALPHADYNAMO` $= 2\times0.314$, `ALPHABETA` $= 2\times6.28$, `THETAANGLE` $=
0.25$, `BETASATURATED` $= 0.1$, `EXPECTEDHR` $= 0.3$ (used only when
`calchronthego = false`), and the `alphaflipssign` / `dampbeta` / `calchronthego`
switches (all default on). The per-cell law is the pure function
`dynamoDeltaA`, gated in isolation by `koral/dynamo_tests.zig`.

**Equatorial sign flip** (`ALPHAFLIPSSIGN`): the dynamo $\alpha$ is made
antisymmetric about the midplane, as the physical $\alpha$-effect requires:
$$\alpha_{\rm eff} = -\frac{\pi/2 - \theta}{H_{\rm d\theta}/2}\,\alpha_{\rm dynamo}.$$

The increment $\Delta A_\phi$ is curled (`ct.curlFromA`) into a poloidal $B$ that is
superimposed on the field, then `p2u` updates the conserveds.

### 7.3 $\beta$-damping of the azimuthal field

Once the plasma-$\beta$ exceeds a saturation value, $B^\phi$ is damped toward zero
without overshoot (`dampBphi`, KORAL `DAMPBETA`):
$$\Delta B^\phi = -\alpha_\beta\,f_{\rm radius}\,f_{z H}\,\frac{dt}{P_K}\,\frac{\max(0,\beta-\beta_{\rm sat})}{\beta_{\rm sat}}\,B^\phi,$$
with $\beta = \tfrac12 b^2 / p_{\rm tot}$, $p_{\rm tot} = (\Gamma-1)u +
\hat E_{\rm ff}/3$ (gas + radiation pressure), $\beta_{\rm sat} = 0.1$, and a
clamp so $|B^\phi|$ never crosses zero.

---

## 8. Numerical scheme

The driver ([`koral/sim.zig`](../koral/sim.zig), `Sim`) advances the finite-volume
system. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module wiring.

### 8.1 Finite-volume discretization

The conserved update per cell is (`opExplicit`):
$$U^{n+1} = U^n - \sum_d \frac{F^{d}_{i+1/2} - F^{d}_{i-1/2}}{\Delta x_d}\,dt + S\,dt,$$
where the geometric (metric) source (`metricSource`, KORAL
`f_metric_source_term_arb`, `GDETIN=1`) is the Christoffel contraction
$$S_\nu = \sqrt{-g}\,T^k{}_l\,\Gamma^l{}_{\nu k}$$
(plus the analogous $R^k{}_l$ term for the radiation rows). The advective fluxes
$F^d$ of every conserved row are built by `flux.fFluxPrime`, all multiplied by
$\sqrt{-g}$.

### 8.2 PPM reconstruction

Cell-average primitives are reconstructed to left/right face states
([`koral/fv/recon.zig`](../koral/fv/recon.zig); the scheme is the tagged union
`Scheme = {donor, linear{theta}, ppm}` dispatched by `reconstruct`/
`reconstructN`). PUFFY uses **PPM** (Colella & Woodward 1984) on a five-point
non-uniform stencil (`ppm`): the interface value (eq 1.6), the monotonized slope
(eqs 1.7–1.8), and the parabola monotonicity limiter (eq 1.10) that flattens the
profile to first order at local extrema. A linear minmod-$\theta$ limiter
(PUFFY $\theta = 1.5$) and donor-cell are the lower-order options. PPM sets the
ghost depth to $N_G = 3$. With `reduceorderatbh` set (C `REDUCEORDERATBH`), cells
whose BL radius is inside the horizon drop one order (PPM → linear → donor); the
MC/Superbee variants of C's `FLUXLIMITER` are not implemented.

### 8.3 Approximate Riemann solvers (LAXF / HLL)

One-sided fluxes are combined per face
([`koral/fv/laxf.zig`](../koral/fv/laxf.zig); the face loop is
`sim.zig::fluxesAtFaces`). **Lax–Friedrichs** (PUFFY):
$$F^\star = \tfrac12\big(F_L + F_R\big) - \tfrac12\,a_g\,(U_R - U_L),$$
with $a_g$ the local maximum characteristic speed. **HLL**:
$$F^\star = \begin{cases} F_L, & a_l > 0\\ F_R, & a_r < 0\\ \dfrac{-a_l F_R + a_r F_L + a_l a_r (U_R - U_L)}{a_r - a_l}, & \text{else}\end{cases}$$
Hydro and radiation are treated as **two independent hyperbolic systems**, each
with its own wave speeds — the split point is the `ee` row (`fluxesAtFaces`).

### 8.4 Constrained transport ($\nabla\!\cdot\!B = 0$)

Magnetic divergence-freeness is maintained by Tóth flux-CT
([`koral/magn/ct.zig`](../koral/magn/ct.zig), `fluxCt`): corner EMFs are formed by
averaging the transverse $B$-rows of the face fluxes, e.g.
$$\mathrm{EMF}_x = c_{\rm emf}\Big[\big(F^y[B^z] + \ldots\big) - \big(F^z[B^y] + \ldots\big)\Big],$$
and the $B$-face-fluxes are rebuilt as the curl of the averaged edge EMFs,
$$F^x[B^y] = \tfrac12\big(\mathrm{EMF}_z(i,j)+\mathrm{EMF}_z(i,j{+}1)\big),\quad F^x[B^x] = 0.$$
Initial fields are set from a vector potential $A_i$ (stored transiently in the
$B$ slots) via the discrete curl $B^i = \mathrm{curl}(A)^i/\sqrt{-g}$
(`calcBfromA`, `calcBfromACore`). The divergence diagnostic is `calcDivB`.

### 8.5 RK2 IMEX time integrator

The stiff radiation source is integrated implicitly while the hydro/MHD fluxes are
explicit, using the Pareschi–Russo SSP2(2,2,2) IMEX scheme with
$$\gamma = 1 - \frac{1}{\sqrt2}\approx 0.2929$$
(`step`, KORAL `problem.c:141-402`). Writing $U(1), U(2)$ for the stage states,
the explicit stage derivative is $\mathrm{dut} = [F(U)]$ and the implicit one is
$\mathrm{drt} = [\text{source}]/(\,dt\,\gamma)$:

**Stage 1 (implicit then explicit):**
$$\mathrm{drt}_1 = \frac{U^{(1)} - U^n}{dt\,\gamma}\ \text{after } \mathtt{opImplicit}(dt\,\gamma),\qquad
\mathrm{dut}_1 = \frac{U - U'}{dt}\ \text{after } \mathtt{opExplicit}(dt).$$

**Intermediate:**
$$U = U^n + dt\,\mathrm{dut}_1 + dt\,(1 - 2\gamma)\,\mathrm{drt}_1.$$

**Stage 2:** compute $\mathrm{drt}_2$, $\mathrm{dut}_2$ analogously, then combine
$$U^{n+1} = U^n + \frac{dt}{2}\big(\mathrm{dut}_1 + \mathrm{dut}_2\big) + \frac{dt}{2}\big(\mathrm{drt}_1 + \mathrm{drt}_2\big).$$
The choice $\gamma = 1-1/\sqrt2$ makes the implicit tableau L-stable
($R(\infty)=0$), damping the stiff radiation source.

**Implicit source solve** ([`koral/solve/implicit.zig`](../koral/solve/implicit.zig),
`solveImplicitLab`): each `opImplicit` drives the conserved-variable residual to
zero with a 6-rung ladder of 4-primitive Newton solves (energy/entropy $\times$
lab/fluid-frame $\times$ RAD/MHD-iterated), each using a one-sided finite-difference
Jacobian, a backtracking damping ladder, and an optional 1-D bisection warm start.
The residual enforces conservation, e.g. for the radiation-iterated energy equation
$$f_0 = R^t{}_t - R^t{}_t{}^{(0)} + dt\,\sqrt{-g}\,G_0 = 0,$$
with the opposite sign for the gas — the $G$ sign convention of §1. The
conservation glue `applyConstraints` reconstructs the non-iterated fluid from
total energy-momentum conservation before every residual/Jacobian evaluation.

Two extensions to the ladder: the FD Jacobian's four perturbed residuals can be
evaluated in one `@Vector(4, f64)` batch (`ImplicitParams.simd_jacobian`,
default on, bit-identical to scalar); and the whole 6-rung ladder can retry
inside an outer **opacity-damping loop** (C `OPDAMPINIMPLICIT`; params keys
`opdamp_maxlevels`/`opdamp_factor`, default off / AGN preset 3 levels ×10):
level $\ell$ scales the four-force by $\mathrm{opdamp} = f^{-\ell}$ (§5.2),
accepting a weaker coupling rather than a fixup when the fully-coupled solve
fails. Level 0 is bit-identical to the plain single pass.

**CFL timestep** (`saveWavespeeds`):
$$\frac{1}{dt} = \frac{1}{\mathrm{TSTEPLIM}}\left(\frac{w_x}{\Delta x} + \frac{w_y}{\Delta y} + \frac{w_z}{\Delta z}\right),$$
with $w_d$ the maximum wave-speed magnitude (radiation uses the *unlimited*
speeds here), and PUFFY $\mathrm{TSTEPLIM} = 0.5$.

---

## 9. Floors and ceilings

Floors keep the inversion well-posed and the evolution physical; they are applied
after every inversion ([`koral/solve/invert.zig`](../koral/solve/invert.zig),
`checkFloorsMhd`; [`koral/solve/invert_rad.zig`](../koral/solve/invert_rad.zig),
`checkFloorsRad`). PUFFY thresholds (`FloorParams.puffy`, `RadParams.puffy`):

**Density and internal energy:**
$$\rho \ge \rho_{\rm floor}\ (10^{-30}),\qquad
u_{\rho,\min}\,\rho \le u \le u_{\rho,\max}\,\rho\ (10^{-8},\,1).$$
These bound the coldest/hottest gas so the EOS and entropy stay finite.

**Magnetization ceilings:** in the default **drift frame** (Ressler et al. 2017,
`FloorParams.b2rhofloorframe = .driftframe`), if $b^2 > b^2_{\rho,\max}\rho$ or
$b^2 > b^2_{u,\max}u$ (both $= 50$ for PUFFY), mass and internal energy are
*added in the drift frame* so total momentum is preserved, capping the
force-free-like magnetization. This is the numerical floor that limits how
force-free the jet funnel can get. The alternative **ZAMO frame**
(`.zamoframe`, params key `zamo_floor_frame`, used by the AGN preset) triggers on
the $b^2/\rho$ ceiling only (the $b^2/u$ ceiling is disabled): the injected mass
$d\rho = \rho(f-1)$ is given the normal-observer 4-velocity
$\eta^\mu = -\alpha g^{\mu 0}$, converted to a conserved delta by `p2uMhd`,
added to the pre-floor conserveds, and re-inverted (hot, then entropy). It is
isentropic ($dU = 0$; C `ISENTROPIC_B2RHOFLOORS`), with a fluid-frame
$\rho \to f\rho,\ u \to fu$ fallback if the re-inversion fails
(`B2RHOFLOOR_BACKUP_FFFRAME`).

**Velocity ceiling:** the Lorentz factor is capped by rescaling the relative
velocity so $\gamma \le \gamma_{\max}^{\rm hd}$ ($=10$ for PUFFY):
$$\gamma^2 = 1 + \tilde u^i \tilde u^j g_{ij} \le (\gamma_{\max}^{\rm hd})^2.$$

**Radiation floors** (`checkFloorsRad`): $\hat E \ge E_{\rm rad,floor}$
($=10\times10^{-80}$), plus ceilings/floors on $\hat E_{\rm ff}/\rho$
($10^{-20}$–$10^{4}$) and $\hat E_{\rm ff}/u$ ($10^{-20}$–$10^{20}$) that rescale
$\hat E$, $\rho$, or $u$. After any floor trips, the entropy primitive is
re-synced from $(\rho,u)$. The radiation Lorentz-factor ceiling
$\gamma_{\max}^{\rm rad} = 10$ is *not* applied here — it is enforced inside the
closed-form M1 inversion `u2pRad` (the non-failure test and the two-branch cold
fallback in [`koral/solve/invert_rad.zig`](../koral/solve/invert_rad.zig)).

**Cell fixups** are the second "keep it physical" mechanism: cells whose
inversion or implicit solve failed are flagged (`hd_fixup`, `rad_fixup`, and —
off in the validated build, on in the AGN preset via `doradimpfixups` — the
post-implicit `radimp_fixup`) and replaced by the arithmetic mean of their
non-flagged face neighbours (≥1/2/3 valid neighbours required in 1D/2D/3D),
followed by `p2u`. Fixups never touch $\rho$ or $B$: `hd_fixup` averages $u$ and
the velocities, `rad_fixup` only the radiation slots, `radimp_fixup` both fluids.
Polar-axis-corrected rows are skipped.

All of these thresholds are compiled presets (`FloorParams.puffy`,
`RadParams.puffy`) but most are runtime-overridable from the params file
(`rhofloor`, `uurhoratio*`, `eerhoratio*`, `eeuuratio*`, `b2*ratiomax`,
`gammamaxhd`/`gammamaxrad`, `zamo_floor_frame`, …) — the AGN and Sgr A* presets
use this instead of a recompile.

---

## 10. The PUFFY problem

The PUFFY initial condition and driver live in
[`koral/problems/puffy/puffy.zig`](../koral/problems/puffy/puffy.zig) and
[`koral/problems/puffy/main.zig`](../koral/problems/puffy/main.zig); the
arbitrary-precision enthalpy integral uses
[`koral/math/quad.zig`](../koral/math/quad.zig). It is a Penna-style
**limotorus** — a relativistic, radiation-supported, geometrically thick
equilibrium torus. The reference (golden-validated) configuration is a
$10\,M_\odot$ Schwarzschild ($a=0$) hole in MKS2 with `MKSR0 = 0.1`,
`MKSH0 = 0.9`, on a $384\times360\times1$ grid over $r\in[1.85, 500]$. The mass,
spin, grid, and domain are runtime-retargetable: `rminForSpin(a) =
0.925\,r_h(a)` keeps the inner edge tracking the Kerr horizon (exactly 1.85 at
$a=0$), 3D runs subdivide a fixed $\varphi$-wedge `PHIWEDGE` $= \pi/2$ with
periodic $\varphi$, and the shipped presets cover 3D (`puffy3d.toml`), Sgr A*
(`puffy3d_sgra.toml`, `puffy3d_sgra_spin.toml` at $a = 0.9375$), and a $10^9\,
M_\odot$ AGN (`puffy_agn.toml`; see `docs/PUFFY_AGN_DIVERGENCES.md` for how that
preset deliberately departs from the C reference run).

### 10.1 Broken-power-law angular momentum

The torus is sub-Keplerian with a broken-power-law specific angular momentum in
the von-Zeipel cylinder radius $\lambda$ (`l3d`, KORAL `tools.c`):
$$\ell_{3d}(\lambda) = \xi\,\ell_K\!\big(\mathrm{clamp}(\lambda, \lambda_1, \lambda_2)\big),$$
with sub-Keplerian factor $\xi = 0.995$ and the equatorial Keplerian value
$$\ell_K(r,a) = \sqrt{r}\,\frac{1 - 2a/r^{3/2} + (a/r)^2}{1 - 2/r + a/r^{3/2}}.$$
The two break radii $r_1 = 20$, $r_2 = 350$ (mapped to $\lambda_1,\lambda_2$)
produce the C$^1$ kinks that make $\ell$ constant beyond the breaks. $\lambda(R)$
is found by bisection on the von-Zeipel relation (`lamBL`), and
$\Omega(\ell) = -(g_{tt}\ell + g_{t\phi})/(g_{\phi\phi} + g_{t\phi}\ell)$
(`omega3d`).

### 10.2 Enthalpy and density

The relativistic-torus enthalpy $f(r)$ follows from integrating the von-Zeipel
integrand (`LnfIntegrand`, quadrature `quad.integrate` over $[r_{\rm in},
r_{\rm ml}]$):
$$\frac{d\ln f}{dr} = -\frac{\ell}{1 - \Omega\ell}\,\frac{d\Omega}{dr}.$$
With the gravitational factor $A_{\rm grav} = |u_t|$ of the circular orbit
(`computeAgrav`), the enthalpy ratio and density are (`initDsandvels`):
$$h = \frac{f_{\rm in}A_{\rm grav}}{f\,A_{\rm grav,in}},\qquad
\varepsilon = \frac{h-1}{\Gamma_t},\qquad
\rho = \left[\frac{(\Gamma_t-1)\varepsilon}{\kappa}\right]^{1/(\Gamma_t-1)},\qquad
u = \frac{\kappa\,\rho^{\Gamma_t}}{\Gamma_t - 1},$$
using the **torus** polytropic index $\Gamma_t = 4/3$ (distinct from the gas
$\Gamma = 5/3$ used everywhere else) and entropy constant $\kappa = 60$
(runtime-overridable as `lt_kappa`; the AGN preset uses $8\times10^{-2}$). Cells
with $R = r\sin\theta < r_{\rm in} = 35$, or $\varepsilon < 0$, or a non-convergent
integral, are flagged "outside" ($\rho = -1$) and filled with a Bondi-like
atmosphere.

### 10.3 LTE gas + radiation pressure split

Inside the torus the total pressure $P = (\Gamma-1)u$ is split into ideal-gas and
radiation contributions assuming local thermodynamic equilibrium (`prepInitCell`,
`tFromPtot`):
$$P = \underbrace{\frac{k_B\rho}{\mu_{\rm gas}m_p}\,T}_{\text{gas}} + \underbrace{\frac{4\sigma_{\rm rad}}{3}\,T^4}_{\text{radiation}},$$
solved for the temperature $T$ (positive real root of the quartic). The internal
energy is then recomputed from $T$ (`thermo.uFromTrho`) and the radiation energy
$\hat E = 4\sigma_{\rm rad}T^4$; the fluid-frame radiation state is lifted to the
lab frame (`radiation.pradFf2Lab`) with $u_r^\mu = u^\mu$ (radiation comoving with
the gas, zero flux). All primitives are then transformed BL → MKS2
(`frames.transPallCoco`). This is what makes the torus **radiation-supported**:
a large fraction of the pressure that holds it up geometrically thick is
$P_{\rm rad}$.

### 10.4 Initial poloidal field and $\beta$-normalization

A poloidal seed field is set from a "quad-loops" azimuthal vector potential
(`prepInitCell`):
$$A_\phi = \max\!\big(\text{base}^2 - 0.02,\, 0\big)\,\sqrt{10^{-23}}\;\sin\!\frac{\pi/2 - \theta}{0.1},\qquad
\text{base} = \frac{\rho\, r_{\rm BL}}{4\times10^{-22}},$$
stored in the $B_3$ slot, then curled into a divergence-free $B$ by `calcBfromA`
(§8.4). Finally the whole field is rescaled to a target plasma-$\beta$
(`postinit`, KORAL `BETANORMFULL`):
$$f = \sqrt{\frac{\beta_{\max}}{\max_{\rm domain}(p_{\rm mag}/p_{\rm tot})}},\qquad
B^i \to f\,B^i,\qquad \beta_{\max} = \frac{1}{20},$$
with $p_{\rm mag} = b^2/2$, $p_{\rm tot} = (\Gamma-1)u + \hat E_{\rm ff}/3$, and
$\beta_{\max}$ runtime-overridable as `maxbeta` (AGN preset: $1/30$). So the
initial field is weak; the $\alpha$–$\Omega$ dynamo (§7) then sustains it.
`maxPmagPtot` ([`koral/io/scalars.zig`](../koral/io/scalars.zig)) reports
whether that magnetization is held over the run.

### 10.5 Boundaries, driver, diagnostics

The radial-outer boundary is outflow with $r$-rescaling and a no-inflow clamp; the
radial-inner boundary is a plain copy (the inner edge $r_{\rm in,grid} =
0.925\,r_h$ is inside the horizon, computed per run from the spin); the
$\theta$ boundaries are polar reflection
with sign flips of $v^\theta, B^\theta, F^\theta$ (`Bc.calc`). The driver runs the
RK2IMEX loop with the CFL $dt = 1/\mathrm{tstepdenmax}$, emitting `scalars.dat`
diagnostics — total mass, accretion rate $\dot M$ at the horizon shell,
radiative/total luminosity at the outer shell, disk scale height, maximum
magnetization — and periodic binary primitive dumps
([`koral/io/scalars.zig`](../koral/io/scalars.zig),
[`koral/io/dump.zig`](../koral/io/dump.zig)). Key diagnostic definitions:
$$M = \int\rho\sqrt{-g}\,dx\,dy\,dz,\qquad
\dot M = -\oint \rho\, u^r\sqrt{-g}\,d\theta\,d\phi,\qquad
L = \oint (-R^r{}_t)\sqrt{-g}\,d\theta\,d\phi.$$

---

## 11. Appendix: constants and unit conversions

### 11.1 Physical constants (CGS literals)

Exact literals from KORAL's `ko.h`, kept verbatim for bit-comparability
([`koral/units.zig`](../koral/units.zig),
[`koral/physics/thermo.zig`](../koral/physics/thermo.zig)):

| symbol | value | meaning |
|--------|-------|---------|
| $G$ | $6.674\times10^{-8}$ | gravitational constant |
| $c$ | $2.9979246\times10^{10}$ | speed of light (cm/s) |
| MSUNCM | $147700.0$ | $GM_\odot/c^2$ in cm |
| $k_B$ | $1.3806488\times10^{-16}$ | Boltzmann (erg/K) |
| $m_e$ | $9.1094\times10^{-28}$ | electron mass (g) |
| $m_p$ | $1.67262158\times10^{-24}$ | proton mass (g) |
| $\sigma_{\rm rad}$ | $5.670367\times10^{-5}$ | Stefan–Boltzmann |
| $h$ | $6.6260755\times10^{-27}$ | Planck (erg·s) |
| $\pi_c$ | $3.141592654$ | truncated $\pi$ (MKS2 metric only) |

### 11.2 Geometrized-unit scaling

The length unit is $GM/c^2$; with `MASSCM = MASS × MSUNCM` (PUFFY: $10\times147700
= 1.477\times10^6$ cm). Representative conversions (`units.zig`):
$$\text{length: } x/\mathrm{masscm},\qquad
\text{time: } x\,c/\mathrm{masscm}\ (GM/c^3),\qquad
\text{velocity: } x/c,$$
$$\text{density: } x\,\frac{G\,\mathrm{masscm}^2}{c^2}\ (\propto M^{-2}),\qquad
\text{energy density: } x\,\frac{G\,\mathrm{masscm}^2}{c^4}\ (\propto M^{-2}).$$
The Boltzmann constant and Stefan–Boltzmann constant in GU are
$k_B^{\rm GU} = k_{B,\rm CGS}\,G/c^4/\mathrm{masscm}$ and
$\sigma_{\rm rad}^{\rm GU} = \sigma_{\rm rad,CGS}\,G/c^5\,\mathrm{masscm}^2$, with
$a = 4\sigma_{\rm rad}$. Because $a_{\rm code}\propto M^2$ exactly, the radiative
shock-tube tests pick the BH mass to match a paper's radiation constant
([`koral/testing/tubes.zig`](../koral/testing/tubes.zig), `massForA`).

### 11.3 Where to look next

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — module graph, comptime configuration,
  storage layout, the C-oracle test strategy.
- [`USER_GUIDE.md`](USER_GUIDE.md) — building, configuring a run, parameters.
- The subsystem source files referenced throughout this document are the
  authoritative statement of each equation; they carry per-function citations to
  the KORAL C source they transcribe.
