# PUFFY AGN preset — divergences from `koral_lite_puffy`

`koral/problems/puffy/puffy_agn.toml` retargets koral-zig to the configuration
currently committed in the sibling C tree
`koral_lite_puffy/PROBLEMS/PUFFY/define.h` (PROBLEM 147): a **10⁹ M☉ AGN torus**
with metallicity composition, MESA opacities, wider floors, a different floor
frame, and different coordinates.

This is a **different, newer configuration** than the one koral-zig is validated
against. koral-zig's bit-for-bit goldens track
`koral_lite/PROBLEMS/PUFFY/define.h` — the 10 M☉ radiation-supported torus of
Lančová et al. (2019). The AGN preset is applied purely through the runtime
override fields in `koral/params.zig`, so **the validated `.puffy` constants and
every golden are untouched** (`zig build test` is unaffected).

> **The AGN run is NOT bit-comparable to `koral_lite_puffy`.** Several C settings
> are physics features that are not ported to koral-zig; no toml value can
> reproduce them. This document is the honest ledger of what matches and what
> does not.

Comparison basis: `diff koral_lite/PROBLEMS/PUFFY/define.h
koral_lite_puffy/PROBLEMS/PUFFY/define.h`.

---

## 1. Matched by the preset (runtime overrides)

These map cleanly onto koral-zig knobs and are set in `puffy_agn.toml`:

| C `#define` | validated | AGN | koral-zig knob |
|---|---|---|---|
| MASS | 10 | 1e9 | `mass` |
| TNY | 360 | 384 | `ny` |
| MKSR0 / MKSH0 | 0.1 / 0.9 | −1.5 / 0.875 | `mksr0` / `mksh0` |
| RHOFLOOR | 1e-30 | 1e-40 | `rhofloor` |
| UURHORATIOMIN / MAX | 1e-8 / 1e0 | 1e-10 / 1e2 | `uurhoratiomin` / `uurhoratiomax` |
| EERHORATIOMAX | 1e4 | 1e20 | `eerhoratiomax` |
| B2UURATIOMAX | 50 | 1e3 | `b2uuratiomax` |
| RADIMPEPS | 1e-6 | 1e-5 | `radimpeps` |
| RADIMPMAXITER | 40 | 50 | `radimpmaxiter` |
| BREMSSTRAHLUNG | on | **off** | `bremsstrahlung` |
| KLEINNISHINA | on | **off** | `kleinnishina` |
| MU_* → HFRAC/HEFRAC/MFRAC | 1/2/2 | .70/.28/.02 | `hfrac` / `hefrac` |
| LT_KAPPA | 6e1 | 8e-2 | `lt_kappa` |
| MAXBETA | 1/20 | 1/30 | `maxbeta` |
| RHOATMMIN | 1e-24 | 1e-20 | `rhoatmmin` |
| TGASATMMIN | 1e10 | 1e4 | `atm_tgas` |
| ATMTRADINIT + ERADATMMIN form | 3e5, Brandon | 3e4, (1e-8)·LTE | `atm_trad_init` / `atm_erad_factor` |

Already equal in both configs (no override needed): BHSPIN 0, GAMMA 5/3, TNX 384,
TNZ 1, TSTEPLIM 0.5, DTOUT1 100, DTOUT2 500, RMIN 1.85, RMAX 500, PHIWEDGE π/2,
B2RHORATIOMAX 50, GAMMAMAXHD/RAD 10, EERHORATIOMIN 1e-20, EEUURATIO 1e-20/1e20,
RADIMPCONV 1e-10, RADIMPCONVREL 1e-8, RADIMPLICITDAMPINGFACTOR 3,
MAXRADIMPDAMPING 1e-3, RADIMP_START_WITH_BISECT, SCALE_JACOBIAN,
ALLOWRADCEILINGINIMPLICIT, ALLOWFORENTRINF4DPRIM, RADVISCOSITY (ALPHARADVISC 0.1,
MAXRADVISCVEL 0.1), MIMICDYNAMO (THETAANGLE 0.25, ALPHADYNAMO, BETASATURATED 0.1,
ALPHABETA), COMPTONIZATION, SYNCHROTRON, INT_ORDER 2 (PPM), CORRECT_POLARAXIS,
NCCORRECTPOLAR 2, SPECIFIC_BC.

---

## 2. NOT ported — cannot be matched (physics features)

The AGN run **diverges** from `koral_lite_puffy` on all of the following. None is
a knob; each needs new koral-zig code to close.

| C `#define` | what it does | koral-zig status | impact |
|---|---|---|---|
| **`MESA_KAPPA`** | absorption opacity from MESA tables (with brems OFF, this is *the* absorption source) | **no loader** | **largest.** Absorption is synchrotron-only here — MESA free-free / bound-free / line opacity is absent, changing the radiation–matter coupling. |
| `USE_SYNCHROTRON_BRIDGE_FUNCTIONS` | Ramesh NR bridge added to synchrotron opacity | bridge blocks off (vestigial) | synchrotron opacity lacks the non-relativistic bridge component. |
| `B2RHOFLOORFRAME ZAMOFRAME` (+ `B2RHOFLOOR_BACKUP_FFFRAME`, `ISENTROPIC_B2RHOFLOORS`) | magnetic floor applied in the ZAMO frame, isentropic | koral-zig floors are **DRIFTFRAME** | the b²/ρ floor injects mass/energy in a different frame — differs where the field is strong (funnel/corona). |
| `MAXDIFFTRADS 1e4` / `MAXDIFFTRADSNEARBH 1e2` | clamp on Trad/Te difference (Comptonization limiter) | absent | Compton term unclamped; can differ in the hot, scattering-dominated interior. |
| `OPDAMPINIMPLICIT 1` (`OPDAMPMAXLEVELS 3`, `OPDAMPFACTOR 10`) | opacity-damping ladder inside the implicit solver | absent (`OPDAMPINIMPLICIT 0` path only) | stiff cells that C rescues by damping opacity may instead fall through koral-zig's damping ladder. |
| `UUEERATIOMAX 10` | ceiling on u_int / Ê ("remove extra uu from ehat") | absent | no such ceiling is applied. |
| `DAMPRADWAVESPEEDNEARAXIS` (`…NCELLS 2`) | damp radiative wavespeed within 2 cells of the pole | absent | near-axis rad CFL/wavespeed not damped → different dt / axis behavior. |
| `DORADIMPFIXUPS 1` | post-implicit radiation fixup pass | pass exists but **disabled** | failed-implicit cells are not fixed up as in C. |
| `REDUCEORDERATBH` | reduce reconstruction order inside the horizon | absent | full-order reconstruction inside r_h (excised region; effect limited). |

---

## 3. No effect / irrelevant to this run

- **`u2pconv` 1e-10 → 1e-8** — the C run loosens the u2p Newton tolerance.
  koral-zig hardwires `u2pconv = 1e-10` (`solve/invert.zig`) and it is not
  threaded through a runtime knob, so the AGN run uses the *tighter* 1e-10. This
  is a convergence tolerance, not a physical parameter; the effect is at the
  inversion round-off level. (Plumb it into `FloorParams` if you want an exact
  match.)
- **`EXPECTEDHR` 0.3 → 0.7** — only used when `CALCHRONTHEGO` is *off*. PUFFY
  sets `CALCHRONTHEGO`, so `EXPECTEDHR` is dead in both configs.
- **`POLARAXISAVGIN3D`** — 3D-only; this preset is 2D (`nz = 1`).
- **`EVOLVEPHOTONNUMBER` / `NPH_*` / `RADIMPLICITMAXNPHCHANGE`** — photon-number
  evolution is off.
- **`RESTARTNUM` / `RESTORETORUS` / `PERTURB*`** — restart bookkeeping;
  koral-zig restarts via `--restart <file|dir>`.
- **`NTX` / `NTY` tiling** — MPI decomposition; koral-zig is single-node
  (shared-memory `nthreads`).
- **`SILOOUTPUT` / `RADOUTPUT` / `SCAOUTPUT` / `NOUTSTOP` / `PRINT_*`** — C
  output-format flags; koral-zig writes `scalars.dat`, KDMP, and optional
  `.silo` (`-Dsilo`).
- **`B2UURATIOMIN` / `B2RHORATIOMIN` 0** — "not implemented anyway" in C.
- **`FLUXLIMITER` 0 → 1** — only affects the linear (INT_ORDER 1) reconstruction
  path; both configs use PPM (INT_ORDER 2), which ignores it.

---

## Closing the gap

To make the AGN run bit-comparable to `koral_lite_puffy` you would port §2, in
rough priority: **MESA opacities** first (it defines the absorption physics),
then the ZAMO floor frame, then the implicit-solver refinements (opacity
damping, MAXDIFFTRADS, radimp fixups), then the smaller limiters. Until then,
treat this preset as *"the koral_lite_puffy setup as closely as koral-zig can
currently express it"* — a useful AGN-scale experiment, not a validated
reproduction.
