# How the PUFFY AGN preset differs from `koral_lite_puffy`

`koral/problems/puffy/puffy_agn.toml` retargets koral-zig to the configuration
committed in the sibling C tree `koral_lite_puffy/PROBLEMS/PUFFY/define.h`
(PROBLEM 147): a **10⁹ M☉ AGN torus** with metallicity composition, MESA
opacities, wider floors, hardened b²/ρ floors, and different coordinates.

This preset does not use the configuration behind koral-zig's golden tests. Those
tests track the 10 M☉ torus in `koral_lite/PROBLEMS/PUFFY/define.h`. The AGN
settings enter through runtime fields in `koral/params.zig`, so they do not change
the `.puffy` defaults or the golden records.

> **History note (2026-07-09).** An earlier version of this document listed eight
> "unportable" §2 divergences. A verbatim re-reading of the C source
> (`opacities.c`, `u2p.c`, `rad.c`, `finite.c`, `problem.h`) overturned several of
> those claims and found one the document had missed. The physics code is
> now **ported and wired through the preset**; the corrected ledger is below.

> **History note (2026-08-12).** Restored, the file was accidentally deleted in
> the 2026-08-09 docs cleanup (commit 099d476) while eight references still
> pointed here, and refreshed against the fork's big 2026-08-11 update (HEAD
> `ed6d2f2`): the fork's floor policy is now **drift-frame** (ZAMO retained here
> as an option), `radimp_lag_opac` (C: `copy_state_opac`) and
> `reduceorderafterfixup` are ported, the fork's ENTROPYEQ Sgas fix is adopted
> *unconditionally* (the port's first deliberate divergence from the validated
> baseline), and `puffy_debora.toml` was added as the full fork-mirror preset.

> **The AGN run is not bit-comparable to `koral_lite_puffy`.** See §4. It
> it matches the C *physics*, not just "as closely as a few knobs allow."

Comparison basis: `koral_lite_puffy/PROBLEMS/PUFFY/{define.h,problem.h}`,
`opacities.c`, `u2p.c`, `rad.c`, `finite.c` at fork HEAD `ed6d2f2` (2026-08-11).
The sibling checkout `../koral_lite_puffy` is 3 commits behind (`5905f8a`).

---

## 1. Matched by the preset (runtime overrides)

Scalar knobs that map cleanly onto koral-zig, set in `puffy_agn.toml`:

| C `#define` | validated | AGN | koral-zig knob |
|---|---|---|---|
| MASS | 10 | 1e9 | `mass` |
| TNY | 360 | 384 | `ny` |
| MKSR0 / MKSH0 | 0.1 / 0.9 | −1.5 / 0.875 | `mksr0` / `mksh0` |
| RHOFLOOR | 1e-30 | 1e-40 | `rhofloor` |
| UURHORATIOMIN / MAX | 1e-8 / 1e0 | 1e-10 / 1e2 | `uurhoratiomin` / `uurhoratiomax` |
| EERHORATIOMAX | 1e4 | 1e20 | `eerhoratiomax` |
| B2UURATIOMAX | 50 | 1e3 | `b2uuratiomax` (but see §3, the trigger is disabled) |
| RADIMPEPS | 1e-6 | 1e-5 | `radimpeps` |
| RADIMPMAXITER | 40 | 50 | `radimpmaxiter` |
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
MAXRADVISCVEL 0.1), MIMICDYNAMO, COMPTONIZATION (but κ_es = 0 → term is 0, §2),
SYNCHROTRON, INT_ORDER 2 (PPM), CORRECT_POLARAXIS, NCCORRECTPOLAR 2, SPECIFIC_BC.

---

## 2. Physics implemented for the preset

All are default-off (so the validated goldens are untouched) and turned on by
`puffy_agn.toml`:

| C `#define` | what it does | koral-zig implementation | knob |
|---|---|---|---|
| **`MESA_KAPPA`** | MESA Rosseland opacity table | `physics/mesa.zig` loads the table and performs bilinear lookup in log T and log R. The free-free Rosseland channel becomes `max(κ_MESA − κ_es, 0)`. MESA does not replace Planck absorption; that remains free-free plus synchrotron, matching the C branch. Near log T ≈ 5.2, the analytic Rosseland formula can be about 10 times too transparent. | `mesa_table` |
| **scattering = 0** | `PR_KAPPAES` is undefined for PROBLEM 147 | `calc_kappaes` returns zero, which removes scattering opacity and the Comptonization term proportional to κ_es. | `scattering = false` |
| `USE_SYNCHROTRON_BRIDGE_FUNCTIONS` | Ramesh NR bridge | in `opacities.zig`: clamp Trad→Trad_lim=TradBB^(4/3)/Te^(1/3); add the NR component (∝ B²/T³) to the gas/rad synchrotron absorption; multiply the number opacity by the Te/Tc_n crossover; drop the Terelfactor suppression | `synchrotron_bridge` |
| **b²/ρ floor policy** (`B2RHOFLOORFRAME`, `ISENTROPIC_B2RHOFLOORS`) | frame used when injecting floor mass | The 2026-08-11 fork uses drift-frame, isentropic b²/ρ floors, disables the b²/u trigger, and leaves velocity unchanged inside the horizon. koral-zig mirrors those choices. The older ZAMO path remains available through `zamo_floor_frame`. `checkFloorsMhd` keeps the pre-floor velocity when C would read uninitialized `ucont`; `n_floorguard` counts those recoveries. | `zamo_floor_frame`, `isentropic_b2rhofloors`, `b2uufloor`, `fluid_floor_inside_horizon` |
| `OPDAMPINIMPLICIT` (`OPDAMPMAXLEVELS 3`, `OPDAMPFACTOR 10`) | opacity-damping ladder | in `implicit.zig solveImplicitLab`: wrap the 6-rung ladder in an outer level loop; on total failure retry with the four-force scaled by `factor⁻ˡᵉᵛᵉˡ` (via `radforce.Params.opdamp`) | `opdamp_maxlevels` / `opdamp_factor` |
| **`copy_state_opac`** (fork, 2026-08-11) | opacity lagging in the implicit solver | Freezes opacities at the pre-solve state for the full Newton solve. This keeps the MESA iron bump out of the Jacobian. | `radimp_lag_opac` |
| `DAMPRADWAVESPEEDNEARAXIS` (`…NCELLS 2`) | damp rad wavespeed near the pole | in `sim/timestep.zig wavespeedRows`: within N cells of either pole, zero τ so the rad wavespeed keeps its undamped 1/3 (equivalent to C's override) | `dampradwavespeednearaxis` |
| `DORADIMPFIXUPS 1` | post-implicit radiation fixup | the neighbour-averaging pass already matched C; now enabled | `doradimpfixups` |
| `REDUCEORDERATBH` | drop reconstruction order inside r_h | `Core.init` (`sim/core.zig`) computes the radial threshold once (`reduce_order_ix_max`, r_BL(cell) < r_h) and `sim/explicit.zig` drives PPM→linear there | `reduceorderatbh` |
| **`REDUCEORDERAFTERFIXUP`** (fork, 2026-08-11) | fixup-flagged cells reconstruct one order lower on the next sweep | in `sim/explicit.zig`; flags are consumed from the u2p/implicit pass just before the sweep *within* the step (RK2IMEX runs `calcU2p` before `opExplicit`) | `reduceorderafterfixup` |

**Unconditional ENTROPYEQ fix.** The fork's Sgas fix in `rad.c`
uses `Sgas` where the achael baseline reads `Tgas`) is adopted *unconditionally*
in `solve/implicit.zig`, the port's first deliberate divergence from the
validated-baseline C. Versus `koral_lite_puffy` this is **parity**, not a
divergence; versus the 10 M☉ oracle it flips ~21/336 entropy-rung records,
counted and bounded (≤30) by `implicit_golden_tests`. Energy rungs stay
bit-identical.

---

## 3. Settings with no effect

- **`MAXDIFFTRADS` 1e4 / `MAXDIFFTRADSNEARBH` 1e2.** All three C sites are inside
  `#ifdef EVOLVEPHOTONNUMBER`, which is **off** for PUFFY (`define.h:26`
  commented). The clamp is compiled out; with Trad ≡ TradBB it would be a no-op
  anyway. Implementing a clamp would *diverge* from C, not converge. Not ported.
- **`UUEERATIOMAX` 10** is a dead macro. It is defined in `define.h:165` but
  referenced **nowhere** in the C tree. No clamp exists in C. Not ported.
- **`u2pconv` 1e-10 → 1e-8.** The C run loosens the u2p Newton tolerance;
  koral-zig hardwires the *tighter* 1e-10 (`solve/invert.zig:28`). Convergence
  tolerance, not physics, effect at inversion round-off level. Still a genuine
  (tiny) divergence; see §4.
- **`EXPECTEDHR` 0.3 → 0.7.** This is used only when `CALCHRONTHEGO` is off. PUFFY sets
  it, so `EXPECTEDHR` is dead in both configs (the `expectedhr` params knob
  exists for `puffy_debora.toml`'s record).
- **`POLARAXISAVGIN3D`**, 3D-only; this preset is 2D (`nz = 1`).
- **`EVOLVEPHOTONNUMBER` / `NPH_*` / `RADIMPLICITMAXNPHCHANGE`**, photon-number
  evolution is off (`define.h:26`). koral-zig has no NF variable either.
- **`RESTARTNUM` / `PERTURB*`**, restart bookkeeping; koral-zig restarts via
  `--restart <file|dir>` (`RESTORETORUS` is a real gap, §5).
- **`NTX` / `NTY` tiling**, MPI decomposition (single-node here; the MPI port
  tracks it separately).
- **`SILOOUTPUT` / `RADOUTPUT` / `SCAOUTPUT` / `NOUTSTOP` / `PRINT_*`**, C output
  flags; koral-zig writes `scalars.dat`, KDMP, and optional `.silo` (`-Dsilo`).
- **`B2UURATIOMIN` / `B2RHORATIOMIN` 0**, "not implemented anyway" in C. And the
  b²/uu (`B2UURATIOMAX`) *ceiling* trigger is commented out in the fork's u2p.c,
  so `b2uuratiomax` in the toml is set for the record but is dead (§2, floor row).
- **`FLUXLIMITER` 1.** This affects only INT_ORDER 1 reconstruction; both configs use
  PPM (INT_ORDER 2), which ignores it.

---

## 4. Remaining true divergences (why it is still not bit-comparable)

The ports match the C *physics*, but a bit-for-bit reproduction would still differ
on:

- **libm ULP.** The MESA lookup takes `log10`, and the opacity/bridge use
  `pow`/`cbrt`/`exp`. Zig's LLVM intrinsics vs the C libm differ at the
  sub-ULP level; over many steps this diffuses. (The MESA *table interpolation* is
  bit-faithful, same bilinear formula on the same values.)
- **Exact π.** The port uses one exact `std.math.pi` where C mixes the truncated
  `#define Pi = 3.141592654` and exact `M_PI` (MKS2 metric flavor vs coco point
  transforms; `fourpi` vs `fourmpi` in the emission constants). θ-sensitive MKS2
  quantities differ from C at ~1e-9 (see `PHYSICS.md` §2.4). Unified 2026-07-06.
- **`u2pconv` 1e-10 vs 1e-8.** koral-zig uses the tighter tolerance (§3). Plumb it
  into the solver if an exact match is wanted.
- **`REDUCEMINMODTHETA 1`.** `koral_lite_puffy` reduces the minmod-θ limiter
  near the axis (`define.h:104`). This small reconstruction change is not ported
  and matters only where order is already reduced.
- **Floor guard ladder on unhealthy states.** Where C's floor path reads
  uninitialized `ucont` (γ ≫ GAMMAMAXHD + strong field), koral-zig's bail-outs
  keep the pre-floor velocity instead. C's behavior there is undefined, so
  bit-comparison is meaningless on exactly those cells; healthy states are
  bit-identical (`solve/invert.zig`, gated in `state_tests`).
- **Entry-conserveds quirk.** C's `check_floors_mhd` adds the floor delta to the
  conserveds computed at function *entry* (before the scalar ρ/u floors);
  koral-zig reproduces this, but it is a subtle C behavior worth flagging.

Treat this preset as koral-zig's expression of the `koral_lite_puffy` model. It
is not a byte-identical oracle. `zig build test` for the 10 M☉ goldens is
unaffected: every
preset feature is default-off, and the one unconditional change (ENTROPYEQ Sgas,
§2) is excluded record-by-record in `implicit_golden_tests`.

---

## 5. Still missing vs the fork (HEAD `ed6d2f2`, 2026-08-11)

Gaps broader than this preset needs, catalogued for the full fork-mirror preset
`koral/problems/puffy/puffy_debora.toml`:

- **`RESTORETORUS`** + the per-step problem hook it needs (C's "my_finger"
  per-step density restore toward the initial torus).
- **Live magnetic rad-floor block.** The fork flipped `check_floors_rad`'s
  magnetic block from `SKIP_MAGNFIELD` to `MAGNFIELD`.
- **Midplane-κ text output** (fork diagnostic).
- **INT_ORDER 1.** The koral-zig PUFFY binary uses compile-time PPM.
- **`REDUCEMINMODTHETA`**, **`BALANCEENTROPYWITHRADIATION`** (and `UUEERATIOMAX`,
  dead in C, §3).
- The fork's entropy-u2p Jacobian change `dwmrho0dW` 1/γ² → 1/γ is deliberately
  **not** adopted, the baseline's 1/γ² matches harmpi and the fork change is
  unverified (`solve/invert.zig:13`).

---

## Smoke check

(2026-07-09) A 60×56, eight-step run exercised MESA lookup, the ZAMO floor,
the synchrotron bridge, the opacity-damping ladder,
reduce-order-at-BH, near-axis wavespeed damping, radimp fixups, scattering-off,
with `nan=0`, `radimpfail=0`, clean exit. The 2026-08-12 additions (drift-frame
policy + guard ladder, lagged opacities, after-fixup reduce-order, unconditional
Sgas) landed with all tests and goldens green; `puffy_tests` parses every shipped
preset. The full production run (`./zig-out/bin/puffy
koral/problems/puffy/puffy_agn.toml`, 384², ReleaseFast) is a long AGN-scale
experiment; use `nout_step` to force early frames while the disk is still near
t=0.
