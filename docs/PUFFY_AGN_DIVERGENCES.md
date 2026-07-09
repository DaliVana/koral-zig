# PUFFY AGN preset — divergences from `koral_lite_puffy`

`koral/problems/puffy/puffy_agn.toml` retargets koral-zig to the configuration
committed in the sibling C tree `koral_lite_puffy/PROBLEMS/PUFFY/define.h`
(PROBLEM 147): a **10⁹ M☉ AGN torus** with metallicity composition, MESA
opacities, wider floors, a ZAMO floor frame, and different coordinates.

This is a **different, newer configuration** than the one koral-zig is validated
against. koral-zig's bit-for-bit goldens track `koral_lite/PROBLEMS/PUFFY/define.h`
— the 10 M☉ radiation-supported torus of Lančová et al. (2019). The AGN preset is
applied purely through the runtime override fields in `koral/params.zig`, so **the
validated `.puffy` constants and every golden are untouched** (`zig build test` is
unaffected — verified after this work).

> **History note (2026-07-09).** An earlier version of this document listed eight
> "unportable" §2 divergences. A verbatim re-reading of the C source
> (`opacities.c`, `u2p.c`, `rad.c`, `finite.c`, `problem.h`) overturned several of
> those claims and surfaced one the document had missed. The physics features are
> now **ported and wired through the preset**; the corrected ledger is below.

> **The AGN run is still NOT bit-comparable to `koral_lite_puffy`** — see §4 — but
> it now matches the C *physics*, not just "as closely as a few knobs allow."

Comparison basis: `koral_lite_puffy/PROBLEMS/PUFFY/{define.h,problem.h}`,
`opacities.c`, `u2p.c`, `rad.c`, `finite.c`.

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
| B2UURATIOMAX | 50 | 1e3 | `b2uuratiomax` (but see §3 — dead in the ZAMO build) |
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

## 2. Now PORTED (physics features implemented for the preset)

Each of these was "no knob, needs new code" in the old ledger. All are now
implemented, default-off (so the validated goldens are untouched), and turned on
by `puffy_agn.toml`.

| C `#define` | what it does | koral-zig implementation | knob |
|---|---|---|---|
| **`MESA_KAPPA`** | MESA Rosseland opacity table | new `physics/mesa.zig` (loader + logT/logR bilinear lookup, transcribed from `read_MESA_table`/`return_MESA_kappa`); the **free-free Rosseland** channel becomes `κ_MESA − κ_es`, floored at 0. **Correction:** MESA feeds only the *Rosseland* channels, NOT absorption — the free-free *Planck* absorption is the bremsstrahlung formula, computed regardless of the (commented-out) `BREMSSTRAHLUNG` macro, exactly as C's MESA branch does. So absorption = FF-Planck + synchrotron. | `mesa_table` |
| **scattering = 0** | (not a `#define` — `PR_KAPPAES` is *undefined* for PROBLEM 147) | **The old ledger missed this.** `calc_kappaes` ≡ 0 in koral_lite_puffy, so the scattering opacity AND the Comptonization four-force term (∝ κ_es) both vanish. koral-zig's AGN preset now sets `kappaes = .none`. | `scattering = false` |
| `USE_SYNCHROTRON_BRIDGE_FUNCTIONS` | Ramesh NR bridge | in `opacities.zig`: clamp Trad→Trad_lim=TradBB^(4/3)/Te^(1/3); add the NR component (∝ B²/T³) to the gas/rad synchrotron absorption; multiply the number opacity by the Te/Tc_n crossover; drop the Terelfactor suppression | `synchrotron_bridge` |
| `B2RHOFLOORFRAME ZAMOFRAME` (+ `ISENTROPIC_B2RHOFLOORS`, `B2RHOFLOOR_BACKUP_FFFRAME`) | b²/ρ floor mass injected in the ZAMO frame | in `invert.zig checkFloorsMhd`: build the normal observer n^μ=−α g^{μ0}, inject Δρ as a delta-conserved added to the entry conserveds (isentropic ⇒ ΔU=0), recover prims; fluid-frame backup on u2p failure. The b²/uu (CASE 2) floor is commented out in C's ZAMO build, so it is skipped here too. | `zamo_floor_frame` |
| `OPDAMPINIMPLICIT` (`OPDAMPMAXLEVELS 3`, `OPDAMPFACTOR 10`) | opacity-damping ladder | in `implicit.zig solveImplicitLab`: wrap the 6-rung ladder in an outer level loop; on total failure retry with the four-force scaled by `factor⁻ˡᵉᵛᵉˡ` (via `radforce.Params.opdamp`) | `opdamp_maxlevels` / `opdamp_factor` |
| `DAMPRADWAVESPEEDNEARAXIS` (`…NCELLS 2`) | damp rad wavespeed near the pole | in `sim.zig wavespeedRows`: within N cells of either pole, zero τ so the rad wavespeed keeps its undamped 1/3 (equivalent to C's override) | `dampradwavespeednearaxis` |
| `DORADIMPFIXUPS 1` | post-implicit radiation fixup | the neighbour-averaging pass already matched C; now enabled | `doradimpfixups` |
| `REDUCEORDERATBH` | drop reconstruction order inside r_h | in `sim.zig`: a once-computed radial threshold (r_BL(cell) < r_h) drives `reconstr_par = 1` → PPM→linear there | `reduceorderatbh` |

---

## 3. Non-divergences / no effect (do NOT implement)

- **`MAXDIFFTRADS` 1e4 / `MAXDIFFTRADSNEARBH` 1e2** — all three C sites are inside
  `#ifdef EVOLVEPHOTONNUMBER`, which is **off** for PUFFY (`define.h:26`
  commented). The clamp is compiled out; with Trad ≡ TradBB it would be a no-op
  anyway. Implementing a clamp would *diverge* from C, not converge. Not ported.
- **`UUEERATIOMAX` 10** — a **dead macro**: `#define`d in `define.h:165` but
  referenced **nowhere** in the C tree. No clamp exists in C. Not ported.
- **`u2pconv` 1e-10 → 1e-8** — the C run loosens the u2p Newton tolerance;
  koral-zig hardwires the *tighter* 1e-10 (`solve/invert.zig`). Convergence
  tolerance, not physics — effect at inversion round-off level. Still a genuine
  (tiny) divergence; see §4.
- **`EXPECTEDHR` 0.3 → 0.7** — only used when `CALCHRONTHEGO` is off; PUFFY sets
  it, so `EXPECTEDHR` is dead in both configs.
- **`POLARAXISAVGIN3D`** — 3D-only; this preset is 2D (`nz = 1`).
- **`EVOLVEPHOTONNUMBER` / `NPH_*` / `RADIMPLICITMAXNPHCHANGE`** — photon-number
  evolution is off (`define.h:26`). koral-zig has no NF variable either.
- **`RESTARTNUM` / `RESTORETORUS` / `PERTURB*`** — restart bookkeeping; koral-zig
  restarts via `--restart <file|dir>`.
- **`NTX` / `NTY` tiling** — MPI decomposition; koral-zig is single-node.
- **`SILOOUTPUT` / `RADOUTPUT` / `SCAOUTPUT` / `NOUTSTOP` / `PRINT_*`** — C output
  flags; koral-zig writes `scalars.dat`, KDMP, and optional `.silo` (`-Dsilo`).
- **`B2UURATIOMIN` / `B2RHORATIOMIN` 0** — "not implemented anyway" in C. And the
  b²/uu (`B2UURATIOMAX`) *ceiling* is commented out in the ZAMO build, so
  `b2uuratiomax` in the toml is set for the record but is dead (§2, ZAMO row).
- **`FLUXLIMITER` 1** — only affects INT_ORDER 1 reconstruction; both configs use
  PPM (INT_ORDER 2), which ignores it.

---

## 4. Remaining true divergences (why it is still not bit-comparable)

The ports match the C *physics*, but a bit-for-bit reproduction would still differ
on:

- **libm ULP** — the MESA lookup takes `log10`, and the opacity/bridge use
  `pow`/`cbrt`/`exp`. Zig's LLVM intrinsics vs the C libm differ at the
  sub-ULP level; over many steps this diffuses. (The MESA *table interpolation* is
  bit-faithful — same bilinear formula on the same values.)
- **`u2pconv` 1e-10 vs 1e-8** — koral-zig uses the tighter tolerance (§3). Plumb it
  into `FloorParams`/the solver if an exact match is wanted.
- **`REDUCEMINMODTHETA 1`** — koral_lite_puffy also reduces the minmod-θ limiter
  near the axis (`define.h:104`). Discovered during this work; NOT ported (a minor
  near-axis reconstruction tweak, only bites where order is already reduced).
- **ZAMO entry-conserveds quirk** — C's `check_floors_mhd` adds the ZAMO delta to
  the conserveds computed at function *entry* (before the scalar ρ/u floors);
  koral-zig reproduces this, but it is a subtle C behavior worth flagging.

Treat this preset as **"koral_lite_puffy's physics as koral-zig now expresses
it"** — a faithful AGN-scale reproduction of the C *model*, not a byte-identical
oracle. `zig build test` (the 10 M☉ validated goldens) is unaffected: every new
feature is default-off and gated behind the preset's knobs.

---

## Smoke check

A tiny 60×56, 8-step run of the preset (scratch config) exercises every new path —
MESA lookup, ZAMO floor, synchrotron bridge, opacity-damping ladder,
reduce-order-at-BH, near-axis wavespeed damping, radimp fixups, scattering-off —
with `nan=0`, `radimpfail=0`, clean exit. The full production run
(`./zig-out/bin/puffy koral/problems/puffy/puffy_agn.toml`, 384², ReleaseFast) is a
long AGN-scale experiment; use `nout_step` to force early frames while the disk is
still near t=0.
