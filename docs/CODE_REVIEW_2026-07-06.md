# koral-zig core review — 2026-07-06 (outstanding, prioritized)

This is the pruned + re-prioritized survivor of the full-codebase review (original: 1 high / 37
medium / 72 low across naming, structure, purity, hot-path/inlining, data-oriented design,
idiomatic Zig, best-practices, test hygiene). Findings fixed in earlier passes were removed; what
remains is the open work, re-ordered by *actionability now* rather than by the original per-dimension
grouping. **Exception:** the just-completed **P1 tier is kept in place with `Fixed` annotations**
(2026-07-08) so the record of what changed stays visible — prune it once it's old news. Line numbers
are from the pre-fix tree and may have drifted under the applied renames/extractions — re-grep the
named symbol before editing.

## Already resolved

- **P1 — Correctness & latent bugs (all 7)** ✅ **2026-07-08** — done and verified (207/216 default,
  215/216 `-Dslow-tests`, goldens byte-identical). Kept **annotated in place** in the P1 tier below
  (each finding has a `Fixed` note) rather than removed, since the tier was just completed; prune when
  it's old news. Covers: gcon col-4 zeroing, driver dt-guard + non-zero NaN exit, radviscosity⇒opac
  reject, `Sim.init` precondition validation, `cflDt()` + seeded-CFL assert, `SpecificBc` widened to
  `relele.Error!`, and the `fFluxPrime` `NanInFlux` guard.

The rest below were removed (fixed in earlier passes):

- **High — params.zig string ownership**: uniform heap-owned string fields + `deinit` + `errdefer`. ✔
- **Hot path (7)**: dynamo/radvisc BL-geometry sidecar; `MetricCache.krBlock` + inline index accessors;
  `fFluxPrime` four-velocity/b-field dedup (`calcTijFromState`); slim implicit-residual state
  (`fillRadStateSlim`/`calcGiFromStateSlim`/`OpacLevel`); `calcChiSlim`; `solve4dPrimCore` state00 hoist;
  driver Debug-build warning. ✔
- **Naming (all 5 medium + ~25 low)**: `maxPmagPtot`, `gamma_adiab` unification, `Acov/Bcov` restore,
  `VelType` merge, `State.gamma_adiab`, `bfield.zig`, `calcKappaFromState`, `calcRadViscCoeff`,
  `opac`/`consts`/`channels`/`units`/`par`, `allocator`, `cellSize`, `Dual3.constant`, `active_dims`,
  `from`/`to`, `dyn_a`, `implicit.Solver`, `comptonGiCoeff`, `pi_truncated`, `serializePrimDump`, etc. ✔
- **Project structure (3 medium + 2 low)**: test-file naming scheme unified; PUFFY `kappaes` default →
  `.none`; `sim/{storage,bc,threading}.zig` extraction; `koral/flux`→`koral/riemann`;
  `rHorizonBL`/`rIscoBL`/`stepFunction` refiled to metric/math. ✔
- **Idiomatic — `geometryBLat`** single BL-geometry helper. ✔
- **Won't-do (decided)**: `fillGeometry`/`fillGeometryFace` → `geometry`/`faceGeometry` rename —
  ~130 call sites / 17 files for a modest gain, verifier-downgraded; left as-is by request.

Severity tag on each item below is the reviewer's original (**high** = latent bug / perf cliff,
**medium** = concrete improvement, **low** = polish); the **P#** is the new priority.

---

# P1 — Correctness & latent bugs ✅ DONE (2026-07-08)

These can silently produce wrong physics, non-determinism, or a crash/UB far from the cause. Do
these first; several are cheap.

**All 7 applied and verified (2026-07-08).** Each carries a `Fixed` note below. Verification:
`zig build test` → 207/216 pass (9 slow-gated skips); `zig build test -Dslow-tests` → 215/216
pass (1 skip) including the 384×360 PUFFY / M11 keystone / M12–M13 goldens — all byte-for-byte
unchanged, since every fix is either golden-safe by construction (gcon col-4 zeroing matches the
on-the-fly `geometryAt` convention; the radvisc capture is bit-identical; `cflDt()` is the same
value; the flux NaN guard never fires on healthy states) or signature/validation-only (SpecificBc
widening, `Sim.init` checks — which reject only configs no valid caller constructs). Two regression
tests added: `metric_tests.zig` now pins `GG[i][4]` on both cell and face paths, and a new
`sim_tests.zig` pins all five `Sim.init` rejections + the clean-config path.

#### `koral/metric/precompute.zig:224` — storeBlocks leaves gcon/gconb slots [0..2][4] uninitialized; garbage copied into every Geometry.GG *(medium)*

MetricCache.init allocates self.gcon (line 147) and self.gconb (line 157) with `a.alloc`
(undefined contents) and only memsets them in the faces=false skip path (lines 158-161) — the
normal fill path never zeroes them. storeBlocks (224-234) writes the 16 metric entries plus
off+3*5+4 (gttpert) but never touches off+0*5+4, off+1*5+4, off+2*5+4 — 3 of the 20 slots per
block stay heap garbage. geometryFromBlocks (341-346) copies all 20 slots into Geometry.GG, so
GG[0][4], GG[1][4], GG[2][4] carry uninitialized memory in every cell/face Geometry built from the
cache — inconsistent with geometryAt (line 103), which sets `GG[i][4] = if (i==3) gttpert else 0`.
No kernel indexes those three slots *today*, but the whole 4×5 GG block is passed by pointer
throughout relele/frames/wavespeeds, so any future loop, struct dump, or Geometry comparison reads
nondeterministic values — breaking the deterministic-run guarantee and defeating msan/valgrind.
This is exactly the partially-filled-undefined-memory bug class the C code suffered from.

**Fix:** In storeBlocks explicitly zero the three unused gcon extras — `for (0..3) |j| dst_gcon[off + j*5 + 4] = 0;`
— matching geometryAt (:103), or `@memset` self.gcon/self.gconb unconditionally right after
allocation in init (cover centers *and* faces). Add a test asserting `fillGeometry(...).GG[i][4] == 0`
for i < 3 (and likewise fillGeometryFace).

**Fixed (2026-07-08):** Added `for (0..3) |i| dst_gcon[off + i*5 + 4] = 0;` to `storeBlocks`
(covers centers and faces in one place, since both go through it). Extended the existing
"fillGeometry/fillGeometryFace agree with direct computation" gate in `metric_tests.zig` to assert
`GG[i][4] == 0` for i < 3 and `GG[3][4] == gttpert` on both the cell and the face path — the prior
loop only checked columns 0..3, so the garbage was invisible. Bit-identical to the on-the-fly
`geometryAt`, so goldens unaffected.

#### `PROBLEMS/puffy/main.zig:176` — NaN abort path can be unreachable after a global blow-up, and the NaN 'abort' exits with success status *(medium)*

Three interacting problems in the driver's failure handling. (1) `step()` resets `tstepdenmax = -1`
(sim.zig:1678) and `saveWavespeeds` only updates it via `if (tsd > self.tstepdenmax)` (sim.zig:904);
NaN compares false, so if a step blows up globally, `tstepdenmax` keeps -1 and the next iteration
computes `dt = 1.0 / -1 = -1.0`. Time marches backwards, `s.t >= next_out` never becomes true, the
NaN check at line 193 never runs, and the run spins doing negative-dt garbage steps until
`nstep_max`, then prints "done". (2) Even when NaN is detected, line 195 does a bare `return;` —
exit code 0, so batch/CI treat a NaN-poisoned run as successful. (3) NaN detection only happens at
output cadence (`dtout1`), so NaNs propagate through many steps before being noticed.

**Fix:** In the driver only (leave the C-mirrored sentinel/update in sim.zig): (a) guard the
timestep before stepping — `if (!(dt > 0) or !std.math.isFinite(dt)) return error.InvalidTimestep;`
after line 177 (catches the all-NaN dt=-1 case and the +inf dt=0 stall); (b) replace the bare
`return;` at 195 with `return error.NanDetected;` (or `std.process.exit(1)`); (c) optionally run the
cheap `s.tstepdenmax > 0` positivity check every step. (Note tstepdenmax can never itself be NaN —
NaN fails the `>` update — so an isFinite check on it would not detect NaN.)

**Fixed (2026-07-08):** Driver at `koral/problems/puffy/main.zig` (the review's `PROBLEMS/` path is
stale). (a) Pre-step guard `if (!(dt > 0) or !std.math.isFinite(dt)) { print…; return error.InvalidTimestep; }`
— `!(dt > 0)` catches both the blow-up (dt = −1) and the diverging-denominator stall (dt = 0); the
isFinite arm is defensive. The dt is computed via the new `s.cflDt()` (finding below). (b) The NaN
output-cadence check now `return error.NanDetected;` (non-zero exit for CI/batch). Every-step
positivity (c) is subsumed by the guard, which runs each iteration.

#### `koral/physics/radvisc.zig:247` — calcRadVisccoeff silently returns nu = 0 when opac is null, diverging from C's SKIPRADSOURCE semantics *(medium)*

`const opp = &(sim.opt.opac orelse return 0);` makes the entire radiative-viscosity machinery a
silent no-op whenever `Options.opac` is null while `opt.radviscosity` is true — the sweep still
calls rijviscFace/addRadViscFlux (sim.zig:1285-1299) but every stored R^i_j_visc is zero and the run
completes with no error/log. Nothing in Sim.init validates the coupling. This is a genuine
behavioral divergence from C: opac==null is the analog of SKIPRADSOURCE (sim.zig:24), and C's
calc_rad_visccoeff under SKIPRADSOURCE sets mfp=1.e50 (rad.c:4516-4518) which clamps to mfplim =
r_BL — i.e. C keeps viscosity *active* (nu = ALPHARADVISC·mfplim·stepfunction), while Zig produces
exactly zero.

**Fix:** Make the semantics explicit — either (a) reject the combination at Sim.init (radviscosity=true
requires opac != null, clear error), or (b) transcribe C's SKIPRADSOURCE branch (mfp = 1e50 before
the mfplim clamp, cite rad.c) so the config keeps viscosity active exactly like C. The silent nu=0
no-op with the sweep still running is the one outcome to eliminate. Independently (idiomatic nit,
formerly its own low finding): replace `&(sim.opt.opac orelse return 0)` with
`const opp = if (sim.opt.opac) |*o| o else return 0;` — the `orelse`-of-optional copies the whole
~300-byte Params to a stack temporary per cell per step and takes its address; the capture form
points at the field (cf. sim.zig:823 `if (self.opt.opac) |*op|`).

**Fixed (2026-07-08):** Took option (a) — `Sim.init` now rejects `radviscosity && opac == null`
(gated on `comptime L.hasVar(.ee)`) with `error.InvalidConfig` (see the Sim.init finding below), so
the silent ν = 0 no-op config can no longer be constructed; the naming already landed as
`calcRadViscCoeff`. Also applied the idiomatic capture: `const opac = if (sim.opt.opac) |*o| o else return 0;`
(copy-free; the `else return 0` stays as a defensive guard for any direct call). All current
radviscosity call sites set opac, so no behavior change; the visc golden is unaffected.

#### `koral/sim.zig:317` — Sim.init validates none of its runtime preconditions *(medium)*

Several assumed invariants are unchecked and each violation fails far from the cause: (a) `Grid.ng`
is never checked against `cfg.ghostCells()` (PPM needs 3, linear 2). With PPM, `sweep()`'s
unconditional i-2 load (sim.zig:1266-1270) drives `Field.cellOffset`'s `@intCast(ix + ngx)` negative
if ng=2 → safety panic in Debug/ReleaseSafe, OOB UB in ReleaseFast. (b) `setBcCell` unwraps
`self.opt.specific_bc.?` (line 616); `bc_x = .specific` with null `specific_bc` panics on first
setBc. (c) `Options.coords` (runtime, drives MetricCache) duplicates `cfg.coords` (comptime, used by
dynamo/scalars/puffy); divergence silently mixes two coordinate systems — wrong physics, no
diagnostic. (d) `correct_polaraxis` with `ny <= 2*nccorrectpolar` makes every row 'corrected' and the
polar overwrite sources lie inside the overwritten band — order-dependent garbage on small test
grids.

**Fix:** Validate at the top of Sim.init, returning `error.InvalidConfig` (not `std.debug.assert`,
so ReleaseFast is covered): (1) `g.ng >= cfg.ghostCells()`; (2) `opt.specific_bc != null` whenever any
of bc_x/bc_y/bc_z is `.specific`; (3) `opt.coords == cfg.coords` — or better, drop `Options.coords`
entirely and pass `cfg.coords` to the MetricCache init (sim.zig:333-334), since every other consumer
already uses `Cfg.coords`; (4) `!opt.correct_polaraxis or nyi() > 2*opt.nccorrectpolar`.

**Fixed (2026-07-08):** Five `error.InvalidConfig` checks at the top of `Sim.init`, before any
allocation: (1) ghost depth `g.ng >= cfg.ghostCells()`; (2) any `.specific` axis requires a non-null
`specific_bc`; (3) `opt.coords == cfg.coords` (kept the equality check rather than dropping the field —
verified all ~30 Sim.init call sites already pass matching coords, so the assert is safe and the
drop's blast radius wasn't worth it); (4) `radviscosity ⇒ opac != null` (gated on `L.hasVar(.ee)`,
folds in the radvisc finding above); (5) `correct_polaraxis ⇒ ny > 2·nccorrectpolar`. New
`sim_tests.zig` pins each rejection and a clean-config success. Errors (not asserts) so ReleaseFast
is covered. Every existing valid config passes untouched (full suite green). Documented in
ARCHITECTURE §5.1 and the USER_GUIDE driver template.

#### `koral/sim.zig:1673` — step() silently produces dt=inf unless initTimestepGuess was called *(medium, purity)*

`self.own_dt = 1.0 / self.tstepdenmax` (and the driver's independent `dt = 1.0 / s.tstepdenmax`).
`tstepdenmax` is 0 after Sim.init and nothing asserts it was seeded: `finishInit()` bundles
setBc+initTimestepGuess, but `puffy.initAll()` does NOT call it — main.zig must remember the extra
`s.initTimestepGuess()` on line 156. Forgetting it (or copying initAll's example into a new driver)
gives dt = inf on step 1 → NaNs across the grid → a wasted run diagnosed only by the n_nan check.
Two sources of truth for the CFL dt (step's own_dt vs the driver's recomputation) can also drift.

**Fix:** (1) At the top of step(), `std.debug.assert(forced_dt != null or self.tstepdenmax > 0);`
converting the silent inf/NaN first step into a named failure. (2) Expose
`pub fn cflDt(self: *const Self) f64 { return 1.0 / self.tstepdenmax; }` and use it in both step()
(1673) and the puffy driver (main.zig:176); and/or fold `initTimestepGuess` into `initAll` so new
drivers cannot forget the seeding step.

**Fixed (2026-07-08):** Both. (1) `step()` opens with `std.debug.assert(forced_dt != null or self.tstepdenmax > 0);`
— the driver always forces dt so it's never tripped there; it guards standalone/test callers of
`step(null)`. (2) Added `pub fn cflDt(self: *const Self) f64` and routed both `step()`'s `own_dt` and
the driver's per-loop dt through it, so the two CFL expressions cannot drift. Left
`initTimestepGuess` as an explicit call (folding it into `initAll` is a larger API change; the assert
already turns the "forgot to seed" mistake into an immediate failure). Documented in ARCHITECTURE §5.3
and the USER_GUIDE API-points list.

#### `koral/problems/puffy.zig:671` — Bc.calc swallows numerical errors with `catch unreachable` (6 sites) — panic in safe builds, UB in ReleaseFast *(low; merges the SpecificBc-error idiomatic finding)*

Lines 671, 696, 700, 705, 712, 717 use `catch unreachable` on `frames.transPmhdCoco` and
`relele.convVels`, which return `relele.Error`. At the PUFFY outer boundary these are arguably
unreachable for finite inputs, but NaN inputs slide through and this Bc is the template future
problems (or an xlo boundary inside the ergoregion, where g00 flips sign and utInUcon's discriminant
CAN go negative) will copy; in ReleaseFast a triggered `unreachable` is UB, not a diagnosable abort.
The only reason for `catch unreachable` is that `SpecificBc` (sim.zig:195) forbids error returns —
even though its sole caller `setBcCell` (sim.zig:616) already returns `Error!void` and uses `try`
right below.

**Fix:** Widen `SpecificBc` to `Error![NV]f64` (sim.zig's `Error` already includes `relele.Error`),
change the one call site to `try self.opt.specific_bc.?(...)`, and replace the six `catch unreachable`
with `try`. puffy.Bc and the other comptime-constant BCs are unaffected. If instead kept as-is, at
minimum add a comment at each site stating why it is unreachable at the outer boundary.

**Fixed (2026-07-08):** Widened `SpecificBc` to `relele.Error![NV]f64` (chose the narrower
`relele.Error` over sim's full `Error` — BCs only produce conversion failures, never OOM, and this
avoids widening `bc.Error`). `setBcCell` now `try`s the callback, and puffy's `Bc.calc` replaces its
six `catch unreachable` with `try`. The three test BCs (`tubeBc`, `BondiCtx.bc`, `radtubeBc`)
updated to the new return type (they do no conversions, so `return pp;` still coerces). puffy/radtube
step goldens unchanged (signature-only). Documented in ARCHITECTURE §9 and the USER_GUIDE BC-callback
section.

#### `koral/physics/flux.zig:34` — C's hard NaN abort on the stress tensor was silently dropped from fFluxPrime *(low, verifier-downgraded)*

C's f_flux_prime (physics.c:1230-1247) checks `isnan(T[ii][jj])` for all 16 components and
hard-exits with the cell coords ('nan in flux_prime', exit(-1)); compiled in for PUFFY. The Zig
fFluxPrime computes calcTij (line 34) and assembles fluxes with no NaN check and no comment marking
the omission — unlike every other documented divergence here. Upstream silent NaN producers are
deliberately preserved (wavespeeds.lrCore, relele.calcAlpgam) but *those drop C's diagnostic print
too*, so a NaN first reaching a flux propagates into the conserved update, where cellFixup neighbor-
averaging can silently 'heal' the cell — the run completes with quietly wrong physics and no trace of
origin. Golden tests only exercise healthy states, so nothing pins this.

**Fix:** Add a NaN check over T (or the assembled ff) — a debug-mode assert or a distinct
`error.NanInFlux` that the sweep's existing `try` (sim.zig:1283,1295) propagates so the driver's
step-failure path reports it with cell context — and note the divergence from C's isnan/exit in the
flux.zig header. Do NOT restore the prints in lrCore or calcAlpgam (those omissions are intentional).

**Fixed (2026-07-08):** `fFluxPrime` checks the assembled flux (`for (ff) |v| if (!isFinite(v)) return error.NanInFlux;`)
— the assembled vector catches the radiative rows too, not just T. Added `NanInFlux` to
`relele.Error` (not a flux-local set): it's the shared evolution error base every stage unions, so it
threads through the sweep, the threading `ChunkResult.err`, and the driver's step-failure path with
no other signature edits — and it sits naturally beside `SpacelikeVelocity`, whose doc already says
"we refuse [to propagate a NaN] instead". Divergence from C's `exit(-1)` noted in the flux.zig
header. Healthy states never trip it, so the f_flux_prime and all step goldens are byte-identical.

---

# P2 — Performance

Threading (`sim.zig:959`, `:1402`) is the highest-leverage item and directly on the active P1 perf
plan (persistent pool / parallel explicit path). The memory-traffic and hot-loop items are all
bit-identical refactors pinned by the existing golden + threading determinism gates.

#### `koral/sim.zig:959` — parallelRows spawns and joins fresh OS threads on every pass (~10/step) with a static row partition *(high)*

Each `parallelRows` (937-972) does `std.Thread.spawn` for up to min(nthreads,64,ny) threads and joins
them. A step executes ~10 such passes (2× opImplicit + up to 8× calcU2p), so at nthreads=16 that is
~160 create/destroy cycles/step (~20-60 µs each) — several ms/step of fixed overhead that grows with
thread count. Separately, the contiguous iy-range split assigns equal row counts, but implicit-solver
cost per cell varies by orders of magnitude and is spatially concentrated (disk body vs polar funnel),
so a static partition lets the slowest chunk gate every pass.

**Fix:** Create a persistent worker pool once in Sim.init (`std.Thread.Pool`, or long-lived threads
on a condvar) and dispatch all ~10 passes to it. (main.zig:9 already documents nthreads as driving
"an optional std.Thread.Pool", and C uses a persistent OpenMP pool.) For opImplicit, replace the
static split with dynamic chunking (atomic row counter, 1-4 rows/grab) so expensive rows load-balance;
the implicit worker cannot error and ChunkResult counters are order-independent integer sums, so the
result stays bit-identical and the threading determinism gate still holds.

#### `koral/sim.zig:1402` — Entire explicit operator is serial; only the u2p/implicit inversions use parallelRows *(high)*

Per step, parallelRows threads 8 passes but the dominant explicit work is single-threaded:
calcWavespeeds (:796), sweep (:1220), fluxesAtFaces (:1318), the metricSource/flux-divergence update
loop (:1418-1444), calcRijViscTotal, and stage arithmetic (:1459-1491). All write disjoint per-cell/
per-face slots exactly like u2pRows, so running them through parallelRows is bit-identical at any
thread count. The only shared state is tstepdenmax/tstepdenmin in saveWavespeeds — accumulate
per-ChunkResult and combine with @max/@min after join (order-independent). With nthreads>1 Amdahl
caps speedup hard while this path stays serial.

**Fix:** Route the explicit path through parallelRows. Two mechanical adaptations: parallelRows hands
workers domain rows [0,ny) but calcWavespeeds/calcRijViscTotal cover a ghost row per side and wide
sweeps use -1..n+1 cross ranges — give parallelRows an optional row-range (or let boundary chunks
absorb ghost rows); and the dim=1 sweep splits along the sweep direction itself (still race-free —
face k gets pb_l from k-1 and pb_r from k, disjoint slots) but needs its own worker shape rather than
the iy-row one. Move tstepdenmax/tstepdenmin into ChunkResult as per-chunk max/min combined after join.

#### `koral/sim.zig:1072` — cellFixup performs four full-grid memcpys even when no cell is flagged *(high, DoD)*

cellFixup unconditionally copies u→u_bak and p→p_bak (1072-1073) and back (1138-1139) before/after
the flagged-cell averaging. With NV=13 each Field is several MB, so ~4×6 MB traffic per invocation;
calcU2p runs cellFixup(.hd_fixup) ~6×/step plus cellFixup(.radimp_fixup) after each opImplicit —
~150-250 MB/step of memcpy. In the common case nothing is flagged and the whole thing is a no-op.

**Fix:** Add a zero-flagged-cells early-out: a quick scan of the flag array before the memcpys, or
track a flagged-cell count while flags are written (u2pRows/implicitRowsWorker already aggregate
per-chunk tallies in ChunkResult, so a count slots in naturally). Removes ~350 MB/step of no-op
traffic at production size while preserving the C cell_fixup structure. (Optionally collect fixed
cells into a small (index,pp,uu) list instead of full-array staging — bit-for-bit equivalent since
scan reads come from live self.p — but the early-out captures nearly all the win.)

#### `koral/sim.zig:1430` — Per-component Field/FaceStore get/set loops recompute the cell offset NV times per cell *(medium)*

Several domain-wide inner loops fetch/store one variable at a time, re-running the full 3D offset
math (casts + assert + 3 mul/add + bounds-checked index) per component: (1) the conserved update in
opExplicit (1430-1441) does 6 FaceStore.get + 1 Field.get + 1 Field.set per iv; (2) stageDeriv
(1459-1474) and stageCombine (1477-1491) do 3 offset computations per iv, 7×/step over the domain;
(3) the flux combine in fluxesAtFaces (1381-1394); (4) fluxCt's EMF loops (ct.zig:60-127) at lower
weight. In ReleaseFast LLVM usually hoists the offset, but in Debug (default `zig build test`) and
ReleaseSafe every call re-derives + re-checks — a large fraction of test wall time.

**Fix:** Load each accessed cell/face into stack `[NV]f64` buffers once per cell via the existing
Field/FaceStore `.load`/`.store` and index the buffers in the iv loop (the convention field.zig's
header already mandates, and sweep/u2pRows/fluxesAtFaces' pb_l/pb_r already use). Do NOT use raw
`data[off..][0..NV]` slices (violates the documented "never raw storage pointers" AoSoA convention).
Per-component arithmetic unchanged → bit-identical. Leave ct.zig's EMF gather as-is (single components
from flPin-varied neighbors; whole-cell loads wouldn't help).

#### `koral/sim.zig:1234` — y- and z-sweeps iterate the strided direction innermost; x is never inner for dim!=0 *(medium, DoD)*

sweep() (1230-1235) and fluxesAtFaces() (1331-1336) nest c1→c0→i (sweep dim innermost). Field storage
is iv-fastest then x, so dim=0 streams contiguously but dim=1's inner loop advances iy, striding
sx()*NV*8 = 40,560 B/iteration on the PUFFY grid — the five-point stencil, pb_l/pb_r/fl_l/fl_r stores,
and scal/flb accesses all take that stride → out of L2, poor prefetch, ~60% line utilization. For 3D
the z-sweep inner loop strides ~15 MB — every stencil load a TLB+DRAM miss, a genuine cliff (latent:
current targets are 2D). Every i-iteration is independent, so interchanging loops is bit-identical.

**Fix:** For dim != 0, interchange so x (the contiguous cross-coordinate) is innermost (dim=1:
c1=iz outer, i=iy middle, c0=ix inner; dim=2 analogous). Stores go to disjoint slots of distinct face
arrays and all per-iteration inputs are read-only → traversal order is bit-identical. dx5/grid.size
become inner-loop invariants; the 5-point stencil becomes five contiguous row streams. Add a comment
that the iterations are order-independent so the order stays free.

#### `koral/sim.zig:1281` — Sweep refills identical face geometry twice per face and recomputes loop-invariant dx5/stencil loads *(low)*

Inside sweep's innermost i-loop: (a) the `dor` branch fills face geometry at i+1 (:1293), byte-identical
to the `dol` fill at the next iteration (:1281) — every interior face geometry is filled twice, and
fluxesAtFaces (:1356) fills it a third time; each fillGeometryFace copies 40 f64 + a divide + sqrt.
(b) dx5 (:1244-1250) depends only on (i,dim) but is recomputed per (c0,c1). (c) the stencil (:1259-1270)
reloads all rows per i although consecutive i share all but one row.

**Fix:** The worthwhile piece is the face geometry: carry the dor-branch Geometry (face i+1) into the
next iteration as the dol-branch geometry (seed at i=-1), halving fillGeometryFace calls in sweep;
bit-identical (pure cache read). dx5 hoist and stencil rolling-window are optional micro-polish
(p.load is a single ~104-byte L1-hot memcpy, marginal). The third fill in fluxesAtFaces is a separate
pass, fine to leave.

#### `koral/physics/flux.zig:35` — Hot functions compute values no caller consumes: full-tensor lowering for one row, dead Qdotnp chain, eager sgas *(medium, DoD)*

Provably-unread work in per-cell hot functions (golden-safe to drop — unconsumed values can't shape
compared outputs): (1) fFluxPrime lowers both full 4×4 tensors (indices2221 = 64 madds each, :35/:88)
but consumes only row idim+1 — 3 of 16 gas entries and 4 of 16 rad entries; row-restricted variants
are bit-identical, ~25-30% of every face-flux call is dead. (2) invert.zig:194-200/:235-238:
Qcovp/Qconp/betasqoalphasq/Dfactor/Qdotnp exist solely to fill cons[6], which neither fU2pHot nor
fU2pEntropy reads (U2P_EQS_NOBLE is hardwired) — ~30 dead flops incl. a division on every u2pSolverW
call. (Sub-parts on the always-boost Gi and eager RadState.sgas were closed by the slim-residual fix.)

**Fix:** (1) Add `indices2221Row(t1, gg, row) -> [4]f64` plus row variants of calcTij/calcRij for
fFluxPrime; keep the full-tensor versions for metricSource/fillRadState/scalars. (2) In u2pSolverW,
guard the Qcovp/... block behind the (absent) U2P_EQS_JON path with a comment (preserving C
diffability) or delete it. Each tensor entry is computed independently → bit-identical.

#### `koral/physics/flux.zig:42` — Hot per-cell chains re-derive bit-identical kinematic state up to 3× because APIs cannot accept precomputed pieces *(medium, DoD; fFluxPrime sub-part already done)*

Deep per-cell functions re-derive the same transients from pp+geom instead of sharing them; pure
functions of identical inputs, so every recomputation is bit-identical. Remaining occurrences (the
fFluxPrime internal double-eval was fixed via calcTijFromState): (a) `sim.zig` calcWavespeeds loop
(:824-835) — calcChi triggers a full fillRadState, then gasWavespeedsLr reconverts the gas velocity
and calcRadWavespeeds reconverts the rad velocity, per cell per stage; (b) implicit.zig:726/728/526 —
solveImplicitLab runs p2u(pp00), calcFfRtt(pp00) (conversion + full Rij), then solve4dPrim's
fillRadState(pp00) repeats the conversion and Rij build — 3 conversions + 2 Rij tensors of identical
values per cell entry before the Newton loop; (c) implicit.zig:201 vs :210 — applyConstraints computes
uconUcovFromPrims, then p2uMhd redoes convVelsBoth on the same velocity slots.

**Fix:** Extend the state-passing pattern: (d-wavespeeds) fill RadState once per cell and feed
gasWavespeedsLr from st.ucon/bsq, chi from st.kappa + the standalone Te-flavored calcKappaes(pp),
calcRadWavespeeds from the shared urf; (solveImplicitLab) share the ucon/ucov pair and the calcRij
tensor between p2u, calcFfRtt, and fillRadState — but do NOT substitute state00.ehat for calcFfRtt's
value (they contract Rij in different FP shapes and the result decides the startwith rung at
implicit.zig:731); pass calcFfRtt the precomputed u and rij instead. All reused values are bit-identical.

#### `koral/metric/precompute.zig:59` — applyKrisCorrection computes the full CoordData (inverse + 64 Christoffels) at six face points per cell just to read gdet *(low)*

Lines 59-60 call `metric.compute()` at the hi/lo face for each direction (6/cell during fillCenters),
and compute() does the 16-cofactor dual inversion + full Christoffel assembly, all discarded except
`.gdet`. fillFaces (:285-319) then recomputes metric.compute at the same face points again to populate
gb/gconb. For a 3D grid this makes cache construction roughly an order of magnitude more expensive than
needed. Init-time only (no wrong physics) but the dominant startup cost as grids grow.

**Fix:** Add `pub fn gdetAt(coords, mp, x) f64` in metric.zig running gcovDual + det4 only (identical
FP ops to compute()'s gdet path → bitwise-identical), and call it at :59-60. ~2-2.5× faster cache
construction. Optionally, when `InitOpts.faces` is true, run fillFaces first and read face gdet from
`gb[d]` at offset 3*5+4 to eliminate the 6 recomputes (needs a cache-aware variant since
applyKrisCorrection is also used standalone by computeCorrected/tests and gb is zeroed when faces=false).

#### `koral/field.zig:35` — Tiny per-cell index/getter leaf helpers on the hot path are not `inline fn` (Debug/ReleaseSafe only) *(low; partially done)*

The `krBlock`/`cellIndex`/`faceIndex`/`kr` accessors were marked `inline` in the kr fix, but the
other leaf helpers called O(NV × cells × passes)/step are still plain fns: Field.cellOffset/get/set/
load/store (field.zig:35-64), FaceStore.offset/get/set/load/store (now in sim/storage.zig),
Grid.xl/yl/zl/xc/yc/zc (grid.zig:87-119), Sim.flagIdx/scGet/scSet (sim.zig:440-455), and laxf/hll
(riemann/laxf.zig:10/16). ReleaseFast already inlines these (one compilation unit); the win is Debug
(no inlining at all — the default `zig build test` mode) and ReleaseSafe.

**Fix:** Mark those leaf helpers `inline fn`. Debug-only effect, no FP change → golden-safe. Measure
one slow golden test before/after; running `zig build test -Doptimize=ReleaseSafe` may make it moot.
Do NOT add `inline` to mid-size kernels (fFluxPrime, p2u, u2pSolverW, avg2point).

---

# P3 — Purity, hidden state & structural API

Signature-/ownership-level cleanups that remove ambient-state hazards and the repo's remaining
must-call-X-before-Y couplings. Nearly all are bit-identical (indices/plumbing only), pinned by the
existing goldens. Grouped: high-value structural first, then the verifier-downgraded polish.

#### `koral/sim.zig:195` — SpecificBc callback has no context parameter, forcing the only module-level var in the repo *(medium)*

`SpecificBc` is a bare `*const fn (sim, ix, iy, iz, t, ifinit, face) [NV]f64` with no user-context
slot. Any BC needing runtime parameters must use global state — and this already happened:
radtube_tests.zig:52 declares `var active_tube: *const tubes.Tube = undefined;` (the single
module-level mutable var in ~19k lines) so tubeBc can see which tube is running, with runTube
assigning it before each run. Hidden must-set-before-use dependency; concurrent tube runs would read
each other's tube.

**Fix:** Add `bc_ctx: ?*const anyopaque = null` to Options and change SpecificBc to
`*const fn (ctx: ?*const anyopaque, sim, ix, iy, iz, t, ifinit, face) [NV]f64`, with setBcCell
(sim.zig:616) passing `self.opt.bc_ctx`. Fixes TWO existing workarounds: radtube_tests.zig:52
(`var active_tube`) passes `tube` as ctx; evolution_tests.zig:622-624 (BondiCtx's `var michel`/`var rc`,
set at :676) passes `&michel` (its Michel solution is computed at runtime, so a comptime-bound BC can't
express it). puffy.Bc and the other comptime BCs ignore ctx. Compose with the P1 error-union widening
if done together (`Error![NV]f64` with a ctx param).

#### `koral/sim.zig:991` — calcU2p's boundary refresh depends on ambient self.time set only inside step() *(medium)*

calcU2p ends with `try self.setBc(self.time, false)` — depends on a field only step() assigns (:1671).
Invisible in the signature, and the API misleads: opExplicit(t, dtin) accepts t and discards it
(`_ = t;` :1403) while the BC refresh inside its calcU2p uses self.time. A standalone caller (a test
driving opExplicit/calcU2p, or dynamo.applyDynamo which has an explicit t but calls sim.calcU2p()) gets
ghosts evaluated at whatever time the last step() froze — silently wrong for a time-dependent BC.
PUFFY's Bc ignores t, so nothing catches it.

**Fix:** Give calcU2p an explicit time parameter (`pub fn calcU2p(self: *Self, t: f64)`) and thread it
from the four step() sites (:1702/1716/1725/1738), opExplicit (:1448, making its discarded t genuinely
used), and dynamo.applyDynamo (dynamo.zig:274, which already has t). Keep `self.time = t;` in step() as
the C-mirroring record (now write-only documentation). Signature-only, identical value on every path.

#### `koral/sim.zig:263` — ptm1 and ppostimplicit are write-only fields — dead state allocated and copied every step *(medium)*

A repo-wide grep shows ptm1/ppostimplicit are only written: allocated in init (:340-341), copied into
4×/step (:1695,1697,1717,1720), never read. The module header (:27-28) says upreexplicit/ppreexplicit
were *skipped* because "nothing reads them before M12" — same reasoning applies here but these were
kept. Cost: two full ghost-inclusive fields (~15 MB each at production) + ~4 full-array memcpys
(~60 MB traffic)/step, and they look like live RK2IMEX coupling state to a reader.

**Fix:** Delete both fields, their init/deinit entries, and the four copyFull calls, leaving a one-line
comment at the two step() sites noting C copies p into ptm1/ppostimplicit here and the copies are dead
on this path. Update docs/ARCHITECTURE.md 5.2/5.3 (the ptm1 row at :457 wrongly claims it is "handed to
the implicit solver"). If kept for line-by-line C diffability, mark both decls
`// write-only: kept to mirror problem.c; no Zig consumer`.

#### `koral/physics/radvisc.zig:61` — calcShearLab mixes an explicit center-cell pp0 with ambient sim.p neighbours; the radvisc chain carries redundant (ix,iy,iz) alongside geom *(medium)*

calcShearLab takes pp0 for the center cell but loads the ±1 neighbours from sim.p (:112-113) — the
result depends on ambient sim.p that the signature hides behind pp0. Today every caller passes exactly
sim.p's values, but the API permits a trial/modified pp0 (future implicit/fixup path) that would
silently mix trial center with stale neighbours. Related: calcRadVisccoeff (:236), calcRadShearviscosity
(:291), calcRijVisc (:310) all take both (ix,iy,iz) and a `*const Geometry`, yet Geometry already
carries .ix/.iy/.iz — a mismatched pair compiles fine and reads the wrong cell's spacing/Christoffels.

**Fix:** (1) Drop pp0 from calcShearLab and load the center prims from sim.p internally (every caller
already passes exactly that); or, if a divergent pp0 must stay possible, add a debug assert that pp0
matches sim.p at (ix,iy,iz). (2) Drop the (ix,iy,iz) params from calcRadVisccoeff/calcRadShearviscosity/
calcRijVisc and read geom.ix/iy/iz (set by fillGeometry), making geometry-position consistency
structural — also mirrors C, which extracts ix/iy/iz from geom->ix. Neither touches any FP expression.

#### `koral/metric/precompute.zig:191` + `koral/field.zig:35` — Ghost-offset index arithmetic duplicated in four places; flags reach through p.cellOffset/NV *(medium idiomatic + low DoD, merged)*

MetricCache.cellIndex (:191-198), faceIndex (:202-222), Field.cellOffset (field.zig:35-42), and
FaceStore.offset all implement the identical signed-index → padded-storage mapping, each re-doing the
`@intCast(ix + @as(i64, @intCast(g.ngx)))` dance (FaceStore caches i64 ng copies; Field and MetricCache
don't). The duplication already produced a smell: flagIdx (sim.zig:440) recovers a cell index by
`p.cellOffset(ix,iy,iz) / NV` — coupling flag indexing to the primitives field's layout, which silently
breaks if the documented AoSoA flip of Field (field.zig:6-8) ever happens.

**Fix:** Add `pub fn cellIndex(g: Grid, ix, iy, iz) usize` (with cached i64 ngx/ngy/ngz or a
`toPadded(i, ng)` helper) to Grid; express Field.cellOffset and MetricCache.cellIndex through it, share
a face-index helper for faceIndex/FaceStore.offset, and give flags their own layout-independent index
(`grid.cellIndex(...) * n_flags`) instead of the p.cellOffset/NV reach-through. Integer-only → golden-safe.

#### `koral/metric/precompute.zig:369` — faces=false path: NaN-geometry footgun + ~183 MB dead allocation; and dead dxdx_my2out/out2my *(medium purity + low DoD, merged)*

Two related storage issues. (1) MetricCache.init with `.faces = false` still allocates full-size
gb/gconb and memsets them to zero (:154-162); fillGeometryFace on such a cache returns an all-zero
metric, gdet=0, alpha = sqrt(-1/0) = NaN (:349) — no assert, no error, NaN poisoning downstream. Whether
the function is callable depends on an ambient init-time option, not expressed in types. ~183 MB of
zeroed-never-used memory on the PUFFY 384×360 grid (postprocessing mode; no callers today). (2)
dxdx_out2my is allocated nc*16 f64 and (post the dynamo fix that pointed dxdx_my2out at .bl) is still
write-only with no reader.

**Fix:** Make face storage `gb: [3]?[]f64` / `gconb: [3]?[]f64` — null (or len-0) when faces skipped;
`fillGeometryFace` unwraps with `orelse @panic("MetricCache built with .faces=false")` (or debug assert);
free conditionally in deinit. Misuse then fails loudly and postprocessing mode drops the dead memory.
Separately, drop `dxdx_out2my` (and any still-unread half of the pair) until a coordinate-transformed-
output consumer lands, or allocate/fill only when the target coords differ.

#### `koral/field.zig:55` + `koral/solve/implicit.zig:261` — Out-pointer where returning by value is equally efficient and removes the undefined-buffer step *(low, merged)*

`Field.load(self, ix, iy, iz, out: *[NV]f64)` forces every caller into
`var pp: [NV]f64 = undefined; f.load(ix,iy,iz,&pp);` (~99 sites) — the undefined buffer is briefly
observable and refactors can read it before the fill. Zig result-location semantics make a by-value
return write directly into the caller's variable (zero extra copies for 104 B). Same shape in
`residual` (implicit.zig:261), which fills `f: *[4]f64` and separately returns the max error — the
caller must know both channels compose one result.

**Fix:** `pub fn load(self, ix, iy, iz) [NV]f64 { return self.data[self.cellOffset(...)..][0..NV].*; }`
and migrate call sites mechanically (can add the by-value form alongside the old one and migrate
incrementally). Have `residual` return `relele.Error!struct { f: [4]f64, err: f64 }` (removes the
discarded `_ = residual(...)` at the Jacobian site, :592). Keep `store`'s `*const [NV]f64` (read-only).
Pure data movement → golden-safe.

#### `koral/field.zig:143` — Tautological test assertion: the iv-adjacency property is never actually checked *(low)*

In the "memory layout is iv-fastest then x" test, line 143 is
`try std.testing.expectEqual(f.cellOffset(0,0,0)+1, f.cellOffset(0,0,0)+1);` — both sides identical, so
the assertion is vacuous and the comment's claim is unverified. The other assertions (NV stride, SX*NV,
ghost corner) are real.

**Fix:** Check against the raw flat array:
`f.set(0,0,0,0,1.0); f.set(1,0,0,0,2.0); try std.testing.expectEqual(@as(f64,2.0), f.data[f.cellOffset(0,0,0)+1]);`
— this actually pins iv-fastest ordering (the uniqueness test can't, since it uses the same address
function for write and read and never inspects f.data).

#### `koral/io/scalars.zig:208` — scaleHeightAt mutates the Sim during a logically read-only diagnostic *(low)*

scaleHeightAt takes `sim: *SimT` (every other reduction takes *const) solely because it calls
dynamo.calcScaleHeight, which fills the persistent sim.scaleth array then reads back one entry.
Consequences: the mutability propagates (main.zig scalarRow must take `*SimT`); an output pass
overwrites dynamo-owned state (harmless only because recomputed from unchanged p); diagnostics can
never run concurrently with anything touching scaleth; and the full-grid fill is wasted for one radius.

**Fix:** Add `pub fn scaleHeightAtIx(comptime SimT, sim: *const SimT, ix: i64) f64` in dynamo.zig with
the existing per-column accumulation (incl. the ix==0 unnormalized C quirk), and have calcScaleHeight
delegate its per-ix body to it so the FP-critical code lives once and both paths stay bit-identical.
scaleHeightAt becomes `*const SimT`; scalarRow and its callers uniformly take *const. Safe because
applyDynamo (dynamo.zig:271) already recomputes scaleth before every dynamo use.

#### `koral/magn/ct.zig:222` — curlFromA delivers its result through the sim.vecpot scratch field — an invisible cross-module data channel *(low)*

curlFromA/cornerAverageA/calcBfromACore communicate exclusively through sim.vecpot slots (0..2 =
corner A, 3..5 = B); the consumer contract is comments only. mimicDynamo (dynamo.zig:243-255) calls
curlFromA then reads sim.vecpot.get(3..5); puffy init reads them in calcBfromA's ifoverwrite loop. No
signature says curlFromA has an output, clobbers vecpot, or that mimicDynamo's superimpose loop depends
on the immediately-preceding call — a reorder or interleaved second curl silently corrupts ΔB.

**Fix:** Make the scratch explicit: pass it as a parameter (`scratch: *field_mod.Field(6)`, cycle-free)
for curlFromA/cornerAverageA/calcBfromACore, callers passing `&sim.vecpot`; have curlFromA return the
scratch pointer so consumers visibly read the result, and document that slots 0..2 are clobbered
(corner A) and 3..5 are the output B. Identical arithmetic → goldens unchanged; the channel becomes
visible in the API.

#### `koral/metric/precompute.zig:29` — applyKrisCorrection mutates through a pointer where value-in/value-out is bit-identical; and a stale comment *(low)*

applyKrisCorrection(d: *metric.CoordData, ...) is the only mutate-through-pointer API in the otherwise
value-pure metric layer. A pure form doing the identical FP ops in order returns bit-identical results.
Related nit: the comment at :282-283 claims the correction "lives at module level below" — it actually
sits above at :29 (stale after a move).

**Fix:** Unconditional part: fix the stale comment at :282-283. Optional: change the signature to
`pub fn krisCorrected(coords, mp, g: *const Grid, d: metric.CoordData, ix, iy, iz) metric.CoordData`
(implemented as `var out = d; ...; return out;` — copy before any write, since
`d = krisCorrected(..., d, ...)` may alias the by-ref param with the result location) and assign at the
two call sites (:75, :255). Keeping the in-place form (mirroring C's calc_Krzysie_at_center) is also
defensible; then fix only the comment.

#### `koral/metric/precompute.zig:86` — geometryAt fabricates cell identity (ix=iy=iz=0); three callers duplicate a post-construction patch *(low, verifier-downgraded)*

geometryAt hardcodes `.ix=.iy=.iz=0, .ifacedim=-1` — fabricated values. Three call sites patch by
mutation: fillGeometryBL (puffy.zig:404-406), geomBLat (dynamo.zig:112-114, now folded into the BL
sidecar), and blGeom (io/scalars.zig:48-50). A caller that forgets the patch hands kernels a geometry
claiming to be cell (0,0,0). The fields are currently write-only across the codebase.

**Fix:** Add `geometryAtCell(coords, mp, x, ix, iy, iz)` (or an optional cell parameter) and delete the
patch sites. Indices never enter FP → golden-safe. (Duplication cleanup / future-proofing, not a live
bug — see the ifacedim/ix/iy/iz dead-field finding in P5, which may make this moot.)

#### `koral/physics/opacities.zig:165` — Opac.tot_emissivity is a placeholder-invalid field returned as 0 by calcOpacitiesFromState *(low, verifier-downgraded)*

calcOpacitiesFromState (pub) returns an Opac whose tot_emissivity is a sentinel 0, valid only after
calcKappaFromState or radforce's grey branch overwrite it — a must-call-X-after-Y dependency not in the
types. Any future caller reading tot_emissivity off calcOpacitiesFromState silently gets 0. Currently
write-only across the codebase.

**Fix:** Keep the flat struct (mirrors C's `struct opacities` 1:1). Lightest structural fix: have
calcOpacitiesFromState fill tot_emissivity itself with the exact expression calcKappaFromState uses
(bit-identical, never reaches goldens), making the sentinel impossible; calcKappaFromState's overwrite
becomes a droppable no-op. Leaving as-is with the existing comments is also defensible.

#### `koral/physics/radiation.zig:107` — pradFf2Lab mutates pp in place through a pointer where a by-value transform works equally well *(low, verifier-downgraded)*

pradFf2Lab takes `pp: *[N]f64` and rewrites EE..FZ via u2pRad, inheriting C's in-place shape. The rest
of the layer is functional (transPmhdCoco/transPradCoco/transPallCoco take ppin by value precisely to
neutralize C's aliasing quirk), and the sole caller (puffy.zig:497, init-time only) has no need for
in-place. Pointer signature hides which slots change.

**Fix:** Take pp by value and return the updated vector (copy into a local, pass &local to u2pRad,
return local). Update the single call site to `pp = try radiation.pradFf2Lab(cfg, pp, geomBL, ...);`.
Bit-identical incl. the u2pRad-failure path. Leave u2pRad's pointer signature (conditional-write-on-
failure + hot callers).

#### `koral/physics/radvisc.zig:335` — calcRijViscTotal fills a hidden per-step store (sim.rijvisc) with no freshness guarantee at the read site *(low, verifier-downgraded)*

calcRijViscTotal writes R^i_j_visc into sim.rijvisc; consumed later by rijviscFace (sim.zig:1206, from
sweep) inside opExplicit. Nothing ties the fill to the read: step() happens to call the fill before both
explicit stages, but opExplicit is independently callable (evolution_tests.zig:120), and with
radviscosity on a direct caller silently gets zeros on step 1 or the *previous* step's tensor after —
a must-call-before pattern (C's Rijviscglobal). The global_dt lead itself is clean (explicit parameter
through the chain).

**Fix:** Add a freshness stamp — `rijvisc_stamp: ?struct { step: u64, dt: f64 } = null` on Sim, set by
calcRijViscTotal, and a `std.debug.assert` in rijviscFace (or at opExplicit entry when radviscosity and
L.hasVar(.ee)) that the stamp matches the current step/dt. Turns a silent zero/stale tensor on a direct
opExplicit call into a loud debug failure; golden output unchanged. Do NOT thread the Field(16) through
opExplicit's signature (burdens hydro-only configs, departs from the C mirror).

#### `koral/sim.zig:315` — Transient impl_dt field on Sim is ambient state feeding the implicit solve *(low, verifier-downgraded)*

opImplicit (:1540) writes `self.impl_dt = dtin` and implicitRowsWorker (:1570) reads it — every per-cell
inversion depends on a field set moments earlier by a different function (the C global_dt pattern), and
it makes opImplicit non-reentrant. Root cause: parallelRows' worker signature has no way to carry
per-pass arguments. (radvisc's dt and thermo.Consts are already threaded explicitly — impl_dt is the one
remaining exception.)

**Fix:** Make parallelRows context-generic —
`fn parallelRows(self, ctx: anytype, comptime worker: fn (*Self, @TypeOf(ctx), i64, i64, *ChunkResult) void)`
— spawn with `.{ self, ctx, iy0, iy1, &results[i] }`, pass `dtin` from opImplicit and `{}` from calcU2p,
and delete impl_dt. Makes the dt dependency visible in the worker signature. Pure plumbing; no
correctness hazard today (single-pass by design), so this is cleanliness only. (Compose with the
P2 threading rework, which touches the same dispatch.)

#### `koral/solve/invert.zig:171` — u2pSolverW's pp parameter is an undocumented in/out: it must hold the current primitives as the Newton initial guess *(low)*

u2pSolverW reads pp[rho], pp[uu], pp[vx..vz] (:241-251) to build the initial W before overwriting pp on
success — the C ppin==ppout contract, but nothing at the signature says so (the doc implies pure output).
Same for u2pMhd (:361), whose ppbak dance relies on the incoming pp being the previous good state to
restore on failure. A caller passing a zeroed buffer gets W=0, 50 stuck iterations, returns −150 →
entropy fallback → −150 → silent fixup flagging that looks like a physics failure. Unenforceable by
assert (any positive rho is "valid"), and the guess-dependence is load-bearing for golden agreement.

**Fix:** Keep the name `pp` (mirrors C's u2p_solver_W) but state the in/out contract in both doc comments
— u2pSolverW (:169): "pp is in/out — must contain the cell's current primitives on entry; they seed the
Newton iteration (C: previous step's p). Anything else diverges or fails with −150." u2pMhd (:358): note
that on total failure the incoming pp is restored unchanged. Drop any isFinite assert (rho=0 is finite,
would not catch the failure mode).

---

# P4 — Test hygiene

The suite is the codebase's strongest safety net for these refactors — harden the two structural gaps
(silent coverage loss, duplicated helpers) before doing large P2/P3 work, then the reporting polish.

#### `koral/koral.zig:107` — Test registration is fully manual with no guard against a silently dropped file *(medium)*

The entire suite hangs off the single `test { _ = @import(...) }` block in koral.zig. All test files are
referenced today, but nothing enforces it: a new `foo_tests.zig` that compiles cleanly runs zero tests
unless someone adds `_ = @import("foo_tests.zig");`, and neither build nor CI notices — the classic
silent-coverage-loss trap, made likely by the suite's size and the milestone cadence.

**Fix:** Add a configure-time guard in build.zig: iterate koral/ entries (same openDir/iterate pattern
used for PROBLEMS/), collect basenames matching `*_tests.zig` / `*_golden_tests.zig` plus the special
cases (metric_tests.zig, testing/tubes.zig), read koral.zig, and fail the build unless the literal
`@import("<name>")` token appears for each. ~20 lines, makes the manual registration contract
self-enforcing.

#### `koral/metric_golden_tests.zig` (was golden_test.zig) — duplicates testing/golden.zig helpers verbatim; Dev/FieldDev re-implemented in three golden files *(medium)*

testing/golden.zig holds the shared KGLD reader + deviation trackers; every golden file uses it except
the metric golden, which carries private copies: `Golden`, `readGolden`, `coordsFromId`, and a
byte-for-byte duplicate `DevTracker`. A KGLD format bump now has two readers to update, and the two
DevTrackers can diverge. The field-scale tracker is re-implemented three times: `Dev` in
visc/dynamo golden tests (identical) and the richer `FieldDev` in the puffy golden test.

**Fix:** Port the metric golden to the shared helpers — replace local Golden/readGolden with
`gold.readGolden(a, "metric/" ++ name, nin, nout)` and use `gold.DevTracker`/`gold.coordsFromId`
(metric.Coords is a re-export of config.Coords → drop-in). Move one field-scale tracker (the FieldDev
superset) into testing/golden.zig and delete the identical `Dev` copies (their add() sites are in cell
loops, so ix/iy are in scope; or give the shared tracker a location-less add overload). Leave the
puffystep golden as-is (its per-variable-scale/cell-exclusion logic is genuinely different).

#### `koral/state_tests.zig:32` — Theory-test helpers copy-pasted: expectClose ×4, velr-with-gamma ×4, ppFromTemps ×2, run-to-tend ×5 *(medium)*

Identical 7-line `expectClose` appears verbatim in state_tests.zig:32, flux_tests.zig:20,
radiation_tests.zig:51, opacity_tests.zig:53 (metric_tests.zig:55 has a deliberate rtol/atol variant).
The controlled-Lorentz VELR sampler is written 4× (velrWithGamma/velrGamma + two inline re-derivations).
`ppFromTemps` is byte-identical in opacity/implicit tests. The "advance to t_end with the C dt=t1−t
clamp" idiom is repeated at 5 sites (C-provenance comment on only one). Six near-identical PUFFY
`SimP.Options` builders recur across radvisc/dynamo/golden tests, differing only in toggles.

**Fix:** Create koral/testing/theory.zig (peer of golden.zig/tubes.zig) holding expectClose, one unified
velrWithGamma, ppFromTemps, and a `stepTo(sim, t_end)` carrying the "C: dt=t1-t clamp" comment once
(ppFromTemps/stepTo need a comptime layout/Sim-type parameter since the copies close over file-local
cfg — mechanical, no math change). Give problems/puffy.zig (or testing/) one
`simOptions(.{ .radviscosity, .dynamo, .implicit, .opac })` builder to replace the six Options literals.

#### `koral/evolution_tests.zig:86` — Bare expect() in per-cell sweep loops reports nothing on failure *(low)*

The uniform-static and conservation gates assert inside cell loops with plain `try expect(...)` on a
computed deviation: evolution_tests.zig:86/:129/:149, mhd_evolution_tests.zig:144, radiation_tests.zig:477.
On failure they print nothing — cell, variable, deviation — so a regression in a 100-step gate starts as
a blind bisect. Below the suite's own standard (opacity_tests.zig:356 does the identical sweep with a
diagnostic print).

**Fix:** Apply the opacity_tests.zig:356 pattern: on violation print ix/iy, iv, expected, got, dev, then
return the error — or route through the shared expectClose once extracted (P4 above; it already prints
both values).

#### `koral/radtube_tests.zig:322` — Fast radtube battery asserts tube 3a's gates before running tube 4a, hiding 4a's measurements *(low)*

The slow set (:351) deliberately runs all six configs first then asserts, with a comment explaining a
single failed gate must not hide the others; golden_puffystep_test.zig:246 adopts the same print-all-
then-assert pattern. But the default battery (:322-349) inlines runTube+asserts sequentially: if 3a
trips a gate, 4a — several minutes of the most expensive default-suite compute, incl. the only
default-suite ODE-residual gate — never runs and its metrics are never printed.

**Fix:** Mirror the slow set (:404-418): run 3a and 4a into a Metrics array, print both, then evaluate
all gates and fail once. Simpler alternative: split into two test blocks ("tube 3a"/"tube 4a") — Zig's
runner runs every test even after a failure, so 4a's metrics and ODE gate still execute.

#### `koral/radvisc_tests.zig:63` — Data-dependent silent skip: torusCell() orelse SkipZigTest can mask a regression *(low)*

Both M12 radviscosity limiter tests bail with `torusCell(&s) orelse return error.SkipZigTest` (:63, :90).
Unlike the suite's other skips (slow-gating, missing-golden — both expected, the latter with a message),
this converts an unexpected runtime condition (the 32×40 PUFFY init producing no cell with ρ > 1e-20 in
the scanned window) into a silent skip with no message. A refactor that shifts/empties the torus quietly
retires the only tests pinning RADVISCNUDAMP/RADVISCMAXVELDAMP; the skip-count change is easy to miss.

**Fix:** The init is deterministic, so no torus cell is a test failure, not an environment condition:
`orelse return error.NoTorusCellFound` (or unwrap with a message). If skip semantics are truly intended,
print a diagnostic first, as readGolden does.

---

# P5 — Idiomatic polish & minor

Low-risk cleanups; batch them when touching the relevant file. None affect goldens.

#### `koral/sim.zig:143` — Split allocator convention: FaceStore.deinit(allocator) vs Field.deinit() in the same aggregate *(low)*

`FaceStore.deinit(self, a)` is unmanaged-style while `Field.deinit(self)` stores its allocator. Sim.deinit
mixes `@field(...).deinit()` for Fields with `self.pb_l[d].deinit(a)` for FaceStores — the kind of
inconsistency that produces a wrong-allocator/double-free bug on a later refactor. **Fix:** Store
`a: std.mem.Allocator` in FaceStore (set in init), change deinit to `fn deinit(self: *Self) void`, add
the `self.* = undefined` poisoning Field.deinit already does; then Sim.deinit's FaceStore loop matches
the Field pattern. (Now lives in sim/storage.zig after the extraction.)

#### `koral/relele.zig:240` — convVelsCore's conversion dispatch ends in an else-catchall instead of an exhaustive switch *(low)*

The if/else chain enumerates five (which1,which2)/(from,to) pairs and the final `else` is documented only
by `// VELR -> VEL3`. If VelType grows or a pair is mistyped, the catchall silently routes to VELR→VEL3
math (C had an explicit `my_err` fallback the port lost). **Fix:** Replace with nested exhaustive switches
so the compiler proves coverage, branch bodies byte-identical; fold same-type cases in (mirrors C) or keep
the `from == to` fast path in front and mark the diagonal arms `unreachable`.

#### `koral/recon/recon.zig:26` — avg2pointScalar's else-catchall silently degrades any unknown order to donor cell *(low, verifier-downgraded)*

`order` is a raw u8, switch is `1 => linear, 2 => ppm, else => donor`. A caller passing 3 is silently
reconstructed at order 0. **Fix:** `0 => .{ .ul = u[2], .ur = u[2] }, 1 => linearMinmod(...), 2 => ppm(...),
else => unreachable`. Pure API hardening; the production path is already compile-time safe (sim.zig
exhaustively switches the 3-member config.Reconstruction enum).

#### `koral/physics/radiation.zig:151` — Hand element-copy loops where whole-array assignment / @bitCast is idiomatic *(low)*

`for (0..6) |i| aval[i] = a0[i];` (:151), `aval[6+i] = a1[i];` (:168), and the [4][4]→[16] flatten in
radvisc.zig:358-361. **Fix:** `aval[0..6].* = a0;`, `aval[6..12].* = a1;`, and
`const t: [16]f64 = @bitCast(rvisc);` (layout-identical) passed to store.

#### `PROBLEMS/puffy/main.zig:113` — Dead parameters/diagnostic: writeScalars' unused allocator, collectDiag's unreported n_rad_fixup *(low)*

`writeScalars` (:110) discards `a` (`_ = a;` :113); both call sites thread it for nothing. `collectDiag`
counts `n_rad_fixup` (:55, :75-77) but scalarRow never reads it. **Fix:** Drop the `a` parameter (update
:170, :187). For n_rad_fixup: the flag is genuinely populated (sim.zig:1051 sets .rad_fixup regardless
of do_u2prad_fixups), so either add an n_radfix column to ScalarRow/header/appendScalarLine and wire it
through, or delete the Diag field + counting branch.

#### `koral/geometry.zig:14` — Geometry.ifacedim/ix/iy/iz are write-only dead data (and ifacedim is a -1 sentinel) *(low)*

Grepping shows no code reads geo.ifacedim/ix/iy/iz — only assigned (precompute.zig:89/333,
testing/golden.zig:216, puffy.zig:404, dynamo.zig:112, scalars.zig:48). The C branch that consumed
ifacedim was ported differently (rijviscFace takes `dim` explicitly). **Fix:** Delete the four fields and
their assignments across the five write sites. (Subsumes the geometryAt-fabricates-cell-identity P3 item
if done — no cell parameter needed once nothing reads the fields.) If a face marker is later needed,
reintroduce `face: ?Face = null` with `Face = enum(u2){x,y,z}` rather than a -1 sentinel.

#### `koral/state.zig:10` — State(cfg) is dead code with a stale doc promising a design that was never built *(low)*

The module doc promises an M3+ comptime composition (base fields + each module's StateFields) that never
happened; the type still holds only the six M0 fluid fields, is exported as koral.State, and has zero
users — the real per-cell state is radforce.RadState. **Fix:** Either delete state.zig + the koral.zig
re-export, or rewrite the doc to state the actual design ("superseded by radforce.RadState; kept only for
X") pointing C's struct_of_state readers at radforce.zig.

#### `koral/testing/golden.zig:152` — readKstp panics on nrec==0; readKini leaks vars / coordsFromId turns corrupt data into unreachable *(low)*

readKstp computes `per_rec = (raw.len-32)/8/k.nrec` before validating nrec — a truncated .kstp with
nrec=0 is a div-by-zero panic instead of `error.BadGoldenFile`. readKini allocates `k.vars` (:327) then
can `return error.BadGoldenFile` (:334) without freeing (leak report buries the real error under
std.testing.allocator). coordsFromId (:228-235) ends in `else => unreachable` on a value read straight
from the file. **Fix:** `if (k.nrec == 0 or ncell == 0) return error.BadGoldenFile;` after computing
ncell; `errdefer a.free(k.vars);` after the vars alloc; change coordsFromId to `!config.Coords` with
`else => error.BadGoldenFile`.

#### `koral/sim.zig:332` — Sim.init (~35 allocs) and MetricCache.init (11 allocs) leak all prior allocations if a later one fails; duplicated field-name list *(low, verifier-downgraded)*

~35 sequential `try` allocations with no errdefer; an OOM mid-init leaks everything before it. For a
driver that exits on init failure this is cosmetic, but Sim is constructed repeatedly in the test suites
(std.testing.allocator reports these as confusing secondary failures). The field-name tuple is also
duplicated verbatim between init (:338) and deinit (:395-399). **Fix (decide once for the codebase):**
either explicitly accept "startup OOM is fatal, OS reclaims", or back Sim's + MetricCache's storage with
a single internal `std.heap.ArenaAllocator` (failed init and deinit both a single arena release — also
collapses the hand-maintained per-field deinit). At minimum, hoist the duplicated field-name tuple into
one comptime const referenced by both init and deinit. (No failing-allocator injection exists in the
suite, so per-allocation errdefer would guard a path that never executes.)

#### `koral/geometry.zig:8` — Geometry is a ~416-byte value re-gathered up to ~5×/cell per stage; could be materialized once *(low, verifier-downgraded)*

Every fillGeometry/fillGeometryFace gathers 40 f64 from two heap arrays, recomputes alpha (sqrt+divide)
and xxvec. Per stage a cell-center Geometry is rebuilt ~5× and each face Geometry ~3×. Cost is <1-2% of a
stage (polish). **Fix (if it ever matters):** Materialize precomputed Geometry values at init (centers +
3 face sets, ~+1.7 KB/cell) and have fillGeometry/fillGeometryFace return `*const Geometry` — kernels
already take `*const Geometry`, and the from-scratch constructors keep working. Avoid the pointer-view
variant (gg/GG as `*const [4][5]f64` into the cache): it turns a hazard-free value into a lifetime-managed
view for the same gain.

#### `koral/metric/precompute.zig:109` — MetricCache stores ~2 KB/cell; symmetric tensors kept full-size *(low, accounting only)*

Per-cell: g 160 B + gcon 160 + kris 512 + (dead dxdx, see P3) → ~1088 B at centers + ~183 MB of face
blocks → ~338 MB on the PUFFY grid (vs 104 B of primitives). gcov/gcon are symmetric and kris is
symmetric in its lower indices; compacting to triangles would cut the cache ~33% (matters much more at
3D, tens of GB) with bit-identical stored values (cache layout is not part of the golden contract).
**Fix:** No action now — if memory pressure appears at 3D milestones, store g/gcon/kris as triangles
behind the existing storeBlocks/geometryFromBlocks/kr accessors. Otherwise a one-line comment noting the
deliberate C-shaped (gSIZE=20/gKr) memory-for-simplicity trade suffices. Do not touch CoordData.kris /
applyKrisCorrection.

#### `koral/sim.zig:83` — Scal AoS record scatters each direction's six wavespeeds across a 160-byte record *(low, DoD)*

scal is Field(20) — the enum groups all-left/all-right/all-max across dimensions, so one dim-pass of
fluxesAtFaces reads six 8-byte values scattered over all three cache lines, for two cells per face. C
stores these as separate SoA arrays. **Fix:** Reorder the enum per-dimension —
`{ahdxl, ahdxr, ahdx, aradxl, aradxr, aradx, ahdyl, ...}` — so each flux dim-pass reads one contiguous
48-byte run per cell. All access goes through the ahd_*/arad_* tables or named literals, so the reorder
needs no other changes and cannot affect goldens. (Optionally gate the 9 arad slots on L.hasVar(.ee) —
larger change.)

#### `build.zig:50` — Per-problem executable installed through two InstallArtifact steps; -Dmpi option plumbed but unread *(low, project structure)*

`b.installArtifact(exe)` (:48) then a second independent `b.addInstallArtifact(exe, .{})` (:50-51) install
the same binary to the same path — redundant, and they'll silently disagree if per-problem install options
ever diverge. Separately, `-Dmpi` (:6) is stored into build_options (:10) but no koral/ file reads
`build_options.mpi` — dead plumbing until the MPI backend exists. **Fix:** Create the InstallArtifact once
and share it (`const install = b.addInstallArtifact(exe, .{}); b.getInstallStep().dependOn(&install.step);
b.step(entry.name, ...).dependOn(&install.step);`). For -Dmpi, make misuse loud: either remove the option
until the backend lands, or add a comptime guard
`if (@import("build_options").mpi) @compileError("MPI backend not implemented; build without -Dmpi");`.

---

## Provenance

- Original review: 18 reviewer agents + 131 verifier agents, ~5.7M tokens, ~47 min; 138 raw → 137 deduped
  → 120 survived adversarial verification. Full machine-readable results in the session transcript journal.
- This pruned/prioritized revision (2026-07-08): removed every finding carrying a `Fixed` annotation
  (1 high + 6 hot-path + 1 idiomatic + all 5 medium naming + 3 project-structure medium + ~25 low naming +
  build-mode warning + 2 project-structure low), kept the one won't-do (fillGeometry rename), merged
  cross-dimension duplicates (gcon col-4, faces=false + dead dxdx, index arithmetic, flux dead-work,
  golden-test naming straggler), and re-ordered the 53 remaining findings by actionability
  (P1 correctness → P2 performance → P3 purity/API → P4 test hygiene → P5 polish). Priorities are
  editorial, not from the original verifier. Line numbers predate the applied renames/extractions.
