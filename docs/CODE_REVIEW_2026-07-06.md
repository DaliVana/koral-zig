# koral-zig core review — 2026-07-06

A full-codebase review (~19k lines, all source + test files) along the dimensions requested:
**naming**, **project structure**, **function purity**, **inline functions in hot paths**,
**datatype sizes / data-oriented design**, **idiomatic Zig**, plus general best practices and
test hygiene.

**Method.** 18 independent reviewer agents (one per dimension x file-group, plus whole-project
structure and test-hygiene reviewers) produced 138 raw findings; after dedup, every finding was
re-read in the code by an adversarial verifier told to refute it. 11 were rejected as factually
wrong or protected by the C-diffability contract (intentionally transcribed C quirks / FP shapes).
6 verifiers were cut off by a rate limit; those findings were hand-verified afterwards. Identical
issues found independently by several reviewers are merged below (the params.zig high-severity
issue was found by three separate reviewers).

**Result: 1 high / 37 medium / 72 low findings.**

**Naming dimension status (2026-07-06): fixed.** All 5 medium + 25 of 26 low naming
findings were applied (see "Fixed" annotations inline below); the one skipped —
`koral/metric/precompute.zig:354` fillGeometry/fillGeometryFace — was left as-is by
request (huge blast radius, ~130 call sites/17 files, for a modest gain; also the one
naming finding the adversarial verifier itself downgraded). `zig build test
-Dslow-tests` (the full suite, including the radtube battery, the M11 keystone
eps12 regeneration, and the M12/M13 golden comparisons) passed clean after the
rename pass — exit 0, no panics, every golden deviation still inside its bound.

| Dimension | high | medium | low |
|---|---|---|---|
| Best practices & latent bugs | 0 | 4 | 6 |
| Purity & hidden state | 0 | 6 | 12 |
| Hot path & inlining | 0 | 10 | 5 |
| Data-oriented design | 0 | 4 | 5 |
| Idiomatic Zig | 1 | 2 | 11 |
| Naming | 0 | 5 | 26 |
| Project structure | 0 | 3 | 3 |
| Test hygiene | 0 | 3 | 4 |

Severity: **high** = latent bug / real perf cliff; **medium** = clear concrete improvement;
**low** = worthwhile polish. File references are repo-relative, lines 1-indexed.

---

## Overall assessment: project structure

This is an unusually disciplined project layout for a ~19k-line numerical code. The three-tier split — `koral/` (library), `PROBLEMS/<name>/main.zig` (one auto-discovered executable per problem), and the validation apparatus (`oracle/` C harnesses, `tools/gen_golden.sh`, committed `tests/golden/` binaries) — is principled, and crucially it is *documented*: README, docs/README, ARCHITECTURE §module-map, and the USER_GUIDE "new problem" recipe all cross-reference the same layout, and `tests/golden/manifest.json` records the koral_lite SHA, compiler, and generation timestamp for every golden file. Git hygiene is clean (dumps/, oracle/build/, zig-cache, zig-out all ignored; nothing generated is committed except the goldens, which is the point). build.zig is small and idiomatic: one library module, one test artifact with `-Dtest-filter`/`-Dslow-tests`, and a PROBLEMS/ directory scan that gives each problem `install`, `<name>`, and `run-<name>` steps.

Inside `koral/` the nesting rule is only implicit — roughly "core data model and C-core-file transcriptions at the root (config, layout, grid, field, geometry, state, units, params, relele, frames, p2u, sim), subsystems in directories" — and it has soft spots: `flux/` and `recon/` are single-file directories (flux/laxf.zig is 21 lines) while 300-line relele.zig/frames.zig sit at the root, and the p2u/u2p transform pair is split across levels (p2u.zig top-level, its inverse in solve/invert.zig). The worst naming trap is that `koral/flux/laxf.zig` (Riemann flux combination) coexists with `koral/physics/flux.zig` (the physical flux f_flux_prime), and the barrel then exposes the flux/ directory under a *third* name, `koral.riemann`. The barrel itself (koral.zig) is otherwise complete and consistent — every module is re-exported through tidy namespace structs with convenience aliases, nothing internal leaks, and the comptime Config/VarLayout machinery is well placed.

The problems/puffy.zig (library: ICs, `Bc(SimT)` boundary hook, constants — all unit-testable) vs PROBLEMS/puffy/main.zig (driver: arg parsing, time loop, I/O only) split is principled, documented, and the executable side is admirably thin (208 lines, plus a nice comptime C-diffability assertion). The boundary leaks only in one direction: PUFFY-specific preset constants are scattered as `.puffy` decls across six generic library modules, and physics/radforce.zig goes further — its generic `Params` *defaults* to the PUFFY Klein–Nishina scattering hook, and that hook is a closed enum in the library rather than a problem-supplied hook like `specific_bc`.

Testing is both the strongest and the structurally weakest area. Strong: two meaningful test families (theory vs C-golden) with an excellent strategy, committed goldens that skip visibly when absent, and slow-test gating. Weak: 27 test files sit flat in the koral/ root and outnumber the ~12 core source modules they test, one subsystem (metric) nests its tests instead, and three naming schemes coexist (`X_tests.zig`, `golden_X_test.zig`, `metric/tests.zig`, plus the unqualified `golden_test.zig` which is actually the *metric* golden test). Registration is a hand-maintained 60-line list in koral.zig's test block — I verified it is currently complete, but nothing enforces that, and the documented new-problem recipe depends on contributors remembering to edit it. sim.zig at 1751 lines is internally well-sectioned with clear `// ----` headers mapping to C source files, but it has clean extraction seams (BC block, threading dispatch, FaceStore) that will matter when MPI arrives.

---

## High severity

#### `koral/params.zig:45` — Mixed string ownership in Params (literal default vs heap dupe) with no deinit — latent invalid free

*Idiomatic Zig, Best practices & latent bugs, Purity & hidden state — flagged independently by 3 reviewers*

Params.parse dupes string values with the caller's allocator (parseValue, params.zig:124-128), but the default `out_dir = "dumps"` is a static literal. The struct has no deinit, so callers must guess ownership. PROBLEMS/puffy/main.zig:143 does `defer a.free(p.out_dir);` unconditionally with `init.gpa` — if a user's params file omits `out_dir` (relying on the documented default), that frees a string literal: invalid free / UB under DebugAllocator or any GPA. Two smaller leaks in the same design: (1) a key assigned twice leaks the first dupe (setField overwrites without freeing, params.zig:103); (2) any parse error after a string was duped leaks it (no errdefer in parse). The test at params.zig:157 only works because it happens to know out_dir was set from the file — that knowledge cannot exist at the API boundary.

**Fix:** Make ownership uniform: in parse(), dupe every []const u8 field up-front (including defaults) so all strings are always allocator-owned, add `pub fn deinit(self: *Params, a: std.mem.Allocator) void` that frees them, free the previous value before overwriting on repeated keys, and put `errdefer` cleanup in parse so error returns don't leak. Then main.zig's cleanup becomes `defer p.deinit(a);` and is correct for every input file.

---

## Medium severity

### Best practices & latent bugs

#### `PROBLEMS/puffy/main.zig:176` — NaN abort path can be unreachable after a global blow-up, and the NaN 'abort' exits with success status

*Best practices & latent bugs*

Three interacting problems in the driver's failure handling. (1) `step()` resets `tstepdenmax = -1` (koral/sim.zig:1678) and `saveWavespeeds` only updates it via `if (tsd > self.tstepdenmax)` (sim.zig:904); NaN compares false, so if a step blows up globally (every domain cell's wavespeed denominator NaN), `tstepdenmax` keeps the -1 sentinel and the next loop iteration computes `dt = 1.0 / s.tstepdenmax = -1.0`. Time then marches backwards, `s.t >= next_out` (line 183) can never become true, so the NaN check at line 193 never executes and the run spins doing garbage negative-dt steps until `nstep_max`, then prints "done". (2) Even when NaN is detected, line 195 does a bare `return;` — process exit code 0, so batch scripts/CI treat a NaN-poisoned run as successful. (3) NaN detection only happens at output cadence (`dtout1`), so NaNs can propagate through many steps of localized blow-up before being noticed at all.

**Fix:** In PROBLEMS/puffy/main.zig (Zig-native driver only — leave the C-mirrored tstepdenmax sentinel/update in sim.zig untouched): (a) guard the timestep before stepping, e.g. `if (!(dt > 0) or !std.math.isFinite(dt)) { ... return error.InvalidTimestep; }` after line 177 — this catches both the all-NaN case (tstepdenmax stuck at -1 → dt = -1) and the +inf case (dt = 0 stall); (b) replace the bare `return;` at line 195 with `return error.NanDetected;` (or std.process.exit(1)) so a NaN abort yields a non-zero exit status for batch scripts/CI; (c) optionally run the cheap positivity check `s.tstepdenmax > 0` every step rather than waiting for output cadence — note tstepdenmax itself can never be NaN since NaN fails the `>` update, so an isFinite check on it would not detect NaN.

#### `koral/metric/precompute.zig:224` — storeBlocks leaves gcon/gconb slots [0..2][4] uninitialized; garbage is copied into every Geometry.GG handed to kernels

*Best practices & latent bugs, Data-oriented design — flagged independently by 2 reviewers*

MetricCache.init allocates self.gcon (line 147) and self.gconb (line 157) with a.alloc — undefined contents — and only memsets them in the faces=false skip path (lines 158-161), i.e. the normal fill path never zeroes them. storeBlocks (lines 224-234) writes the 16 metric entries plus off+3*5+4 (gttpert) but never touches off+0*5+4, off+1*5+4, off+2*5+4 — 3 of the 20 slots per block stay as heap garbage. geometryFromBlocks (lines 341-346) then copies all 20 slots with `for (0..5) |j|` into Geometry.GG, so GG[0][4], GG[1][4], GG[2][4] carry uninitialized memory in every cell/face Geometry built from the cache. This is inconsistent with geometryAt (line 103), which explicitly sets `GG[i][4] = if (i == 3) gttpert else 0`. Today no kernel indexes those three slots (verified by grep), but the whole 4x5 GG block is passed by pointer (*const [4][5]f64) throughout relele/frames/wavespeeds, so any future loop over the block, a struct dump, or a Geometry comparison reads nondeterministic values — silently breaking the `deterministic` run guarantee and defeating msan/valgrind. This is exactly the partially-filled-undefined-memory bug class the C code suffered from.

**Fix:** In storeBlocks, explicitly zero the three unused gcon extras — `for (0..3) |j| dst_gcon[off + j * 5 + 4] = 0;` — matching geometryAt (precompute.zig:103) and the codebase's own explicit-slot convention (see radforce.zig:39). Alternatively @memset self.gcon and self.gconb unconditionally right after allocation in init. Add a test asserting fillGeometry(...).GG[i][4] == 0 for i < 3 (and likewise for fillGeometryFace). Note self.gcon is never memset even in the faces=false path, so the fix must cover centers as well as faces.

#### `koral/physics/radvisc.zig:247` — calcRadVisccoeff silently returns nu = 0 when opt.opac is null, diverging from C's SKIPRADSOURCE semantics

*Best practices & latent bugs, Purity & hidden state — flagged independently by 2 reviewers*

`const opp = &(sim.opt.opac orelse return 0);` makes the entire radiative-viscosity machinery a silent no-op whenever Options.opac is null while opt.radviscosity is true — the sweep still calls rijviscFace/addRadViscFlux (sim.zig:1285-1299) but every stored R^i_j_visc is zero, and the run completes with no error or log. Nothing in Sim.init or Options validates the coupling (sim.zig:217 `opac: ?radforce.Params = null`, :231 `radviscosity: bool = false` are independent). This is also a genuine behavioral divergence from C: opac==null is documented as the analog of SKIPRADSOURCE (sim.zig:24), and C's calc_rad_visccoeff under SKIPRADSOURCE sets mfp=1.e50 (rad.c:4516-4518), which then clamps to mfplim = r_BL — i.e. C keeps viscosity ACTIVE (nu = ALPHARADVISC·mfplim·stepfunction) in that configuration, while Zig produces exactly zero. Secondary nit on the same line: `&(optional orelse return)` takes the address of a stack temporary copy of the ~250-byte Params struct per cell per step rather than pointing at the field; `if (sim.opt.opac) |*o| o else ...` yields a *const into the actual storage and is clearer.

**Fix:** Make the opac==null semantics explicit for radviscosity — either (a) reject the combination at Sim.init (radviscosity=true requires opac != null, with a clear error), or (b) transcribe C's SKIPRADSOURCE branch (rad.c:4516-4518: mfp = 1e50 before the mfplim clamp, skipping calcChi; the chi<SMALL test is moot since mfp>mfplim always fires) with a comment citing rad.c, so the config keeps viscosity active exactly like C. Option (b) preserves C-diffability; (a) is acceptable if that config is deliberately unsupported, but the silent nu=0 no-op with the sweep still running is the one outcome to eliminate. Independently, replace `&(sim.opt.opac orelse return 0)` with `const opp = if (sim.opt.opac) |*o| o else return 0;` to point at the field instead of a stack temporary copy of Params (cf. the copy-free lvalue form `&(self.opt.opac.?)` already used at sim.zig:1550).

#### `koral/sim.zig:317` — Sim.init validates none of its runtime preconditions (ghost depth vs reconstruction, specific_bc presence, coords consistency, nccorrectpolar vs ny)

*Best practices & latent bugs*

Several assumed invariants are unchecked and each violation fails far from the cause: (a) `Grid.ng` is caller-supplied and never checked against `cfg.ghostCells()` (PPM needs 3, linear 2). With PPM+MHD, `sweep()` loads primitives at dim index i-2 = -3 with cross index -1; if ng=2 that makes `Field.cellOffset`'s `@intCast(ix + ngx)` go negative — a safety panic in Debug/ReleaseSafe and out-of-bounds UB in ReleaseFast, deep inside the sweep. (b) `setBcCell` unwraps `self.opt.specific_bc.?` (line 616); configuring `bc_x = .specific` with a null `specific_bc` is a runtime unwrap panic on the first `setBc`. (c) `Options.coords` (runtime, drives the MetricCache) duplicates `cfg.coords` (comptime, used by dynamo/scalars/puffy via `SimT.Cfg.coords` for coco transforms); if they ever diverge the code silently mixes two coordinate systems — wrong physics with no diagnostic. (d) `correct_polaraxis` with `ny <= 2*nccorrectpolar` makes every row 'corrected' and the polar overwrite sources (`iysrc = nc`) lie inside the overwritten band — order-dependent garbage on small test grids (the module explicitly supports reduced ny for tests).

**Fix:** Validate preconditions at the top of Sim.init (sim.zig:317): (1) g.ng >= cfg.ghostCells() — with PPM and ng=2, sweep()'s unconditional i-2 load (sim.zig:1266-1270) drives Field.cellOffset's @intCast negative (panic in safe builds, UB in ReleaseFast); note PPM alone triggers this, MHD only widens the cross range. (2) opt.specific_bc != null whenever any of bc_x/bc_y/bc_z is .specific (avoids the unwrap panic at sim.zig:616). (3) opt.coords == cfg.coords — or better, drop Options.coords entirely and pass cfg.coords to the MetricCache init (sim.zig:333-334), since every other consumer (io/scalars.zig, magn/dynamo.zig, physics/radvisc.zig, problems/puffy.zig) already uses the comptime Cfg.coords; divergence currently produces silently wrong physics. (4) !opt.correct_polaraxis or nyi() > 2*opt.nccorrectpolar — otherwise the polar overwrite sources lie inside the overwritten band and every row is flagged corrected (sim.zig:1600-1638, 476-480). Prefer returning errors (e.g. error.InvalidConfig) over std.debug.assert so ReleaseFast builds are covered too; cost is negligible either way.

### Purity & hidden state

#### `koral/metric/precompute.zig:369` — fillGeometryFace validity depends on the init-time `faces` option — silent NaN geometry when faces=false, and ~137 MB of dead allocation

*Purity & hidden state*

MetricCache.init with `.faces = false` still allocates full-size gb/gconb arrays and memsets them to zero (precompute.zig:154-162). fillGeometryFace on such a cache then returns a Geometry with an all-zero metric, gdet=0, and alpha = sqrt(-1/0) = NaN (line 349) — no assert, no error, just NaN poisoning downstream FP. Whether the function is callable depends on an ambient option chosen at init, not expressed in types and not asserted — category (c)/(e) of the purity hunt, and directly analogous to the C global-state surprises the author is trying to eliminate. The dead allocation is also material: for the PUFFY 384x360 grid the three face arrays x2 are about 429k faces x 20 x 8 B x 2 = ~137 MB zeroed and never used in the faces=false (postprocessing) mode.

**Fix:** Make gb/gconb `[3]?[]f64`: set null and skip the allocation entirely when opts.faces is false (saving ~183 MB on the PUFFY 384x360 grid — the z-face array is doubled in 2D by the sz+1 slice), unwrap in fillGeometryFace with `orelse @panic("MetricCache built with .faces=false")` (or a debug assert), and free conditionally in deinit. Note faces=false currently has no callers in the repo, so this is hardening the planned postprocessing mode, not fixing a live bug; the faces=true path and all FP shapes are untouched, so golden fidelity is unaffected.

#### `koral/physics/radvisc.zig:61` — calcShearLab mixes an explicit center-cell pp0 with neighbour prims read ambiently from sim.p, and the radvisc chain carries redundant (ix,iy,iz) alongside geom

*Purity & hidden state*

calcShearLab takes pp0 for the center cell but loads the ±1 neighbours from sim.p (lines 112-113) — the result depends on ambient sim.p state that the signature hides behind pp0, which suggests the caller controls the input. Today every caller (calcRijVisc ← calcRijViscTotal, line 355-357) passes exactly sim.p's values, but the API permits passing a trial/modified pp0 (e.g. from a future implicit or fixup path), which would silently produce a stencil mixing trial center with stale neighbours — a C calc_shear_lab quirk (rad.c reads global arrays) transcribed into a Zig signature that no longer signals it. Related instance of consistency-by-convention in the same chain: calcRadVisccoeff (line 236), calcRadShearviscosity (:291) and calcRijVisc (:310) all take both (ix,iy,iz) and a *const Geometry, yet Geometry already carries .ix/.iy/.iz (precompute.zig:330-333); a mismatched pair compiles fine and reads the wrong cell's grid spacing or Christoffels. Reentrancy itself is fine — sim.cache.fillGeometry/kr are *const reads of precomputed arrays — so the only ambient mutable input is sim.p.

**Fix:** Two golden-safe signature fixes, neither touching any FP expression: (1) drop pp0 from calcShearLab and load the center prims from sim.p internally — every caller (calcRadShearviscosity:302 via calcRijVisc:321 via calcRijViscTotal:355-357, plus golden_visc_test.zig:108-110) already passes exactly sim.p's values, and calcRijVisc keeps its own pp for erad/calcChi; or, if a divergent pp0 must stay possible, add a debug assert that pp0 matches sim.p at (ix,iy,iz) so a trial-center/stale-neighbour stencil cannot arise silently. (2) Drop the (ix,iy,iz) parameters from calcRadVisccoeff/calcRadShearviscosity/calcRijVisc and read geom.ix/geom.iy/geom.iz (set by fillGeometry, precompute.zig:330-332), making geometry-position consistency structural — this also mirrors C more closely, since C's calc_rad_visccoeff extracts ix/iy/iz from geom->ix rather than taking separate index arguments.

#### `koral/sim.zig:195` — SpecificBc callback has no context parameter, forcing the only module-level var in the repo

*Purity & hidden state*

SpecificBc is a bare function pointer `*const fn (sim, ix, iy, iz, t, ifinit, face) [NV]f64` with no user-context slot. Any boundary condition that needs runtime parameters cannot be expressed without global state — and this has already happened: koral/radtube_tests.zig:52 declares `var active_tube: *const tubes.Tube = undefined;` (the single module-level mutable var in ~19k lines) purely so tubeBc can see which tube is running, with `runTube` assigning it before each run. That is a hidden must-set-before-use dependency, and it makes concurrent tube runs (e.g., a future parallel test runner) silently read each other's tube. PUFFY's Bc gets away with comptime constants only by luck of its problem definition.

**Fix:** Add a context slot to the BC interface: `bc_ctx: ?*const anyopaque = null` in Options and `SpecificBc = *const fn (ctx: ?*const anyopaque, sim: *const Self, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: BcFace) [NV]f64`, with setBcCell (sim.zig:616) passing `self.opt.bc_ctx` (alternatively, make specific_bc a two-field struct {ctx, call}). This fixes TWO existing global-state workarounds, not one: radtube_tests.zig:52 (`var active_tube`, the repo's only file-scope mutable var) passes `tube` as ctx and recovers it via `@ptrCast(@alignCast(ctx.?))`; evolution_tests.zig:622-624 (BondiCtx's struct-scoped `var michel`/`var rc`, set at line 676) passes `&michel` — its Michel solution is computed at runtime, so a comptime-bound BC cannot express it. puffy.Bc and the other comptime-constant BCs simply ignore ctx. Purely Zig-native API surface; no C-shape or golden fidelity involvement.

#### `koral/sim.zig:263` — ptm1 and ppostimplicit are write-only fields — dead state allocated and copied every step

*Purity & hidden state*

A repo-wide grep shows ptm1 and ppostimplicit are only ever written: allocated in init (line 340-341), copied into four times per step (lines 1695, 1697, 1717, 1720), and never read anywhere. The module header (lines 27-28) explicitly says upreexplicit/ppreexplicit copies were *skipped* because "nothing reads them before M12" — the same reasoning applies to these two, but they were kept. Cost: two full ghost-inclusive fields (~15 MB each at production 384×360×13) plus ~4 full-array memcpys (~60 MB traffic) per step, and — worse for maintainability — they look like live coupling state to a reader tracing the RK2IMEX data flow ("what reads the post-implicit primitives?" has the answer "nothing", which takes a whole-repo search to establish).

**Fix:** Delete both fields, their init/deinit list entries, and the four copyFull calls, leaving a one-line comment at the two step() sites noting that C (problem.c) copies p into ptm1/ppostimplicit here and that the copies are dead on this path (the Zig implicit solver reads live p directly; FORCEUEQPINIMPLICIT regeneration happens inside solveImplicitLab from its pp argument). Also update docs/ARCHITECTURE.md 5.2/5.3: the ptm1 row (line 457) currently claims it is "handed to the implicit solver", which is untrue of the Zig code — nothing reads it, and at the 2nd-implicit site it is snapshotted before doCorrect() mutates p, so it does not even equal the solver's input. If you prefer keeping the buffers for line-by-line diffability against problem.c, at minimum mark both declarations "// write-only: kept to mirror problem.c; no Zig consumer" so readers tracing the RK2IMEX data flow do not hunt for a consumer.

#### `koral/sim.zig:991` — calcU2p's boundary refresh depends on ambient self.time set only inside step()

*Purity & hidden state*

calcU2p ends with `try self.setBc(self.time, false)` — its result depends on a field that only step() assigns (line 1671, `self.time = t`). The dependency is invisible in the signature, and the surrounding API actively misleads: opExplicit(t, dtin) accepts t and immediately discards it (`_ = t;` line 1403) while the BC refresh buried inside its calcU2p call uses self.time instead. A standalone caller (a test driving opExplicit or calcU2p directly, or dynamo.applyDynamo which already receives an explicit t but then calls sim.calcU2p()) gets ghost cells evaluated at whatever time the last step() froze — for a time-dependent specific BC this produces silently wrong ghosts that look like a physics bug. PUFFY's Bc ignores t, so nothing catches it today.

**Fix:** Give calcU2p an explicit time parameter (`pub fn calcU2p(self: *Self, t: f64)`) and thread it from the four step() call sites (sim.zig:1702/1716/1725/1738), opExplicit (sim.zig:1448, making its currently-discarded t parameter genuinely used), and dynamo.applyDynamo (dynamo.zig:274, which already has t in scope and already passes it to its own setBc call two lines earlier). Keep `self.time = t; // C: global_time = t` in step() as the C-mirroring record — after the change it becomes write-only documentation. No FP change: the identical value is passed on every existing path. This is signature-only; C's calc_u2p signature already differs in the port, so C-diffability is unaffected.

#### `koral/sim.zig:1673` — step() silently produces dt=inf unless initTimestepGuess was called — ordering dependency not asserted

*Purity & hidden state*

step() computes `self.own_dt = 1.0 / self.tstepdenmax` and the PUFFY driver (PROBLEMS/puffy/main.zig:176) independently computes `dt = 1.0 / s.tstepdenmax`. tstepdenmax is 0 after Sim.init, and nothing asserts it was ever seeded: finishInit() bundles setBc+initTimestepGuess, but puffy.initAll() does NOT call it — main.zig must remember the extra `s.initTimestepGuess()` on line 156. Forgetting it (or writing a new problem driver from initAll's example) gives dt = inf on step 1 → NaNs across the grid → a wasted run diagnosed only by the n_nan output check. There are also two sources of truth for the CFL dt (step's own_dt vs. the driver's recomputation), which can drift apart if one changes.

**Fix:** Two small changes, neither affecting any evolved value: (1) at the top of step(), assert the CFL denominator was seeded — `std.debug.assert(forced_dt != null or self.tstepdenmax > 0);` (the forced_dt escape keeps always-forced drivers legal; an unconditional assert is also safe for every current caller) — converting the silent inf/NaN first step into an immediate named failure; (2) unify the duplicated CFL expression by exposing `pub fn cflDt(self: *const Self) f64 { return 1.0 / self.tstepdenmax; }` and using it both in step() (line 1673) and the puffy driver (PROBLEMS/puffy/main.zig:176), and/or fold initTimestepGuess into initAll (it is part of C's solve preamble, and main.zig's own comment already groups them as "ko.c:140-263 + the solve preamble dt guess") so new problem drivers cannot forget the seeding step.

### Hot path & inlining

#### `koral/magn/dynamo.zig:106` — Dynamo and radvisc passes recompute static per-cell BL geometry, coordinate transforms, and Jacobians every sub-step

*Hot path & inlining*

The grid and metric are time-independent, yet per-cell quantities derived purely from cell coordinates are recomputed on every pass: (1) geomBLat (dynamo.zig:106-116) calls precompute.geometryAt -> metric.compute(.bl), which evaluates the full dual-number Kerr-BL metric including Christoffels and dlgdet that Geometry then discards — hundreds of flops per cell, invoked per cell per mimicDynamo pass (twice per step over domain + ring) via fieldAngle (dynamo.zig:123); (2) fieldAngle -> transPmhdCoco (dynamo.zig:126) computes two coco.dxdx Jacobians per cell per pass, although MetricCache already has a per-cell Jacobian store (dxdx_my2out) — just not for the BL target; (3) coco.cocoN (exp/trig) per cell per pass at dynamo.zig:93 and :182 (calcScaleHeight needs only theta_BL, mimicDynamo only r/theta_BL) and per cell per step at radvisc.zig:274 (calcRadVisccoeff needs only r_BL, and the stepFunction/rmin weight there is fully static per cell); (4) ct.calcBfromACore (ct.zig:277) and calcDivB (ct.zig:330 and the pB helper) build a full ~400-byte Geometry (40 loads + sqrt) per access just to read gdet.

**Fix:** Precompute once (in MetricCache or a dynamo-owned sidecar at init): per-cell xxbl (BL r, theta), the BL Geometry blocks, the MYCOORDS<->BL Jacobians, and the static radial weights (radvisc's stepFunction(r-rmin) factor, dynamo's facradius). Note MetricCache.dxdx_my2out/out2my are currently filled with identity matrices (sim.zig:334 sets out_coords = opt.coords) and never read — pointing out_coords at .bl (C's usual OUTCOORDS) makes the existing store directly usable for transPmhdCoco. Add a cheap MetricCache.gdet(ix,iy,iz) accessor (self.g[cellIndex*20 + 19]) for calcBfromACore (ct.zig:277, per-sub-step hot path) and calcDivB (ct.zig:300/330, diagnostic-only polish). Two corrections to the detail: dlgdet is retained in Geometry.gg[i][4] (only the Christoffels are computed and discarded), and calcDivB is called only from tests, so that part is minor. All cached values are bit-identical to the recomputed ones, so golden-file agreement is unaffected.

#### `koral/metric/precompute.zig:323` — MetricCache.kr recomputes the full 3D cell index on every one of 128 calls per cell in metricSource

*Hot path & inlining*

kr(i,j,k,ix,iy,iz) calls cellIndex (3 signed->unsigned casts, an assert, 3 multiplies/adds) and then does i*16+j*4+k for every scalar fetched. metricSource (sim.zig:1166-1176) calls it 64 times per cell for the hydro rows plus 64 more for the radiation rows, per domain cell, per explicit op (2 per step) — order 10^7 redundant cellIndex evaluations per step on PUFFY grids, none of which LLVM is guaranteed to CSE through the accessor in Debug/ReleaseSafe. calcShearLab in physics/radvisc.zig:184-188 has the same pattern (~80 kr calls per cell, once per step over domain+ring). This is pure Zig-native accessor design; no C FP expression shape is involved (the multiply order gdetu*t[k][l]*Kr stays untouched).

**Fix:** Add a per-cell block accessor to MetricCache, e.g. `pub inline fn krBlock(self: *const MetricCache, ix: i64, iy: i64, iz: i64) *const [64]f64 { return self.kris[self.cellIndex(ix, iy, iz) * 64 ..][0..64]; }`. In metricSource (sim.zig:1163-1178) fetch the block once before the k/l/nu loops and index it with `kr_blk[l * 16 + nu * 4 + k]`; same in calcShearLab (radvisc.zig:181-190) with `[k*16 + i*4 + j]` / `[i*16 + i*4 + k]`. Mark cellIndex/faceIndex/kr `inline fn` so Debug/ReleaseSafe builds also skip the per-call recomputation. Keep kr() itself as the convenience accessor for the test call sites (metric/tests.zig:429,438). Fetched values are bit-identical, so golden agreement and the C-mirrored multiply order are unaffected.

#### `koral/physics/flux.zig:42` — fFluxPrime computes the gas four-velocity and magnetic four-vector twice per call

*Hot path & inlining*

hydro.calcTij (called at flux.zig:34) internally runs relele.uconUcovFromPrims + mhd.bconBcovBsqFrom4vel on the same pp/geom, then fFluxPrime immediately recomputes the identical relele.convVelsBoth(vcon,.velr,...) at line 42 and bconBcovBsqFrom4vel at lines 48-53. convVelsBoth alone is a 3x3 qsq contraction + sqrt + a 4x4 index-lowering; bcon/bcov/bsq is another ~30 flops + 4x4 lowering. This runs twice per face side, i.e. 4x per cell per active dim per explicit op — one of the hottest call sites in the code. Because the duplicated work is the *same expression evaluated on the same inputs*, reusing the first result is bit-for-bit identical: no C-mirrored FP shape changes, only the redundant second evaluation disappears. Same pattern one level up: checkFloorsMhd (solve/invert.zig:444-453) computes ConCov+bcon for the same face state immediately before fFluxPrime in the sweep (sim.zig:1282-1283), and p2u.calcUtp1 (p2u.zig:23-30) re-derives the qsq that convVelsBoth->calcAlpgam already computed.

**Fix:** Add a calcTij variant that accepts (or returns) a precomputed relele.ConCov + mhd.Bfield4; in fFluxPrime compute u and b once at the top and pass them down. Keep the existing calcTij signature for its five other callers (sim.zig:1155, io/scalars.zig:166, tests). Since convVelsBoth and bconBcovBsqFrom4vel are pure, the result is bit-identical — verify with the existing golden flux tests. Optionally extend the same threading to checkFloorsMhd->fFluxPrime in the sweep and to calcUtp1's qsq, but note checkFloorsMhd rewrites vx/vy/vz when the magnetization floor fires (invert.zig:498-499), so its cached u/b must be invalidated on that path (its currently-discarded return code signals it).

#### `koral/physics/radforce.zig:264` — Dead transcendental work in every implicit residual evaluation: unused RadState/opacity fields and a discarded Lorentz boost

*Hot path & inlining*

fillRadState + calcGiFromState run once per residual evaluation of the 4-prim Newton (base + 4 FD-Jacobian perturbations + every damping re-apply, so ~6-10x per Newton iteration per cell per opImplicit pass, plus every f1dErr call of the 1-D bisection). A significant fraction of each call computes values no solver-path consumer ever reads, and all of it can be skipped bit-identically: (1) radforce.zig:134 computes sgas via hydro.sFromU (one @log + two std.math.pow) — residual() only reads st0.sgas from the once-per-rung state00; the per-eval st.sgas is never read (the transcribed C bug at implicit.zig:339 reads st.tgas instead); (2) opacities.zig:129-134 (rad_num: two cbrt + divide) and opacities.zig:142-144 (gas_ross: two pow + exp + divide) fill Opac channels that calcGiFromState and the residual never touch (only gas_abs, rad_abs, rad_ross, kappaes are read); (3) radforce.zig:264 performs the full lab->ff Lorentz boost (lorentzLab2Ff = convVelsBoth + normalObsConCov + omega/omega^2 build + multiply2, ~200 flops + 3 sqrt), yet gi_ff[0] is immediately overwritten by the exact fluid-frame expression on line 265 and gi_ff[1..3] are read only by tests/golden comparisons (opacity_tests.zig, golden_opac_test.zig) — never by implicit.zig's residual() or f1dErr(); (4) inside that boost, frames.zig:68-72 recomputes the fluid u^mu/u_mu that fillRadState already stored in st.ucon/st.ucov, and frames.zig:63/89 recompute alpha = sqrt(-1/GG[0][0]) although geom.alpha holds the bit-identical value. Roughly 8 of ~22 transcendentals plus ~250 flops per evaluation are pure waste on the single highest-call-count path in the code.

**Fix:** Add comptime-slimmed variants used only by the solver path, keeping the existing full functions for the golden/state tests (golden_opac_test.zig reads gas_num/rad_num/gas_ross and gi.ff[1..3], so the full paths must remain): a fillRadState(comptime full: bool) that skips sgas when full == false, a gated calcOpacitiesFromState that skips the rad_num (opacities.zig:129-134) and gas_ross (142-144, incl. zbg) channels, and a slim calcGiFromState that returns gi_lab plus the directly-computed ff[0] (radforce.zig:265), skipping the lab->ff boost at line 264 entirely and taking st.ucon/ucov and geom.alpha instead of rebuilding them from vprim. All consumed outputs keep their exact expressions, so golden files are unaffected, and golden_implicit_test/golden_puffystep_test exercise the slim path end-to-end. Expected win is roughly 15-25% of the residual-evaluation inner loop (highest in the f1dErr bisection, which has no constraint-solve overhead); worth a quick profile of opImplicit before/after to confirm.

#### `koral/physics/radforce.zig:320` — calcChi runs the full fillRadState (Rij, ehat, tradbb, all six opacity channels) although chi needs none of the radiation-frame state

*Hot path & inlining*

calcChi = calcKappa + calcKappaes is called per cell (including a ghost layer) in calcWavespeeds (sim.zig:824, twice per step via both opExplicit calls) and per cell in calcRadVisccoeff (radvisc.zig:249, once per step via calcRijViscTotal). calcKappa (radforce.zig:294-303) performs the entire fillRadState: urfCon + calcRij (16 components + sqrt), ehat contraction, lteTfromE (2 sqrt), the state-path kappaes with its Klein-Nishina pow(tkn/4.5e8, 0.86), sgas (log + 2 pow), and all six opacity channels. But kappa = opac.gas_abs depends only on (rho, te, ne, bsq) — no trad/tradbb/Rij term enters kappagasff or kappagassyn — and the standalone calcKappaes (radforce.zig:307-317) uses trad = te, recomputing tempsFromUrho a second time. So the Rij/ehat/tradbb machinery, the state-path kappaes, and the four trad-dependent opacity channels (rad_abs log1p + zeta pows, rad_ross, rad_num cbrt terms, gas_ross pows/exp) are all dead work for the wavespeed tau-limiter and the viscosity mean-free-path — the two callers that dominate this function's call count.

**Fix:** Add a slim chi path used by the wavespeed tau-limiter (sim.zig:824) and radvisc (radvisc.zig:249): compute tempsFromUrho once, uconUcov + bsq (still needed for kappagassyn's bmagcgs), then only the kappagasff + kappagassyn expressions and the standalone kappaes(te) term. Share the existing formula code with calcOpacitiesFromState behind a comptime channel-subset flag so the FP expression shapes — and hence golden bits — stay identical, and keep the current calcKappa/calcChi as the C-parity entry points for golden_opac_test.zig. This skips urfCon/calcRij, the ehat contraction, lteTfromE, sgas, the state-path Klein-Nishina kappaes, and the four trad-dependent opacity channels (roughly a dozen transcendental calls per cell) in loops that run three times per cell per step.

#### `koral/sim.zig:959` — parallelRows spawns and joins fresh OS threads on every pass (~10 passes per RK2IMEX step) with a static row partition

*Hot path & inlining, Idiomatic Zig — flagged independently by 2 reviewers*

Each call to parallelRows (sim.zig:937-972) does std.Thread.spawn for up to min(nthreads, 64, ny) threads and joins them. A full step() executes ~10 such passes (2x opImplicit + up to 8x calcU2p including the dynamo and final inversions), so at nthreads = 16 that is ~160 thread create/destroy cycles per step (~20-60 us each on macOS/Linux) — several ms/step of fixed overhead that grows with thread count and shrinks the useful grain as grids get smaller. Separately, the contiguous iy-range split assigns equal row counts, but the implicit solver's cost per cell varies by orders of magnitude (Newton iteration count, rung ladder depth) and is spatially concentrated (disk body vs polar funnel in PUFFY), so with a static partition the slowest chunk gates every pass.

**Fix:** Create a persistent worker pool once in Sim.init (std.Thread.Pool, or long-lived threads parked on a condition variable) and dispatch the row ranges of all ~10 parallelRows passes per step to it, instead of spawning/joining fresh OS threads each pass. Note PROBLEMS/puffy/main.zig:9 already documents nthreads as driving "an optional std.Thread.Pool" — so this also makes the code match its own docs, and matches the C original's OpenMP behavior (persistent pool). For the opImplicit pass, replace the static equal-row split with dynamic chunking (atomic row counter, 1-4 rows per grab) so expensive disk-body rows load-balance against cheap funnel rows; the implicit worker cannot error (sim.zig:1546-1549) and the ChunkResult counters are order-independent integer sums, so the result stays bit-identical to the serial path and the threading_tests.zig determinism gate still holds.

#### `koral/sim.zig:1072` — cellFixup performs four full-grid memcpys even when no cell is flagged

*Hot path & inlining, Data-oriented design — flagged independently by 2 reviewers*

cellFixup unconditionally copies u -> u_bak and p -> p_bak (sim.zig:1072-1073) and back (sim.zig:1138-1139) before/after the flagged-cell averaging. With NV = 13 on a PUFFY-sized grid each Field is several MB, so that is ~4 x 6 MB of memory traffic per invocation; calcU2p runs cellFixup(.hd_fixup) on each of its ~6 calls per step plus cellFixup(.radimp_fixup) after each opImplicit, i.e. on the order of 150-250 MB/step of memcpy. In the common case no cell has the fixup flag set and the whole exercise is a no-op.

**Fix:** Add a zero-flagged-cells early-out to cellFixup: either a quick scan of the flag array before the memcpys, or track a flagged-cell count while flags are written (u2pRows/implicitRowsWorker set the flag for every non-polar cell each pass, and parallelRows' ChunkResult already aggregates per-chunk tallies, so a count slots in naturally). This removes ~350 MB/step of no-op memcpy traffic at production PUFFY size (6 hd_fixup invocations x 4 full-field copies of ~15 MB each; the rad_fixup/radimp_fixup variants are already gated off by their default-false options) while preserving the C cell_fixup structure. Optionally, the full-array staging could be replaced by collecting fixed cells into a small (index, pp, uu) list written back after the scan — bit-for-bit equivalent since neighbor/center reads during the scan come from live self.p, never the backup arrays — but the early-out alone captures nearly all the win with minimal divergence from the C shape.

#### `koral/sim.zig:1402` — Entire explicit operator is serial; only the u2p/implicit inversions use parallelRows

*Hot path & inlining*

Per step, parallelRows threads 8 passes (6 calcU2p + 2 opImplicit), but the dominant explicit work is single-threaded: calcWavespeeds (sim.zig:796, per-cell gasWavespeedsLr + calcChi + calcRadWavespeeds), sweep (sim.zig:1220, per cell: NV-wide avg2point + 2x checkFloorsMhd + 2x fFluxPrime + radvisc), fluxesAtFaces (sim.zig:1318, 2x p2u per face + NV-wide flux combine), the metricSource/flux-divergence update loop (sim.zig:1418-1444), calcRijViscTotal, and the stage arithmetic (1459-1491). All of these write disjoint per-cell/per-face slots, exactly like the already-parallel u2pRows, so running them through parallelRows is bit-identical at any thread count. The only shared state is tstepdenmax/tstepdenmin in saveWavespeeds — accumulate per ChunkResult and combine with @max/@min after join (order-independent, still bit-identical). With nthreads>1 on PUFFY-sized grids, Amdahl's law caps speedup hard while the explicit path stays serial.

**Fix:** Route the explicit path through parallelRows as well: calcWavespeeds, the three sweep passes, fluxesAtFaces, the metricSource/flux-divergence update loop, calcRijViscTotal, and stageDeriv/stageCombine. All of them write disjoint per-cell/per-face slots and read only *const geometry and fields not written in the same pass, so the result stays bit-identical at any thread count — the same argument that justifies u2pRows today, already pinned by the threading_tests.zig determinism gate (golden tests keep running at nthreads=1). The one shared accumulator, tstepdenmax/tstepdenmin in saveWavespeeds (sim.zig:904-905), moves into ChunkResult as per-chunk max/min fields combined with @max/@min after the join (order-independent). Two mechanical adaptations: parallelRows currently hands workers domain rows [0, ny), while calcWavespeeds/calcRijViscTotal cover one ghost row per side and wide sweeps use -1..n+1 cross ranges, so give parallelRows an optional row-range (or let the boundary chunks absorb the ghost rows); and for the dim=1 sweep the split falls along the sweep direction itself, which remains race-free (face k receives pb_l from iteration k-1 and pb_r from iteration k — disjoint slots) but needs its own worker shape rather than reusing the iy-row one.

#### `koral/sim.zig:1430` — Per-component Field/FaceStore get/set loops recompute the cell offset NV times per cell

*Hot path & inlining*

Several domain-wide inner loops fetch/store one variable at a time, re-running the full 3D offset math (casts + assert + 3 mul/add + bounds-checked index) per component instead of using the codebase's own load/store-into-[NV]f64 convention: (1) the conserved update in opExplicit (sim.zig:1430-1441) does 6 FaceStore.get + 1 Field.get + 1 Field.set per iv — ~8*NV offset computations per cell where 8 per cell suffice; (2) stageDeriv (1459-1474) and stageCombine (1477-1491) do 3 offset computations per iv, and run 7 times per step over the domain; (3) the flux combine in fluxesAtFaces (1381-1394) does fl_l.get + fl_r.get + flb.set per iv on top of 12 scGet calls; (4) fluxCt's EMF loops (magn/ct.zig:60-127) are the same pattern at lower weight. In ReleaseFast LLVM can usually hoist the pure offset out of the iv loop, but in Debug (default build, where the golden tests run) and ReleaseSafe every call re-derives it and re-checks bounds — these loops are a large fraction of test wall time.

**Fix:** In the sim.zig loops, load each accessed cell/face into stack [NV]f64 buffers once per cell via the existing Field/FaceStore .load/.store (the convention field.zig's header already mandates for per-cell kernels, and which sweep/u2pRows/fluxesAtFaces' pb_l/pb_r already use) and index the buffers in the iv loop: opExplicit's update needs 6 face loads + u.load + u.store per cell instead of ~8*NV offset computations; stageDeriv/stageCombine (7 domain-wide passes per step) become load/load(/load)/compute/store with unchanged per-component arithmetic, so golden results are bit-identical. Do not use raw data[off..][0..NV] slices — that violates field.zig's documented "never raw storage pointers" AoSoA-flexibility convention. Leave ct.zig's EMF gather as-is: it reads single components from flPin-varied neighbors, so whole-cell loads would not help there.

#### `koral/solve/implicit.zig:526` — solveImplicitLab redoes the state00 fill and the entire 1-D bisection identically for every ladder rung

*Hot path & inlining*

solve4dPrim (implicit.zig:526-529) computes state00 = fillRadState(pp00) and, with PUFFY's start_with_bisect, runs solve1dPrim — a bracket search (up to 50 iterations) plus bisection to 1e-3, each iteration being one f1dErr = fillRadState + calcGiFromState (the two most expensive per-cell kernels). None of its inputs (pp00, state00, geom, dt, gam, opp) depend on the rung's whichprim/whicheq/whichframe, so all six rungs of solveImplicitLab (implicit.zig:745-747) recompute a bit-identical state00 and bisection result. Every cell that fails rung 0 — exactly the cells that already dominate step time — pays ~10-15 redundant fillRadState+calcGi evaluations per extra rung attempted, and a cell whose state00 fill errors walks all 6 rungs to fail identically each time.

**Fix:** Hoist state00 = fillRadState(pp00) and the bisect-adjusted starting guess pp0 out of solve4dPrim into solveImplicitLab, computing them once before the rung loop and passing them in (e.g. as *const RadState and *const [NV]f64 parameters); early-out of the ladder entirely when the state00 fill fails, since all six rungs would fail identically at that point. Both values are pure functions of inputs unchanged across rungs, so results are bit-identical and golden-safe. Note solve4dPrim is also called directly with explicit rung parameters from koral/implicit_tests.zig:109 and :329-330 — either update those call sites to precompute the two values or keep the existing signature as a thin wrapper around the new inner function.

### Data-oriented design

#### `koral/metric/precompute.zig:149` — dxdx_my2out/dxdx_out2my are precomputed and stored for every cell but never read anywhere

*Data-oriented design*

MetricCache allocates nc*16 f64 for each Jacobian array (precompute.zig:149-150) and fillCell fills them with two coco.dxdx calls plus a cocoN call per cell (precompute.zig:269-279). A repo-wide grep shows no accessor and no reader — not in sim.zig, io, or tests. That is 256 B/cell (~36.5 MB on the PUFFY grid) of dead storage plus nontrivial init-time transcendental work. Worse, Sim.init hardcodes out_coords = opt.coords (sim.zig:334), so in the only production use these arrays hold identity matrices. Related in the same file: when opts.faces == false ('postprocessing mode') the six face arrays are still fully allocated and memset (precompute.zig:154-162) — ~137 MB of zeros on a PUFFY-sized grid.

**Fix:** Drop dxdx_my2out/dxdx_out2my (and the out_coords InitOpts field) until a consumer (coordinate-transformed output) lands, or allocate/fill them only when out_coords != coords. They are write-only today — no accessor, no reader anywhere — and in the sole production call (sim.zig:334 passes out_coords = opt.coords) coco.dxdx/cocoN short-circuit, so the arrays hold identity matrices costing 256 B/cell (~36.5 MB on the 384x360 PUFFY grid). Secondary polish: the faces=false branch (precompute.zig:158-161) still allocates and zeroes all six face arrays (~183 MB on that grid); leave gb/gconb as len-0 slices with a debug assert in fillGeometryFace — noting that no caller currently passes faces=false, so this is preventive rather than a live cost.

#### `koral/physics/flux.zig:35` — Hot functions compute values no caller ever consumes: full-tensor lowering for one row, unconditional ff-boost of Gi, eager RadState.sgas, dead Qdotnp chain

*Data-oriented design*

Several per-cell hot functions do work whose results are provably never read (golden-safe to drop — unconsumed values cannot shape compared outputs): (1) fFluxPrime lowers both full 4x4 tensors (indices2221 = 64 madds each, lines 35 and 88) but consumes only row idim+1 — 3 of 16 gas entries (t[idim+1][1..3], lines 74-76; the energy row is assembled separately) and 4 of 16 rad entries (line 90). calcTij and calcRij likewise build 16 entries where only row idim+1 is needed here. Every entry of these tensors is computed independently, so row-restricted variants are bit-identical; roughly 200-250 flops (~25-30%) of every face-flux call is dead work, twice per face per dim per RK stage. (2) koral/physics/radforce.zig:264-265: calcGiFromState always performs the lab->ff boost (lorentzLab2Ff = one convVelsBoth + two 4x4 matrix builds + 64-madd contraction + multiply2), but the implicit residual's lab-frame rungs — the FIRST rung tried, i.e. the common success path — consume only gi_pair.lab (implicit.zig:284-285; giff is read only in the whichframe==.ff branches at :327-341). (3) koral/physics/radforce.zig:134: fillRadState eagerly computes sgas via hydro.sFromU (2 pow + 1 log) on every fill; it is consumed only by the rarely-reached entropy rungs (implicit.zig:338), yet fillRadState runs ~5x per Newton iteration plus once per cell in the wavespeed loop. (4) koral/solve/invert.zig:194-200 and :235-238: Qcovp, Qconp (a 16-madd indices12), betasqoalphasq, Dfactor and Qdotnp exist solely to fill cons[6], which neither fU2pHot nor fU2pEntropy reads (they use cons[0..5]; U2P_EQS_NOBLE is hardwired per the file header) — ~30 dead flops including a division on every u2pSolverW call, which the implicit constraint loop invokes repeatedly.

**Fix:** Four golden-safe dead-work eliminations in hot paths (unconsumed values cannot shape compared outputs): (1) Add a row-restricted lowering indices2221Row(t1, gg, row) -> [4]f64 plus row variants of calcTij/calcRij for fFluxPrime (koral/physics/flux.zig:35/:88) — each tensor entry is computed independently, so results are bit-identical; keep the full-tensor versions for metricSource/fillRadState/scalars. (2) In calcGiFromState (koral/physics/radforce.zig:264), the lab->ff boost result is entirely dead in the implicit residual: the only ff component the residual and f1dErr ever read is giff[0], which line 265 overwrites with the exact fluid-frame expression independent of the boost. Split into a lab+ff0 fast path used by implicit.zig, keeping the full boosted Gi for the golden/opacity tests (mirrors C's type selector in calc_all_Gi_with_state). (3) Make RadState.sgas lazy — compute it in the entropy-rung branch (implicit.zig:338) from st0.rho/st0.uint with the same sFromU expression (same bits); this drops 2 pow + 1 log from every fillRadState in the Newton loop and the wavespeed path. (4) In u2pSolverW (koral/solve/invert.zig:194-200, 235-238), Qcovp/Qconp/betasqoalphasq/Dfactor/Qdotnp feed only cons[6], which neither fU2pHot nor fU2pEntropy reads; guard the block behind the (absent) U2P_EQS_JON path with a comment — preserving C diffability — or delete it.

#### `koral/physics/flux.zig:42` — Hot per-cell call chains re-derive bit-identical kinematic state (4-velocity, b-field, Rij) up to 3x because APIs cannot accept precomputed pieces

*Data-oriented design*

The deep per-cell functions each re-derive the same transient values from pp+geom instead of sharing them, and since they are pure functions of identical inputs every recomputation is bit-identical (the golden/ulp contract is NOT at stake — only Zig-native signatures change). Occurrences: (1) fFluxPrime: calcTij (koral/physics/hydro.zig:45-60) internally runs uconUcovFromPrims + bconBcovBsqFrom4vel; flux.zig:42 then redoes convVelsBoth on the same vcon, :48 redoes bconBcovBsqFrom4vel, :62 calcUtp1 re-derives the same 9-term qsq that convVelsCore/calcAlpgam already computed, and :87 calcRij re-converts the rad velocity. Upstream, sim.zig:1282/1294 checkFloorsMhd (koral/solve/invert.zig:444-453) already computed the identical u and b for the same pl/pr+geom (reusable whenever no floor fires). Net: the gas 4-velocity is solved 3x and the b four-vector built 3x per face-side flux evaluation. (2) koral/physics/radforce.zig:264: calcGiFromState receives st but boost2Lab2Ff(vprim) -> lorentzLab2Ff -> convVelsBoth recomputes exactly st.ucon/st.ucov (fillRadState:137 derived them from the same vprim/geom); lorentzFromPair could take st.ucon/st.ucov directly. (3) sim.zig calcWavespeeds loop (:824-835): calcChi triggers a full fillRadState (gas + rad conversions inside), then gasWavespeedsLr (wavespeeds.zig:94) reconverts the gas velocity and calcRadWavespeeds (radiation.zig:144) reconverts the rad velocity — per cell, per stage. (4) koral/solve/implicit.zig:726/728/526: solveImplicitLab runs p2u(pp00) (conversion), calcFfRtt(pp00) (conversion + full Rij build + lowering), then solve4dPrim's fillRadState(pp00) repeats the conversion and the Rij build — 3 conversions and 2 Rij tensors of identical values per cell entry, before the Newton loop even starts. (5) implicit.zig:201 vs :210: applyConstraints computes uconUcovFromPrims, then p2uMhd redoes convVelsBoth on the same untouched velocity slots (only rho/entr changed in between). Each convVelsBoth costs a sqrt + ~40 flops + a 16-madd indices21; in fFluxPrime the duplicates are roughly 20-30% of the call, and fillRadState/calcGiFromState pairs run ~5x per Newton iteration in the implicit solver — the dominant cost center of a rad-MHD step.

**Fix:** Extend the existing fillRadState/calcGiFromState state-passing pattern so hot per-cell paths stop re-deriving bit-identical values: (a) add calcTij/fFluxPrime variants accepting precomputed relele.ConCov + b-field (and reuse them from the sweep when checkFloorsMhd fired no floor); (b) let calcGiFromState build the boost via lorentzFromPair(st.ucon, st.ucov, w...) instead of reconverting vprim — this alone removes one convVelsBoth per residual call (~5x per Newton iteration); (c) add calcRijFromUrf and reuse one urf per cell; (d) in the wavespeed loop fill RadState once per cell and feed gasWavespeedsLr from st.ucon/bsq, chi as st.kappa + the standalone Te-flavored calcKappaes(pp) (the TradBB-flavored st.kappaes is a different quantity), and calcRadWavespeeds from the shared urf; (e) in solveImplicitLab/solve4dPrim, share the ucon/ucov pair and the calcRij tensor between p2u, calcFfRtt, and fillRadState — but do NOT substitute state00.ehat for calcFfRtt's value: the two contract Rij in different FP shapes (raise-then-contract vs lower-then-contract) and the result decides the startwith rung at implicit.zig:731, so keep calcFfRtt's contraction while passing it the precomputed u and rij. All reused values are bit-identical outputs of the same functions, so golden tests pin the refactor.

#### `koral/sim.zig:1234` — y- and z-sweeps iterate the strided direction innermost; x is never the inner loop for dim!=0

*Data-oriented design*

In sweep() (sim.zig:1230-1235) the loop nest is c1 (cross[1]) -> c0 (cross[0]) -> i (sweep dim innermost). fluxesAtFaces() (sim.zig:1331-1336) has the identical nest. Field storage is iv-fastest then x (field.zig:41), so for dim=0 the inner loop streams contiguously, but for dim=1 the inner loop advances iy, striding sx()*NV*8 = 390*13*8 = 40,560 B per iteration on the PUFFY grid (each 104 B cell record ~2.5 16KB pages from the next). The five-point stencil loads, the pb_l/pb_r/fl_l/fl_r stores (FaceStore is also x-fastest, sim.zig:152), and the scal/flb accesses in fluxesAtFaces all take that stride, so the y-pass runs out of L2 with poor prefetch and ~60% cache-line utilization instead of streaming from L1. For a 3D problem the z-sweep inner loop would stride sx()*sy()*NV*8 (~15 MB) — every stencil load a TLB+DRAM miss, a genuine cliff (latent: current targets are 2D). Because every i-iteration of both loops is independent (loads from p/scal, stores to disjoint face slots; no carried state), interchanging the loops so the contiguous cross-direction (x) is innermost for dim=1/2 sweeps is bit-identical — no C-diffability cost.

**Fix:** For dim != 0 in sweep() and fluxesAtFaces(), interchange the loops so the x cross-coordinate is innermost (dim=1: c1=iz outer, i=iy middle, c0=ix inner; dim=2 analogous). All per-iteration inputs (p, scal, geometry cache) are read-only during the pass and stores go to disjoint slots of distinct face arrays (pb_r/fl_r at face i, pb_l/fl_l at face i+1, flb at cf), so traversal order is bit-identical — golden agreement is unaffected. After interchange the 5-point stencil becomes five contiguous row streams and the dx5/grid.size values become inner-loop invariants. Add a comment stating the iterations are order-independent so the traversal order stays free.

### Idiomatic Zig

#### `koral/io/scalars.zig:44` — Three near-identical copies of the "BL geometry at a cell center" helper across modules

*Idiomatic Zig, Naming — flagged independently by 2 reviewers*

The same 8-line computation — build `[4]f64{0, xc, yc, zc}`, `coco.cocoN(xx, coords, .bl, mp)`, `precompute.geometryAt(.bl, mp, xxbl)`, stamp ix/iy/iz — exists as `blGeom` (koral/io/scalars.zig:44-52), `geomBLat` (koral/magn/dynamo.zig:106-116), and `fillGeometryBL` (koral/problems/puffy.zig:400-408). scalars.zig's own comment admits the duplication ("Mirrors puffy.fillGeometryBL but pulls ... from the sim options"). If OUTCOORDS handling ever changes (e.g. a problem where OUTCOORDS != BL, or geometryAt gains a parameter), three call sites in three modules must be found and edited identically; a divergence here would silently skew only some diagnostics (dynamo weighting vs luminosity vs BCs).

**Fix:** Extract one pub helper in koral/metric/precompute.zig next to geometryAt, e.g. `pub fn geometryAtBL(g: *const Grid, coords: config.Coords, mp: MetricParams, ix: i64, iy: i64, iz: i64) Geometry` — the file already imports Grid/coco/config (no new imports) and applyKrisCorrection there already uses the same (coords, mp, *const Grid, ix, iy, iz) shape. Have scalars.blGeom and dynamo.geomBLat call it with (&sim.grid, SimT.Cfg.coords, sim.opt.mp, ...), and puffy.fillGeometryBL with its module-level mp; keep puffy's pub wrapper as a thin forwarding alias if tests depend on its signature.

**Fixed (2026-07-06):** Added one `pub fn geometryBLat` in `metric/precompute.zig`; `blGeom` (scalars), `geomBLat` (dynamo), and `fillGeometryBL` (puffy) now delegate to it, so the reduction has a single implementation.

#### `koral/metric/precompute.zig:191` — Ghost-offset index arithmetic duplicated between MetricCache.cellIndex and Field.cellOffset — hoist to Grid

*Idiomatic Zig*

MetricCache.cellIndex (precompute.zig:191-198) and Field.cellOffset (koral/field.zig:35-42) implement the identical signed-index → padded-storage mapping, including the same triple `@intCast(ix + @as(i64, @intCast(g.ngx)))` conversion dance and the same bounds asserts; faceIndex (precompute.zig:202-222) repeats the conversions a third time. Both depend only on Grid data. The duplication has already produced a smell downstream: sim.zig:441 recovers a cell index by dividing a Field offset by NV (`(self.p.cellOffset(ix, iy, iz) / NV) * n_flags`).

**Fix:** Add `pub fn cellIndex(g: Grid, ix: i64, iy: i64, iz: i64) usize` (and optionally a `toPadded(i, ng) usize` helper for the per-axis conversion) to Grid. Field.cellOffset becomes `self.grid.cellIndex(ix,iy,iz) * NV`, MetricCache.cellIndex delegates, and sim.zig's flag index becomes `grid.cellIndex(...) * n_flags` with no division. One place owns the tricky signed/unsigned conversion and its asserts.

### Naming

#### `koral/io/scalars.zig:216` — max(pmag/ptot) is called "magnetization", "beta", and "beta-inverse" in three modules
**Fixed (2026-07-06):** renamed `maxMagnetization` -> `maxPmagPtot` and `ScalarRow.maxbeta_inv` -> `max_pmag_ptot`.

*Naming*

The same physical quantity — the domain maximum of pmag/ptot — carries three contradictory names: `maxMagnetization` (scalars.zig:216), `ScalarRow.maxbeta_inv` "max pmag/ptot (β⁻¹)" (koral/io/dump.zig:72), and `maxbeta`/`maxb` in postinit where MAXBETA=1/20 IS the pmag/ptot bound (koral/problems/puffy.zig:60,575,607). So one module says this ratio is β, another says it is β⁻¹, and the function name says "magnetization", which in GRMHD conventionally means σ = b²/ρ — actively misleading for a physicist reader. The scalars.dat header already uses the unambiguous term: "maxPmag/Ptot".

**Fix:** Standardize on the scalars.dat header's unambiguous term: rename `maxMagnetization` → `maxPmagPtot` (koral/io/scalars.zig:216 and module doc line 15; callers at PROBLEMS/puffy/main.zig:92 and koral/scalars_tests.zig:80,88) and `ScalarRow.maxbeta_inv` → `max_pmag_ptot` (koral/io/dump.zig:72,88; uses at PROBLEMS/puffy/main.zig:103,191). Keep `maxbeta` in puffy.zig since it transcribes C's MAXBETA, but extend its comment to state it bounds pmag/ptot (not plasma β). No FP expressions or output strings change, so golden files are unaffected.

#### `koral/physics/radforce.zig:125` — The adiabatic index is spelled three different ways across sibling APIs: gamma, gam, and pgamma
**Fixed (2026-07-06):** unified the adiabatic-index parameter name to `gamma_adiab` across p2u/hydro/thermo/flux/wavespeeds/radforce/implicit; `Options.gam`/`Params.gam` kept as-is (config-file key).

*Naming*

The same physical parameter (adiabatic index, C's GAMMA) is `gamma` in p2u.zig (lines 47, 140), hydro.zig (15, 23, 39), thermo.zig (152, 159, 168), flux.zig (27), and wavespeeds.zig (72, 89); but `gam` in every radforce.zig function (125, 283, 298, 310, 324), sim Options (sim.zig:208), and params.zig:19; and `pgamma` in solve/invert.zig (C-traceable to u2p.c, arguably fine to keep). A caller composing e.g. flux.fFluxPrime(..., gamma) with radforce.calcChi(..., gam, ...) must remember two names for one quantity, and grep-ability suffers. The problem is compounded by state.zig's `gamma` field claiming to be a Lorentz factor (separate finding).

**Fix:** Unify the adiabatic-index parameter name across p2u.zig, physics/{hydro,thermo,flux,wavespeeds,radforce}.zig, solve/implicit.zig, and sim.zig Options. Prefer `gamma_adiab` (or uniform `gam`) over bare `gamma`: state.zig, invert.zig's `k.gamma`, and several test locals use `gamma`/`gam` for the Lorentz factor, so standardizing on bare `gamma` would deepen that collision. Keep `pgamma` in solve/invert.zig where it mirrors u2p.c's local for C-diffability. Note that params.zig documents "keys match field names," so renaming `Params.gam` changes the config-file key — either keep that field as `gam` or accept the run-file format change deliberately.

#### `koral/physics/wavespeeds.zig:30` — lrCore lowercases C's Acov/Bcov into acov/bcon/bcov/bsq, colliding with the magnetic-field vocabulary used in the same file
**Fixed (2026-07-06):** restored capitalized `Acov/Acon/Asq/Au/Au2/AB` and `Bcov/Bcon/Bsq/Bu/Bu2/AuBu` in `lrCore`.

*Naming*

C's calc_wavespeeds_lr_core deliberately uses capitalized `Acov[4], Acon[4], Bcov[4], Bcon[4], Asq, Bsq, Au, Bu, AB` (oracle physics.c:657-658) to keep the direction/time basis vectors distinct from the magnetic four-vector bcon/bcov. The Zig transcription (lines 30-48: `bcov = {1,0,0,0}`, `bcon`, `bsq`, `bu`, `acov`, `acon`, `asq`, `au`, `ab`) collapses that distinction: here `bsq = relele.dot(bcon, bcov)` is g^tt-related, NOT magnetic b-squared — yet 70 lines down in the same file, gasWavespeedsLr (line 99-108) computes a genuine magnetic `bsq` via mhd.bconBcovBsqFrom4vel, and every other module (mhd.zig, p2u.zig, flux.zig, radforce.zig) uses bcon/bcov/bsq exclusively for the magnetic four-vector. A reader skimming lrCore will misread these as B-field quantities. This is a Zig-side transformation that destroyed C's own disambiguation; renaming locals does not affect FP expression shapes.

**Fix:** Restore C's own disambiguation by keeping the capitalized names verbatim — Acov/Acon/Asq/Au/Au2/AB and Bcov/Bcon/Bsq/Bu/Bu2/AuBu — matching how the function already preserves GG; this both removes the collision with the magnetic bcon/bcov/bsq vocabulary (used at lines 72 and 99-107 of the same file and throughout mhd.zig/p2u.zig/flux.zig) and makes the code grep-diffable against physics.c:657ff. If capitalized locals are unwanted, use descriptive snake_case instead (e.g. time_cov/time_con/time_sq for B*, dir_cov/dir_con/dir_sq for A*) with a comment mapping to C's names (// C: Acov/Bcov, physics.c:657). Renaming locals has zero effect on FP expression shapes.

#### `koral/relele.zig:15` — Duplicate VelType enums, and Config.velprim/velprim_rad are dead knobs that silently do nothing
**Fixed (2026-07-06):** `config.VelType` now re-exports `relele.VelType`; `Config.validate()` compile-errors if `velprim`/`velprim_rad` are set to anything but `.velr`.

*Naming*

Two distinct enums exist for the same concept: relele.VelType (enum(u8) { vel4 = 1, vel3 = 2, velr = 3 }, values pinned for golden records) and config.VelType (config.zig:47, enum { vel4, vel3, velr }). The config one exists only to type Config.velprim / Config.velprim_rad (config.zig:55-56), which are never read anywhere in the codebase — every call site hardcodes .velr (relele.uconUcovFromPrims:271, frames.zig:69/76, p2u.zig:57/117, radiation.zig:37, radvisc.zig:302, ...). A user who sets velprim = .vel4 in their Config gets VELR behavior with no diagnostic, and the two types cannot be interchanged if the fields ever do get wired up.

**Fix:** Keep a single VelType (config.zig can re-export relele.VelType; the explicit 1/2/3 values are harmless there), and either delete velprim/velprim_rad or make Config.validate() @compileError on anything other than .velr until other parametrizations are actually implemented.

*Hand-verified: config.zig:47 defines a second, incompatible VelType (plain enum) next to relele.zig:15 (enum(u8) with C values), and grep confirms Config.velprim/velprim_rad are read nowhere — changing them silently does nothing.*

#### `koral/state.zig:15` — State.gamma is documented as "Lorentz factor" but its C counterpart struct_of_state.gamma is the adiabatic index
**Fixed (2026-07-06):** renamed the field to `gamma_adiab` and corrected the doc comment (it mirrors C's adiabatic index, not a Lorentz factor).

*Naming*

state.zig declares `gamma: f64 = 0, // Lorentz factor w.r.t. normal observer`, in a struct explicitly documented as mirroring C's struct_of_state (ko.h:437). In C, that field is the adiabatic index: fill_struct_of_state does `gamma = GAMMA; ... state->gamma = gamma;` (oracle physics.c:30-37); C has no per-cell Lorentz-factor field by that name (the radiation one is `relgamma`). So either the comment is wrong (the field is the adiabatic index, like every other `gamma` parameter in the Zig physics layer) or the field is a repurposed name that now means the opposite of its C namesake — either way the exported struct (re-exported as koral.State) will mislead whoever fleshes out the M3+ comptime composition. Note also that nothing in the codebase constructs this struct; radforce.RadState is the real struct_of_state port.

**Fix:** If the field is meant to be C's adiabatic index, fix the comment and consider `gamma_adiab`; if a Lorentz factor is genuinely intended, rename the field to `lorentz` so it cannot be confused with the adiabatic-index `gamma` used everywhere else. Alternatively delete the unused stub now that RadState (radforce.zig:97) supersedes it.

### Project structure

#### `koral/koral.zig:143` — Test files: three naming schemes, two placement rules, and one misleading name

*Project structure*

27 test files sit flat in the koral/ root, outnumbering the ~12 core source modules (grid, field, layout, geometry, state, sim, ...) they surround — `ls koral/` buries the load-bearing files. Meanwhile the metric subsystem alone nests its theory tests as koral/metric/tests.zig. Naming mixes three conventions: suffix-plural `X_tests.zig` (state_tests, flux_tests, ...), prefix-singular `golden_X_test.zig` (golden_state_test, ...), and bare `tests.zig` (metric). The two families (theory vs C-golden) are meaningful and worth keeping distinguishable, but the current mix means a subsystem's tests don't sort together and grep patterns need three shapes. Worst single case: koral/golden_test.zig is actually the M1 *metric* golden test (its own doc comment says so) — the generic name suggests it is the golden-test framework, which actually lives in koral/testing/golden.zig.

**Fix:** Pick one placement and one naming scheme for the two test families. Concretely: keep tests next to the code, named uniformly `<subsystem>_tests.zig` (theory) and `<subsystem>_golden_tests.zig` (C-comparison) so each subsystem's files sort adjacently; move koral/metric/tests.zig to koral/metric_tests.zig (or nest all tests — but pick one rule); and rename koral/golden_test.zig → metric_golden_tests.zig, since its generic name suggests the golden-test framework, which actually lives in koral/testing/golden.zig. Alternatively, move all 26 root test files into koral/tests/ to declutter the root. Either way, state the chosen rule in the repo docs (README.md, or the zig-rewrite-architecture doc in koral_lite that koral.zig already references) and update the test import block in koral/koral.zig accordingly.

**Fixed (2026-07-06):** Unified to one scheme — theory gates `<subsystem>_tests.zig`, C-oracle goldens `<subsystem>_golden_tests.zig` — so each subsystem's files sort adjacently. Moved `metric/tests.zig` → `metric_tests.zig`, renamed the misleading `golden_test.zig` → `metric_golden_tests.zig` (and the other ten `golden_<x>_test.zig` → `<x>_golden_tests.zig`). The rule is stated in koral.zig's `test` block and the USER_GUIDE test inventory; the Cyrillic "локate" typo was fixed too.

#### `koral/physics/radforce.zig:61` — PUFFY leaks into the generic library layer: scattered .puffy presets and a problem-specific default

*Project structure*

Problem-specific constants live inside six generic library modules as `.puffy` decls: config.zig:108 (Config), solve/invert.zig:48 (FloorParams), solve/invert_rad.zig:48 (RadParams), solve/implicit.zig:85 (ImplicitParams), physics/thermo.zig:46 (Composition), physics/radforce.zig:69 (Params.puffy(), which hardcodes MASS=10 M☉ into the physics layer). The preset-catalog pattern (each next to a `cdefault`) is defensible since the values annotate define.h lines, but two aspects cross the boundary: (1) radforce.zig:61 `kappaes: KappaesMode = .puffy` — the *default* of a generic parameter struct is one specific problem's Klein–Nishina hook, so any non-PUFFY problem that forgets to override it silently gets PUFFY scattering physics; (2) KappaesMode/KappaMode are closed enums in the library (radforce.zig:51), so a future problem with a nontrivial kappaes.c-style hook must edit the generic module — in C this is problem-supplied code (PR_KAPPAES), and the Zig BC layer already solves the same shape correctly with the `specific_bc` function pointer and the `puffy.Bc(SimT)` generic-struct pattern.

**Fix:** Change the default to the C-faithful neutral value (`kappaes: KappaesMode = .none` — C without PR_KAPPAES returns 0, per this file's own doc comment) and have `Params.puffy()` set `.kappaes = .puffy` explicitly. This is behavior-preserving for every current call site (all use `Params.puffy()` or `Params.grey(...)`, which set kappaes), so no golden-file risk. Optionally: add a function-pointer variant to KappaMode/KappaesMode so future problems supply kappa hooks without editing the library (mirroring the `specific_bc` pattern), and consolidate the six scattered `.puffy` presets as re-exports from koral/problems/puffy.zig so the problem's parameter surface lives in one file.

**Fixed (2026-07-06):** `kappaes` default changed to the C-faithful `.none`; `Params.puffy()` now sets `.kappaes = .puffy` explicitly. Behavior-preserving for every current call site (all go through `puffy()`/`grey()`). The optional preset-consolidation / function-pointer-hook was left out as non-essential.

#### `koral/sim.zig:539` — sim.zig (1751 lines) has clean extraction seams that will matter before MPI lands

*Project structure*

The file is internally well-organized (section headers, C-source mapping in the module doc), but it is 2.3× the next-largest source file and aggregates several separable concerns inside the comptime `Sim(cfg)` struct: (1) the boundary-condition block — setBc/setBcCell/p2uCell/copyCellP/avgCellP/fillCorners2d, lines 539-791, ~250 lines — which is exactly the code that grows when the planned MPI backend adds ghost exchange (the comm/ seam already anticipates this); (2) the threading dispatch — parallelRows/u2pRowsWorker/implicitRowsWorker/ChunkResult, lines 925-983 and 1546-1588; (3) FaceStore plus the Flag/Scal bookkeeping enums, lines 59-173, which are generic storage utilities not evolution logic. The codebase already has the right pattern for extracting from a comptime struct: problems/puffy.zig's `pub fn Bc(comptime SimT: type) type`.

**Fix:** Extract in the order the code will churn: (1) sim/bc.zig first — setBc/setBcCell/p2uCell/copyCellP/avgCellP/fillCorners2d (lines 539-791), written SimT-generic like the existing house pattern (puffy.zig:639 `Bc(comptime SimT: type)`, magn/ct.zig `fluxCt(comptime SimT: type, sim: *SimT)`); this is the region the planned MPI backend (comm/serial.zig doc: "the MPI backend implements the same API later") will grow, and a dedicated file maps directly onto C finite.c:2805/3203. Note call sites change from method form to module-function form since Zig 0.16 has no usingnamespace mixins. (2) parallelRows/ChunkResult/the two row workers into sim/threading.zig. (3) FaceStore + Flag/Scal (lines 59-173) into sim/storage.zig — these are already file-scope declarations outside Sim(cfg), so the move is mechanical. Keep Sim.step and the operators (opExplicit/opImplicit/calcU2p) in sim.zig.

**Fixed (2026-07-06):** Extracted `sim/storage.zig` (FaceStore + the Flag/Scal bookkeeping enums), `sim/bc.zig` (SimT-generic `setBc`+ghost/corner helpers — the MPI-growth seam), and `sim/threading.zig` (generic `parallelRows` + `ChunkResult`). `sim.zig` keeps a thin `setBc` delegator method and the sim-specific row workers, and re-exports `BcKind`/`BcFace`/`Flag`/`FaceStore` so problems/tests are unchanged. (The two thin row-worker bodies were kept in sim.zig rather than moved, to avoid pulling the implicit-solver imports into the dispatch module.)

### Test hygiene

#### `koral/golden_test.zig:30` — golden_test.zig duplicates testing/golden.zig helpers verbatim; Dev/FieldDev tracker re-implemented in three more golden files

*Test hygiene*

koral/testing/golden.zig exists precisely to hold the shared KGLD reader and deviation trackers, and every golden file uses it — except golden_test.zig, which carries its own private copies: `Golden` (lines 30–49), `readGolden` (51–76), `coordsFromId` (78–86), and a byte-for-byte duplicate `DevTracker` (103–126). A KGLD format bump (version 2, header change) now has two readers to update, and the two DevTrackers can silently diverge. Separately, the field-scale tracker is re-implemented three times: `Dev` in golden_visc_test.zig:41–51 and golden_dynamo_test.zig:53–63 (identical), and the richer `FieldDev` in golden_puffy_test.zig:54–73 (same core + location/report). golden_puffystep_test.zig avoids it only by inlining the max-diff/max-scale logic.

**Fix:** Port golden_test.zig to the shared helpers in testing/golden.zig: replace the local Golden/readGolden with gold.readGolden(a, "metric/" ++ name, nin, nout) (its internal expectEquals subsume the local nin/nout checks at lines 131-132, 198-199, 224-225), and use gold.DevTracker and gold.coordsFromId (metric.Coords is a re-export of config.Coords, so this is drop-in); keep the local expectGolden if desired. Separately, move one field-scale tracker into testing/golden.zig — the FieldDev superset from golden_puffy_test.zig:54-73 — and delete the identical Dev copies in golden_visc_test.zig:41-51 and golden_dynamo_test.zig:53-63 (their add() call sites sit in cell loops, so the ix/iy arguments FieldDev wants are in scope; alternatively give the shared tracker a location-less add overload). Leave golden_puffystep_test.zig as is — its per-variable-scale/cell-exclusion logic is a genuinely different pattern.

#### `koral/koral.zig:107` — Test registration is fully manual with no guard against a silently dropped file

*Test hygiene, Project structure — flagged independently by 2 reviewers*

The entire suite hangs off the single `test { _ = @import(...) }` block in koral/koral.zig (build.zig registers only the koral module as test root). I verified all 27 test files and all 12 in-module test-bearing files are currently referenced — coverage is complete today. But the mechanism is unguarded: a new `foo_tests.zig` that compiles cleanly will run zero tests unless someone remembers to add `_ = @import("foo_tests.zig");` to this block, and nothing in the build or CI would notice. With 27 externally-registered files and an active milestone cadence, this is the classic silent-coverage-loss trap the suite's own size makes likely.

**Fix:** Add a configure-time guard in build.zig: iterate koral/ top-level entries (same openDir/iterate pattern already used for PROBLEMS/), collect basenames matching *_tests.zig / *_test.zig, plus the special cases metric/tests.zig and testing/tubes.zig (test-bearing but not pattern-matching), read koral/koral.zig, and fail the build unless the literal token @import("<name>") appears for each collected file. ~20 lines, runs at build-graph configure time, makes the manual registration contract self-enforcing.

#### `koral/state_tests.zig:32` — Theory-test helpers copy-pasted across files: expectClose ×4, velr-with-gamma ×4, ppFromTemps ×2, run-to-tend loop ×5

*Test hygiene*

Identical 7-line `expectClose` (dev = |a−b|/max(1,|a|,|b|), print, error) appears verbatim in state_tests.zig:32, flux_tests.zig:20, radiation_tests.zig:51, opacity_tests.zig:53 (metric/tests.zig:55 has a deliberate rtol/atol variant). The controlled-Lorentz-factor VELR sampler is written four times: `velrWithGamma` (state_tests.zig:54), `velrGamma` (radiation_tests.zig:86, same math, shuffled args), plus inline re-derivations in opacity_tests.zig:252–260 and implicit_tests.zig:386–395. `ppFromTemps` is byte-identical in opacity_tests.zig:62–87 and implicit_tests.zig:51–76. The "advance to t_end with the C dt=t1−t clamp" idiom (`if (s.t + 1.0/s.tstepdenmax > tend) dt = tend - s.t`) is repeated at evolution_tests.zig:402,486,692 and mhd_evolution_tests.zig:290,348 — the C-provenance comment exists on only one copy. Near-identical PUFFY `SimP.Options` builders also recur in radvisc_tests.zig:19, dynamo_tests.zig:38, golden_visc_test.zig:53, golden_dynamo_test.zig:36, golden_puffy_test.zig:82, golden_puffystep_test.zig:56, differing only in the radviscosity/dynamo/implicit toggles.

**Fix:** Create koral/testing/theory.zig (peer of testing/golden.zig and testing/tubes.zig) holding expectClose, one unified velrWithGamma (pick one of the two existing signatures and update call sites), ppFromTemps, and a stepTo(sim, t_end) helper carrying the "C: dt=t1-t clamp" comment once. Note ppFromTemps and stepTo need a comptime layout/Sim-type parameter since the current copies close over file-local cfg (identical in both files) — mechanical consolidation, no math or tolerance changes. Separately, give problems/puffy.zig (or testing/) one simOptions(.{ .radviscosity = ..., .dynamo = ..., .implicit = ..., .opac = ... }) builder to replace the six near-identical SimP.Options literals in radvisc_tests.zig:19, dynamo_tests.zig:38, golden_visc_test.zig:53, golden_dynamo_test.zig:36, golden_puffy_test.zig:82, golden_puffystep_test.zig:56.

---

## Low severity

### Best practices & latent bugs

#### `koral/metric/precompute.zig:59` — applyKrisCorrection computes the full CoordData (inverse + 64 Christoffels) at six face points per cell just to read gdet

*Best practices & latent bugs*

Lines 59-60 call metric.compute() at the hi/lo face for each of the three directions (6 calls per cell during fillCenters), and metric.compute performs the 16-cofactor dual inversion and the full Christoffel assembly, all of which is discarded except `.gdet`. On top of that, fillFaces (lines 285-319) recomputes metric.compute at the same face points again to populate gb/gconb. For a production 3D grid this makes cache construction roughly an order of magnitude more expensive than needed. Init-time only, so no wrong physics — but it is the dominant startup cost as grids grow.

**Fix:** Add a cheap module-level helper in metric.zig, e.g. `pub fn gdetAt(coords: Coords, mp: MetricParams, x: [4]f64) f64` that runs gcovDual + det4 and returns @sqrt(-det.v) — identical FP operations to compute()'s gdet path, so the value is bitwise-identical and golden agreement is preserved — and call it at precompute.zig:59-60. This skips inv4 (16 dual cofactor expansions) and the Christoffel assembly, the dominant per-call cost. Expect roughly 2-2.5x faster cache construction, not an order of magnitude, since the center and face full computes remain. Optionally, when InitOpts.faces is true, run fillFaces before the correction and read face gdet from gb[d] at offset 3*5+4 to eliminate the 6 recomputations entirely; note this needs a cache-aware variant since applyKrisCorrection is also used standalone by computeCorrected (tests) and gb is zeroed when faces=false.

#### `koral/physics/flux.zig:34` — C's hard NaN abort on the stress tensor was silently dropped from fFluxPrime

*Best practices & latent bugs — downgraded by verifier*

C's f_flux_prime (physics.c:1230-1247) checks isnan(T[ii][jj]) for all 16 components right after calc_Tij and hard-exits with the offending cell coordinates ('nan in flux_prime', my_err + exit(-1)); the check is compiled in for PUFFY (FORCEFREE undefined). The Zig fFluxPrime computes calcTij (line 34) and assembles fluxes with no NaN check and no comment marking the omission — unlike every other intentional divergence in this codebase, which is meticulously documented ('C quirk', 'dead code', 'mirrored'). This matters because the code deliberately preserves silent NaN producers upstream: wavespeeds.lrCore (koral/physics/wavespeeds.zig:53-55) mirrors C's sqrt of a negative discriminant but additionally drops C's diagnostic print, and relele.calcAlpgam likewise drops its print. In C the first NaN reaching a flux halts the run pinpointing the cell; in Zig it propagates into the conserved update, where u2p flags a fixup and cellFixup neighbor-averaging can smear or silently 'heal' the cell — the run completes with quietly wrong physics and no trace of where the NaN originated. Since golden tests only exercise healthy states, nothing pins this behavior.

**Fix:** Add a NaN check over T (or the assembled ff) in fFluxPrime — either a std.debug-mode assertion or a distinct error such as error.NanInFlux that the sweep's existing `try` (sim.zig:1283,1295) propagates so the driver's step-failure path can report it with cell context — and note the divergence from C's isnan/exit(-1) (physics.c:1230-1245) in the flux.zig module header, matching the transcription notes in solve/implicit.zig and wavespeeds.zig. Note the driver already aborts when primitives go non-finite at output rows (PROBLEMS/puffy/main.zig:193-196); the check's value is pinpointing the origin cell and closing the between-outputs window where cellFixup can silently heal a transient NaN. Do not restore the prints in wavespeeds.lrCore or relele.calcAlpgam — those omissions are documented as intentional.

#### `koral/problems/puffy.zig:671` — Bc.calc swallows numerical errors with `catch unreachable` (6 sites) — panic in safe builds, UB in ReleaseFast

*Best practices & latent bugs*

Lines 671, 696, 700, 705, 712, 717 use `catch unreachable` on `frames.transPmhdCoco` and `relele.convVels`, which return `relele.Error` (utInUcon can return SpacelikeVelocity when b*b - g00*c < 0; utInVel3 paths return VelocityConversionFailed). At the PUFFY outer boundary (r~500, g00<0, positive-definite spatial metric) these are arguably unreachable for finite inputs, and NaN inputs slide through without erroring — but nothing enforces that: this Bc is the template that future problems (or an xlo boundary inside the ergoregion, where g00 flips sign and utInUcon's discriminant CAN go negative) will copy, and in ReleaseFast a triggered `unreachable` is undefined behavior rather than a diagnosable abort. The only reason for `catch unreachable` is that the `SpecificBc` function-pointer type (`fn(...) [NV]f64`, sim.zig:195) forbids error returns, even though its sole caller `setBcCell` already returns `Error`.

**Fix:** Widen `SpecificBc` (sim.zig:195) to return `Error![NV]f64` — sim.zig's `Error` already includes `relele.Error`, `setBcCell` already returns `Error!void`, and the fn pointer has exactly one call site (sim.zig:616), so the change is one `try` there plus replacing the six `catch unreachable` in puffy.zig with `try`. Note the mechanism precisely: `VelocityConversionFailed` comes from `convVelsCore`'s `ut < 1.0 or isNan(ut)` checks (relele.zig:219/223), not `utInVel3` — meaning a NaN reaching these boundary conversions errors and hits the `unreachable` today (panic in safe builds, UB in ReleaseFast), so propagation also buys a diagnosable failure during blow-up debugging.

#### `koral/sim.zig:332` — Sim.init (and MetricCache.init) leak all prior allocations if a later allocation fails

*Best practices & latent bugs — downgraded by verifier*

init performs ~30 sequential allocations (MetricCache with 11 internal buffers, 14 Fields, 15 FaceStores, scal, flags, 3 emf arrays, vecpot, rijvisc, dynA, scaleth) with `try` and no `errdefer`. An OOM at, say, the flags alloc (line 353) leaks the cache and every Field/FaceStore allocated before it. precompute.MetricCache.init has the identical pattern internally (g/gcon/kris/dxdx/gb/gconb). For a driver that exits on init failure this is cosmetic, but Sim is also constructed repeatedly inside the test suites (std.testing.allocator will report these leaks as confusing secondary failures on the OOM path), and library users get no clean failure mode.

**Fix:** Back Sim's (and MetricCache's) storage with a single internal std.heap.ArenaAllocator so a failed init and deinit are both a single arena release — this fixes the error-path leak and also collapses the hand-maintained per-field deinit (sim.zig:393-418), which is the more real hazard when fields are added. Skip the per-allocation errdefer variant: no allocation-failure injection exists in the test suite (no FailingAllocator/checkAllAllocationFailures anywhere), and the sole driver exits on init failure, so ~30 errdefers would guard a path that never executes.

#### `koral/state.zig:10` — State(cfg) is dead code with a stale doc promising a design that was never built

*Best practices & latent bugs, Data-oriented design — flagged independently by 2 reviewers*

state.zig's module doc says 'M0 stub: only the base fluid fields exist. From M3 onward this becomes a comptime composition — base fields merged with each active module's StateFields (bcon/bsq/pmag from MHD; Ehat/Rij/Gi from radiation)'. All milestones M0-M13 are complete, yet the promised composition never happened: the type still holds only the six M0 fluid fields, is exported as koral.State (koral.zig:29-30), and has zero users anywhere in the tree — the real per-cell derived state was implemented as radforce.RadState (koral/physics/radforce.zig:97-117) instead. A reader (or a future contributor implementing a new module) following the doc will look for the StateFields mechanism and find nothing; the export suggests API surface that is actually abandoned.

**Fix:** Either delete state.zig and the koral.zig re-export, or rewrite the doc to state the actual design ('superseded by radforce.RadState; kept only for X') — ideally pointing C's struct_of_state (ko.h:437) readers at radforce.zig.

#### `koral/testing/golden.zig:152` — readKstp panics on a truncated/corrupt file with nrec==0; readKini leaks vars on the BadGoldenFile path

*Best practices & latent bugs*

readKstp computes `per_rec = (raw.len - 32) / 8 / k.nrec` before validating nrec; a corrupt or truncated .kstp with nrec=0 in the header produces an integer division-by-zero panic instead of the intended `error.BadGoldenFile`, which is a confusing failure mode for the exact situation these validity checks exist for (half-regenerated goldens from tools/gen_golden.sh). Separately, readKini allocates `k.vars` (line 327) and then can `return error.BadGoldenFile` at the length check (line 334) without freeing it — under std.testing.allocator the real diagnostic gets buried under a leak report.

**Fix:** In readKstp, validate the header-derived divisors before use: add `if (k.nrec == 0 or ncell == 0) return error.BadGoldenFile;` after computing ncell (line 151) — nrec==0 panics at line 152 and ncell==0 panics at line 154's `/ ncell`. In readKini, add `errdefer a.free(k.vars);` immediately after the allocation at line 327 so the BadGoldenFile return at line 334 (and an OOM at line 335) do not leak under std.testing.allocator.

### Purity & hidden state

#### `koral/field.zig:55` — Field.load uses an out-pointer where returning [NV]f64 by value is equally efficient and removes the undefined-buffer step

*Purity & hidden state*

load(self, ix, iy, iz, out: *[NV]f64) forces every caller into the two-step `var pp: [NV]f64 = undefined; f.load(ix, iy, iz, &pp);` pattern (~99 `.load(` call sites across koral/ and PROBLEMS/). The undefined buffer is briefly observable and refactors can accidentally read it before the fill. Zig's result-location semantics make a by-value return write directly into the caller's variable — zero extra copies for a 104-byte array at NV=13. This is pure data movement (memcpy), no FP expression shape involved, so golden agreement cannot be affected. store's `pp: *const [NV]f64` in-param is fine as-is (it only reads).

**Fix:** Change to `pub fn load(self: *const Self, ix: i64, iy: i64, iz: i64) [NV]f64 { return self.data[self.cellOffset(ix, iy, iz)..][0..NV].*; }` and update call sites mechanically: `var pp = f.load(ix, iy, iz);` (or `const pp = ...` at the many sites that never mutate the buffer, and plain assignment `ppn[in_n] = f.load(...)` for array-element and reused-buffer sites). This matches the codebase's existing by-value [NV]f64 style (p2u already returns by value) and is pure data movement, so golden agreement is unaffected. Can be done incrementally by adding the by-value form alongside the old one. Keep store's `*const [NV]f64` parameter as-is.

#### `koral/field.zig:143` — Tautological test assertion: compares an expression with itself, so the iv-adjacency property is never actually checked

*Purity & hidden state, Best practices & latent bugs, Data-oriented design — flagged independently by 3 reviewers*

In the "memory layout is iv-fastest then x" test, line 143 reads `try std.testing.expectEqual(f.cellOffset(0, 0, 0) + 1, f.cellOffset(0, 0, 0) + 1);` — both sides are the same expression, so the assertion is vacuous and the comment's claim ("Adjacent iv within one cell -> adjacent memory") is unverified. The other assertions in the test (NV stride in x, SX*NV in y, ghost corner at 0) are real; only this line asserts nothing.

**Fix:** Replace the tautological assertion with a check against the raw flat array, e.g.: `f.set(0, 0, 0, 0, 1.0); f.set(1, 0, 0, 0, 2.0); try std.testing.expectEqual(@as(f64, 2.0), f.data[f.cellOffset(0, 0, 0) + 1]);` — this actually pins iv-fastest ordering, which the uniqueness test cannot (it uses the same address function for both write and read and never inspects f.data).

#### `koral/io/scalars.zig:208` — scaleHeightAt mutates the Sim during a logically read-only diagnostic

*Purity & hidden state*

scaleHeightAt takes `sim: *SimT` (every other reduction in this file takes *const) solely because it calls dynamo.calcScaleHeight, which fills the persistent sim.scaleth array, then reads back one entry. Consequences: (1) the mutability requirement propagates — PROBLEMS/puffy/main.zig scalarRow must take `*SimT`, so the whole diagnostics row is non-const; (2) an output pass overwrites state the dynamo owns (harmless today only because the values are recomputed from unchanged p, i.e. identical); (3) diagnostics can never run concurrently with anything touching scaleth. For a single radius the full-grid fill is also wasted work — only column ix is needed.

**Fix:** Add a pure per-radius helper in dynamo.zig — `pub fn scaleHeightAtIx(comptime SimT: type, sim: *const SimT, ix: i64) f64` — containing the existing per-column accumulation (Σρ√g(π/2−θ)²/Σρ√g with the ix==0 unnormalized C quirk), and have calcScaleHeight delegate its per-ix loop body to it so the FP-shape-critical code exists in exactly one place and both paths stay bit-identical. Then scalars.scaleHeightAt becomes `sim: *const SimT` (calling radialShellIndex then scaleHeightAtIx), and scalarRow/its transitive callers can uniformly take *const SimT. This is safe: applyDynamo (dynamo.zig:271) already recomputes scaleth before every dynamo use, so nothing depends on the diagnostic-side fill.

#### `koral/magn/ct.zig:222` — curlFromA delivers its result through the sim.vecpot scratch field — an invisible cross-module data channel

*Purity & hidden state*

curlFromA/cornerAverageA/calcBfromACore communicate exclusively through sim.vecpot slots (0..2 = corner A, 3..5 = B), and the consumer contract is only stated in comments: dynamo.mimicDynamo (magn/dynamo.zig:243-255) calls ct.curlFromA and then reads sim.vecpot.get(3..5), and puffy-style init reads them inside calcBfromA's ifoverwrite loop (ct.zig:163-165). Nothing in any signature says curlFromA has an output, that it clobbers whatever was in vecpot, or that mimicDynamo's superimpose loop depends on the immediately preceding call — a reordering or an interleaved second curl (e.g., a diagnostic calling calcBfromA mid-dynamo) silently corrupts the ΔB. This mirrors C's pvecpot global, but unlike the FP expressions there is no diffability reason to keep the shape: C reads/writes a global array, Zig can name the buffer.

**Fix:** Make the scratch explicit in the signatures: pass it as a parameter (e.g. `scratch: anytype` alongside the existing `src: anytype`, or `scratch: *field_mod.Field(6)` after importing the field module — cycle-free since sim.zig already imports it) for curlFromA/cornerAverageA/calcBfromACore, with callers passing &sim.vecpot; have curlFromA return the scratch pointer so consumers visibly read the function's result, and state in the doc comment that slots 0..2 are clobbered (corner A) and 3..5 are the output B. Identical arithmetic and goldens; the cross-module data channel becomes visible in the API and reentrancy analysis becomes local. This also strengthens ct.zig's existing design goal of not depending on Sim internals.

#### `koral/metric/precompute.zig:29` — applyKrisCorrection mutates CoordData through a pointer where a value-in/value-out form is bit-identical

*Purity & hidden state*

applyKrisCorrection(d: *metric.CoordData, ...) is the only mutate-through-pointer API in the metric layer; everything else (metric.compute, forms, coco, dual, quad) is already value-pure. It rewrites only the trace-adjacent kris entries; a pure form performing the identical FP operations in identical order returns bit-identical results, so the C-diffability contract is untouched. Both call sites collapse to single assignments (computeCorrected at line 75, fillCell at line 255). Related nit: the comment at precompute.zig:282-283 claims the correction "lives at module level below" — the function actually sits above at line 29; stale after a move. Audit result for the whole assignment: no module-level `var` or `threadlocal` exists in any of the 16 base-layer files reviewed (build.zig, koral.zig, config, layout, field, grid, params, units, geometry, comm/serial, math/dual, math/quad, metric/{metric,forms,coco,precompute}); units.zig keeps the ko.h constant globals as a proper Units value struct, and quad.integrate is stack-only and fully reentrant — this layer is clean for the row-parallel threading.

**Fix:** Two parts, first is unconditional: fix the stale comment at precompute.zig:282-283 — the correction sits above at line 29, not "below", and tests actually go through computeCorrected (golden_test.zig:251) rather than calling it directly. Optionally, for consistency with the otherwise value-pure metric layer, change the signature to `pub fn krisCorrected(coords: Coords, mp: MetricParams, g: *const Grid, d: metric.CoordData, ix: i64, iy: i64, iz: i64) metric.CoordData` and assign at the two call sites (lines 75, 255); implement it as `var out = d; ...; return out;` — copying to a local before any write matters because `d = krisCorrected(..., d, ...)` may alias the by-reference parameter with the result location. FP ops stay identical in order, so golden agreement is unaffected. Keeping the in-place form (mirroring C's calc_Krzysie_at_center) is also defensible; in that case fix only the comment.

#### `koral/metric/precompute.zig:86` — geometryAt fabricates cell identity (ix=iy=iz=0) — callers must mutate-to-fix, and a forgotten patch silently aliases cell (0,0,0)

*Purity & hidden state — downgraded by verifier*

geometryAt hardcodes .ix=0, .iy=0, .iz=0, .ifacedim=-1 into the returned Geometry (precompute.zig:86-89) — plausible-looking but fabricated values. Both production callers already have to patch this by post-construction mutation: koral/problems/puffy.zig:404-406 (fillGeometryBL) and koral/magn/dynamo.zig:112-114 (geomBLat) each do `geom.ix = ix; geom.iy = iy; geom.iz = iz;` — duplicated fix-up code. A third caller that forgets the patch hands kernels a geometry claiming to be cell (0,0,0); anything that uses geo.ix/iy/iz to index the metric cache (e.g. Christoffel lookups via MetricCache.kr) then silently reads the wrong cell's data. The result of geometryAt is not fully determined by its arguments in any honest sense — the caller must supply the missing state by mutation afterwards.

**Fix:** geometryAt fabricates cell identity (.ix/.iy/.iz = 0), and three call sites duplicate the same post-construction patch: puffy.zig:404-406 (fillGeometryBL), dynamo.zig:112-114 (geomBLat), and io/scalars.zig:48-50 (blGeom). Add a geometryAtCell(coords, mp, x, ix, iy, iz) variant (or optional cell parameter) and delete the three patch sites; indices never enter FP so goldens are unaffected. Note the fields are currently write-only across the codebase — no consumer reads Geometry.ix/iy/iz today, so this is duplication cleanup and future-proofing, not a latent-bug fix.

#### `koral/physics/opacities.zig:165` — Opac.tot_emissivity is a placeholder-invalid field: calcOpacitiesFromState returns it as 0, filled only by specific callers

*Purity & hidden state — downgraded by verifier*

calcOpacitiesFromState (pub) returns an Opac whose tot_emissivity is a sentinel 0, valid only after kappaFromState (opacities.zig:178) or radforce's grey branch (radforce.zig:193) overwrite it. The struct's validity thus depends on which constructor produced it — a must-call-X-after-Y dependency not expressed in types; any future caller of calcOpacitiesFromState that reads tot_emissivity (e.g. a diagnostics dump or the eventual emission bookkeeping) silently gets 0. This mirrors C's fill-order coupling between calc_opacities_from_state and calc_kappa_from_state, but Zig can express it structurally at zero cost.

**Fix:** Keep the flat Opac struct (it mirrors C's struct opacities 1:1 and golden_opac_test maps r.out[6..11] onto the six fields). The lightest structural fix is to have calcOpacitiesFromState fill tot_emissivity itself using the exact expression kappaFromState uses (gas_abs * fourpi * sigma_rad_over_pi*te^4 — identical value since gas_abs >= 0 on this path, and the field never reaches goldens), making the sentinel impossible; kappaFromState's overwrite then becomes a no-op it can drop. Note the field is currently write-only across the codebase, so leaving it as-is with the existing comments is also defensible.

#### `koral/physics/radiation.zig:107` — pradFf2Lab mutates pp in place through a pointer where a by-value transform works equally well

*Purity & hidden state — downgraded by verifier*

pradFf2Lab takes `pp: *[N]f64` and rewrites the EE..FZ slots via invert_rad.u2pRad, inheriting C's in-place prad_ff2lab shape. Everything else in this layer is functional — frames.transPmhdCoco/transPradCoco/transPallCoco take ppin by value and return the new vector precisely to neutralize C's aliasing quirk (frames.zig header comment), and the sole caller (problems/puffy.zig:497, init-time only) has no need for in-place semantics. The pointer signature is the odd one out: it hides which slots change (only a doc comment says "EE..FZ in place, MHD untouched"), and denies the caller a before/after pair without a manual copy. Golden fidelity does not depend on this shape — the FP operations are identical either way.

**Fix:** For consistency with the module's otherwise functional style (transPmhdCoco/transPradCoco/transPallCoco and every other pp-taking function in radiation.zig), change pradFf2Lab to take pp by value and return the updated vector: copy pp into a local, pass &local to u2pRad, return local. Update the single call site (puffy.zig:497) to `pp = try radiation.pradFf2Lab(cfg, pp, geomBL, invert_rad.RadParams.puffy);`. Bit-identical results, including the u2pRad-failure path. Leave u2pRad's pointer signature as is — its conditional-write-on-failure semantics and hot-loop callers justify it.

#### `koral/physics/radvisc.zig:335` — calcRijViscTotal fills a hidden per-step store (sim.rijvisc) with no freshness guarantee at the read site

*Purity & hidden state — downgraded by verifier*

calcRijViscTotal(sim, global_dt) writes R^i_j_visc into sim.rijvisc; the values are consumed much later by sim.zig's rijviscFace (sim.zig:1206, called from sweep at :1286/:1298) inside opExplicit. Nothing in the types or asserts ties the fill to the read: step() (sim.zig:1683-1685) happens to call the fill before both explicit stages, but opExplicit is independently callable (evolution_tests.zig:120 already calls it directly), and with opt.radviscosity on a direct caller silently gets zeros on the first step (Field.init memsets to 0, field.zig:25) or the *previous* step's tensor afterwards — computed from old prims and an old dt via the RADVISCNUDAMP nulimit. This is a must-call-X-before-Y dependency expressed only by convention, exactly the C global-state pattern (Rijviscglobal + problem.c:127 ordering) this rewrite is trying to escape. Note the global_dt lead itself is resolved cleanly here — dt is an explicit parameter through calcRadVisccoeff/calcRadShearviscosity/calcRijVisc, visible in every signature; the store is the one remaining ambient link. (Broader audit result for this file set: no module-level `var` or `threadlocal` exists in any of the 14 assigned files, and thermo.zig's Consts are fully parameterized — no findings needed there.)

**Fix:** Add a freshness stamp next to the store: a field like `rijvisc_stamp: ?struct { step: u64, dt: f64 } = null` on Sim, set by calcRijViscTotal, and a `std.debug.assert` in rijviscFace (or at opExplicit entry when opt.radviscosity and L.hasVar(.ee)) that the stamp exists and matches the current step/dt. This turns a silent zero/stale viscous tensor on a direct opExplicit call into a loud debug failure, with golden output unchanged. Do not thread the Field(16) through opExplicit's signature — that burdens hydro-only configs and departs from the deliberate C structural mirror.

#### `koral/sim.zig:315` — Transient impl_dt field on Sim is ambient state feeding the implicit solve

*Purity & hidden state — downgraded by verifier*

opImplicit (line 1540) writes `self.impl_dt = dtin` and implicitRowsWorker (line 1570) reads it — the result of every per-cell implicit inversion depends on a field set moments earlier by a different function, exactly the C global-state pattern (global_dt) this rewrite is trying to kill. It also makes opImplicit non-reentrant: two overlapping passes (or a future nested/pipelined use) would silently share one dt. The root cause is that parallelRows' worker signature `fn (self, iy0, iy1, res)` has no way to carry per-pass arguments, so the value is smuggled through Sim. For contrast, the other two suspected ambient-state leads are already clean: radvisc receives global_dt as an explicit parameter all the way down (physics/radvisc.zig calcRijViscTotal → calcRadVisccoeff), and thermo.Consts is a value struct threaded through radforce.Params — both dependencies are visible in signatures. impl_dt is the one remaining exception.

**Fix:** Optional polish: make parallelRows context-generic — `fn parallelRows(self: *Self, ctx: anytype, comptime worker: fn (*Self, @TypeOf(ctx), i64, i64, *ChunkResult) void) Error!void` — spawn with `.{ self, ctx, iy0, iy1, &results[i] }`, pass `dtin` from opImplicit and `{}` from calcU2p, and delete the impl_dt field. This makes the dt dependency visible in the worker signature instead of flowing through Sim. Pure plumbing, no FP or golden impact. Note this is cleanliness only: the current field is documented, written once before spawn (no race), and Sim is single-pass by design, so no correctness hazard exists today.

#### `koral/solve/implicit.zig:261` — residual() splits its result between an out-param and a return value where a single returned struct works identically

*Purity & hidden state*

residual fills `f: *[4]f64` and separately returns the max normalized error — the caller must know both channels compose one result (solve4dPrim juggles f1/f2 buffers plus the returned errbase). Since [4]f64 is a tiny value type and the FP expressions inside are untouched by how results leave the function, this is a C shape (C fills a caller array) kept where the contract allows a value return. The same pattern appears in sim.zig's FaceStore.load (line 161) and Field.load, which take `out: *[NV]f64` where returning `[NV]f64` by value is equally efficient in Zig (comptime-known size, copy either way) and would make call sites read as pure expressions (`const pp = self.p.loadAt(ix,iy,iz);`) instead of declare-undefined-then-fill.

**Fix:** Core change: have residual return `relele.Error!struct { f: [4]f64, err: f64 }` — mechanical, bit-identical arithmetic, and it removes the discarded `_ = residual(...)` return at the Jacobian call site (line 592) where only f2 is wanted. Optional follow-on: if the declare-undefined-then-fill pattern at load call sites bothers you, add a value-returning `loadAt(ix,iy,iz) [NV]f64` alongside load() on Field/FaceStore and migrate opportunistically — but only if you accept two coexisting accessors across ~97 call sites; store() stays as-is since it genuinely writes into the field.

#### `koral/solve/invert.zig:171` — u2pSolverW's pp parameter is an undocumented in/out: it must hold the current primitives as the Newton initial guess

*Purity & hidden state*

u2pSolverW reads pp[rho], pp[uu], pp[vx..vz] (lines 241-251) to build the initial W before overwriting pp on success — the classic C ppin==ppout contract, but nothing at the signature says so (the doc comment only says "on 0 the MHD primitive slots of pp are overwritten", implying pure-output). Same for u2pMhd (line 361), whose ppmhd/ppbak dance additionally relies on the incoming pp being the previous good state to restore on failure. A caller passing a fresh/zeroed buffer gets W=0, the ×10 bound loop stays at 0 for 50 iterations, and the solver returns −150 → entropy fallback → −150 again → silent fixup flagging that looks like a physics failure rather than an API misuse. This is unenforceable by assert (any positive rho is "valid"), and the guess-dependence is load-bearing for golden agreement (the Newton path must match C), so the contract deserves to be loud.

**Fix:** Keep the parameter name `pp` (it mirrors C's u2p_solver_W signature), but state the in/out contract in both doc comments. For u2pSolverW (line 169): "pp is in/out — it must contain the cell's current primitives on entry; they seed the Newton iteration (C: u2p.c uses the previous step's p). Passing anything else diverges from C's iteration path or fails with -150." For u2pMhd (line 358): note that on total failure the incoming pp is restored unchanged, so callers must pass the previous good primitives. Drop the isFinite assert — a zeroed buffer has finite rho=0, so it would not catch the failure mode.

### Hot path & inlining

#### `build.zig:5` — Default build mode is Debug; problem executables have no preferred release mode

*Hot path & inlining — downgraded by verifier*

standardOptimizeOption(.{}) leaves the default at Debug, so `zig build puffy` / `zig build run-puffy` silently produces a Debug binary: every std.debug.assert in Field.cellOffset, FaceStore.offset, MetricCache.cellIndex fires per access, all slice indexing is bounds-checked, and nothing is inlined. For a production GRRMHD run that is a many-fold slowdown that can quietly cost days. docs/USER_GUIDE.md:82 does tell users to pass -Doptimize=ReleaseFast, but the footgun is the default. Zig's ReleaseFast does not relax IEEE float semantics (no ffast-math unless @setFloatMode(.optimized) is used), so golden/ulp agreement is unaffected by mode. Re bounds checks (the (f) question): in ReleaseFast asserts and bounds checks vanish, so slice hoisting is moot there; in ReleaseSafe/Debug the existing load/store-into-[NV]f64 convention already reduces per-cell checks to one slice bound — the remaining offenders are the per-component get/set loops listed in the separate finding.

**Fix:** Do NOT pass .preferred_optimize_mode = .ReleaseFast: in Zig 0.16 that still defaults to Debug (it only maps -Drelease/--release to the preferred mode) and it replaces the -Doptimize option with a bool -Drelease, breaking the documented `zig build run-puffy -Doptimize=ReleaseFast` (docs/USER_GUIDE.md:82,95,700). Instead, make an accidental Debug production run visible: in the problem executables (e.g. PROBLEMS/puffy/main.zig or shared runner code), log the active mode at startup — `std.log.info("build mode: {t}", .{@import("builtin").mode});` — and optionally print a prominent warning when the mode is .Debug for a non-test run.

#### `koral/field.zig:35` — Tiny per-cell index/getter helpers on the hot path are not `inline fn`

*Hot path & inlining — downgraded by verifier*

The author's stated goal is inline functions in hot paths; currently only relele.dot/kron and metric.dpart are marked. The tiny leaf helpers called O(NV x cells x passes) per step are plain fns: Field.cellOffset/get/set/load/store (field.zig:35-64), FaceStore.offset/get/set/load/store (sim.zig:147-168), MetricCache.cellIndex/faceIndex/kr (precompute.zig:191/202/323), Grid.xl/yl/zl/xc/yc/zc/size (grid.zig:87-119), Sim.flagIdx/scGet/scSet (sim.zig:440-455), and laxf/hll (flux/laxf.zig:10/16). Calibration per Zig guidance: the whole koral tree compiles as one compilation unit, so in ReleaseFast LLVM will almost certainly inline all of these anyway — explicit `inline` there is documentation, not speed. Where it is load-bearing is Debug and ReleaseSafe: Debug performs no inlining at all, and that is the default mode for `zig build test`, i.e. every golden test pays a function call + argument spill for each of the millions of get/set/offset invocations. Do NOT spray inline on the mid-size kernels (fFluxPrime, p2u, u2pSolverW, avg2point) — that bloats code and preempts better compiler heuristics.

**Fix:** If Debug-mode `zig build test` turnaround matters, mark the tiny index-math/getter leaf helpers `inline fn`: Field.cellOffset/get/set/load/store (koral/field.zig:35-64), FaceStore.offset/get/set/load/store (koral/sim.zig:147-168), MetricCache.cellIndex/faceIndex/kr (koral/metric/precompute.zig:191/202/323), Grid.xc/yc/zc/xl/yl/zl/size (koral/grid.zig:87-119), Sim.flagIdx/scGet/scSet (koral/sim.zig:440-455), and laxf/hll (koral/flux/laxf.zig:10/16). This only affects Debug — ReleaseSafe and ReleaseFast already inline these via LLVM since the tree is one compilation unit — and inlining does not change FP results, so golden agreement is unaffected. Measure one slow golden test before/after to confirm the win; the alternative of simply running `zig build test -Doptimize=ReleaseSafe` may make the change unnecessary. Do not add `inline` to larger kernels (fFluxPrime, p2u, u2pSolverW, avg2point).

#### `koral/metric/precompute.zig:59` — applyKrisCorrection computes the full CoordData (inverse + 64 Christoffels) at six face points per cell just to read gdet

*Hot path & inlining*

Lines 59-60 call metric.compute() at the hi/lo face for each of the three directions (6 calls per cell during fillCenters), and metric.compute performs the 16-cofactor dual inversion and the full Christoffel assembly, all of which is discarded except `.gdet`. On top of that, fillFaces (lines 285-319) recomputes metric.compute at the same face points again to populate gb/gconb. For a production 3D grid this makes cache construction roughly an order of magnitude more expensive than needed. Init-time only, so no wrong physics — but it is the dominant startup cost as grids grow.

**Fix:** Add a cheap module-level helper in metric.zig, e.g. `pub fn gdetAt(coords, mp, x) f64` that evaluates gcovDual + det4 only (skip inv4 and the Christoffel loops), and use it in applyKrisCorrection. Optionally reorder so fillFaces runs first and the correction reads face gdet from gb[d] (it is stored at offset 3*5+4), eliminating the recomputation entirely.

*Hand-verified: lines 59-60 call metric.compute(...).gdet — full CoordData incl. 64 AD Christoffels for one scalar, 6x per cell. Init-time only (fillCenters), so low.*

#### `koral/metric/precompute.zig:224` — MetricCache gcon/gconb column-4 slots (rows 0..2) are never initialized but are copied on every fillGeometry

*Hot path & inlining*

init() allocates self.gcon and self.gconb[d] without @memset (the memset at precompute.zig:158-161 only runs when opts.faces is false), and storeBlocks (line 224) writes only the 4x4 gcon block plus slot [3][4]=gttpert — slots i*5+4 for i in 0..2 stay undefined. geometryFromBlocks (341-345) then copies all 4x5 entries into Geometry.GG on every fillGeometry/fillGeometryFace call, so undefined heap f64s are read and propagated on the hottest accessor in the code. Nothing currently reads GG[0..2][4] downstream, so results are unaffected — but geometryAt (line 103) explicitly zeroes those same slots, so the cached and on-the-fly paths return different Geometry bytes, valgrind/msan will flag the reads, and any future consumer of GG[i][4] gets garbage only on the cached path.

**Fix:** In storeBlocks, zero the three unused gcon column-4 slots to match geometryAt (e.g. `for (0..3) |i| dst_gcon[off + i * 5 + 4] = 0;`), or alternatively @memset self.gcon and self.gconb[d] unconditionally after allocation in init(). Either makes cached Geometry bytes deterministic and identical to the on-the-fly path.

#### `koral/sim.zig:1281` — Sweep refills identical face geometry twice per face and recomputes loop-invariant dx5/stencil loads

*Hot path & inlining*

Inside sweep's innermost i-loop: (a) the `dor` branch fills face geometry at i+1 (sim.zig:1293) which is byte-identical to the `dol` fill at the next iteration (1281) — every interior face geometry is filled twice, and fluxesAtFaces (1356) fills it a third time; each fillGeometryFace copies 40 f64 plus a divide and sqrt for alpha. (b) dx5 (1244-1250) depends only on (i, dim) but is recomputed for every (c0,c1) cross-cell — 10 xl() evaluations per cell that could be built once per sweep direction. (c) the 3-to-5-point stencil (1259-1270) reloads all rows per i although consecutive i share all but one row (pm1 of i+1 == p0 of i); a rolling window keeps 4 of 5 loads. All three are pure hoists of identical values — bit-identical results, no C FP-shape change. Individually small, together they are a measurable slice of the sweep, which runs per cell per active dim twice per step.

**Fix:** The worthwhile piece is the face geometry: carry the dor-branch Geometry (face i+1) into the next iteration as the dol-branch geometry (seed at i = -1, where only dor runs), halving fillGeometryFace calls in sweep — each call copies 40 f64 plus a divide and sqrt, and results are bit-identical since it is a pure cache read. The dx5 hoist (10 xl flops/cell) and stencil rolling window are optional micro-polish: p.load is a single contiguous ~104-byte memcpy of L1-hot data, so savings there are marginal and may not justify diverging the loop structure from the C original it mirrors. The third fill in fluxesAtFaces is a separate pass and fine to leave.

### Data-oriented design

#### `koral/field.zig:35` — Cell-index math duplicated in four places with recurring usize/i64 cast churn

*Data-oriented design*

The ghost-offset+stride mapping is independently implemented in Field.cellOffset (field.zig:35-42), MetricCache.cellIndex (precompute.zig:191-198) and faceIndex (precompute.zig:202-222), FaceStore.offset (sim.zig:147-153), and indirectly flagIdx (sim.zig:440-442), which derives a cell number by dividing p's cellOffset by NV — an awkward reach-through that couples flag indexing to the primitives field's layout. Each copy re-does the @intCast(ix + @as(i64, @intCast(g.ngx))) dance because Grid stores ghost depths as usize while cell indices are signed; FaceStore already solved this by caching i64 copies of ngx/ngy/ngz, Field and MetricCache did not. If the documented plan to flip Field to AoSoA (field.zig:6-8) ever happens, MetricCache/FaceStore/flags each encode the x-fastest order separately and must be changed in lockstep.

**Fix:** Add Grid.cellIndex(ix,iy,iz) -> usize (with signed ghost offsets or cached i64 ngx/ngy/ngz on Grid) and express Field.cellOffset and MetricCache.cellIndex through it; MetricCache.faceIndex and FaceStore.offset can likewise share a face-index helper (their sizings already agree). Most importantly, give flags their own layout-independent index instead of p.cellOffset/NV — that reach-through is the one place that silently breaks if the documented AoSoA flip of Field (field.zig:6-8) ever happens; MetricCache/FaceStore have independent storage and are duplication, not coupling. Integer-only change, so golden-file/C-diff agreement is untouched.

#### `koral/geometry.zig:8` — Geometry is a ~416-byte value re-gathered from the cache up to 3x per face and ~5x per cell each stage; it could be a pointer view into the cache blocks

*Data-oriented design — downgraded by verifier*

Geometry is ~416 bytes (gg+GG = 320B, xxvec 32B, ix/iy/iz 24B, scalars + padding). Every fillGeometry/fillGeometryFace call gathers 40 f64 from two separate heap arrays, recomputes alpha (sqrt + divide) and xxvec (three grid coordinate calls). Per RK stage a cell-center Geometry is rebuilt ~5 times (calcWavespeeds sim.zig:813, u2pRows :1010, metricSource :1148, implicitRowsWorker :1569, updateEntropy :1523), and each face Geometry is built 3 times (the flux sweep builds face i and face i+1 in each cell iteration, sim.zig:1281/:1293, and fluxesAtFaces:1356 builds the same face again for p2u of the biased states). The cost is small relative to the physics per call (<1-2% of a stage), so this is polish — but the fix also simplifies the design: kernels already take *const Geometry, and relele/invert already take *const [4][5]f64, while the cache stores blocks in exactly that [4][5] layout. Geometry.gg/GG could therefore be *const [4][5]f64 pointers straight into the cache, with alpha precomputed into the currently-undefined gcon col-4 slots (see the storeBlocks finding), shrinking Geometry to ~80B and eliminating all repeated gathers and sqrts. Alternatively, materialize Geometry arrays once at init (~416B/cell + 3 face sets).

**Fix:** If the ~1-2%/stage geometry-rebuild cost ever matters, prefer materializing precomputed Geometry values at init (cell centers + 3 face sets, ~+1.7KB/cell on top of the existing ~4KB/cell metric cache) and have fillGeometry/fillGeometryFace return *const Geometry: kernels already take *const Geometry, Geometry stays a plain value type, and the from-scratch constructors (geometryAt, geomFromRecord in the golden harness, puffy.fillGeometryBL) keep working unchanged. Avoid the pointer-view variant (gg/GG as *const [4][5]f64 into the cache): it would break those three self-contained constructors and turn a hazard-free value into a lifetime-managed view (dangling-pointer risk) for the same perf gain.

#### `koral/metric/precompute.zig:109` — MetricCache stores ~2 KB/cell; symmetric tensors kept full-size (accounting requested by review)

*Data-oriented design*

Per-cell precomputed bytes: g 20*8=160, gcon 160, kris 64*8=512, dxdx 2*128=256 (dead, see separate finding) = 1088 B at centers; faces add ~6*160=963 B -> ~2.05 KB/cell, ~293 MB on the 390x366 PUFFY grid (vs 104 B of primitives). gcov/gcon are symmetric (10 unique + 5 extras = 15 slots, stored as 20, mirroring C's gSIZE=20) and kris is symmetric in its lower indices (40 unique of 64, stored full 64 mirroring C's gKr; applyKrisCorrection at precompute.zig:66 explicitly maintains the mirror copy). Compacting to triangles would cut the cache ~35% (and much more matters for eventual 3D grids where this scales to tens of GB) with zero change to stored values — memory layout of the cache is not part of the golden-file contract, only the values are. Also note the explicit-sweep per-cell working set this feeds: p/u 208 B + face-array stores ~1.2 KB + face metric blocks 640 B + center block 320 B + kris 512 B + scal 152 B + flags 24 B + rijvisc 128 B, roughly 3.3 KB/cell per opExplicit; kris redundancy (192 B) and the scal/flag items above are the avoidable inflators.

**Fix:** Optional, no action needed now: the metric cache costs ~338 MB (~2.37 KB/cell) on the 390x366 stored PUFFY grid — centers 1088 B/cell (g 160 + gcon 160 + kris 512 + dxdx 256) plus ~183 MB of face blocks (note dim-2 faces are 2x cells when nz=1) — vs 104 B/cell of primitives. If memory pressure appears at 3D milestones (~60+ GB at 384x360x192), store gcov/gcon as triangles (g: 10+4 extras = 14 slots, gcon: 10+1 = 11) and kris as a 40-slot lower triangle behind the existing storeBlocks/geometryFromBlocks/kr accessors, cutting the cache ~33% with bit-identical stored values (cache layout is not part of the golden-file contract). Do not touch CoordData.kris or applyKrisCorrection — the symmetric writes there live in C-shaped compute code and are unaffected by cache layout. Otherwise, a one-line comment noting full-size storage is a deliberate C-shaped (gSIZE=20/gKr) trade of memory for accessor simplicity would suffice.

#### `koral/metric/precompute.zig:158` — faces=false still allocates all six face arrays full-size and zero-fills them; fillGeometryFace then silently returns NaN geometry

*Data-oriented design*

With `InitOpts.faces = false` (postprocessing mode) the gb/gconb arrays are still allocated at full size and memset to zero (precompute.zig:154-161). For the PUFFY padded grid (sx=390, sy=366) that is ~180 MB of dead zeroed memory across the six arrays. Worse, calling fillGeometryFace on such a cache is not an error: geometryFromBlocks reads the zeros and produces gdet=0 and alpha=@sqrt(-1.0/0.0)=NaN (precompute.zig:347-349) — garbage that propagates silently into downstream physics instead of failing at the misuse site. This is a sentinel-shaped hole where an optional fits.

**Fix:** Make the face storage optional: `gb: [3]?[]f64` / `gconb: [3]?[]f64` set to null when faces are skipped (or store empty slices), and have fillGeometryFace unwrap with `orelse @panic("MetricCache built without faces")` or a std.debug.assert. Misuse then fails loudly and the postprocessing-mode cache drops the ~180 MB.

*Hand-verified: gb/gconb are allocated full-size unconditionally at lines 156-157; faces=false merely zero-fills. Postprocessing-mode-only waste, so low.*

#### `koral/sim.zig:83` — Scal AoS record scatters each direction's six wavespeeds across a 152-byte record

*Data-oriented design*

scal is Field(19) — 152 B/cell (~21.7 MB). The enum groups all-left, all-right, all-max across dimensions (ahdxl..ahdzr, ahdx..ahdz, aradxl.., sim.zig:83-104), so one dim-pass of fluxesAtFaces reads slots {0,1,6,9,10,15} for dim=0 (sim.zig:1344-1375) — six 8-byte values scattered over all three cache lines of the record, for two cells (cf and cm) per face. C stores these as separate SoA global arrays (ahdxl[] etc., as the comment at sim.zig:82 notes), so the AoS regrouping is a Zig-native choice that reduced locality relative to C. Simply reordering the enum so each dimension's sextet (ahd l/r/m + arad l/r/m) is contiguous makes each flux pass touch one 48-byte run per cell. Minor additional note: the 9 arad slots exist even for configs without the radiation module (72 B/cell wasted there; PUFFY itself uses them).

**Fix:** Reorder the Scal enum per-dimension — {ahdxl, ahdxr, ahdx, aradxl, aradxr, aradx, ahdyl, ...} — so each flux dim-pass reads one contiguous 48-byte run per cell instead of six values scattered across the 160 B (20-slot, not 19 as originally stated) record. All access already goes through the ahd_l/ahd_r/ahd_m/arad_l/arad_r/arad_m tables or named enum literals, so the reorder needs no other code changes and cannot affect golden output. Optionally gate the 9 arad slots on L.hasVar(.ee) to save 72 B/cell in non-radiation configs, noting that requires making the enum (or the field count) comptime-conditional and is a larger change than the reorder.

### Idiomatic Zig

#### `PROBLEMS/puffy/main.zig:113` — Dead parameters and a dead diagnostic in the driver: writeScalars' unused allocator, collectDiag's unreported n_rad_fixup

*Idiomatic Zig*

`writeScalars` (line 110) accepts `a: std.mem.Allocator` and immediately discards it with `_ = a;` (line 113) — both call sites (171, 187) thread the allocator through for nothing. And `collectDiag` counts `n_rad_fixup` per cell (lines 55, 75-77) but `scalarRow` never reads it: the ScalarRow gets only n_hd_fixup, s.n_radimp_failures, and n_nan, so the rad-fixup census is computed every output cadence and dropped. Either it was meant to be in scalars.dat (C's scalars machinery does track it) or it is dead weight.

**Fix:** Drop the unused `a: std.mem.Allocator` parameter from writeScalars (main.zig:110, `_ = a;` at 113) and update the two call sites (lines 170 and 187). For n_rad_fixup: the flag is genuinely populated (sim.zig:1051 sets .rad_fixup = -1 on u2pRad corrections regardless of do_u2prad_fixups), so the counted census is meaningful — either add an n_radfix column to dump.ScalarRow/scalar_header/appendScalarLine and wire diag.n_rad_fixup through scalarRow, or delete the Diag field and the counting branch (main.zig:55, 75-77) if the rad-fixup census is not wanted in scalars.dat.

#### `koral/geometry.zig:14` — Geometry.ifacedim is a -1 sentinel and, together with ix/iy/iz, is write-only dead data

*Idiomatic Zig*

`ifacedim: i8` uses -1 as a "cell center" sentinel where an optional fits. Moreover, grepping the repo shows no code ever reads geo.ifacedim, geo.ix, geo.iy, or geo.iz — they are only assigned (precompute.zig:89/333, testing/golden.zig:216, problems/puffy.zig:404, magn/dynamo.zig:112). The C branch that consumed ifacedim (rad.c:3782 f_flux_prime_rad_total) was ported differently in Zig (sim.zig rijviscFace takes `dim` explicitly), so mirroring the C struct field buys nothing here — C-diffability is not at stake for a never-read struct member.

**Fix:** Delete the four dead fields (ix, iy, iz, ifacedim) from Geometry and their assignments at precompute.zig:86-89/330-333, testing/golden.zig:213-216, problems/puffy.zig:404-406, magn/dynamo.zig:112-114, and io/scalars.zig:48-50 (a fifth write site). Simple deletion is preferable to the optional-enum replacement since no kernel consumes a face marker today (rijviscFace already takes dim explicitly); if a face marker is later needed, reintroduce it as a named `pub const Face = enum(u2) { x, y, z };` with a `face: ?Face = null` field rather than a -1 sentinel.

#### `koral/metric/precompute.zig:146` — MetricCache.init has 11 allocations and no errdefer — partial failure leaks and leaves gb/gconb undefined

*Idiomatic Zig, Purity & hidden state — flagged independently by 2 reviewers; downgraded by verifier*

init performs five `try a.alloc` calls (precompute.zig:146-150) followed by six more in the `for (0..3)` loop (156-157). If any allocation after the first fails, everything allocated so far leaks, and because `gb`/`gconb` start as `undefined` (151-152) there is no way to even write a recovery path: the partially-filled arrays cannot be distinguished from garbage. This is exactly the multi-allocation init pattern errdefer exists for. (Contrast Field.init, which is single-allocation and fine.)

**Fix:** MetricCache.init makes 11 allocations with no errdefer, so a mid-init OOM leaks the earlier slices — but note this matches a codebase-wide convention (Sim.init, its only non-test caller, makes ~35 errdefer-less allocations around it, and OOM at startup is fatal anyway). Worth deciding once for the codebase rather than spot-fixing: either explicitly accept "startup OOM is fatal, OS reclaims", or make inits OOM-safe. If fixing, the cleanest form here is a single backing allocation sliced into g/gcon/kris/dxdx_my2out/dxdx_out2my/gb/gconb (total length is computable up-front from cellCount and faceCount), reducing init to one try and deinit to one free; otherwise add errdefer a.free(...) after each scalar alloc and initialize gb/gconb to empty slices (.{ &.{}, &.{}, &.{} }) with an errdefer loop freeing them before the face loop.

#### `koral/physics/radiation.zig:151` — Hand element-copy loops where whole-array assignment / @bitCast is idiomatic

*Idiomatic Zig*

Pure data copies (not C FP expression shapes) written as index loops: radiation.zig:151 `for (0..6) |i| aval[i] = a0[i];` and radiation.zig:168 `for (0..6) |i| aval[6 + i] = a1[i];`; radvisc.zig:358-361 flattens the [4][4]f64 rvisc into a [16]f64 with a nested loop before sim.rijvisc.store. Zig has direct forms for all three that state intent and avoid the index bookkeeping.

**Fix:** radiation.zig: `aval[0..6].* = a0;` and `aval[6..12].* = a1;`. radvisc.zig: `const t: [16]f64 = @bitCast(rvisc);` ([4][4]f64 and [16]f64 are layout-identical) and pass &t to store.

#### `koral/physics/radvisc.zig:247` — `&(sim.opt.opac orelse return 0)` copies the whole Params struct to a stack temporary per cell

*Idiomatic Zig*

sim.opt.opac is ?radforce.Params; `orelse` yields the payload by value, so this line copies the entire Params (which embeds thermo.Consts — roughly 30 f64 fields plus Units and Composition, ~300 bytes) into an anonymous temporary and takes its address, once per cell per step inside calcRadVisccoeff. Besides the needless copy, pointing opp at a block-lifetime temporary is subtler than pointing into sim.opt. The codebase already uses the idiomatic zero-copy form elsewhere: sim.zig:823 does `if (self.opt.opac) |*op|`.

**Fix:** Use optional pointer capture: `const opp = if (sim.opt.opac) |*o| o else return 0;` — no copy, opp is *const radforce.Params into sim.opt, and it matches the sim.zig idiom.

#### `koral/recon/recon.zig:26` — avg2pointScalar's else-catchall silently degrades any unknown order to donor cell

*Idiomatic Zig — downgraded by verifier*

`order` is a raw u8 and the switch is `1 => linear, 2 => ppm, else => donor`. Only 0/1/2 are meaningful (C: INT_ORDER), but a caller passing 3 (e.g. a future scheme wired into config.Reconstruction but not implemented here, or a bad base_order/param combination) is silently reconstructed at order 0 instead of failing. The order-reduction arithmetic in avg2point (line 43) is careful about underflow, so the only unguarded entry point is this catchall.

**Fix:** Make the domain explicit in avg2pointScalar: `0 => .{ .ul = u[2], .ur = u[2] }, 1 => linearMinmod(u, theta), 2 => ppm(u, dx), else => unreachable`. This is pure API hardening: the production path is already compile-time safe (sim.zig:181-185 exhaustively switches over the 3-member config.Reconstruction enum, and recon.zig:43 clamps eff to 0), so the change only documents the accepted domain and traps misuse of the pub scalar entry point in safe builds. Behavior for legal orders 0/1/2 and golden records are unchanged.

#### `koral/relele.zig:240` — convVelsCore's conversion dispatch ends in an else-catchall instead of an exhaustive switch

*Idiomatic Zig*

The if/else chain enumerates five (which1, which2) pairs and the final `else` is documented only by the comment `// VELR -> VEL3`. Correctness currently rests on manual enumeration of the 6 ordered pairs; if VelType ever grows (or a pair is mistyped during a refactor) the catchall silently routes to the VELR→VEL3 math. This is control flow, not transcribed FP arithmetic, so restructuring it costs nothing in golden fidelity — the branch bodies stay byte-identical.

**Fix:** Replace the if/else chain with nested exhaustive switches so the compiler proves coverage: `switch (which1) { .vel4 => switch (which2) { ... }, .vel3 => ..., .velr => ... }`, keeping every branch body byte-identical (golden fidelity is unaffected — this is control flow only). Handle the diagonal either by folding the same-type cases into the switch (this mirrors C, which enumerates VEL3->VEL3, VEL4->VEL4, VELR->VELR explicitly) or by keeping the `which1 == which2` fast path in front and marking the diagonal switch arms `unreachable`. Note the C original also ends with an explicit `else { my_err("velocity conversion not supported"); return -1; }` fallback, so the current Zig catchall silently routing unmatched pairs into the VELR->VEL3 math is a defensive property C had that the port lost; the exhaustive switch restores it at compile time.

#### `koral/sim.zig:143` — Split allocator convention: FaceStore.deinit(allocator) vs Field.deinit() in the same aggregate

*Idiomatic Zig*

`FaceStore.deinit(self, a: std.mem.Allocator)` (sim.zig:143-145) is unmanaged-style, while `field_mod.Field` stores its allocator and exposes `deinit(self)` with no argument (koral/field.zig). Sim.deinit (sim.zig:393-418) consequently mixes `@field(self, name).deinit()` for Fields with `self.pb_l[d].deinit(a)` for FaceStores, plus raw `a.free` for flags/emf/scaleth. Both conventions are valid Zig 0.16, but mixing them inside one owner struct is exactly the kind of inconsistency that produces a wrong-allocator or double-free bug when someone later refactors deinit or adds a buffer following the "other" pattern.

**Fix:** Unify the deinit convention for Sim-owned storage types. FaceStore documents itself as "like Field" (sim.zig:113) but diverges in ownership style. Smallest change: store `a: std.mem.Allocator` in FaceStore (set in init, sim.zig:126-141), change deinit to `fn deinit(self: *Self) void`, and add the `self.* = undefined` poisoning that Field.deinit already does (field.zig:29-32). Then Sim.deinit's FaceStore loop (sim.zig:402-408) matches the Field pattern. (Alternatively adopt unmanaged as house style and strip the allocator from Field — but that ripples into params.zig and tests, so the managed direction is the cheaper unification.) Note: the mismatch is signature-enforced, so this is consistency polish, not a latent-bug fix.

#### `koral/sim.zig:195` — SpecificBc callback cannot return errors, forcing `catch unreachable` on fallible physics in the PUFFY boundary condition

*Idiomatic Zig — downgraded by verifier*

`pub const SpecificBc = *const fn (...) [NV]f64` (sim.zig:195-203) has no error union, so the PUFFY implementation swallows genuinely reachable failures with `catch unreachable`: koral/problems/puffy.zig lines 671 and 696 (`frames.transPmhdCoco`), 700, 705, 712, 717 (`relele.convVels`). These calls return `relele.Error{VelocityConversionFailed, SpacelikeVelocity}` — exactly the states a boundary-adjacent cell can transiently reach in a hard radiation-MHD run (the whole fixup machinery exists because primitives do go unphysical). If the outermost domain cell copied by the .xhi outflow BC ever carries a spacelike VELR, this panics in Debug and is undefined behavior in ReleaseFast, instead of propagating like every other inversion failure. The call site `setBcCell` (sim.zig:616) is already `Error!void` and uses `try` for the p2u right below, so the plumbing exists; this is a Zig-native API-boundary choice, not a C-fidelity constraint (C's set_bc returns an int).

**Fix:** Future-proofing, not a live bug: none of the six catch unreachable in puffy.zig's Bc.calc (lines 671, 696, 700, 705, 712, 717) can actually fire — velr→vel4 goes through calcAlpgam which clamps instead of erroring (C-mirrored), transPmhdCoco's vel4→velr uses convVelsUt which skips utInUcon, and the two bare vel4→velr calls cannot see delta<0 outside the ergosphere. But that invariant is undocumented and spans three modules, and SpecificBc is generic framework API. Since setBcCell (sim.zig:607) is already Error!void, consider changing SpecificBc to return Error![NV]f64, using `try self.opt.specific_bc.?(...)` at sim.zig:616, and replacing the six catch unreachable with try; alternatively, keep the signature and add a brief comment at the catch unreachable sites stating why each is unreachable at the outer boundary.

#### `koral/sim.zig:317` — Sim.init has ~35 fallible allocations and zero errdefer — any mid-init failure leaks everything allocated so far

*Idiomatic Zig — downgraded by verifier*

`init` (sim.zig:317-391) sequentially allocates: the MetricCache (332), 14 Fields via `inline for` (338-344), 15 FaceStores (345-351), `scal` (352), `flags` (353), 3 `emf` arrays (355-359), `vecpot`/`rijvisc`/`dynA` (360-362), and `scaleth` (363) — every one with `try` and no `errdefer`. If e.g. the FaceStore alloc at index 7 fails, the cache plus 14 Fields plus 7 FaceStores leak. For a production PUFFY grid this is hundreds of MB per failed init, and any DebugAllocator-based test of init failure will report leaks. Relatedly, the field-name list `.{ "p", "u", "ut0", ... }` is duplicated verbatim between init (338) and deinit (395-399), so adding a stage buffer requires editing two lists in lockstep.

**Fix:** Hoist the duplicated field-name tuple into a single comptime const (e.g. `const field_names = .{ "p", "u", "ut0", ... };`) referenced by both init (sim.zig:338) and deinit (sim.zig:395) so the two lists cannot drift when a stage buffer is added. Optionally, if init-failure cleanup is ever wanted (e.g. for future failing-allocator tests), add errdefer unwinding or move the grid-lifetime storage to an arena — but note Sim is returned by value, so an owned arena must use the std.heap.ArenaAllocator.State promote pattern rather than storing a derived Allocator. Given that Sim.init runs once at startup and failure is fatal, the errdefer work is not urgent.

#### `koral/testing/golden.zig:327` — readKini leaks k.vars on malformed-file error paths; coordsFromId turns corrupt file data into unreachable

*Idiomatic Zig*

In `readKini`, `k.vars = try a.alloc(i32, nvars);` (line 327) has no errdefer, so the subsequent `return error.BadGoldenFile` when `raw.len != off + nvals * 8` (line 334) and a failure of `k.data = try a.alloc(f64, nvals);` (line 336) both leak the vars slice — under std.testing.allocator a truncated/corrupt .kini golden reports a leak on top of the real error, muddying the failure. (readKstp and readGolden are clean; readKstp even has the right `errdefer aw.deinit()` pattern.) Separately, `coordsFromId` (lines 228-235) ends in `else => unreachable` on a value read straight from the golden file, so a corrupt coordinate id is checked-panic in Debug and UB in ReleaseFast test builds rather than the `error.BadGoldenFile` every other validation in this file produces.

**Fix:** Add `errdefer a.free(k.vars);` immediately after the vars allocation. Change `coordsFromId` to return `!config.Coords` (or accept a fallback) with `else => error.BadGoldenFile`, matching the file's existing validation style.

### Naming

#### `PROBLEMS/puffy/main.zig:119` — Local `writePrimDump` shares its name with dump.writePrimDump but does a different job
**Fixed (2026-07-06):** renamed the pure serializer to `dump.serializePrimDump`.

*Naming*

main.zig defines `fn writePrimDump(io, out_dir, a, s, idx)` (writes a .kdmp file to disk) which internally calls `dump.writePrimDump(SimT, s, bytes)` (pure serialization into a caller buffer). Two functions with the identical name at different layers of the same operation force the reader to check the namespace to know whether filesystem I/O happens — exactly the situation the pure/impure split in dump.zig was designed to make obvious.

**Fix:** Rename the pure serializer to dump.serializePrimDump (one definition, one call site at main.zig:122) — this keeps main.zig's local file-writing wrappers (writeScalars, writePrimDump) consistently named while making the pure/impure layer split visible at the call site. Alternatively rename the main.zig wrapper to writePrimDumpFile, but then rename writeScalars analogously for consistency.

#### `koral/config.zig:47` — Enum `VelType` is vaguer than its sibling config enums and mismatches its own fields
**Fixed (2026-07-06):** superseded by the `relele.VelType` merge (koral/relele.zig:15 above) -- `config.VelType` no longer exists as a separate type.

*Naming*

The sibling config enums are named after the concept they select: `Reconstruction`, `FluxMethod`, `TimeStepping`, `Coords`. `VelType` ('type of velocity'?) does not say what it parametrizes — its own doc comment ('Velocity primitive parametrization (C: VELPRIM)') and the two fields typed with it (`velprim: VelType`, `velprim_rad: VelType` at lines 55-56) both use the VELPRIM vocabulary. A reader grepping for the C macro VELPRIM finds the fields but not the type.

**Fix:** Rename config.zig's `VelType` to `VelPrim` so the type, the `velprim`/`velprim_rad` fields, and the C macro VELPRIM share one greppable name. This also removes the name collision with the unrelated `relele.VelType` (enum(u8) mirroring C mnemonics.h values, used throughout the physics layer), which stays as-is. config.VelType has no references outside config.zig, so the rename touches only lines 47 and 55-56.

#### `koral/field.zig:21` — Allocator struct fields and parameters named single-letter 'a'
**Fixed (2026-07-06):** renamed to `allocator` across field.zig, precompute.zig (MetricCache), params.zig, sim.zig, io/dump.zig, testing/golden.zig, PROBLEMS/puffy/main.zig.

*Naming*

The allocator is stored/passed as `a` in several public APIs: Field.a field (koral/field.zig:21, used as `self.a.free(...)` at line 30), MetricCache.a field (koral/metric/precompute.zig:110, freed through at lines 170-178), and the `a: std.mem.Allocator` parameters of Params.load/parse/setField/parseValue (koral/params.zig:62, 69, 100, 114). A struct field has the longest possible live range, and `a` is opaque at use sites far from the declaration (e.g. `self.a.alloc(...)` inside MetricCache.init). Zig std and most Zig codebases use `allocator` (or `gpa`); the rename is mechanical and makes every alloc/free/deinit site self-explanatory.

**Fix:** Rename the allocator fields/parameters from `a` to `allocator` (or `gpa`) across the codebase (field.zig, precompute.zig, params.zig, sim.zig, io/dump.zig, testing/golden.zig, PROBLEMS/puffy/main.zig). Beyond matching Zig std convention, this frees the identifier `a` for its canonical GR meaning: MetricParams.a is the BH spin (C: BHSPIN), and MetricCache currently has both `self.a` (allocator) and `self.mp.a` (spin) in one struct. Leave MetricParams.a untouched — it mirrors C/GR notation.

#### `koral/grid.zig:112` — `Grid.size(i, dim)` reads as grid extent, not per-cell width
**Fixed (2026-07-06):** renamed `Grid.size` -> `Grid.cellSize` (all ~35 call sites).

*Naming*

`size` returns the width of one cell (C: get_size_x, per the doc comment), but sits on a type whose other methods are extent queries — `sx()`, `sy()`, `sz()`, `cellCount()` — so `grid.size(i, 0)` at a call site is easy to misread as 'size of the grid in dimension 0'. The doc comment carries the C cross-reference, so the name itself does not need to preserve C's 'size' wording.

**Fix:** Rename `Grid.size` to `cellSize` (keep the `C: get_size_x (finite.c:2457)` doc comment for greppability). Mechanical rename across ~35 call sites; it disambiguates sites like puffy_tests.zig:314 `s.grid.size(0, 0)` that currently read as a whole-grid extent query alongside `sx()/sy()/sz()/cellCount()`.

#### `koral/math/dual.zig:17` — `Dual3.con` abbreviation collides with 'con = contravariant' and breaks symmetry with `variable`
**Fixed (2026-07-06):** renamed `Dual3.con` -> `Dual3.constant` across dual.zig, forms.zig, metric.zig, coco.zig.

*Naming*

`con(x)` means 'constant (zero derivative)', but everywhere else in this codebase 'con' means contravariant (CoordData.gcov/gcon, MetricCache.gcon/gconb, Geometry.GG) — so `Dual3.con` at heavy-use sites like forms.zig (Dual3.con(0), Dual3.con(-1), Dual3.con(1)) and metric.zig momentarily reads as metric vocabulary. Its sibling constructor `variable(x, slot)` (line 22) is fully spelled out, making the abbreviation inconsistent within the same pair. Relatedly, the constant-operand helpers pair oddly: multiply-by-constant is `scale` (line 59) while add-constant is `addc` (line 64) — one descriptive word, one single-letter suffix.

**Fix:** Rename `Dual3.con` to `Dual3.constant` (Zig reserves `const`, not `constant`), matching the fully spelled-out sibling `variable` and avoiding the misread as "contravariant" — the dominant meaning of "con" in this codebase (gcon, gconb, GG, ConCov, normalObsNcon). Mechanical rename touching only dual.zig, metric/forms.zig, metric/metric.zig, metric/coco.zig, and metric/tests.zig. Optionally rename `addc` to `addConst` (or `shift`, pairing with `scale`).

#### `koral/metric/precompute.zig:251` — Local `g` (grid) collides with field `self.g` (metric blocks) inside MetricCache methods
**Fixed (2026-07-06):** renamed the local bindings from `g` to `grid` in all MetricCache methods; `self.g` (the metric-block field) is untouched.

*Naming*

MetricCache has a field `g: []f64` holding the 4x5 covariant metric blocks (line 118), yet its methods bind `const g = &self.grid;` — so `g` means 'grid' and `self.g` means 'metric array' in the same function body. fillCell is the worst case: `const g = &self.grid;` at line 251 and `storeBlocks(self.g, ...)` at line 258, eight lines apart. The same local-vs-field double meaning recurs in cellIndex (line 192), faceIndex (line 203), fillCenters (line 237), fillFaces (line 286), fillGeometry (line 355), and fillGeometryFace (line 370). The field name `g` mirrors C's global array and is fine to keep; the locals are the free-to-fix side.

**Fix:** Rename the local bindings from `g` to `grid` (i.e. `const grid = &self.grid;`) in all MetricCache methods, reserving `g`/`self.g` for the metric-block array as in C.

#### `koral/metric/precompute.zig:354` — fillGeometry/fillGeometryFace have a mutating 'fill' prefix but are pure accessors
**Not fixed -- skipped by request (2026-07-06):** the rename touches ~130 call sites across 17 files for a modest gain, and the finding was itself downgraded by the review's own verifier; left as-is.

*Naming — downgraded by verifier*

MetricCache.fillGeometry (line 354) and fillGeometryFace (line 369) take `self: *const MetricCache` and return a Geometry by value — they mutate nothing. The name is transcribed from C's fill_geometry, which fills a caller-supplied out-struct, so in C the verb is accurate; in the Zig API it misleads. The problem is sharpened by two in-file contrasts: (1) fillCenters (line 236), fillFaces (line 285), and fillCell (line 250) in the same struct DO mutate the cache, so 'fill' carries real mutation semantics right next door; (2) the module-level geometryAt (line 82) is a similarly named, similarly purposed function with different semantics (computes the metric from scratch instead of reading the cache), so a reader has to open both bodies to learn which one recomputes. The doc comments already carry the C cross-references ('C: fill_geometry (metric.c:1884)'), so renaming loses no C-diffing traceability.

**Fix:** Consider renaming MetricCache.fillGeometry -> geometry and fillGeometryFace -> faceGeometry, keeping the existing "C: fill_geometry (metric.c:1884)" / "C: fill_geometry_face (metric.c:1971)" doc comments for traceability — this matches the precedent already set by geometryAt (a rename of C's fill_geometry_arb). Note the trade-off: ~130 call sites across 17 files lose visual name correspondence with C when diffing hot loops, so weigh that against the clearer read-only semantics. Drop the geometryAt -> computeGeometryAt rename; a doc-comment note distinguishing cached vs recomputed is sufficient.

#### `koral/physics/flux.zig:34` — Three different conventions for marking tensor index placement in local names (bare/`_con`/`_up`)
**Fixed (2026-07-06):** adopted `tij_uu`/`tij_ud` and `rij_uu`/`rij_ud` in flux.zig's fFluxPrime and radiation.zig's calcFfRtt/pradFf2Lab.

*Naming*

Locals holding the same up-up vs mixed tensor pairs are suffixed three different ways: flux.zig lines 34-35 use `tij` for T^munu and bare single-letter `t` for T^mu_nu (with a 40-line live range), while lines 87-88 of the same function use `rij_con` for R^munu and `rij` for the mixed form; radiation.zig calcFfRtt (lines 79-80) uses `rij_up`/`rij` for the identical pair. In GR code the index placement is exactly the information a local's name should carry, and the current mix means `rij` is up-up in one function and mixed in another.

**Fix:** Adopt one suffix convention for index placement in these Zig-native locals — e.g. `tij_uu`/`tij_ud` and `rij_uu`/`rij_ud` — eliminating the bare `t` in fFluxPrime. Also apply it in radiation.zig's pradFf2Lab, where a single `var rij` is reassigned through up-up, boosted, and mixed index states under one name.

#### `koral/physics/mhd.zig:1` — Module named mhd.zig contains only magnetic-field helpers while the MHD stress tensor lives in hydro.zig
**Fixed (2026-07-06):** renamed `koral/physics/mhd.zig` -> `koral/physics/bfield.zig` (14 import sites updated).

*Naming*

physics/mhd.zig is the port of C's magn.c (its own doc header says "Magnetic-field four-vector helpers (C: magn.c)") and exports only bcon/bcov/B-field conversions. Meanwhile the actual MHD stress-energy tensor calc_Tij — explicitly documented as "MHD stress-energy tensor T^munu ... + b^2 terms" — lives in physics/hydro.zig (line 33). A reader hunting for the MHD tensor will open mhd.zig first and find only field-vector algebra; the module names are effectively swapped relative to their content.

**Fix:** Rename koral/physics/mhd.zig to koral/physics/bfield.zig — do NOT use magn.zig, since koral/magn/ already exists and holds the other magn.c ports (ct.zig, dynamo.zig), so a second "magn" name would add confusion. Alternatively move calcTij into it. The rename is a mechanical import-alias update across 14 files: p2u.zig, frames.zig, koral.zig, solve/invert.zig, io/scalars.zig, problems/puffy.zig, magn/dynamo.zig, physics/{hydro,flux,wavespeeds,radforce}.zig, and golden_state_test.zig / puffy_tests.zig / state_tests.zig.

#### `koral/physics/opacities.zig:173` — Inconsistent calc- prefix policy between sibling functions transcribed from identically-shaped C names
**Fixed (2026-07-06):** renamed `kappaFromState` -> `calcKappaFromState` to match its sibling `calcOpacitiesFromState`.

*Naming — downgraded by verifier*

Functions ported from C `calc_*` names sometimes keep the prefix and sometimes drop it, with no discernible rule, even within one file: opacities.zig has calcOpacitiesFromState (C: calc_opacities_from_state, line 63) directly feeding kappaFromState (C: calc_kappa_from_state, line 173) — adjacent functions, same C naming shape, opposite treatment; plus tautot/tauabs (C: calc_tautot/calc_tauabs) and kappaEsPuffy dropping it. hydro.zig keeps calcTij (line 35) but drops it for sFromU/uFromS (C: calc_Sfromu/calc_ufromS, lines 15/23). The sharpest pair is the two wavespeed entry points that the sweep calls side by side: wavespeeds.gasWavespeedsLr (drops, C: calc_wavespeeds_lr_pure) vs radiation.calcRadWavespeeds (keeps, C: calc_rad_wavespeeds). Predictability of the C-to-Zig name mapping matters in a codebase whose whole review posture is C traceability.

**Fix:** Document the calc-prefix convention for C calc_* ports (one line in a CONTRIBUTING note or module doc: e.g. "drop `calc` when the remainder is a self-standing noun phrase") and apply it to the two most jarring pairs only: rename kappaFromState -> calcKappaFromState to match its same-file sibling calcOpacitiesFromState (or rename that one to opacitiesFromState), and align the sweep's side-by-side entry points calcRadWavespeeds/gasWavespeedsLr to one style. A full-codebase rename is not warranted since every port already records its C name in a `/// C:` doc comment.

#### `koral/physics/radforce.zig:58` — Single-letter struct fields/params `c`, `p`, `u`, `ch`, `s` shadow core physics symbols (speed of light, pressure, internal energy) in Zig-native APIs
**Fixed (2026-07-06):** renamed `Params.c` -> `consts`, `Params.ch` -> `channels`, `u: Units` -> `units`, and the `p: *const Params` parameters -> `par` throughout radforce.zig.

*Naming — downgraded by verifier*

In a GR-RMHD code where `CCC` is the speed of light, `p`/`pre` is pressure (hydro.zig:63), `u`/`uu` is internal energy / conserved vector, and `pp` is primitives, the Zig-native parameter structs reuse exactly those letters for unrelated things: radforce.Params has field `c: thermo.Consts` (line 58), read as `p.c` / `const c = &p.c;` (radforce.zig:129, 244) — "p.c" reads like pressure-times-lightspeed; every radforce entry point takes `p: *const Params` (lines 242, 284, 299, 311, 325); thermo's free functions take `c: *const Consts` (thermo.zig:152, 159, 168, 180), as do opacities.calcOpacitiesFromState/kappaFromState/kappaEsPuffy (opacities.zig:63, 173, 186); Consts.init and Params.init take `u: Units` (thermo.zig:101, radforce.zig:63) where `u` elsewhere is internal energy (hydro.sFromU's `u` param); Channels is threaded as `ch` and StateIn as `s`. These are all Zig-side inventions with long live ranges, not C transcriptions, so descriptive names are free.

**Fix:** Rename the Zig-native struct fields and signature parameters, but keep the terse locals inside formula-heavy kernels. Concretely: Params field `c: thermo.Consts` -> `consts` and `ch` -> `channels` (radforce.zig:58-59); `u: Units` -> `units` in Consts.init (thermo.zig:101) and Params.init/grey (radforce.zig:63, 78); and `p: *const Params` -> `par` or `params` — highest value where it sits next to `pp` in the same signature (radforce.zig:121-127, 279-285), since p/pp both mean primitives in KORAL. Inside kernel bodies, re-alias (`const c = &par.consts;`) so multi-line opacity/four-force expressions keep the minimal `c.` prefix and stay visually diffable against the C globals (fourpi, kappaCGS2GU). Optionally disambiguate thermo.zig, where `c` currently means Composition in method receivers (lines 52-64) but Consts in free functions (152/168/180).

#### `koral/physics/radforce.zig:168` — Local `kr` (a KappaResult) collides with the codebase-wide meaning of Kr = Christoffel symbols
**Fixed (2026-07-06):** renamed the local to `kappa_result`.

*Naming*

fillRadState binds `const kr = switch (p.kappa) ...` holding an opacities.KappaResult, live from line 168 to line 216. Everywhere else in the codebase `Kr`/`kr` is the Christoffel-symbol vocabulary inherited from KORAL — e.g. sim.cache.kr(k, i, j, ix, iy, iz) is the Christoffel accessor called from radvisc.zig:184/188. Reusing the identifier for "kappa result" invites misreading in the one file most concerned with opacities and geometry together.

**Fix:** Rename the local to `kappa_result` (or `kres`).

#### `koral/physics/radforce.zig:229` — comptComptonCoeff stutters ("compt Compton") and obscures its relation to Gi
**Fixed (2026-07-06):** renamed `comptComptonCoeff` -> `comptonGiCoeff` (radforce.zig + 3 test call sites).

*Naming*

The Zig-invented name `comptComptonCoeff` duplicates the same word twice in two spellings (C's name is calc_Compt_Gi_with_state, so "compt" already means Compton). The doc comment has to explain that it returns the coefficient shared by both frames of the Compton correction to Gi — none of which the name conveys.

**Fix:** Rename `comptComptonCoeff` to `comptonGiCoeff` (the scalar coefficient of the Compton correction applied to both Gi frames). Update the 4 call sites: koral/physics/radforce.zig:268, koral/golden_opac_test.zig:122, koral/opacity_tests.zig:293/297/299. The doc comment's `C: calc_Compt_Gi_with_state (rad.c:2907)` reference preserves C traceability.

#### `koral/physics/radvisc.zig:53` — Private `delta` duplicates relele.kron under a second name for the Kronecker delta
**Fixed (2026-07-06):** deleted `delta`; calcShearLab now calls `relele.kron` directly.

*Naming*

radvisc.zig defines `fn delta(i, j)` (line 53, used once at line 202) which is byte-for-byte the same function as the public `relele.kron` (relele.zig:33) already imported by this file and used by frames.zig for the identical projection-tensor purpose. Two names for one concept makes greps and refactors miss half the uses.

**Fix:** Delete `delta` and call `relele.kron(i, j)` at line 202 (as frames.zig:55 does).

#### `koral/physics/radvisc.zig:236` — Half-camelized function names calcRadVisccoeff and calcRadShearviscosity are inconsistent with their properly camelized siblings in the same file
**Fixed (2026-07-06):** renamed `calcRadVisccoeff` -> `calcRadViscCoeff`, `calcRadShearviscosity` -> `calcRadShearViscosity`.

*Naming*

C's calc_rad_visccoeff and calc_rad_shearviscosity were ported as `calcRadVisccoeff` (line 236) and `calcRadShearviscosity` (line 291) — "Visccoeff" and "Shearviscosity" treat two words as one — while the same file correctly camelizes calcRijVisc (line 310), calcRijViscTotal (line 335), and addRadViscFlux (line 381). So `Visc` is a word boundary in three names and not in the other two.

**Fix:** Rename calcRadVisccoeff → calcRadViscCoeff and calcRadShearviscosity → calcRadShearViscosity to match the file's other camelized names (calcRijVisc, calcRijViscTotal, calcShearLab). Mechanical: update file-local calls (lines 304, 321) plus koral/radvisc_tests.zig (lines 2, 71, 74) and koral/golden_visc_test.zig (line 110). The doc comments already record the exact C names, so C-traceability is unaffected.

#### `koral/physics/radvisc.zig:247` — `opp` for *const radforce.Params is cryptic and one edit from `pp` in calls that pass both
**Fixed (2026-07-06):** renamed to `opac` (also fixes solve/implicit.zig and sim.zig:1550's `opp`, see below).

*Naming*

calcRadVisccoeff binds `const opp = &(sim.opt.opac orelse return 0);` and immediately passes it in `radforce.calcChi(cfg, pp.*, geom, sim.opt.gam, opp)` (line 249) — `pp` (primitives) and `opp` (opacity params) in one argument list, visually one character apart. The same abbreviation is a convention in sim.zig:1550 and solve/implicit.zig:269/360/401, so this is a codebase-wide Zig-native name, not a one-off.

**Fix:** Standardize the name for `*const radforce.Params` bindings/parameters to `opac` (matching the `Options.opac` field it comes from). Occurrences: radvisc.zig:247, sim.zig:1550 (note sim.zig:823 already uses `op` and radforce.zig's own functions use `p` — three names for the same struct), and solve/implicit.zig parameters at 269, 360, 401, 514, 720 plus their call sites. Readability-only: the type system already prevents an actual pp/opp swap.

#### `koral/physics/thermo.zig:22` — `pi_c` and `Composition.cdefault` use a cryptic c-for-C-language prefix; pi_c reads as pi times lightspeed
**Fixed (2026-07-06):** renamed `pi_c` -> `pi_truncated` in both thermo.zig and forms.zig; left `Composition.cdefault` as a codebase-wide convention, unchanged.

*Naming*

`pub const pi_c: f64 = 3.141592654` (line 22) uses `_c` to mean "the C code's value", but in a relativity code `pi_c` naturally parses as pi*c (speed of light) — the exact kind of misread the truncated-vs-exact pi split (fourpi vs fourmpi, lines 88-89) is trying to prevent. Likewise `Composition.cdefault` (line 43) reads as one opaque word; the sibling constant is plainly named `puffy`.

**Fix:** Rename `pi_c` -> `pi_truncated` (or `koral_pi`) in BOTH definition sites — koral/physics/thermo.zig:22 and koral/metric/forms.zig:98 (duplicate constants; consider consolidating to one) — keeping the ko.h:28 comments. In a GR code `pi_c` misreads as pi*c, and with c=1 the misread is numerically invisible; coco.zig:126 even mixes exact `pi` with `forms.pi_c` in a single expression. For `cdefault`: it is a codebase-wide convention (also FloorParams/ImplicitParams/RadParams in koral/solve/), so either rename it consistently everywhere (e.g. `c_default` or `default`) or leave it as-is — do not rename only Composition.cdefault.

#### `koral/physics/wavespeeds.zig:26` — Parameter `dims: [3]bool` does not say what true means
**Fixed (2026-07-06):** renamed `dims` -> `active_dims` in wavespeeds.zig, radiation.zig, and the sim.zig call site.

*Naming*

lrCore (line 26), gasWavespeedsLr (line 90), and radiation.calcRadWavespeeds (radiation.zig:142) all take `dims: [3]bool` marking which directions are active (C: TNX/TNY/TNZ > 1). "dims" alone reads as "the dimensions" and the boolean payload is only discoverable from the module doc. Call sites like `.{ true, true, false }` give no hint.

**Fix:** Rename the parameter to `active_dims` in all three signatures (wavespeeds.zig lrCore:26 and gasWavespeedsLr:90, radiation.zig calcRadWavespeeds:142). For consistency, also rename the local `dims` in sim.zig:797 that feeds both calls.

#### `koral/recon/recon.zig:40` — avg2point parameter `param: u8` is maximally uninformative
**Fixed (2026-07-06):** renamed `param` -> `reconstr_par`.

*Naming*

`param` is the per-call order reduction near boundaries (effective order = base_order - param). The name is inherited from C's avg2point signature (finite.c:97 `int param`), but even C's own call sites use the better name `reconstrpar` (finite.c:797). A Zig call site `avg2point(nv, ..., base_order, param, theta)` gives no clue what the argument does, and `param` next to `base_order` invites transposition.

**Fix:** Rename `param` to `reconstr_par` — the name the module doc (recon.zig:13) already uses for it, matching C's call-site name — or `order_reduction` if clarity is preferred over C traceability. Update the doc comment to note the C signature spells it `int param` (finite.c:97). No call-site changes needed; the single caller (sim.zig:1274) is positional.

#### `koral/relele.zig:202` — convVelsCore/convVels parameters `which1`/`which2` do not say which is source and which is target
**Fixed (2026-07-06):** renamed `which1`/`which2` -> `from`/`to` across convVelsCore/convVels/convVelsUt/convVelsBoth.

*Naming*

The velocity-conversion family (convVelsCore line 200, convVels line 251, convVelsUt line 256, convVelsBoth line 263) takes `which1: VelType, which2: VelType`. The names are C's (relele.c conv_vels_core), but they carry zero direction information — a signature reader cannot tell that which1 is the input representation and which2 the output, and swapped arguments type-check silently. Parameter names are not part of the golden-record contract; the C cross-reference in the doc comment preserves traceability.

**Fix:** Rename `which1`/`which2` to `from`/`to` across the family: `convVelsCore(uin, from, to, gg, GG, ut_known)`, and likewise in convVels (line 251), convVelsUt (line 256), and convVelsBoth (line 263, `from` only). The existing `C: relele.c:136 conv_vels_core` comment keeps C traceability; no golden-file impact.

#### `koral/sim.zig:294` — Field `dynA` breaks the snake_case field convention and under-describes its contents
**Fixed (2026-07-06):** renamed `dynA` -> `dyn_a` (sim.zig, magn/dynamo.zig, dynamo_tests.zig).

*Naming*

`dynA: field_mod.Field(3)` is the only camelCase field on Sim — every sibling is snake_case (`pb_l`, `fl_r`, `u_bak`, `min_dx`, `impl_dt`, `rijvisc`, `scaleth`). It holds the dynamo's cell-centered ΔA_φ scratch (only slot 2 is ever written, dynamo.zig:232), which the name does not convey.

**Fix:** Rename `dynA` to `dyn_a` (or `dyn_aphi`) to match the snake_case convention of every other Sim field. `dyn_a` is the most accurate since the Field(3) is passed to ct.curlFromA as a full 3-component vector-potential buffer (slots 0..2 = A_i), even though only slot 2 (phi) is ever nonzero. Rename touches 7 sites: sim.zig:294,362,414; magn/dynamo.zig:160,232,243; dynamo_tests.zig:163.

#### `koral/sim.zig:1384` — Cryptic Zig-native locals in fluxesAtFaces (`vag`/`val`/`var_`) and parallelRows (`use`)
**Fixed (2026-07-06):** renamed to `ag_sel`/`al_sel`/`ar_sel` in fluxesAtFaces and `n_workers` in parallelRows.

*Naming*

In the per-variable flux combination (sim.zig:1384-1386) the selected system speeds are named `vag`, `val`, `var_`: `val` reads as "value", `var_` exists only to dodge the `var` keyword, and the `v` prefix ("per-variable"?) is undocumented. These are Zig-side inventions, not C transcriptions (the C loop indexes NVMHD directly). Same pattern in parallelRows (sim.zig:952): `use` — a verb — holds the worker-thread count.

**Fix:** Rename the three selection locals so they name which system's speed was chosen while keeping the visible link to the C-mirrored sources ag/al/ar: e.g. `ag_sel`/`al_sel`/`ar_sel` (or `sys_ag`/`sys_al`/`sys_ar`). This also removes the `var_` keyword-escape underscore. Avoid `a_max`/`a_left`/`a_right`, which would obscure the correspondence with the C names. In parallelRows, rename `use` to `n_workers` (consistent with the neighbouring n_fail/n_iters style).

#### `koral/solve/implicit.zig:173` — Type constructor `Impl` reads as "implementation" rather than "implicit solver"
**Fixed (2026-07-06):** renamed `implicit.Impl` -> `implicit.Solver` (3 call sites).

*Naming*

`pub fn Impl(comptime cfg: config.Config) type` is the module's main entry point; at the call site it becomes `implicit.Impl(cfg)` and sim.zig aliases it away as `ImplT` (sim.zig:1551), a hint the name doesn't carry its meaning. "Impl" is a standard suffix for "implementation", not "implicit", so `ImplT.solveImplicitLab` reads oddly.

**Fix:** Rename `implicit.Impl` to `implicit.Solver` so call sites read `implicit.Solver(cfg)`, matching the repo's descriptive-noun convention for type constructors (Sim, State, Field, VarLayout, Bc). Update the three aliasing call sites (sim.zig:1551, implicit_tests.zig:46, golden_implicit_test.zig:28), e.g. `const ImplicitSolver = implicit.Solver(cfg);`.

#### `koral/solve/implicit.zig:269` — Opacity-parameter argument named `opp` throughout the implicit solver
**Fixed (2026-07-06):** renamed to `opac` throughout solve/implicit.zig (see koral/physics/radvisc.zig:247 above).

*Naming — downgraded by verifier*

The `*const radforce.Params` opacity/coupling parameter is named `opp` in every signature of the Zig-native API surface: implicit.zig lines 269 (residual), 360 (f1dErr), 401 (solve1dPrim), 514 (solve4dPrim), 721 (solveImplicitLab), and the local at sim.zig:1550 (`const opp = &(self.opt.opac.?)`). `opp` is not an initialism of the type name and reads as "opposite/opponent"; it also clashes with the established name for the very same value elsewhere: the Sim option is `opt.opac` (sim.zig:217) and the C concept is "opacities". Sibling parameter abbreviations at least match their types (`ip` = ImplicitParams, `rp` = RadParams, `mp` = MetricParams), which makes `opp` the one you have to look up.

**Fix:** Rename the `*const radforce.Params` parameter/local from `opp` to `opac` (matching Sim.Options.opac) at implicit.zig:269/360/401/514/720, sim.zig:1550, and also radvisc.zig:247 which uses the same pattern. Purely mechanical; `opp` visually collides with the ubiquitous `self.opt` and radforce.zig itself calls the same value `p`, so one shared name would help.

#### `koral/solve/invert.zig:317` — Local `gam` in u2pSolverW is a Lorentz factor, but `gam` codebase-wide means the adiabatic index
**Fixed (2026-07-06):** renamed the local to `gamma`.

*Naming*

At invert.zig:317 the root-unpacking code does `const gam = @sqrt(gamma2);` — a Lorentz factor. Everywhere else in the codebase `gam` is the adiabatic index Γ (Sim.Options.gam, puffy.gam, the `gam` parameters of p2u/sFromU), and this very function takes the adiabatic index as `pgamma`. The C source names this local `gamma`; abbreviating it to `gam` on the Zig side created the collision. The sibling helper wKinematics already uses `.gamma` for the same quantity.

**Fix:** Rename the local at invert.zig:317 from `gam` to `gamma`, matching both the C source (koral_lite/u2p.c:1312) and the sibling wKinematics helper in the same file, and avoiding collision with the codebase-wide convention that `gam` is the adiabatic index (which this function already holds as `pgamma`). Update the four uses at lines 318, 319, and 324.

#### `koral/testing/golden.zig:263` — Kini.nxi()/nyi() reuse Sim's nxi/nyi names with different semantics
**Fixed (2026-07-06):** renamed `Kini.nxi()`/`nyi()` -> `nSamplesX()`/`nSamplesY()`.

*Naming — downgraded by verifier*

`Sim.nxi()/nyi()/nzi()` (koral/sim.zig:422-430) return the domain size as i64 and are used pervasively as loop bounds. `Kini.nxi()`/`Kini.nyi()` (golden.zig:263-268) return something else entirely: the number of *sampled* points along the axis — nx+2·gx when ghosts are included, or the stride-reduced count — as usize. A test author who knows the Sim API will misread `k.nxi()` as "domain nx"; the accessors sit right next to `cellX`/`cellY`, which correctly advertise the sample→cell mapping.

**Fix:** Rename Kini.nxi()/nyi() to nSamplesX()/nSamplesY() so the sample-grid semantics are explicit and distinct from Sim's domain-size nxi()/nyi(). Small, mechanical change: two declarations in koral/testing/golden.zig (lines 263-268, plus the internal use in base() at line 281 and readKini at line 332) and the handful of call sites in golden_dynamo_test.zig, golden_puffy_test.zig, golden_visc_test.zig, and golden_puffystep_test.zig.

### Project structure

#### `build.zig:50` — Per-problem executable is installed through two separate InstallArtifact steps; -Dmpi option is plumbed but unread

*Project structure, Idiomatic Zig — flagged independently by 2 reviewers*

Line 48 `b.installArtifact(exe)` registers an InstallArtifact under the default install step, then lines 50-51 create a second, independent `b.addInstallArtifact(exe, .{})` for the named `<problem>` step — two steps installing the same binary to the same zig-out/bin path. Harmless today but redundant, and if per-problem install options ever diverge the two copies will silently disagree. Separately, the `-Dmpi` option (line 6) is stored into build_options (line 10) but no file in koral/ reads `build_options.mpi` — it is dead plumbing until the MPI backend exists (the option help text does say "not yet implemented").

**Fix:** Create the InstallArtifact once per exe and share it: `const install = b.addInstallArtifact(exe, .{}); b.getInstallStep().dependOn(&install.step); b.step(entry.name, b.fmt("build {s}", .{entry.name})).dependOn(&install.step);` (drop the separate b.installArtifact call on line 48). For -Dmpi, make misuse loud: since no code reads build_options.mpi yet, either remove the option until the MPI backend lands, or add a comptime check in koral (e.g. comm/serial.zig or koral.zig): `if (@import("build_options").mpi) @compileError("MPI backend not implemented; build without -Dmpi");` so -Dmpi=true fails instead of silently producing a serial binary.

#### `koral/flux/laxf.zig:1` — flux/ directory vs physics/flux.zig vs barrel namespace `riemann` — three names for two things

*Project structure — downgraded by verifier*

koral/flux/laxf.zig holds the Riemann face-flux combination (laxf/hll, 21 lines), while koral/physics/flux.zig holds the physical flux f_flux_prime. The directory name `flux/` collides head-on with `physics/flux.zig`, and the barrel then exposes the directory under a third name: `pub const riemann = struct { pub const laxf = @import("flux/laxf.zig"); }` (koral.zig:75-77). So the on-disk path says flux, the import path says riemann, and grep for "flux" finds both subsystems. A 21-line two-function file also does not need its own directory.

**Fix:** Rename the directory to match the barrel namespace: `git mv koral/flux koral/riemann` so the file becomes koral/riemann/laxf.zig — this mirrors the existing single-file-directory pattern of recon/recon.zig (do not move it to a top-level koral/riemann.zig; recon is not top-level either). Update the three import sites (koral/koral.zig:76, koral/sim.zig:50, koral/flux_tests.zig:12) and the doc path references (docs/USER_GUIDE.md:736, :903; docs/ARCHITECTURE.md:679, :880; docs/PHYSICS.md:524). This removes the flux/ vs physics/flux.zig collision and makes grep for "riemann" find the Riemann solver.

**Fixed (2026-07-06):** `git mv koral/flux koral/riemann` (file is now `koral/riemann/laxf.zig`, mirroring `recon/recon.zig`); the three import sites and the doc path references were updated. Directory name now matches the `riemann` barrel namespace.

#### `koral/physics/radvisc.zig:425` — General metric/math helpers (rHorizonBL, rIscoBL, stepFunction) are misfiled in the radiative-viscosity module

*Project structure*

rHorizonBL (C: metric.c:245), rIscoBL (C: metric.c:252) and stepFunction (C: misc.c:1311) are generic Kerr-geometry / smoothing utilities, not viscosity code. The consequence is a wrong-direction dependency: koral/magn/dynamo.zig:157-196 imports physics/radvisc.zig solely to call all three, and rIscoBL is not used by radvisc.zig at all — it lives here only so dynamo can reach it. In the C sources these live in metric.c/misc.c, so the current placement also breaks the otherwise careful C-file-to-Zig-module correspondence.

**Fix:** Move rHorizonBL/rIscoBL into koral/metric/ (e.g. next to MetricParams in metric/forms.zig or metric/metric.zig) and stepFunction into a small shared math/misc module; import them from radvisc.zig and dynamo.zig from there.

**Fixed (2026-07-06):** `rHorizonBL`/`rIscoBL` moved to `metric/metric.zig` (next to `MetricParams`), `stepFunction` to a new `math/misc.zig`; `radvisc.zig` and `magn/dynamo.zig` import them from there, and dynamo's now-unnecessary `radvisc` import was dropped.

*Hand-verified: rHorizonBL/rIscoBL transcribe metric.c:245/252 and stepFunction misc.c:1311, yet live in radvisc.zig; magn/dynamo.zig:157-158,196 imports them from there.*

### Test hygiene

#### `koral/evolution_tests.zig:86` — Bare expect() in per-cell sweep loops reports nothing on failure

*Test hygiene*

The uniform-static and conservation gates assert inside cell loops with plain `try expect(...)` on a computed deviation: evolution_tests.zig:86 (uniform MINK static), :129 and :149 (conserved totals), mhd_evolution_tests.zig:144 (uniform magnetized static), radiation_tests.zig:477 (uniform rad static). On failure these print nothing — not the cell, the variable, or the observed deviation — so a regression in a 100-step evolution gate starts as a blind `error.TestUnexpectedResult` bisect. This is below the suite's own standard: the same files elsewhere print context before asserting (opacity_tests.zig:356–360 does the identical static sweep with a diagnostic print, and the mirror-symmetry test tracks worst ix/iv before its assert).

**Fix:** Apply the opacity_tests.zig:356 pattern to the five sweeps: on gate violation, print ix/iy, iv, expected, got, dev, then return the error — or route them through the shared expectClose once it is extracted (it already prints both values).

#### `koral/golden_test.zig:1` — Naming stragglers: golden_test.zig breaks the golden_<area>_test.zig scheme and metric/tests.zig is the only nested theory suite

*Test hygiene*

The convention is clean and load-bearing — theory gates in `<area>_tests.zig` at koral/, C-oracle diffs in `golden_<area>_test.zig` — with two stragglers: the metric golden lives in the unqualified `golden_test.zig` (its peers are golden_state_test.zig, golden_flux_test.zig, ...), and the metric theory suite is the sole nested one at `koral/metric/tests.zig` where every other module's suite sits flat (state_tests.zig, flux_tests.zig, ...). Both predate the pattern and make grep/glob-driven navigation (and the registration guard suggested for build.zig) needlessly irregular. Adjacent one-character fix while touching these files: the comment at mhd_evolution_tests.zig:214 reads "локate" — Cyrillic лок typed into an ASCII comment.

**Fix:** Rename koral/golden_test.zig to golden_metric_test.zig and move koral/metric/tests.zig to koral/metric_tests.zig. Full edit list: two import lines in koral.zig (143, 151); in the moved file, strip "../" from the math/dual.zig and grid.zig imports and prefix "metric/" onto the metric.zig, forms.zig, coco.zig, precompute.zig imports; update the stale names in docs/USER_GUIDE.md (lines 358, 369 — the latter can then drop its "(metric)" annotation). While touching test files, fix the Cyrillic "локate" typo in the comment at mhd_evolution_tests.zig:213.

#### `koral/radtube_tests.zig:322` — Fast radtube battery asserts tube 3a's gates before running tube 4a, hiding 4a's measurements — the file's own slow set documents why not to do this

*Test hygiene*

The slow set (line 351) deliberately runs all six configurations first and asserts at the end, with a comment explaining that "a single failed gate must not hide the other tubes' measurements", and golden_puffystep_test.zig:246 adopts the same print-all-then-assert pattern. But the default-suite battery (lines 322–349) inlines runTube+asserts sequentially: if tube 3a trips a gate, tube 4a — several minutes of the most expensive default-suite compute, including the only default-suite ODE-residual gate — never runs and its metrics are never printed.

**Fix:** Mirror the slow set's structure (lines 404-418): run tubes 3a and 4a into a Metrics array, print both lines, then evaluate all gates and fail once at the end. Alternatively — simpler and equally effective — split the battery into two separate test blocks ("tube 3a" / "tube 4a"); Zig's test runner runs every test even after a failure, so 4a's metrics and ODE gate would still execute and print when 3a regresses.

#### `koral/radvisc_tests.zig:63` — Data-dependent silent skip: torusCell() orelse SkipZigTest can mask a regression as a skipped test

*Test hygiene*

Both M12 radviscosity limiter tests bail with `torusCell(&s) orelse return error.SkipZigTest` (lines 63 and 90). Unlike the suite's other skips (slow-gating and missing-golden, both expected states, the latter with an explanatory message), this one converts an unexpected runtime condition — the 32×40 PUFFY init producing no cell with ρ > 1e-20 in the scanned window (iy ≥ ny/2, ix ≥ nx/3) — into a silent skip with no message. A refactor that shifts or empties the torus in that window would quietly retire the only tests pinning the RADVISCNUDAMP and RADVISCMAXVELDAMP limiter properties; the skip count change is easy to miss among the 7 legitimate slow-test skips.

**Fix:** The init is deterministic, so no torus cell is a test failure, not an environment condition: replace with `orelse return error.NoTorusCellFound` (or unwrap with a message). If skip semantics are truly intended, print a diagnostic first as readGolden does.

---

## Stats

- 18 reviewer agents + 131 verifier agents, ~5.7M tokens, ~47 min wall clock (incl. one rate-limit restart).
- 138 raw findings -> 137 after dedup -> 120 survived adversarial verification, 11 rejected, 6 hand-verified after verifier rate-limiting; identical cross-reviewer findings merged into the entries above.
- Full machine-readable results: workflow journal in the session transcript directory.
