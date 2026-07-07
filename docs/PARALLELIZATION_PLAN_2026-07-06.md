# Parallelization & Performance Plan — SIMD, Multicore, MPI, GPU

*2026-07-06. Big-picture strategy for scaling koral-zig from today's serial-plus-row-threads state to 64+ core nodes and multi-node MPI. Grounded in the M13 code and the C reference (`koral_lite`). Line-level cleanups live in `CODE_REVIEW_2026-07-06.md`; this doc is architecture.*

---

## 0. Executive summary

1. **The biggest lever is multicore, not SIMD.** *(P1 DONE 2026-07-07 — §7.4.)* At the time of writing only 2 of ~15 per-step passes were threaded (`u2p`, `implicit`, via per-pass spawn/join). Threading everything behind a persistent worker team with dynamic tiling was the single largest win and the prerequisite to 64-core scaling — **now delivered: the whole PUFFY step runs 7.6× faster on 10 cores (8.2× on 12), up from 1.77×.** SIMD multiplies per-core throughput ~2× (NEON) to ~8× (AVX-512) *only on the kernels it reaches*; cores multiply everything.
2. **One SIMD opportunity stands out because it needs no data-layout change:** the implicit solver's finite-difference Jacobian evaluates 4 independent perturbed residuals per Newton iteration — vectorize those as 4 lanes *within one cell*. The implicit solver is the runtime hog, and this sidesteps the cross-cell divergence problem entirely.
3. **Cross-cell (vertical) SIMD needs the AoSoA flip** the architecture doc (§6, §8.4) already planned. It pays most on AVX-512/SVE cluster CPUs (W=8), least on the Mac (NEON W=2). Do it after multicore, and only for the explicit-path kernels.
4. **MPI: follow C's protocol, not C's deployment.** C runs PUFFY as *flat MPI, 960 ranks of 12×12 cells* — at NG=3 that is >125% ghost-to-interior overhead and 3.7 KB latency-bound messages. The Zig plan is hybrid: 1 rank per NUMA domain × threads inside, tiles hundreds of cells wide, C's exchange *placement* mirrored exactly for diffability.
5. **Mac GPU: no compiler magic exists, and the physics forbids it anyway.** Metal has no `double` type and Apple GPUs have no FP64 hardware. Emulating f64 with f32 pairs fails *numerically*, not just in speed: double-single inherits the f32 exponent range (min normal ~1e-38), and KORAL's floors live at 1e-79/1e-80. The honest Apple-silicon story is CPU: 400–546 GB/s unified memory bandwidth is exactly what this bandwidth-heavy code wants.

Recommended order: **instrument → thread everything → memory diet → SIMD ∥ MPI → overlap/balance.** §7 has the roadmap.

---

## 1. Where the time goes — one RK2IMEX step today

`step()` (`sim.zig:1667`) per-pass inventory, as of 2026-07-06. "Threaded" = ran under the old `parallelRows`, which spawned and joined up to `min(nthreads, 64, ny)` OS threads *per invocation*. **Measured wall-clock per pass in §7.2** (P0) — headline: implicit 45%, dynamo 28% of the serial step. *(Historical: P1 has since put every pass below on the persistent team — see §7.4.)*

| # | Pass | Where | Calls/step | Threaded | Character |
|---|------|-------|-----------|----------|-----------|
| 0 | radvisc `calcRijViscTotal` | `radvisc.zig:333` | 1 | no | per-cell, ±1 stencil, FLOP-heavy |
| 1 | `opImplicit` (rad-gas Newton) | `sim.zig:1535` | 2 | **yes** | per-cell, cell-local, extreme cost variance |
| 2 | `calcU2p` inversion sweep | `sim.zig:984` | 4 | **yes** | per-cell Newton, cell-local |
| 3 | `cellFixup` (hd+rad) | `sim.zig:1062` | 4×2 | no | flagged cells ← 6-neighbor avg |
| 4 | `setBc` + corner fill | `sim.zig:544` | ~7 | no | ghost loops |
| 5 | `doCorrect` polar axis | `sim.zig:1589` | 4 | no | few rows, cheap |
| 6 | `calcWavespeeds` (+dt max-reduce) | `sim.zig:796` | 2 | no | per-cell, quadratics + `calcChi` |
| 7 | `sweep` ×dims (recon+flux) | `sim.zig:1220` | 2 | no | per-face, 5-pt PPM stencil, 2×`fFluxPrime`/face |
| 8 | `fluxesAtFaces` (LAXF combine) | `sim.zig:1318` | 2 | no | per-face, 2×`p2u` |
| 9 | `fluxCt` EMF avg | `ct.zig:33` | 2 | no | 2 passes w/ RAW dependency, memory-bound |
| 10 | conserved update + metric source | `sim.zig:1418` | 2 | no | per-cell, Christoffel contraction (64 dbl read) |
| 11 | `applyDynamo` | `dynamo.zig:262` | 2 | no | θ-reduction per radius + per-cell curl |
| 12 | stage arithmetic: 8×`copyFull`, 4×`stageDeriv`, 3×`stageCombine` | `sim.zig:1454-1491` | — | no | pure memory streaming |
| 13 | `updateEntropy` | `sim.zig:1509` | 1 | no | per-cell p2u∘u2p |

**Compute-bound:** the two Newton solvers (u2p `invert.zig:195-371`; implicit `implicit.zig:507-743` — 6-rung ladder × ≤40 Newton iters × 5 residual evals, each a full `RadState` fill + `calcGi` with transcendental opacities), and the flux/wavespeed tensor builders (`indices2221` = 64-mul 4×4×4 contraction, several per face-side).

**Bandwidth-bound:** stage arithmetic (~0.6 GB of traffic per step at PUFFY 384×360 — 8 full-field copies + 7 triads over 14 `Field(13)` arrays ≈ 208 MB of state), flux-CT gathers, and the metric cache.

**The metric cache is the working-set elephant** (`precompute.zig:125`): per cell 20 (g) + 20 (gcon) + 64 (Christoffels) + 32 (two Jacobians, *dead per the code review*) doubles at centers + 120 at faces ≈ **2 KB/cell ≈ 292 MB** at production 2D resolution — 20× the primitives. Every face computation reads a 20-double block; every source term reads 64 Christoffels.

**Known hazards when threading the rest** (from the hot-path inventory):
- `tstepdenmax/min` running max is fused into the serial wavespeeds loop (`sim.zig:904-905`) → needs per-tile partials + ordered merge.
- `fluxCt` pass 2 reads pass 1's EMFs → team barrier between passes.
- `radvisc` and `dynamo` scale-height are stencil/reduction pre-passes gating the sweep → barrier points.
- `ChunkResult` counters are incremented per cell through a shared-array pointer (`sim.zig:1576-1580`) → false sharing at high thread counts; accumulate in locals, write once.
- `cellFixup` is parallel-safe as-is: flags are frozen during the pass, writes touch only flagged cells, reads only non-flagged neighbors → disjoint.
- No heap allocation in any hot loop (verified) — all per-cell state is stack `[NV]f64` / `[4][4]f64`. This is the property that makes everything below tractable; preserve it.

---

## 2. SIMD / vector strategy

### 2.1 Reality check

- Layout is AoS, `iv`-fastest (`field.zig:35-42`): one cell = 13 contiguous doubles. Nothing auto-vectorizes across cells; kernels are scalar over `[13]f64` stack buffers. This was a deliberate M0 decision (C-diffability + the per-cell solvers touch all NV) with the AoSoA escape hatch designed in (`field.zig:6-8`, arch doc §6 batch view).
- `build.zig` uses `standardTargetOptions` → native CPU on native builds, so NEON/AVX are already *available* to LLVM; but Zig's strict float mode (correctly, for the goldens) forbids FMA contraction and reassociation, so autovectorization of scalar code achieves little. SIMD gains must come from **explicit `@Vector` kernels**.
- Lane math: Apple M-series NEON = 2×f64 (but 4 FP pipes/core, so ILP matters as much as width); AVX2 = 4; AVX-512 (EPYC Zen4/5, SPR) = 8; SVE (Graviton 3/4) = 4. `std.simd.suggestVectorLength(f64)` picks W at comptime. **On the Mac the ceiling is 2×; on the 64-core cluster CPU it's 8× — write the batch kernels once, width-generic.**

### 2.2 The bitwise-compatibility rule (why SIMD doesn't break goldens)

Vertical vectorization — one cell per lane, same op sequence — performs *identical arithmetic per cell* to the scalar kernel: results are bit-identical. Zig lowers `@sqrt` on vectors to hardware sqrt (IEEE-exact) and `@exp`/`@log`/`pow` to per-lane libm calls — also bit-identical. Two things do change bits and must stay out of golden-path kernels:
- `@mulAdd` (FMA) — introduce only behind a `-Dfast` build flag with field-scale-tolerance tests (the FP-contract lesson from M3 in reverse);
- horizontal `@reduce(.Add, …)` — reassociates sums; keep contractions in scalar order or accept tolerance.

So the plan is: **kernels parameterized over `comptime T: type`** where `T` is `f64` or `@Vector(W, f64)` — Zig's arithmetic operators, `@select`, and comparisons work uniformly on both. One source; the scalar instantiation remains the golden-tested reference; the vector instantiation is the production path. This is the Zig-specific advantage — no intrinsics fork, no duplicated physics.

### 2.3 Ranked SIMD opportunities

| Rank | Kernel | Approach | Why / expected gain |
|------|--------|----------|---------------------|
| **1** | **Implicit FD Jacobian** (`implicit.zig:561-603`) | Vectorize the **4 perturbed residual evaluations** of each Newton iteration as 4 lanes of one `@Vector(4, f64)` batch — *within one cell* | No AoSoA needed, no divergence (same code path all lanes, same cell), and it hits ~80% of the dominant kernel's inner cost (4 of 5 residual evals per iteration). Requires the residual chain (`RadState` fill, `calcGi`, opacities) to be `comptime T`-generic. Realistic ~2–3× on the implicit pass even on NEON (`@Vector(4,f64)` = 2 NEON registers, pipelined). |
| 2 | Explicit face kernels: `fFluxPrime`, `p2u`, `calcTij`/`calcRij`, `indices2221` (`flux.zig:22-95`, `relele.zig:64-117`) | Vertical: W faces/cells per batch via AoSoA block view | Pure per-cell arithmetic, few branches → near-width. This is the arch-doc §8.4 plan; needs the batch view (§2.4). |
| 3 | Reconstruction (`recon.zig:60-164`) | Vertical, monotonicity branches → `@select` masks | Bounded branch ladder, no iteration → near-width. |
| 4 | Wavespeeds `lrCore` (`wavespeeds.zig:22-71`) | Vertical; τ-limiter's `calcChi` stays per-lane scalar (transcendentals) | Quadratic + clamps vectorize; partial gain. |
| 5 | LAXF combine + stage arithmetic (`laxf.zig:10-19`, `sim.zig:1454-1491`) | Trivial `@Vector` streaming | Bandwidth-bound — vectorize for free as part of threading, don't expect FLOP wins. |
| 6 | u2p Newton (`invert.zig:195-371`) | Vertical with convergence masks + scalar remainder for divergent cells | 1-D Newton with inner damping loop and entropy-fallback re-run: heavy divergence. Modest payoff; threads already cover it. Do last, or not at all. |

### 2.4 The AoSoA batch view (for ranks 2–5)

Per arch doc §6: blocks of W cells, layout `[NV][W]f64` per block (each variable row 64-byte aligned; W from `suggestVectorLength`). `Field` gains `block(b)` returning `[NV]@Vector(W, f64)`; `load/store` become strided gather/scatter internally so **every existing scalar kernel keeps working unchanged** during the migration. Batch kernels are new code paths for the explicit sweep only. Ghost/remainder cells and divergent solver cells use the scalar path. Keep the AoS layout as a comptime build alternative until the batch path has burned in against the step goldens.

**Priority note:** on the user's 12-core Mac, item 1 is worth doing now; items 2–5 change the picture mainly on AVX-512 hardware — sequence them with the MPI/cluster work, not before multicore (§3).

---

## 3. Multicore: scaling one node to 64+ cores

### 3.1 What's wrong today (three structural issues)

1. **Coverage.** Only the two inversion passes are threaded. Even if the implicit solver is ~50–60% of serial runtime, Amdahl caps the whole-step speedup at ~2–2.5× regardless of core count. Every per-cell/per-face pass in the §1 table must run on the team.
2. **Per-pass spawn/join.** `parallelRows` creates and joins OS threads on every call — ~6 invocations/step today, ~20+ once everything is threaded. At 64 threads × ~20 regions × O(10 µs) spawn cost, that's milliseconds per step of pure overhead, on par with the target step time. Also `maxt = 64` is a hardcoded ceiling.
3. **Static row bands.** Implicit-solver cost varies by orders of magnitude between the τ≫1 torus core and the funnel/floor cells, and the expensive cells cluster spatially (equatorial band, r∈[~35, ~200]). Contiguous `iy` bands guarantee imbalance; at 64 threads on 360 rows the slowest band dominates.

### 3.2 The design

- **Persistent worker team.** Spawn `nthreads` workers once at `Sim.init`; park on a futex/condvar; main thread publishes `(kernel, range)` work descriptors and a generation counter; team barrier at region end. (Zig 0.16 removed `std.Thread.Pool` — hand-rolling this is ~150 lines and gives exact control over the barrier, which a generic pool wouldn't.) All §1 passes become team regions with barriers where the pass table shows dependencies (fluxCt pass1→pass2; radvisc/scale-height before their consumers; BC faces before corner fill).
- **Dynamic tiling.** Decompose the domain into fixed tiles (start: 8 `iy`-rows × full `ix` for the 2D case; square-ish 32² tiles for 3D), dispatched by an atomic ticket counter. Cheap tiles fly, expensive tiles straggle-balance naturally. Tile size is a tunable; 8+ rows keeps the stencil passes' neighbor reads mostly in-tile.
- **Deterministic reductions by construction.** Results indexed **per tile** (not per thread), merged in fixed tile order after the barrier: `tstepdenmax/min` (max/min are order-insensitive anyway), implicit counters (integer sums), dynamo scale-height per-`ix` sums (each `ix` column's θ-sum stays within whatever tiles own it — merge in `iy`-order). This preserves the bit-identity property `threading_tests.zig` already pins, at any thread count and any schedule.
- **Fix the known hazards** from §1: per-tile local accumulators (kills the `ChunkResult` false sharing), lift the dt-reduce out of `saveWavespeeds` into the merge step.
- **NUMA & pinning (matters at 64+).** EPYC-class nodes are 8–12 CCDs with 12 memory channels; a 64-core socket only reaches its ~460 GB/s if pages are spread. First-touch: initialize fields **and the metric cache** from the team using the same tile partition as the runtime loops; pin workers to cores; default `nthreads` = physical cores (SMT gains ~10–20% on the Newton passes, nothing on bandwidth passes — make it a knob).
- **Overlap the serial tails.** Polar-axis `doCorrect` is a few rows (parallelize over `ix·iz` columns trivially); scalars/dump I/O move to a background thread (gzip especially); `saveTimesteps` folds into the wavespeeds region.

### 3.3 The bandwidth wall (and the memory diet)

Once compute passes scale, the step becomes bandwidth-bound on stage arithmetic + metric traffic. Attack working-set and traffic *before* buying SIMD for the explicit path:

- **Delete dead weight:** `ptm1`/`ppostimplicit` fields (dead per code review — saves 2 `copyFull`/step + 2 fields), `dxdx_my2out/out2my` arrays (dead — 32 doubles/cell = 35 MB at 2D production, and 23% of the center-cache footprint).
- **Fuse stage arithmetic:** `copyFull(ut1,…)` + `stageDeriv(dut1,…)` pairs and the two final `stageCombine`s can fuse into single passes (fewer full-field sweeps); where a copy exists only to snapshot, pointer-swap double buffers instead of copying.
- **Axisymmetric metric cache** (the C `METRICAXISYMMETRIC`/`SZMET=1` trick): for MKS2/KS the metric is φ-independent — index the cache by `(ix,iy)` only. Irrelevant in 2D (nz=1) but **mandatory for 3D**: at 384×360×192 a 3D-indexed cache would be ~56 GB/node; axisymmetric indexing returns it to ~300 MB.
- (Later, with SIMD) **recompute vs. load:** on 64 cores FLOPs are cheaper than DRAM; recomputing face metrics from the analytic forms instead of caching 120 doubles/cell/face is a real option worth benchmarking in 3D.

### 3.4 Expected shape of the curve

2D PUFFY = 138k cells → ~2,200 cells/thread at 64: still fine-grained enough for dynamic tiles, but per-region barrier overhead (~µs × 20 regions) starts to show; don't expect better than ~40–50× at 64 cores in 2D. In 3D (26M cells) the same design should scale near-linearly to a full socket, hitting the bandwidth wall on the streaming passes — which is what §3.3 and the eventual hybrid-MPI tiling address.

---

## 4. MPI: multi-node decomposition

### 4.1 What the C code does (the reference to mirror — and where to deviate)

Findings from `koral_lite` (`mpi.c`, `choices.h`, `problem.c`):

- **Decomposition:** static uniform tiles, `NX = TNX/NTX` per rank (divisibility enforced), lexicographic rank↔tile map (`mpi.c:3118-3131`), tile-origin globals `TOI/TOJ/TOK`. No Cartesian communicator, no rebalancing. PUFFY production: **NTX×NTY = 32×30 = 960 ranks of 12×12 cells.**
- **Halo exchange:** NG-deep slabs of all NV primitives; `MPI4CORNERS` (force-on with MAGNFIELD) → full 26-neighbor exchange in 3D, 8-neighbor in 2D. All `Isend` + all `Irecv` + one `Waitall`, hand-packed buffers (`mpi.c:7-30`). **Exactly 2 exchanges per step** (before each `op_explicit`: `do_correct → mpi_exchangedata → calc_avgs_throughout → set_bc`, `problem.c:198-204, 314-320`) plus one inside `apply_dynamo` (`finite.c:1376`) and 3 at init.
- **Collectives per step:** `mpi_synchtiming` = 2 barriers + Allreduce(MAX dt) + 4 diagnostic Reduces + Bcast (`mpi.c:2917-2961`); Allreduces for `nfailures` (load-bearing abort check) and solver stats (`problem.c:794, 841-843`); if dynamo: 2× Allreduce over TNX-length scale-height arrays (`mpi.c:3207-3209`). No sub-communicators anywhere.
- **Polar axis:** only `TJ==0`/`TJ==NTY-1` ranks own it; `TRANSMITTING_YBC` routes pole ghosts to the tile π away in φ with sign flips (3D only; PUFFY 2D uses reflection).
- **Known C gaps to fix, not copy:** scalars output (`mdot`, `lum`…) is simply *disabled* under MPI (`fileop.c:112`); per-step barrier+print collectives are latency poison at scale; flags (`pflag`-style) are never exchanged — `cell_fixup` sees ghost flags as always-clean; `calc_avgs_throughout` is O(TNX) per rank and unimplemented for multiple z-tiles.

### 4.2 The Zig plan

**Phase A — correctness (mirror C exactly).**
- `comm/mpi.zig` as thin `extern` bindings (arch doc §8.1); the existing `Serial` API grows `decompose(globalGrid) → localGrid+origin`, `exchangeHalos(fields…)`, `allreduce`. Numerics never learn about ranks.
- Use a real `MPI_Cart_create` communicator (readability + reorder hint) but keep C's uniform-divisibility decomposition initially.
- **Put exchanges exactly where C puts them** — before each `opExplicit` and inside `applyDynamo` — *not* at every `setBc` site. C proves the canonical algorithm tolerates interior-ghost staleness between those points; matching placement is what keeps the forced-dt step goldens meaningful across an MPI split. `setBc` fills **physical** boundaries only (the `mpi_isitBC` test becomes a method on the local grid).
- Exchange all NV primitives, NG deep, faces+corners (2D: 8 neighbors). Don't exchange flags (C-faithful); revisit later as an accuracy improvement with a tolerance-gated test.
- Collectives: **one** `Allreduce(MAX)` for dt per step. `nfailures` abort check: fold into the same Allreduce as a second value (max of failures) instead of a separate collective. Solver stats: reduce every N steps or at output only. Scale-height: keep C's global TNX-array Allreduce (3 KB — cheap) for diffability.
- Determinism (arch doc §8.5): `allreduce` on sums gets the fixed-order combine option; min/max stay native.

**Phase B — deployment model (deviate from C).**
- **Hybrid: 1 rank per NUMA domain, §3 team inside.** This is the whole point of doing threads well: C's 12×12 tiles carry (12+6)²−12² = 180 ghost cells per 144 interior (125% overhead) and 3.7 KB messages (pure latency). A 4-node × 2-NUMA × 32-thread deployment of the same 2D problem has 8 tiles of ~96×180, ~7% ghost overhead, ~100 KB messages (bandwidth regime).
- **Reality check for 2D PUFFY:** one modern 64-core node with §3 done likely *beats the 960-rank C production setup outright* — more than half of the C configuration's parallel work is ghost redundancy and message latency. MPI is really for 3D.
- **3D decomposition order: φ first, then r, θ last.** φ is periodic (no axis complications), uniform in cost (the expensive torus cells are axisymmetrically distributed → near-perfect static balance across φ-ranks), and keeps each rank's θ-column whole so the scale-height reduction and polar handling stay rank-local per wedge. Cross-pole transmitting BC pairs each polar wedge with the wedge π away — a fixed partner exchange, exactly C's `tz = TK + NTZ/2` scheme. Residual r/θ cost imbalance is absorbed by the intra-node dynamic tiles.
- **Fix the C gaps:** scalars under MPI as proper reductions (shell sums = sum over the rank column owning that radius; deterministic fixed-order combine); dumps via MPI-IO collective writes or file-per-rank + merge tool, gzip on the I/O thread.

**Phase C — overlap (only if profiles demand).**
Post halo `Isend/Irecv`, sweep interior tiles, `Waitall`, sweep the boundary shell. The team+tile machinery from §3 makes "interior first, boundary last" a scheduling policy, not a rewrite. Persistent requests (`MPI_Send_init`) for the fixed neighbor pattern.

**Validation ladder** (matches the project's test strategy): 1-rank MPI ≡ serial **bitwise**; fixed decomposition run-to-run bitwise (determinism mode); 2×1 and 1×2 decompositions vs serial at step-golden field-scale tolerances (interior cells away from tile seams should be bitwise for the first steps — the same argument as the threading test); flags agree cell-for-cell.

---

## 5. Mac GPU (the bonus question)

**Short answer: there is no low-effort path, and the obstacle is physics-grade arithmetic, not tooling.**

- **No f64 on Apple GPUs.** Metal Shading Language has no `double` type; the hardware has no FP64 units. OpenCL on macOS is deprecated *and* reports fp64 unsupported on Apple GPUs. No offloading compiler targets Metal for f64: clang's OpenMP offload targets NVPTX/AMDGCN only; OpenACC is NVIDIA-compiler territory; SYCL implementations have no Metal backend. Zig has no GPU story for Metal either.
- **Emulation is numerically disqualified.** Double-single (df64: a f64 emulated as two f32s) costs 10–40 f32 ops per operation — but worse, it inherits the **f32 exponent range** (min normal ~1.2e-38). KORAL's floors and small quantities live at `ERADFLOOR = 1e-79`, `SMALL = 1e-80`, and torus-edge densities plunge many decades below unity in code units. These underflow to zero in df64 — the scheme breaks before it gets slow. Rescaling the entire unit system to dodge this would be a physics-validation project, not a port.
- **What Apple silicon is actually good at for this code:** unified memory at 400–546 GB/s (M3/M4 Max) — server-class bandwidth feeding the §3.3 streaming passes; 12–16 wide cores; NEON with 4 FP pipes per P-core (the §2.3-rank-1 Jacobian vectorization works fine there). The E-cores add ~15–20% throughput under dynamic tiling — they just take more tiles' worth of time per tile, which the ticket scheduler absorbs. In other words: **the Mac's "GPU-class" resource for KORAL is its memory system, and the CPU already has it.**
- **If GPU ever becomes real** (cluster-side, NVIDIA f64 at 1:2 rate): the explicit sweep + closed-form rad inversion are good GPU citizens; the branchy 6-rung implicit ladder is not, and it dominates runtime — Amdahl kills the port. That is the arch doc's principle 6, and nothing in this analysis changes it. The AoSoA batch layout from §2.4 is the same layout a CUDA port would want, so the SIMD work keeps that door open at zero extra cost.

---

## 6. Interactions & constraints (cross-cutting)

- **Goldens stay authoritative.** Everything above is designed to keep the scalar, serial, strict-FP path intact and golden-tested: vertical SIMD lanes reproduce scalar bits (§2.2); threading preserves bit-identity via disjoint writes + ordered merges (§3.2); MPI mirrors C's exchange placement (§4.2). FMA/fast-math variants live behind a build flag with field-scale-tolerance gates only.
- **Determinism mode** (arch doc §8.5) falls out of the tile-ordered reduction design nearly for free — implement it with the team, not after.
- **The `Comm` seam is already right** (`comm/serial.zig`); resist the temptation to let any kernel learn about ranks or threads. Both the team and MPI slot in behind existing pass boundaries.
- **Threading model choice locks the halo buffer design** (the C code's excuse for not doing hybrid): with one rank per NUMA domain, pack/unpack of halo slabs should itself be a team region (each worker packs its tiles' boundary strips).

---

## 7. Roadmap

| Phase | Work | Payoff | Risk |
|-------|------|--------|------|
| **P0** | Per-pass wall-clock instrumentation (cheap timers around each §1 pass, printed with the scalar cadence) — **DONE 2026-07-06**, see §7.2 | Replaces the assumptions in this doc with measurements; sizes P1–P4 → **headline: the dynamo is the #2 serial cost (28%) and the #1 cost once inversions are threaded (50%)** | none |
| **P1** | Persistent worker team; thread all §1 passes; dynamic tiles; per-tile reductions; false-sharing fixes; NUMA first-touch + pinning — **DONE 2026-07-07** (§7.3 first increment, §7.4 the rest): every §1 pass on the futex-parked team with atomic-ticket dynamic tiles; whole step **7.6× at 10 threads, 8.2× at 12** (was 1.77×). NUMA first-touch/pinning deferred to P4 (meaningless on the unified-memory Mac; no affinity API on Darwin) | The big one: unlocks 64-core nodes; expect ~order-of-magnitude on today's serial share → **delivered ~7.6× of the ~10× ideal on 10 cores** | Barrier bugs → caught by the bitwise threading gates, now covering every pass incl. a full PUFFY step |
| **P2** | Memory diet: dead fields + dead Jacobian arrays out; fuse/swap stage arithmetic; axisymmetric metric-cache indexing (3D-critical) | Raises the bandwidth ceiling P1 will hit; −50%+ step traffic | Low; golden-checked deletions |
| **P3a** | `comptime T`-generic residual chain + implicit FD-Jacobian lane vectorization (§2.3 #1) — **DONE 2026-07-06**, see §7.1 | ~2–3× on the dominant solver, on any CPU, no layout change → **measured 1.16× solve-level on NEON** (see §7.1 for why) | Moderate: needs the opacity/RadState chain generic over T |
| **P3b** | AoSoA batch view + `@Vector` explicit-path kernels (§2.4) | Near-width on flux/recon/wavespeeds — 8× lanes on AVX-512 cluster CPUs, 2× on Mac | Highest-touch SIMD item; keep AoS fallback comptime-selectable |
| **P4** | MPI backend phases A+B (§4.2): bindings, cart grid, C-placed exchanges, single dt collective, hybrid rank-per-NUMA deployment, scalars-under-MPI, MPI-IO dumps | Multi-node; 3D-capable | Validation ladder in §4.2; keep 1-rank ≡ serial bitwise as the anchor |
| **P5** | Comm/compute overlap, persistent requests, load-balance tuning, async I/O | Last 10–30% at scale | Only with P0 profiles in hand |

P3 and P4 are independent tracks after P1+P2; sequence by hardware availability (Mac-only → P3a first; cluster access → P4 first).

### 7.1 P3a results (implemented 2026-07-06)

Shipped as designed in §2.2/§2.3 #1: the whole residual chain
(`fillRadStateG` → opacities → `calcGiFromStateG` → `residualG`, plus the
relele/frames/radiation/bfield/thermo/hydro/units helpers underneath) is
`comptime T`-generic over `f64` / `@Vector(W, f64)` (helpers in
`koral/math/simd.zig`), and `solve4dPrim` evaluates the FD Jacobian's 4
perturbed residuals as one `@Vector(4, f64)` batch
(`ImplicitParams.simd_jacobian`, default on; the per-lane perturbation +
`applyConstraints` sign-retry stays scalar, then the 4 lanes batch — valid
because the fill chain is infallible for VELR primitives). Bit-identity
holds exactly as §2.2 predicted: `simd_tests.zig` gates the chain per lane
and the whole solver (scalar vs SIMD bitwise, converged and failed cells
alike), and every existing golden passes with the batch on.

Measured on the M9 golden battery (336 PUFFY cells × full rung ladder,
`zig build bench-implicit`, ReleaseFast, Apple M-series; C = the oracle
build, clang -O2 + GSL, same deterministic battery via
`oracle/harness_bench_implicit.c`):

| Configuration | ns/solve | vs C |
|---|---|---|
| C `solve_implicit_lab` | 109 178 | 1.00× |
| Zig scalar Jacobian | 84 406 | 1.29× |
| Zig SIMD Jacobian | 73 060 | **1.49×** |

SIMD vs scalar: **1.16× solve-level** (1.17× with the bisect pre-pass
disabled — same number, so the dilution is not the bisect). Why below the
§2.3 2–3× estimate: the chain's cost is dominated by (a) ~17 libm
transcendentals per residual evaluation (pow/exp/log1p/cbrt in the
opacities + sFromU), which the bit-identity rule deliberately keeps as
per-lane scalar calls, and (b) the per-lane scalar `applyConstraints`
(p2u/u2p inversions). Only the pure-arithmetic share of the batched 4-of-5
evaluations vectorizes; Amdahl does the rest. The estimate's error was
assuming arithmetic dominance in §2.3, not the design — which is exactly
what it promised: bit-identical, no layout change, and the entire chain is
now T-generic, which is the prerequisite P3b (AoSoA, wider lanes) and any
future `-Dfast` vector-libm build both reuse. Getting past ~1.2× on this
kernel requires breaking one of the two constraints above: a `-Dfast`
build with vector transcendental approximations (field-scale-tolerance
gates instead of goldens), or masked-batch u2p. Both are follow-on items,
not part of P3a.

Side-finding: PUFFY's `RADIMP_START_WITH_BISECT` pays for itself — the
battery runs 84 μs/solve with the bisect vs 113 μs without (the better
starting guess saves more Newton iterations than the bisect costs).

### 7.2 P0 results (implemented & measured 2026-07-06)

**Implementation.** `koral/sim/timers.zig` — `PassTimers`, one bucket per
§1 pass, with *self-time* attribution: `begin()/end()` keep a small pass
stack and charge elapsed time to the innermost active pass, so nested
timed calls (op_implicit → cell_fixup, apply_dynamo → set_bc/calc_u2p,
op_explicit → everything) never double-count and the buckets sum exactly
to the wall time inside `step()`. Always on (~60 begin/end pairs per
step, one monotonic clock read each — µs against a multi-second step; no
FP effect, so goldens/threading bit-identity untouched; attribution unit-
tested with synthetic timestamps). The PUFFY driver prints the table at
each scalar cadence and resets, so every report covers the interval since
the previous row. The untimed remainder (`save_timesteps`,
`copy_entropycount`, inter-pass gaps) prints as `(other)` — measured
0.05%, i.e. the §1 inventory covers the step.

**Measurement.** PUFFY 384×360 (production 2D), ReleaseFast, Apple
M-series (12 cores), first 20 steps from t=0 (dt at the CFL value
3.5e-3 by step 2; flat-`iy`-band row threading = today's `parallelRows`):

| Pass (§1 row) | serial ms/step | % | 10-thread ms/step | % | speedup |
|---|---|---|---|---|---|
| opImplicit (1) | 1152.1 | 45.0 | 155.3 | 10.7 | 7.4× |
| **applyDynamo (11)** | **723.5** | **28.2** | **724.0** | **50.0** | 1× |
| sweep (7) | 196.4 | 7.7 | 198.8 | 13.7 | 1× |
| calcU2p (2) | 145.2 | 5.7 | 25.9 | 1.8 | 5.6× |
| setBc (4) | 72.9 | 2.8 | 73.2 | 5.1 | 1× |
| calcWavespeeds (6) | 68.3 | 2.7 | 68.8 | 4.7 | 1× |
| fluxesAtFaces (8) | 68.1 | 2.7 | 68.2 | 4.7 | 1× |
| radvisc (0) | 50.6 | 2.0 | 50.9 | 3.5 | 1× |
| conserved update (10) | 32.7 | 1.3 | 32.0 | 2.2 | 1× |
| stage arithmetic (12) | 28.7 | 1.1 | 28.5 | 2.0 | 1× |
| cellFixup (3) | 7.9 | 0.3 | 7.9 | 0.5 | 1× |
| updateEntropy (13) | 8.4 | 0.3 | 8.5 | 0.6 | 1× |
| fluxCt (9) | 5.1 | 0.2 | 5.1 | 0.4 | 1× |
| doCorrect (5) | 0.3 | 0.0 | 0.3 | 0.0 | 1× |
| (other) | 1.2 | 0.0 | 1.2 | 0.1 | — |
| **whole step** | **2561** | 100 | **1449** | 100 | **1.77×** |

**What the measurements confirm.** The implicit solver is the dominant
serial pass at 45% — the §1 "~50–60%" ballpark, at a t≈0 state (expect it
to grow as MRI turbulence stiffens the torus). Whole-step speedup from
threading only the two inversion passes is 1.77× at 10 threads — the
Amdahl cap §3.1 predicted. Per-cell implicit cost averages 4.2 µs/cell
here vs 73 µs/solve on the (torus-cell) M9 battery — a ~17× spread
between floor/funnel and torus cells, which is the §3.1-3 imbalance
argument, quantified. The serial tails the plan proposed to leave for
last (fixup, entropy, fluxct, correct, stage, update) total ~6% — 
correctly deprioritized.

**What they refute: `apply_dynamo` is the plan's biggest miss.** §1
listed it as a minor pass; measured, its *self* cost (scale-height +
mimic ΔA_φ + curl + superpose-p2u, excluding its nested set_bc/calc_u2p
which are attributed to their own rows) is 723 ms/step — 28% serial, and
**50% of the whole step once the inversions are threaded**. It runs twice
per step over domain+1-ring doing per-cell MKS2→BL frame transforms
(`calc_angle_brbphibsq`), Ê extraction, and a p2u superpose — per-cell,
cell-local, embarrassingly parallel, just never threaded (in C either).
Consequences for the roadmap: (a) P1 must thread the dynamo like any §1
pass — it was already in scope, but it is the single largest P1 line
item, ahead of sweep; (b) the dynamo is also a P2-grade optimization
target in its own right (the BL transform per cell per call is
recomputable-vs-cacheable, and the two calls per step re-derive identical
geometry). Post-P1 projection from this table: threading everything
threadable at the observed ~7× row-thread efficiency puts the step at
~360 ms (≈7× whole-step) on this hardware before any P2/P3 work; row
imbalance (the 17× cell spread) is what stands between 7.4× and 10× on
the inversion passes, confirming dynamic tiles.

**Threading efficiency today.** implicit 7.4× and u2p 5.6× on 10 threads
with static contiguous row bands: acceptable at this near-axisymmetric
t≈0 state where cost varies mostly with θ-row, but u2p's lower efficiency
already shows the per-pass spawn/join + band-imbalance overheads (u2p's
pass is only 18 ms — spawn/join is a fixed ~ms-scale tax). Both numbers
should rise with the persistent team + dynamic tiles.

### 7.3 First P1 increment — dynamo threaded (2026-07-07)

The §7.2 headline acted on immediately: `parallelRows` generalized to a
banded `parallelRange(lo, hi)` (any loop index — iy rows or ix columns;
identical band math, so the existing u2p/implicit dispatch is
bit-unchanged), and the dynamo's three per-cell passes now run on it:

- the ΔA_φ + DAMPBETA ring loop (iy bands over domain+1-ring; dt reaches
  the workers via a transient `sim.dyn_dt`, the `impl_dt` pattern);
- the superpose+p2u loop (iy bands over the domain);
- `calcScaleHeight` (ix bands — each worker owns whole columns, so every
  radius's θ-sum keeps its serial accumulation order).

The cheap curl stencil between ΔA_φ and superpose stays serial — it is
the natural barrier (reads finished neighbour-row ΔA_φ). Every band
writes only its own cells/columns, so the result is bit-identical to
serial at any thread count: pinned by a new default-suite gate
(`applyDynamo` ×2 on the 48×44 PUFFY torus with injected B³, nthreads
1 vs 4, full p/u/ΔA_φ/scale-height arrays compared to the bit), the M12
dynamo C-golden re-passing with identical deviations, and the production
runs below printing identical scalars and final t at 1 vs 10 threads.

Same P0 measurement, after the change:

| | serial ms/step | 10-thread ms/step | speedup |
|---|---|---|---|
| applyDynamo | 721.2 (28.5%) | **103.2** (12.5%) | **7.0×** |
| **whole step** | **2527** | **823** | **3.07×** (was 1.77×) |

Serial is unchanged within run noise — the banding indirection is free.
The 10-thread profile is now: **sweep 24.1%, implicit 18.7%, dynamo
12.5%, bc 8.8%, wavespeeds 8.3%, fluxes 8.3%, radvisc 6.2%** — ~540
ms/step of still-serial §1 passes dominate the step. Next P1 increments
in measured order: sweep, then wavespeeds/fluxes/radvisc/update — worth
roughly another 2× on this machine before the persistent-team + dynamic-
tile work (which then reclaims the spawn/join tax and the imbalance gap)
even starts.

### 7.4 P1 complete — persistent team, dynamic tiles, every pass threaded (2026-07-07)

**Implementation** (`koral/sim/threading.zig`, rewritten). The §3.2 design,
with one deviation forced by Zig 0.16: `std.Thread.Mutex`/`Condition` moved
behind the `std.Io` event-loop interface, so the team parks directly on the
OS futex (`std.os.linux.futex_*` / darwin `__ulock_*` — the same calls
`std.Io.Threaded` makes internally; ~40 lines, other platforms fall back to
a yield poll).

- **Persistent team.** `Sim.init` spawns `nthreads−1` helpers once; they park
  on a futex'd region-generation counter. A region is `(task, ctx, [lo,hi),
  tile)`; publishing is one release-increment + wake, completion one
  countdown word the main thread parks on. No per-pass spawn/join anywhere.
- **Dynamic tiles.** Workers (main thread included) pull contiguous tiles of
  the banded index through an atomic ticket; tile = range/(8·workers). The
  implicit pass's ~17× torus/floor cell-cost spread straggle-balances
  naturally — measured below as u2p going 5.6×→7.8× and implicit 7.4×→8.0×
  at the same thread count.
- **Reductions.** `ChunkResult` carries the implicit counters *and* the CFL
  `tstepden` max/min (lifted out of `save_wavespeeds`, closing the §1
  hazard); every worker accumulates on its own stack across its tiles and
  publishes once at region end (no false sharing on hot counters — the
  §1 `ChunkResult` hazard), merged in fixed order. All reductions are
  order-insensitive, so any tile schedule reproduces the serial bits.
- **Coverage.** Every §1 pass now dispatches through the ctx-generic
  `parallelRange` (small stack context structs replace the transient
  `impl_dt`/`dyn_dt` fields): implicit + u2p (migrated), sweep and
  fluxesAtFaces (banded over cross[0] per dimension, comptime-dim workers),
  wavespeeds (with the reduce), flux-CT (two regions — the pass-1→2 EMF
  barrier is the region boundary), conserved update + metric source,
  radvisc `calcRijViscTotal`, the three dynamo passes, `setBc`'s three face
  fills (x over iy, y/z over ix; the 2D corner fill stays serial after
  them), cell_fixup (parallel averaging + `parallelCopy` snapshots),
  polar-axis correction (ix columns), update_entropy, and the stage
  arithmetic (`parallelCopy` full-field copies, banded deriv/combine).
  Still serial: the corner fill, the dynamo curl, `saveTimesteps`,
  `copy_entropycount` — each ≪1%.
- **Bit-identity gates.** The MINK-box step gate (nthreads 1 ≡ 4) now also
  pins the dt reduction; a new **full-PUFFY step gate** (48×44 torus,
  startup + CFL step, radviscosity + dynamo + polar correction + specific
  MKS2 BCs) compares p/u/rijvisc/scaleth/flags/counters/dt-reduce to the
  bit. Full suite: 190 passed / 7 skipped; the `-Dslow-tests` golden battery
  (keystone, radtubes, M12/M13 goldens incl. the puffy64 step: flags 100%)
  passes unchanged — all goldens run at nthreads=1, whose pass bodies are
  bit-identical to the pre-P1 code.

**Measurement** (same battery as §7.2: PUFFY 384×360, ReleaseFast, Apple
M-series 12-core, 20 steps from t=0, identical scalars and final t at every
thread count):

| Pass | serial ms/step | 10T ms/step | speedup |
|---|---|---|---|
| opImplicit | 1145.0 | 142.9 | 8.0× |
| applyDynamo | 721.7 | 92.5 | 7.8× |
| sweep | 191.5 | 25.6 | 7.5× |
| calcU2p | 143.6 | 18.5 | 7.8× |
| setBc | 72.3 | 9.9 | 7.3× |
| calcWavespeeds | 68.0 | 8.7 | 7.9× |
| fluxesAtFaces | 65.3 | 11.8 | 5.5× |
| radvisc | 50.4 | 6.5 | 7.7× |
| stage arithmetic | 31.8 | 5.0 | 6.4× |
| conserved update | 31.1 | 4.0 | 7.8× |
| updateEntropy | 8.3 | 1.1 | 7.7× |
| cellFixup | 7.8 | 4.4 | 1.8× |
| fluxCt | 5.1 | 0.8 | 6.2× |
| **whole step** | **2543** | **332.9** | **7.64×** |

12 threads (E-cores in the pool): **308.8 ms/step = 8.24×** — the ticket
scheduler absorbs the slower cores, +8% over 10T, as §5 predicted. Serial
is 2543 vs 2527/2561 in the two earlier runs — unchanged within noise.

**Reading.** The step went 2561 → 1449 (inversions threaded, pre-P0) → 823
(dynamo increment) → **333 ms/step**, and the §7.2 post-P1 projection
(~360 ms) was slightly beaten because dynamic tiles lifted the inversion
passes past their static-band ceiling. Compute-heavy passes cluster at
7.3–8.0× on 10 threads; the outliers are bandwidth-bound (`cellFixup` is
memcpy-dominated at 1.8×, stage 6.4×, fluxes 5.5× — the §3.3 bandwidth wall
showing first). The 10T profile is implicit 42.9% > dynamo 27.8% > sweep
7.7% > u2p 5.5% — i.e. the step is again dominated by the two per-cell
physics passes, which is exactly where P2 (dynamo BL-geometry reuse, memory
diet) and the P3 follow-ons (vector libm behind `-Dfast`) should land next.
NUMA first-touch + pinning deferred to the P4 cluster work: no thread
affinity API on Darwin and no NUMA on unified memory, so it cannot be
validated here.

---

*Sources: koral-zig hot-path inventory (sim.zig, field.zig, precompute.zig, solve/, physics/, magn/) and koral_lite MPI architecture (mpi.c, choices.h, problem.c, fileop.c) — surveyed 2026-07-06. Hardware assumptions: Apple M-series dev box (12 cores, NEON), 64+ core x86/ARM cluster nodes (AVX-512/SVE, 8–12 NUMA-relevant CCDs, ~460 GB/s/socket).*
