# Slow-light GRRT + adaptive photon-ring refinement — 2026-08-11

Second ray-tracing mode for the imager: photons sample the fluid at their own
retarded coordinate time across a KDMP series (slow light), plus a vacuum
pre-pass that concentrates image-plane resolution on the photon ring
(adaptive quadtree). Fast light is untouched — the static-series identity is
gated bit-for-bit.

## Files

| file | role |
|---|---|
| `koral/render/render.zig` | `traceRay` refactored into resumable `advanceRay(cfg, scene, sampler, state, opts, ctl)` — same loop verbatim, generic over sampler (data source) and pause controller. `RayState`/`RayStatus`, `traceRayWith`, `SnapshotSampler`, `sampleData` (grid+data made explicit), `parseHeaderLoose` (v1/v2 header, shared with the scanner). |
| `koral/render/series.zig` | `Series.scan` (header-only dir scan, sort by t, newest frame's shape wins + foreign shapes skipped, dedup equal t, stride from the end), `WindowSampler` (trilinear × linear-in-t, edge clamp counted), `FileSource` (refcounted frame residency), `SliceSource` (in-memory, tests). |
| `koral/render/sweep.zig` | `renderSlow` — the 3-phase streaming sweep; `RaySpec`/`uniformPlan`; `SlowOpts`, `Stats`. |
| `koral/render/adaptive.zig` | `planRays` — vacuum corner grid + per-pixel quadtree → `[]RaySpec` (weights exact powers of ¼, Σ=1/pixel); `renderPlan` (fast-light plan consumer); slow light consumes the same plan. |
| `tools/kdmp2png.zig` | `--slow DIR [ref.kdmp]`, `--tobs`, `--stride`, `--rslow`, `--adapt D`, `--adapt-dt`, `--max-steps`. |

## Physics (what was double-checked)

* **Retarded time is exact, not modeled.** The geodesic integrator always
  carried the full 4-vector; slow light simply feeds `x⁰` to the sampler.
  Gravitational time delay, Shapiro delay and photon-ring lags are properties
  of the integrated null geodesic — no flat-space approximation anywhere.
  Gates: outgoing-radial KS flight time matches Δt = Δr + 4M·ln((r_c−2M)/(r_e−2M))
  to < 0.05 M over 1000 M (the 4M ln term is 19.3 M there); consecutive
  photon-ring windings (Δφ differing by exactly 2π, equatorial a = 0) are
  delayed by the photon-orbit period 2π·3√3 ≈ 32.65 M to < 0.5 M.
* **Monotonicity ⇒ streaming.** In (M)KS, g^tt < 0 outside the horizon, so t
  is a global time function and every future-directed null k has k⁰ > 0:
  backward tracing decreases x⁰ strictly, ergosphere and horizon included
  (KS time is horizon-regular — in BL coordinates slow light would be
  pathological near r_h). Hence one newest→oldest pass over the series
  suffices and each frame is read exactly once.
* **No re-entry.** A Kerr null geodesic has at most one radial turning point
  outside the photon shell, so a ray that leaves r_slow spatially never
  returns — the tail handoff is exact, not heuristic.
* **Frames live on constant-t slices** of the same chart the rays are
  integrated in (KDMP t is MKS2/KS coordinate time, in M), so interpolating
  two frames at fixed spatial index IS sampling the fluid at the event —
  no simultaneity transformation exists to get wrong. Interpolation is
  linear in the primitives (standard slow-light practice); u^μ, ν̂, the M1
  dipole etc. are recomputed from interpolated primitives per sample.
* **Static-zone approximation** (the one physics approximation): outside
  `--rslow` (default 40 M) the flow is sampled from one reference frame.
  Justified for PUFFY: that region is floor atmosphere with orbital time
  ≫ the series span. Series-edge extrapolation holds the end frame and is
  counted (`Stats.clamped_*`, `past_start`).
* **Camera convention:** `t_cam = t_obs + r_cam`, so `--tobs` reads as "sim
  time imaged at the hole". The absolute offset is unobservable; the delay
  DIFFERENCES across the image (far side ~2r later than near side, ~16 M
  per photon-ring winding) are the physics.

## The sweep (sweep.zig)

A. **Free flight** camera → r_slow on the reference frame (Scene.data).
B. **Window sweep**: consecutive frame pair [t_lo, t_hi] slides newest→oldest;
   every active ray advances until its x⁰ < t_lo, then parks; window slides
   when all are parked. ≤ 2 frames + reference resident (FileSource refcount).
   An 18 GB series renders in a ~600 MB footprint.
C. **Tails** finish on the reference frame.

Pause/resume happens BETWEEN integrator steps and touches no arithmetic:
with a static series the sweep is bit-identical to plain `traceRay`
(render_tests.zig gates the full 3-phase path and `traceRayWith` separately;
the WindowSampler lerp is written `a + w·(b−a)` so equal frames reproduce
snapshot samples exactly). Deterministic at any thread count: rays are
independent; the reduction runs sequentially in spec order; clamp counters
are order-insensitive integer sums.

## Adaptive refinement (adaptive.zig)

Vacuum pre-pass (screen tracing, fluid-independent, one plan serves fast and
slow): (W+1)×(H+1) corner rays record (captured?, flight time), endpoint
first-order-corrected to the exact capture/escape sphere — WITHOUT that
correction the final-step overshoot (tens of M of coordinate time at coarse
eps in the log-r outskirts) drowns the few-M signal and marks the whole
frame (found empirically). A pixel refines when corners disagree: capture
flip, or flight-time spread > `--adapt-dt` (default 5 M; a photon
half-orbit is ~16 M, so the criterion is exactly the delay structure slow
light resolves). Marked pixels subdivide DFS to `--adapt D`; leaves = cells
whose corners agree (one center ray, weight = area). Probes get their own
small step budget (`PlanOpts.probe_max_steps`, default 5000): a probe that
can't resolve capture-vs-escape in that many steps is pinned to the shell,
which IS the refine signal.

Cost notes: with matter, near-critical RT rays terminate via τ within a few
torus crossings; in `--screen` (vacuum) they orbit out the whole step
budget — pass `--max-steps 15000` for deep-adapt screen renders. The
refinement self-limits by gradient (a smooth cell of size s stops once
s·|∇T| < adapt-dt), so `--adapt-dt` is the main cost knob: at small fov the
whole near-curve zone has |∇T| > 5 M/px and a 5 M threshold refines a fat
band (measured: fov 16 at 96 px → 592/9216 px refined, ~150k rays);
raising it to ~15 M confines the work to the curve itself.

## Usage

```
# slow light, EHT-ish frame at the default t_obs = t_last − r_slow
kdmp2png koral/problems/puffy/puffy3d_sgra_spin.toml \
    --slow dumps/puffy3d_sgra_spin out.png \
    --size 512 --fov 40 --incl 60 --stride 8

# + photon-ring refinement
... --adapt 6 --adapt-dt 5

# movie: one invocation per frame, sliding --tobs
```

Smoke-validated on dumps/puffy3d_sgra_spin (25 strided frames, 21 sweep
rounds, 22 frame loads, zero edge holds at t_obs = 112.2): slow 96² ss1 in
7.8 s vs 7.6 s for the co-temporal fast frame — the sweep's streaming adds
essentially nothing over fast light at equal data.

## Caveats

* The sgra_spin series is MRI-unresolved (see qmri verdict in the campaign
  notes): its *variability* is numerically stunted, so slow-light images of
  it demonstrate the machinery, not trustworthy dynamics. The 152 M span
  also only just covers one image's delay spread (~130 M at r_slow = 40) —
  a movie needs the longer, better-resolved cluster campaign series.
* Rendering the pre-MESA dumps with the current sgra_spin toml mixes new
  microphysics into old data (see campaign notes) — unchanged by this work.
* `Series.scan` adopts the NEWEST frame's shape and skips foreign frames
  (dumps/puffy3d_sgra_spin contains 3 frames of a 256×320×32 restart).
* stride counts from the newest frame, so the latest frame is always kept.

## 2026-08-11 addendum: FITS export + light curves (EHT tiers 1–2)

* `koral/render/fits.zig` — pure FITS serializer (BITPIX −64, one HDU),
  AIPS/ehtim keywords: CDELT1/2 [deg] (RA negative), CRVAL + OBSRA/OBSDEC
  (both spellings), FREQ, MJD, BUNIT='JY/PIXEL'. Rows flipped to FITS
  bottom-up; +α → −RA (east-left). Values are I_ν·Ω_pix·10²³ = Jy/pixel —
  `ehtim.image.load_fits` → `im.observe(...)` works directly.
  `kdmp2png --fits PATH` (requires --nu/--dist; exports the RAW image,
  before blur/stretch — synthetic-observation pipelines apply their own
  beam). `--ra/--dec` default to Sgr A* J2000, `--mjd` to 57850
  (2017 Apr 7). Cross-check: FITS ΣS_pix ≡ the --dist Jy print, verified
  identical on real data.
* `tools/kdmp2lc.zig` (`zig build kdmp2lc`) — slow-light flux light curve:
  all epochs batched into ONE streaming sweep via the new
  `sweep.SlowOpts.t_cam_of` (per-spec arrival times; each epoch aimed at
  its own pixel block; the series is still read once). Emits
  idx/t[M]/t[s]/S_ν[Jy]/I_max rows + mean, σ, modulation index M = σ/μ —
  the EHT Sgr A* Paper V variability observable. `--frames DIR` writes
  per-epoch PNGs with a SHARED white point (flicker-free movie frames).
  Gated: a two-epoch batched sweep is bit-identical to two independent
  sweeps (flare scene).
* **r_slow lesson (measured)**: the static zone erases variability
  originating outside it. sgra_spin's 230 GHz flux is bulk free-free →
  rslow 40 gave M ~ 1e-6 while fast-light frames 8.5 M apart differ by
  3.5e-3; rslow 200 recovered M = 2.7e-3 over 52 M (secular brightening),
  with 98% of rays honestly flagged as needing pre-series lookback.
  Compact-synchrotron sources are what the default 40 is for.
* Flux reality check: this preset at 230 GHz gives S_ν ≈ 1.2e-5 Jy vs
  Sgr A*'s ~2.4 Jy — the puffy preset is not a Sgr A* mm source (thermal
  T_b ~ 2e5 K, no hot synchrotron electrons; single-temperature T_e = T_gas
  everywhere). The exporter/light-curve machinery is regime-agnostic; the
  physics verdict awaits campaign runs + electron-temperature work.
