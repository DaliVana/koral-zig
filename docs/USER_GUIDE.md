# koral-zig — User & Developer Guide

A practical, recipe-driven guide to building, running, configuring, extending, and
testing **koral-zig** — a Zig 0.16 reimplementation of the KORAL general-relativistic
radiation-MHD C code (`../koral_lite`), targeting the **PUFFY** radiative-MHD
accretion torus (a 2D axisymmetric limotorus around a 10 M☉ Schwarzschild black
hole in MKS2 coordinates).

This document is onboarding-grade reference material. Every API, equation, constant,
and file path below is drawn from the actual source tree. For the *why* behind the
architecture see `ARCHITECTURE.md`; for the physics derivations see `PHYSICS.md`.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Building and running](#2-building-and-running)
3. [The params (TOML) file](#3-the-params-toml-file)
4. [Output formats](#4-output-formats)
5. [Running the tests](#5-running-the-tests)
6. [Regenerating golden data](#6-regenerating-golden-data)
7. [Recipe: creating a new problem](#7-recipe-creating-a-new-problem)
8. [Cookbook: how to change things](#8-cookbook-how-to-change-things)
9. [Codebase map](#9-codebase-map)

---

## 1. Prerequisites

### To build and run problems

- **Zig 0.16.0.** The code uses the Zig 0.16 `std.Io` API (e.g.
  `std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20))` in
  `koral/params.zig`) and the `pub fn main(init: std.process.Init) !void` entry
  signature. Older Zig releases will not compile it.
- Nothing else. The library depends only on `std`; the serial communication
  backend (`koral/comm/serial.zig`) is a no-op single-rank implementation, so no
  MPI is required for a normal run. (The `build.zig` `-Dmpi` flag exists but is
  "not yet implemented".)

### Additionally, to regenerate the C-comparison golden data

- **clang** and **GSL** (GNU Scientific Library). `tools/gen_golden.sh` finds GSL
  via `brew --prefix gsl` and compiles the C oracle harnesses with
  `-O2 -fcommon -w -I$GSL_PREFIX/include`, linking `-lgsl -lgslcblas -lm`.
- A sibling checkout of **koral_lite** at `../koral_lite` (override with
  `KORAL_LITE=<path>`). The oracle *is* koral_lite compiled with tiny extra test
  problems overlaid.
- The script uses `brew --prefix` and BSD `sed -i ''`, so it is effectively
  macOS-only as written; on Linux the GSL path and `sed` invocation need
  adapting. `libomp` is optional and not required for the serial oracle build.

The golden data is committed to `tests/golden/`; you only need the C toolchain if
you are regenerating it after changing `../koral_lite` or adding a new oracle.

---

## 2. Building and running

The build graph (`build.zig`) is small and fully automatic:

- It creates one **library module** `koral` (root `koral/koral.zig`).
- It scans `PROBLEMS/` and builds **one executable per subdirectory** from
  `PROBLEMS/<name>/main.zig`, importing `koral`.
- It wires a single **`test`** step (`zig build test`) that runs *all* library
  tests as one artifact.

### Commands

```sh
# Build everything (library check + every problem executable, installed to zig-out/bin)
zig build

# Build just one problem
zig build puffy

# Build and run a problem, forwarding args after `--` to the executable
zig build run-puffy -- PROBLEMS/puffy/puffy.toml

# Optimized build (recommended for real runs)
zig build run-puffy -Doptimize=ReleaseFast -- PROBLEMS/puffy/puffy.toml
```

The problem name is the directory name under `PROBLEMS/`. For each `<name>` you
get a `<name>` step (build & install) and a `run-<name>` step (build & run). The
executable takes **one optional argument**: the path to a params file. `puffy`
defaults to `PROBLEMS/puffy/puffy.toml` if none is given (see `main` in
`PROBLEMS/puffy/main.zig`).

### Build options

| Flag | Meaning |
|---|---|
| `-Doptimize=Debug\|ReleaseSafe\|ReleaseFast\|ReleaseSmall` | Standard Zig optimize mode. Debug builds enable the many `std.debug.assert` bounds checks in `Field.cellOffset` etc. |
| `-Dtarget=...` | Standard Zig cross-compile target. |
| `-Dmpi=true` | Threads an MPI build option through; **not yet implemented** (the numerics only see the serial `comm` interface). |
| `-Dslow-tests=true` | Enables the slow test bodies (convergence studies, soaks, the full-grid PUFFY t=0 keystone). See §5. |
| `-Dtest-filter=<substr>` | Passed to `addTest` `.filters`; restrict the single test artifact to tests whose name contains the substring. Repeatable. |

### What the `puffy` executable prints

On startup it echoes the derived configuration and the geometrized-unit time scale
(`GM/c³` in seconds via `Units.gmc3()`):

```
puffy: NV=13 grid 384×360×1 (+3 ghosts) | M=10 M☉ (GM/c³=4.9…e-05s) | threads=1
puffy: init done — β-normalization fac = …
```

Then, at each output cadence, one status line per row:

```
puffy: t=… nstep=… dt=… | Ṁ=… L=… H/R=… β⁻¹=… | nan=… hdfix=… radimpfail=…
```

If any primitive becomes non-finite the run prints `NaN detected — aborting` and
stops. All quantities are in **geometrized/code units** (GU); unit conversion to
CGS/Eddington is left to downstream analysis, matching KORAL's convention of
keeping `scalars[]` in GU.

---

## 3. The params (TOML) file

Runtime parameters live in a **flat TOML-subset** file parsed by
`koral/params.zig`. The parser is line-based: `key = value`, `#` comments (honored
outside quoted strings), blank lines and `[section]` headers ignored (sections are
purely cosmetic grouping). **Keys must match `Params` field names exactly** —
an unknown key is a hard `error.UnknownParamsKey` (typo protection, since the
params file is the run record). Values are typed by field: `f64` via `parseFloat`,
`usize` via `parseInt` base 10, `bool` as `true`/`false`, strings de-quoted and
duplicated.

These are *runtime* numbers a physicist tweaks between runs; anything that changes
the generated code (module set, reconstruction, coordinates) is a **comptime**
`Config` and lives in code, not here (§7).

### Every `Params` field

From `koral/params.zig` (defaults shown):

**Physical**

| Field | Default | Meaning (C name) |
|---|---|---|
| `mass` | `1.0/147700.0` | BH mass in solar masses (`MASS`). The default makes `MASSCM=1` (length identity); **PUFFY must set `mass = 10`**. |
| `bhspin` | `0.0` | Dimensionless spin (`BHSPIN`). PUFFY = 0 (Schwarzschild). |
| `gam` | `5.0/3.0` | Gas adiabatic index Γ (`GAMMA`). |

**Grid** (active cells; serial ⇒ equals total)

| Field | Default | Meaning |
|---|---|---|
| `nx`, `ny`, `nz` | `1`, `1`, `1` | Active cell counts (`NX`/`NY`/`NZ`). A dimension of 1 collapses (zero ghosts, `SX=1`). |

**Domain** (in *output* coordinates; the problem derives internal bounds)

| Field | Default | Meaning |
|---|---|---|
| `rmin`, `rmax` | `0.0`, `0.0` | Radial extent. E.g. PUFFY derives `MINX = log(rmin − mksr0)` for MKS2. |
| `mksr0` | `0.0` | MKS2 radial offset `R0` in `r = R0 + e^{x1}` (`MKSR0`). |
| `mksh0` | `1.0` | MKS2 polar squeeze `H0` (`MKSH0`). |
| `miny`, `maxy` | `0.0`, `1.0` | θ-direction internal bounds. |
| `minz`, `maxz` | `0.0`, `1.0` | φ-direction internal bounds. |

**Run control**

| Field | Default | Meaning |
|---|---|---|
| `tstart` | `0.0` | Start time. |
| `tmax` | `0.0` | Stop time (code units). |
| `nstep_max` | `maxInt(usize)` | Max number of steps. |
| `tsteplim` | `0.5` | CFL factor (`TSTEPLIM`). |
| `dtout1` | `0.0` | Scalar-diagnostics output cadence (code time). |
| `dtout2` | `0.0` | Binary-dump cadence; `>0` enables `.kdmp` prim dumps. |
| `out_dir` | `"dumps"` | Output directory. |

**Floors & ceilings**

| Field | Default | Meaning |
|---|---|---|
| `rhofloor` | `1.0e-30` | Minimum density. |
| `uurhoratiomin` / `uurhoratiomax` | `1.0e-8` / `1.0e2` | Bounds on `uint/rho`. |
| `eerhoratiomin` / `eerhoratiomax` | `1.0e-20` / `1.0e20` | Bounds on radiation energy / rho. |
| `b2rhoratiomax` | `100.0` | `b²/rho` ceiling. |
| `b2uuratiomax` | `100.0` | `b²/uint` ceiling. |
| `gammamaxhd` / `gammamaxrad` | `100.0` / `100.0` | Max gas / radiation Lorentz factor. |

**Execution**

| Field | Default | Meaning |
|---|---|---|
| `deterministic` | `false` | Reserved determinism flag. |
| `nthreads` | `1` | Persistent worker team for all per-step passes; `1` is the bit-identical serial path (§8). |

> **Note.** The `Params` floors are the *runtime* record. The PUFFY driver
> additionally selects tuned **comptime** floor presets
> (`FloorParams.puffy`, `RadParams.puffy`) inside `options()` — those are what the
> inversion and radiation solvers actually use (see §8, "Adjust floors").

### PUFFY example (`PROBLEMS/puffy/puffy.toml`)

```toml
# PUFFY — radiative-MHD limotorus, 10 M☉ Schwarzschild, MKS2

[physical]
mass = 10.0        # MASS (solar masses)
bhspin = 0.0       # BHSPIN — Schwarzschild
gam = 1.6666666666666667   # GAMMA 5/3

[grid]
nx = 384           # TNX (radial)
ny = 360           # TNY (theta)
nz = 1             # TNZ — 2D axisymmetric
rmin = 1.85        # RMIN (< r_horizon=2, so the inner boundary is inside the hole)
rmax = 500.0       # RMAX
mksr0 = 0.1        # MKSR0
mksh0 = 0.9        # MKSH0
miny = 0.001       # MINY
maxy = 0.999       # MAXY = 1 - 0.001
minz = -0.7853981633974483   # -pi/4 (PHIWEDGE pi/2)
maxz = 0.7853981633974483    # +pi/4

[run]
tstart = 0.0
tmax = 10000.0     # run duration in GM/c^3 (reduce for a quick test)
nstep_max = 100000000
tsteplim = 0.5     # TSTEPLIM (CFL factor)
dtout1 = 100.0     # scalar-diagnostics cadence (code time)
dtout2 = 500.0     # primitive-dump cadence (0 disables prim dumps)
out_dir = "dumps/puffy"
nthreads = 1       # persistent worker team for all per-step passes (1 = serial)

[floors]
rhofloor = 1e-30
uurhoratiomin = 1e-8
uurhoratiomax = 1.0
eerhoratiomin = 1e-20
eerhoratiomax = 1e4
b2rhoratiomax = 50.0
b2uuratiomax = 50.0
gammamaxhd = 10.0
gammamaxrad = 10.0
```

Section headers (`[physical]`, `[grid]`, …) are cosmetic — the parser keys purely
on field names, so all fields could sit in one section. The production PUFFY grid
is `TNX=384, TNY=360, TNZ=1, NG=3`; `nx`/`ny` can be reduced for cheaper runs (the
coordinate extents stay fixed — `makeGrid` derives them from the PUFFY constants,
not from the params domain bounds). Note the driver runs while `t < tmax` **and**
`nstep < nstep_max`, so `tmax = 0.0` would take zero steps.

---

## 4. Output formats

### `scalars.dat` — diagnostic time series

Written by `koral/io/dump.zig` (`appendScalarLine` / `scalar_header`). A
whitespace-separated text file, one row per output cadence, in **GU/code units**.
The header line is:

```
# t dt nstep mass mdot radlum totallum H/R maxPmag/Ptot n_hdfix n_radimpfail n_nan
```

The 12 columns of a `ScalarRow` (`koral/io/dump.zig`), and where each is computed
(`koral/io/scalars.zig`):

| Column | Meaning | Source |
|---|---|---|
| `t` | Simulation time. | `Sim.t` |
| `dt` | Step size used. | driver |
| `nstep` | Step counter. | `Sim.nstep` |
| `mass` | Total rest mass `Σ ρ √−g dx dy dz` over the domain. | `totalMass` |
| `mdot` | Mass flux through the horizon shell, **sign-flipped so >0 = accretion**. | `−mdot(ix_horizon)` |
| `radlum` | Radiative luminosity `∮ max(0,−R^r_t) √−g dθ dφ` at the outer shell (BL frame). | `lum` |
| `totallum` | Total energy flux `∮ (ρuʳ + T^r_t + max(0,−R^r_t)) √−g dθ dφ` (same clamped/negated radiation term as `radlum`). | `lum` |
| `H/R` | Density-weighted RMS `√(π/2−θ)²` scale height at r≈15. | `scaleHeightAt` |
| `maxPmag/Ptot` | Max magnetization β⁻¹ = pmag/ptot over the domain (`pmag=b²/2`, `ptot=(Γ−1)u [+ Ê/3]`). | `maxMagnetization` |
| `n_hdfix` | Count of cells that needed an HD inversion fixup. | `collectDiag` |
| `n_radimpfail` | Count of implicit radiative-source failures. | `Sim.n_radimp_failures` |
| `n_nan` | Count of cells with a non-finite primitive. | `collectDiag` |

Print precision (from `appendScalarLine`): `t/mass/mdot/radlum/totallum` at
`{e:.10}`, `dt/H/R/β⁻¹` at `{e:.6}`, counters/`nstep` as integers.

Diagnostic radii used by the PUFFY driver: `r_horizon = 2.0`, `r_lum = 5000.0`
(> RMAX ⇒ the outermost shell), `r_scale = 15.0`.

> **Wedge quirk (faithful to C):** `totalMass` uses the *raw* cell dz (the φ-wedge
> width) and does **not** expand to 2π for a `TNZ==1` slice, whereas `mdot`/`lum`
> **do** hardcode `dφ = 2π`. So the PUFFY 2D `mass` is a wedge mass, not a full
> torus mass. This inconsistency is transcribed from `calc_totalmass`.

### `prims#####.kdmp` — binary primitive snapshot

Written by `writePrimDump` (`koral/io/dump.zig`) when `dtout2 > 0`. Little-endian
throughout. Layout:

```
"KDMP"              4 bytes magic
u32 version = 1
u32 nx, ny, nz, nv  (interior cell counts and NV)
f64 t
f64 p[nz][ny][nx][nv]   iv fastest, then ix, then iy, then iz (AoS, matches C get_u)
```

`primDumpSize` returns exactly the number of bytes `writePrimDump` writes:
`grid.nx*ny*nz` are the *interior* (active) cell counts (the padded storage is
`sx()=nx+2*ngx`), and `writePrimDump` loops over exactly those interior cells, so
allocate `primDumpSize` and use the returned count. The `iv`-fastest AoS order
matches KORAL's `get_u`/`set_u` and the KSTP/KINI golden byte order.

---

## 5. Running the tests

```sh
# Run the whole suite (one artifact, all tests)
zig build test

# Enable the slow bodies (convergence studies, soaks, full-grid PUFFY keystone)
zig build test -Dslow-tests=true

# Restrict to tests whose name contains a substring (repeatable)
zig build test -Dtest-filter="radtube"
zig build test -Dtest-filter="golden" -Dtest-filter="implicit"
```

### Reading the results

`zig build test` compiles *one* test artifact from `koral/koral.zig`'s `test {}`
block, which `_ =` imports every unit module and every dedicated test file. If a
test fails, the Zig **build runner** prints a noisy `error: the following command
failed` / `run test` block around it — that framing is boilerplate. The signal is:

- The individual `FAIL`/`error.<Name>` lines with file:line and the printed
  deviation (golden tests print the observed max deviation via `DevTracker.check`,
  so you see *how far off* rather than dying on the first element).
- The process **exit code** (`0` = all passed/skipped).

**Skips are expected and not failures.** The golden readers
(`readGolden`/`readKstp`/`readKini`) return `error.SkipZigTest` when a golden file
is absent, so the suite is green on a machine that never ran `gen_golden.sh`. A
deleted golden shows as a skip, not a pass — keep that in mind when auditing
coverage.

### Test-file inventory

Three complementary layers (there are deliberately **no** run-to-completion
end-to-end tests). Test files live flat in `koral/` next to the code, with one
naming rule per family so a subsystem's files sort adjacently: theory gates are
`<subsystem>_tests.zig`, C-oracle goldens are `<subsystem>_golden_tests.zig`.
New test files must be listed in `koral/koral.zig`'s `test` block (there is no
auto-registration).

**1. Theory gates** — mathematical identities and documented C quirks, no golden
data:

- `metric_tests.zig`, `evolution_tests.zig`, `mhd_evolution_tests.zig`,
  `radiation_tests.zig`, `opacity_tests.zig`, `implicit_tests.zig`,
  `state_tests.zig`, `flux_tests.zig`, `polaraxis_tests.zig`,
  `radstep_tests.zig`, `radtube_tests.zig`, `dynamo_tests.zig`,
  `radvisc_tests.zig`, `scalars_tests.zig`, `threading_tests.zig`,
  `puffy_tests.zig`, plus in-module `test` blocks in nearly every source file.

**2. Function-level C goldens** (`.kgld`) — diff Zig vs C at recorded input
points, with C's *own* geometry embedded per record (`geomFromRecord`) so only the
state algebra is compared:

- `metric_golden_tests.zig`, `state_golden_tests.zig`, `flux_golden_tests.zig`,
  `rad_golden_tests.zig`, `opac_golden_tests.zig`, `implicit_golden_tests.zig`.

**3. Forced-dt step / keystone goldens** (`.kstp`, `.kini.gz`) — load C's post-init
state bit-for-bit, force C's recorded dt sequence, diff the whole domain each step:

- `step_golden_tests.zig` (the ZIG* tube step goldens),
  `puffystep_golden_tests.zig` (reduced-grid full PUFFY pipeline),
  `puffy_golden_tests.zig` (PUFFY t=0 keystone, full-grid only under
  `-Dslow-tests`), `visc_golden_tests.zig`, `dynamo_golden_tests.zig`.

Support files: `koral/testing/golden.zig` (`Golden`/`Kstp`/`Kini`/`DevTracker`
readers) and `koral/testing/tubes.zig` (the Farris/Sadowski radiative shock-tube
battery + the paper→KORAL unit bridge).

> **Expected non-zero floors.** Some goldens gate at `1e-8` (MKS2 derived
> quantities — C's Mathematica metric exports mix truncated `Pi` vs exact π) and
> the PUFFY torus keystone at `~1e-3` (C's `gsl_qags` is only ~1e-3 accurate at
> the ℓ(λ) C¹ kink; Zig's GK21 is the *more* accurate side, proven by the
> `puffyeps12` variant). These are documented, not bugs — do not tighten them.

---

## 6. Regenerating golden data

`tools/gen_golden.sh` (macOS + brew-GSL, run manually) produces everything under
`tests/golden/`. It never mutates `../koral_lite`; it copies the sources into
`oracle/build/<variant>/src`, overlays the koral-zig test problems from
`oracle/problems/`, registers extra `PROBLEM` numbers in `problem.h`, compiles a
serial clang+GSL build (the `OBJS` list of ~20 koral_lite translation units), and
links each `oracle/harness_*.c`.

### The oracle harnesses

Each harness *is* koral_lite compiled with a driver that dumps a golden:

- `harness_metric.c` / `harness_state.c` / `harness_flux.c` / `harness_rad.c` /
  `harness_opac.c` / `harness_implicit.c` / `harness_scalars.c` /
  `harness_visc.c` / `harness_dynamo.c` — function-level `.kgld` goldens for the
  metric, state algebra, flux vector, M1 radiation, opacities, implicit solver,
  scalar reductions, radiative viscosity, and dynamo.
- `harness_step.c` — the explicit/MHD/radiative **forced-dt step** oracle. Built 5×
  (PROBLEM 200–204 = ZIGSOD, ZIGOT, ZIGMHDTUBE, ZIGRADTUBE, ZIGRADPULSE). It
  replicates ko.c init and runs the RK2IMEX stage block verbatim from
  `problem.c:141-402`, dumping the whole domain each step as `.kstp`. PROBLEM 201
  additionally emits `ct.kgld` (flux-CT on PRNG-filled EMFs) and `bfroma.kgld`
  (`calc_BfromA` on PRNG vector potential), using a fixed-seed xorshift64* PRNG the
  Zig side replays.
- `harness_puffy_step.c` — the **full-PUFFY-pipeline** step oracle on a reduced
  64×60 grid (implicit rad source + radiative shear viscosity + dynamo), 4 steps as
  `.kstp` (gzipped).
- `harness_init.c` — the PUFFY **t=0 keystone**. Emits three `.kini` snapshots over
  the full 384×360 grid + ghosts: `puffy_t0_A` (A_φ still in the B slots),
  `puffy_t0_p` (all NV primitives after `calc_BfromA` — *the* keystone), and
  `puffy_t0_pfinal` (β-normalized B, ghosts deliberately stale). The `puffyeps12`
  variant tightens C's `qags` epsrel from 1e-8 to 1e-12 to attribute the torus
  keystone deviation to C's quadrature.

Large snapshots are `gzip -9`'d in the repo; the readers auto-inflate. A
`manifest.json` records the koral_lite SHA, PROBLEM list, compiler, and per-file
schema. After regenerating, commit the updated `tests/golden/**` and
`manifest.json`.

---

## 7. Recipe: creating a new problem

A **problem** is: a comptime `Config`, a runtime `Params` file, an
initial-condition + boundary-condition module, and a small driver executable. The
library (`koral`) provides everything else. This section walks through a complete
minimal example — a **relativistic MHD shock tube in flat (Minkowski) space** — and
then notes the extra pieces PUFFY needs.

Directory layout for a new problem `mytube`:

```
koral/config.zig                 # (optional) add a comptime Config const
koral/problems/mytube.zig        # init conditions + boundary conditions
koral/koral.zig                  # register mytube under `problems` + tests
PROBLEMS/mytube/main.zig         # the driver executable (auto-discovered)
PROBLEMS/mytube/mytube.toml      # runtime params
```

### (a) Choose or define a comptime `Config`

A `Config` (`koral/config.zig`) selects the physics modules and algorithms.
Modules **must** be listed in canonical order — `hydro, electrons, relel, mhd,
forcefree, radiation` — or `validate()` `@compileError`s. `.hydro` is mandatory.

For an ideal-MHD tube (no radiation), define in `koral/config.zig`:

```zig
/// Flat-space ideal-MHD shock tube: hydro + magnetic field, PPM, HLL.
pub const mytube = Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .ppm,   // or .linear (NG=2) / .donor_cell
    .flux = .hll,             // or .laxf
    .timestepping = .rk2imex,
    .coords = .mink,
};
```

This alone fixes the state-vector width and ghost depth: PPM ⇒ `NG=3`;
`{hydro, mhd}` ⇒ `NV=9` with `RHO=0, UU=1, VX=2, VY=3, VZ=4, ENTR=5, B1=6, B2=7,
B3=8` (via `VarLayout(cfg)`; radiation would add `EE..FZ` starting at 9). You can
also just reuse `config.puffy` if its module set fits.

### (b) Extend the layout only if you add a *new* evolved variable

If your problem needs an evolved variable that no existing module carries, you
extend three things in lockstep (see §8, "Add a physics module"): the `VarTag`
enum (`koral/layout.zig`), the owning module's tag list in `moduleTags`, and — from
M3 — that module's `StateFields`. `NV` and all indices then regenerate
automatically at comptime. For a standard hydro/MHD/radiation tube you need none of
this; the tags already exist.

### (c) Write `koral/problems/<name>.zig` — init + boundary conditions

Two responsibilities: build each domain cell's primitive vector, and supply the
`SpecificBc` callback for any axis using `.specific` boundaries.

**Boundary-condition callback signature** (from `Sim(cfg).SpecificBc`,
`koral/sim.zig`):

```zig
pub const SpecificBc = *const fn (
    sim: *const Self,
    ix: i64, iy: i64, iz: i64,   // ghost-cell indices (signed; ghosts are <0 or ≥n)
    t: f64,
    ifinit: bool,                // true during initial set_bc
    face: BcFace,                // .xlo/.xhi/.ylo/.yhi/.zlo/.zhi
) [NV]f64;                       // returns the ghost cell's primitives
```

Because `Sim` is generic over `Config`, the callback lives inside a comptime
factory `Bc(SimT)` returning a struct with a `pub fn calc(...)` — exactly the
PUFFY pattern (`koral/problems/puffy.zig`, `pub fn Bc(comptime SimT: type) type`).

A minimal `koral/problems/mytube.zig`:

```zig
const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const grid_mod = @import("../grid.zig");
const hydro = @import("../physics/hydro.zig");
const sim_mod = @import("../sim.zig");

pub const cfg = config.mytube;
const L = layout.VarLayout(cfg);
const Grid = grid_mod.Grid;

// --- geometry / grid ------------------------------------------------------
// Flat Cartesian box x∈[-0.5,0.5], collapsed in y,z (nz/ny = 1 ⇒ no ghosts there).
pub fn makeGrid(nx: usize) Grid {
    return Grid.init(.{
        .nx = nx,
        .ng = cfg.ghostCells(),   // PPM ⇒ 3
        .minx = -0.5, .maxx = 0.5,
    });
}

// --- one asymptotic tube state -------------------------------------------
const Side = struct { rho: f64, pgas: f64, vx: f64, bx: f64, by: f64 };

fn primsFor(s: Side, gam: f64) [L.count]f64 {
    var pp: [L.count]f64 = @splat(0.0);
    pp[L.index(.rho)] = s.rho;
    const uu = s.pgas / (gam - 1.0);        // ideal gas: p=(Γ-1)u
    pp[L.index(.uu)]  = uu;
    pp[L.index(.vx)]  = s.vx;               // VELPRIM == VELR spatial slots
    pp[L.index(.entr)] = hydro.sFromU(s.rho, uu, gam);
    pp[L.index(.b1)]  = s.bx;               // evolved 'curly B' field B^i
    pp[L.index(.b2)]  = s.by;
    return pp;
}

const left  = Side{ .rho = 1.0,  .pgas = 1.0,  .vx = 0.0, .bx = 0.5, .by = 1.0 };
const right = Side{ .rho = 0.125, .pgas = 0.1, .vx = 0.0, .bx = 0.5, .by = -1.0 };

// --- initial condition ----------------------------------------------------
pub fn initAll(comptime SimT: type, sim: *SimT) !void {
    var ix: i64 = 0;
    while (ix < sim.nxi()) : (ix += 1) {
        const x = sim.grid.xc(ix);           // cell-center internal coord
        const s = if (x < 0.0) left else right;
        try sim.initCell(ix, 0, 0, primsFor(s, sim.opt.gam)); // pp → p2u → store p,u
    }
    try sim.finishInit();                    // set_bc + initial dt guess
}

// --- boundary conditions (outflow / zero-gradient copy) -------------------
pub fn Bc(comptime SimT: type) type {
    return struct {
        pub fn calc(
            sim: *const SimT, ix: i64, iy: i64, iz: i64,
            t: f64, ifinit: bool, face: sim_mod.BcFace,
        ) [SimT.nv]f64 {
            _ = t; _ = ifinit;
            // Copy the nearest domain cell (simple outflow). For a static tube
            // you could instead return the fixed left/right asymptotic state.
            const src: i64 = switch (face) {
                .xlo => 0,
                .xhi => sim.nxi() - 1,
                else => unreachable,          // ny=nz=1: no y/z faces
            };
            var pp: [SimT.nv]f64 = undefined;
            sim.p.load(src, iy, iz, &pp);
            return pp;
        }
    };
}
```

Key API points, all real:

- `sim.initCell(ix, iy, iz, pp)` stores the primitives and derives the conserveds
  via `p2u` (`koral/sim.zig`).
- `sim.finishInit()` runs `setBc(0, true)` then `initTimestepGuess()`.
- `sim.grid.xc(ix)` is the cell-center internal coordinate (the two-face average,
  `0.5*(xl(i)+xl(i+1))` — an intentional ulp-level match to C; do not "simplify").
- `sim.nxi()/nyi()/nzi()` are the interior cell counts.
- `hydro.sFromU(rho, u, gam)` fills the entropy slot; the layout is indexed by
  `L.index(.rho)` etc., which `@compileError`s if the variable is absent for the
  active module set.

If your tube needs the vector-potential path (seed **B** from A rather than
directly), store A_φ in the B slots and call `ct.calcBfromA(SimT, sim, true)` then
`setBc` before `finishInit`, exactly as PUFFY's `initAll` does (which additionally
runs β-normalization in `postinit`). See `koral/magn/ct.zig`.

### (d) Write `PROBLEMS/<name>/main.zig` — the driver

The driver is auto-discovered by `build.zig`. Use `PROBLEMS/puffy/main.zig` as the
template; a minimal version:

```zig
const std = @import("std");
const koral = @import("koral");

const cfg  = koral.config.mytube;
const mytube = koral.problems.mytube;      // after registering it (step e)
const SimT = koral.Sim(cfg);

fn options(p: *const koral.Params) SimT.Options {
    return .{
        .coords = .mink,
        .gam = p.gam,
        .tsteplim = p.tsteplim,
        .floors = koral.solve.invert.FloorParams.cdefault,
        // opac = null  ⇒ no radiation source (SKIPRADSOURCE); fine for MHD-only.
        .bc_x = .specific,
        .specific_bc = &mytube.Bc(SimT).calc,
        .nthreads = p.nthreads,
    };
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next();
    const params_path = args.next() orelse "PROBLEMS/mytube/mytube.toml";

    const p = try koral.Params.load(a, io, params_path);
    defer a.free(p.out_dir);

    var s = try SimT.init(a, mytube.makeGrid(p.nx), options(&p));
    defer s.deinit();

    try mytube.initAll(SimT, &s);

    while (s.t < p.tmax and s.nstep < p.nstep_max) {
        var dt = 1.0 / s.tstepdenmax;          // CFL dt from the previous step
        if (s.t + dt > p.tmax) dt = p.tmax - s.t;
        try s.step(dt);
    }
    std.debug.print("mytube: done (t={d}, {d} steps)\n", .{ s.t, s.nstep });
}
```

The core loop is `dt = 1/s.tstepdenmax` then `s.step(dt)`. `Sim.step` runs one full
RK2IMEX step, alternating the implicit radiative-source operator with the explicit
operator across the stages. The explicit operator (`opExplicit`) itself is:
wavespeeds → reconstruction sweep → face fluxes → flux-CT for MHD → conserved
update (the flux divergence **and** the metric source term `ms[iv]*dt` are added
together here) → `calcU2p` inversion. Polar-axis correction (`doCorrect`) and the
final entropy update (`updateEntropy`) happen at the stage boundaries in `step`
(`koral/sim.zig`). `step` asserts `cfg.timestepping == .rk2imex`.

### (e) Register in `koral/koral.zig`

Add the problem module and (optionally) its tests to the `problems` namespace and
the `test {}` block:

```zig
pub const problems = struct {
    pub const puffy  = @import("problems/puffy.zig");
    pub const mytube = @import("problems/mytube.zig");   // add this
};

test {
    // ...
    _ = @import("problems/mytube.zig");                  // pull in its tests
}
```

### (f) The `.toml`

`PROBLEMS/mytube/mytube.toml`:

```toml
[physical]
mass = 1.0            # flat space: mass only scales units; MASSCM at default
gam  = 1.6666666666666667

[grid]
nx = 400

[run]
tmax     = 0.4
tsteplim = 0.5
dtout1   = 0.05
out_dir  = "dumps/mytube"
```

Build & run:

```sh
zig build run-mytube -Doptimize=ReleaseFast -- PROBLEMS/mytube/mytube.toml
```

---

## 8. Cookbook: how to change things

All snippets are grounded in the real APIs. Cross-reference `ARCHITECTURE.md` §6
(storage/layout insulation) and `PHYSICS.md` (the equations behind each knob).

### Switch reconstruction (donor / linear / PPM)

Set the comptime field on the `Config` (`koral/config.zig`). Ghost depth follows
automatically via `Config.ghostCells()`:

```zig
.reconstruction = .donor_cell,  // NG=2, 1st order everywhere
.reconstruction = .linear,      // NG=2, minmod-θ limiter
.reconstruction = .ppm,         // NG=3, Colella & Woodward
```

The limiter steepness for linear/PPM is the runtime `minmod_theta` on
`Sim.Options` (default `1.5`, between MinMod=1 and MC=2). The dispatch lives in
`recon.avg2pointScalar` (`koral/recon/recon.zig`). To add a *new* scheme, extend
the `Reconstruction` enum and its `ghostCells()` switch, and add a scalar body in
`avg2pointScalar`.

### Switch the Riemann flux (LAXF / HLL)

Comptime field on `Config`:

```zig
.flux = .laxf,   // Lax-Friedrichs: F* = ½(fl+fr) − ½·ag·(uR−uL)
.flux = .hll,    // Harten-Lax-van Leer two-wave
```

`Sim.fluxesAtFaces` switches on `cfg.flux` into `koral/riemann/laxf.zig`. Hydro and
radiation are treated as **two independent hyperbolic systems** with separate
wavespeeds (the split is `iv >= L.index(.ee)`). To add e.g. HLLC, add a function in
`laxf.zig` with the `(fl, fr, ul, ur, speeds)` shape and a case in
`fluxesAtFaces`.

### Switch coordinates

Comptime field on `Config`:

```zig
.coords = .mink,   // flat Cartesian
.coords = .bl,     // Kerr Boyer-Lindquist
.coords = .ks,     // Kerr-Schild (horizon-penetrating)
.coords = .mks2,   // Modified Kerr-Schild type 2 (PUFFY)
```

The runtime metric parameters (spin `a`, `mksr0`, `mksh0`) are the
`MetricParams` on `Sim.Options.mp`. The metric itself is assembled by
`koral/metric/*` via forward-mode dual-number AD.

### Adjust floors / ceilings

Two layers. The **runtime** `Params` fields (§3) are the run record; the **comptime
presets** are what the solvers use, selected in the driver's `options()`:

```zig
// koral/solve/invert.zig
.floors = koral.solve.invert.FloorParams.cdefault,   // bare choices.h defaults
.floors = koral.solve.invert.FloorParams.puffy,      // PUFFY overrides

// koral/solve/invert_rad.zig
.rad = koral.solve.invert_rad.RadParams.cdefault,    // gammamaxrad=100, ratios 1e-30..1e30
.rad = koral.solve.invert_rad.RadParams.puffy,       // gammamaxrad=10, tuned ratios
```

`FloorParams` carries `rhofloor, uurhoratiomin/max, b2rhoratiomax, b2uuratiomax,
gammamaxhd`; `RadParams` carries `gammamaxrad, eradfloor, eerhoratiomin/max,
eeuuratiomin/max`. To add a new floor, add a field (defaulting to the choices.h
value) and a clamp block in `checkFloorsMhd` / `checkFloorsRad`, keeping the
`ret<0 → sFromU` entropy re-sync at the end. To add a problem-specific set, add a
named `pub const` instance (like `.puffy`) and wire it through `options()`.

### Add a coordinate system

Contained in `koral/metric/`:

1. Add the tag to the `Coords` enum (`koral/config.zig`).
2. Write `gcov<New>(x: [4]Dual3, mp) [4][4]Dual3` in `koral/metric/forms.zig`
   returning only the **covariant** metric — the inverse, determinant,
   Christoffels, and `dlgdet` all derive automatically via `metric.compute`
   (dual-number AD).
3. Add the switch arm in `metric.gcovDual`, and a `gttpert` arm in
   `metric.gttpert`.
4. If it participates in coordinate transforms, add point maps and Jacobians in
   `koral/metric/coco.zig` and wire `cocoN`/`dxdx` (route through KS as the hub, as
   C does).

Nothing in `precompute` or downstream needs changing — `MetricCache` and
`Geometry` are coordinate-agnostic.

### Add an opacity channel

In `koral/physics/opacities.zig`:

1. Add a `bool` to `Channels`.
2. Add a guarded term block inside `calcOpacitiesFromState`, following the
   bremsstrahlung/synchrotron pattern (compute the CGS emissivity/opacity, convert
   with `c.kappacgs2gu * ... * rho`, and sum into the right
   `gas_abs/rad_abs/*_num/*_ross` accumulator).
3. Thread the toggle through `radforce.Params`.

For a whole new absorption law, extend `radforce.KappaMode`; for scattering,
`KappaesMode` + `Params.kappaesAt`. Keep the deliberate two-π split (truncated
`pi_c` for the emission path, exact `M_PI` for synchrotron `bsqcgs`) or you lose
bit-comparability.

### Add or modify a boundary-condition type

Per-axis handling is `bc_x/bc_y/bc_z: BcKind` on `Sim.Options`, where
`BcKind = { periodic, copy, specific }` (`koral/sim.zig`). `.copy` clamps to the
domain edge; `.periodic` wraps (with the `NY<NG ⇒ pin to 0` C quirk); `.specific`
calls your `specific_bc` callback (§7c). To add a brand-new *built-in* kind, extend
the `BcKind` enum and handle it in `setBcCell` (and, if it needs corners,
`fillCorners2d`). Arbitrary/boundary-varying conditions should go through the
`.specific` callback rather than a new enum value.

Note: 3D corner filling is unimplemented — `setBc` `@panic`s if `wide && nz>1`;
only 2D (`ny>1, nz==1`) corner filling exists.

### Enable & tune threading

Set `nthreads` in the params file (or directly on `Sim.Options`). `nthreads > 1`
makes `Sim.init` spawn a **persistent worker team** (`koral/sim/threading.zig`,
P1 of the parallelization plan) that every per-step pass dispatches on —
inversions (`u2p`, implicit), the sweep/flux/wavespeed face passes, flux-CT,
the conserved update, radiative viscosity, the dynamo, boundary fills, fixups,
polar correction, and the stage arithmetic:

```toml
[exec]
nthreads = 8
```

Workers park on a futex'd region counter between passes (no per-pass
spawn/join) and pull contiguous **dynamic tiles** of the banded index range
through an atomic ticket counter, so per-cell cost imbalance (the implicit
solver's ~17× torus/floor spread) straggle-balances naturally. Every band
writes only its own cells/columns/faces and the only cross-band reductions are
order-insensitive (integer sums, f64 max/min for the CFL dt), so the result is
**bit-identical to serial** at any thread count and any tile schedule (the
golden tests run at `nthreads = 1`; `threading_tests.zig` pins 1 ≡ 4 on both a
MINK radiation box and a full PUFFY torus step). Serial remainders — the 2D
ghost-corner fill, the flux-CT region boundary, the dynamo's curl — are the
natural barriers between regions. Spawn failure at init just means fewer
helpers. Any new independent per-cell pass dispatches with
`threading.parallelRange(CtxT, ctx, sim.team, lo, hi, worker)` and a worker
matching `fn(*CtxT, i64, i64, *ChunkResult) void`; pass extra per-region
parameters (like the sub-step `dt`) through a small stack context struct.

### Add a new diagnostic scalar

In `koral/io/scalars.zig` and `koral/io/dump.zig`:

1. Write a pure reduction in `scalars.zig` following the `mdot`/`lum` pattern
   (loop `nzi/nyi/nxi`, `sim.p.load`, `sim.cache.fillGeometry` for MYCOORDS or
   `blGeom` for BL-frame quantities, accumulate). Reuse `hydro.calcTij` /
   `radiation.calcRij` + `relele.indices2221` for tensor fluxes rather than
   re-deriving the stress tensors.
2. Add a field to `dump.ScalarRow`.
3. Extend the `appendScalarLine` format string **and** its arg tuple (they must
   stay in sync — the `bufPrint` format has exactly 12 specifiers today), and add
   the column name to `scalar_header`.
4. Populate it in the driver's per-cadence `ScalarRow` construction
   (`scalarRow` in `PROBLEMS/<name>/main.zig`).

Keep reductions in **GU** — unit conversion is the driver's job at print time.

### Add a new physics module and wire it into the step

This is the large change. In canonical order:

1. **`koral/config.zig`** — append a `Module` enum value in its canonical position
   (order matters — `validate()` enforces strictly-increasing ordinals). Add any
   validation coupling rules.
2. **`koral/layout.zig`** — add the module's `VarTag`s to the `VarTag` enum and its
   ordered tag slice to `moduleTags`. `NV`, `index()`, and `hasVar()` regenerate at
   comptime.
3. **`koral/state.zig`** — (from M3) add the module's `StateFields` to the
   comptime-composed `State(cfg)`.
4. **Physics kernels** — add the module's contributions to the stress-energy
   assembly (`calcTij` for a new fluid term), the flux vector (`fFluxPrime`, guarded
   by `L.hasVar(...)`), the wavespeeds, and the `p2u`/`u2p` conversions. Kernels
   dispatch on `cfg.has(m)` / `L.hasVar(tag)` at comptime, so inactive modules cost
   nothing.
5. **`koral/sim.zig`** — if the module needs an explicit source, add it to
   `opExplicit`'s conserved-update (alongside `metricSource*dt`); an implicit source
   goes in `opImplicit` / `solve/implicit.zig`.

Because everything is comptime-gated on the module set, enabling the module for a
problem is just a `Config` change; the layout and indices follow automatically.

---

## 9. Codebase map

Where the major pieces live (all under `koral/` unless noted):

| Area | Files |
|---|---|
| Comptime config / layout / units / grid / storage | `config.zig`, `layout.zig`, `units.zig`, `grid.zig`, `field.zig`, `params.zig`, `state.zig`, `geometry.zig`, `koral.zig` (namespace root), `comm/serial.zig` |
| Metric (dual-AD, cache, coordinate transforms) | `math/dual.zig`, `metric/forms.zig`, `metric/metric.zig`, `metric/coco.zig`, `metric/precompute.zig` |
| Velocities / boosts / index gymnastics | `relele.zig`, `frames.zig` |
| Primitives ↔ conserved | `p2u.zig`, `solve/invert.zig`, `solve/invert_rad.zig` |
| Fluid & MHD physics | `physics/hydro.zig`, `physics/mhd.zig` |
| M1 radiation | `physics/radiation.zig`, `solve/invert_rad.zig` |
| Microphysics (thermo, opacities, four-force) | `physics/thermo.zig`, `physics/opacities.zig`, `physics/radforce.zig`, `units.zig` |
| Implicit rad-gas source | `solve/implicit.zig` |
| Reconstruction / wavespeeds / flux / Riemann | `recon/recon.zig`, `physics/wavespeeds.zig`, `physics/flux.zig`, `riemann/laxf.zig` |
| Evolution driver | `sim.zig` |
| Constrained transport, dynamo, radiative viscosity | `magn/ct.zig`, `magn/dynamo.zig`, `physics/radvisc.zig` |
| PUFFY problem + quadrature | `problems/puffy.zig`, `PROBLEMS/puffy/main.zig`, `math/quad.zig` |
| Diagnostics / I/O | `io/scalars.zig`, `io/dump.zig` |
| Build / tests / oracle | `build.zig`, `koral.zig` (`test {}`), `tools/gen_golden.sh`, `testing/golden.zig`, `testing/tubes.zig`, `oracle/harness_*.c`, `tests/golden/` |

The library imports almost nothing outward — the foundation (`config`/`layout`/
`grid`/`field`/`units`/`params`) is leaf infrastructure consumed by every physics
and solve module; `sim.zig` is the orchestrator that ties them together. Problems
sit on top and are the only executables.

---

*Related documents:* `ARCHITECTURE.md` (design rationale, storage-layout
insulation, C-diffability contract) and `PHYSICS.md` (GR rad-MHD equations, M1
closure, unit system).
