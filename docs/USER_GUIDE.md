# koral-zig user and developer guide

This guide covers the work around the solver: building it, configuring runs,
reading output, extending a problem, and running the tests. The examples use the
PUFFY radiative-MHD torus in MKS2 coordinates.

Read `ARCHITECTURE.md` for control flow and `PHYSICS.md` for the equations.

---

## Contents

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
  MPI is required for a normal run. `-Dmpi` swaps in `koral/comm/mpi/` (φ-only
  ring; see "Running under MPI" below). With `-Dsilo` the build also
  fetches the lazy sibling `silo-zig` dependency (§2, §4).

### To regenerate the C-comparison golden data

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
- It scans `koral/problems/` and builds **one executable per subdirectory** from
  `koral/problems/<name>/main.zig`, importing `koral`.
- It wires a single **`test`** step (`zig build test`) that runs *all* library
  tests as one artifact, and enforces at configure time that every
  `koral/tests/*_tests.zig` and `koral/tests/golden/*_golden_tests.zig` is
  registered in `koral/koral.zig`'s `test {}` block (missing registration fails
  the build with `error.UnregisteredTestFile`).
- It registers conversion and analysis tools: `res2kdmp`, `kdmp2silo`, `qmri`,
  `kdmp2png`, `kdmp2lc`, and `goldtest`. `bench-implicit` measures the scalar and
  SIMD implicit Jacobians. MPI builds also add the `mpi-gates` harness.

### Commands

```sh
# Build everything (library check + every problem executable, installed to zig-out/bin)
zig build

# Build just one problem
zig build puffy

# Build and run a problem, forwarding args after `--` to the executable
zig build run-puffy -- koral/problems/puffy/puffy.toml

# Optimized build (recommended for real runs)
zig build run-puffy -Doptimize=ReleaseFast -- koral/problems/puffy/puffy.toml
```

The problem name is the directory name under `koral/problems/`. For each `<name>` you
get a `<name>` step (build & install) and a `run-<name>` step (build & run). The
`puffy` CLI is:

```
puffy [params.toml] [--restart <file|dir>]
```

The optional positional is the params-file path (default
`koral/problems/puffy/puffy.toml`). `--restart`, `-r`, and `--restart=<x>`
resume from a KDMP checkpoint. Given a directory, the driver picks the latest
`prims#####.kdmp` inside. Any second positional is `error.BadArgs`. See `main`
in `koral/problems/puffy/main.zig`. Besides `puffy.toml`, five presets ship in
the same directory: `puffy3d.toml` (3D wedge), `puffy3d_sgra.toml` (Sgr A*
mass), `puffy3d_sgra_spin.toml` (Sgr A* with a = 0.9375), and `puffy_agn.toml`
(10⁹ M☉ AGN with MESA opacities). `puffy_debora.toml` mirrors the broader
`koral_lite_puffy` fork configuration. Read `docs/PUFFY_AGN_DIVERGENCES.md`
before using either AGN-oriented preset.

### Build options

| Flag | Meaning |
|---|---|
| `-Doptimize=Debug\|ReleaseSafe\|ReleaseFast\|ReleaseSmall` | Standard Zig optimize mode. Debug builds enable the many `std.debug.assert` bounds checks in `Field.cellOffset` etc. |
| `-Dtarget=...` | Standard Zig cross-compile target. |
| `-Dmpi=true` | Link the system MPI and swap the comm seam to the real backend (`koral/comm/mpi/`). `build.zig` probes `mpicc` for the lib dirs and detects the ABI family (MPICH vs Open MPI) from `mpi.h`; overrides: `-Dmpi-family=mpich\|ompi`, `-Dmpi-lib=<dir>`, `-Dmpi-include=<dir>`. See "Running under MPI" below. |
| `-Dsilo=true` | Build the optional `.silo` field export (`koral/io/silo.zig`, §4). The build compiles LLNL Silo 4.12 from source with the PDB driver and no HDF5, then links it statically through `silo-zig`. A system Silo or VisIt installation is not required. The dependency remains untouched without this flag. |
| `-Dslow-tests=true` | Enables the slow test bodies (convergence studies, soaks, the full-grid PUFFY t=0 keystone, the t = 10 repeat of the KS Bondi gates). See §5. |
| `-Dtest-filter=<substr>` | Passed to `addTest` `.filters`; restrict the single test artifact to tests whose name contains the substring. Repeatable. |

### Running under MPI (P4a+P4b, 3D wedges, node-to-node only)

The MPI layer decomposes only φ. `nx`, `ny`, and `nz`
in the params file stay global; each rank evolves one contiguous
z-slab of `nz / <ranks>` φ-cells, and 2D runs (`nz = 1`) never decompose.
`nz` must divide by the rank count with `nz/ranks ≥ 3` (the ghost depth).
`ntz = 0` (the default) takes the launched world size; a nonzero `ntz` must
match it. MPI is meant to be inter-node, one rank per node, `nthreads` =
cores per node.

```sh
# macOS dev loop: brew install open-mpi, then
zig build puffy -Dmpi -Doptimize=ReleaseFast
mpiexec -n 4 zig-out/bin/puffy koral/problems/puffy/puffy3d.toml

# The validation ladder (gates 2-6; gate 1 is in `zig build test`)
zig build mpi-gates -Dmpi -Doptimize=ReleaseSafe
for n in 1 2 3 4; do mpiexec -n $n zig-out/bin/mpi-gates || break; done
```

I/O under MPI (P4b, plan §8):

* **`scalars.dat`.** Every column is a ring-global value (two collectives
  per output row: one SUM carrying the diagnostic counters, one MAX); rank 0
  writes the file. Serially the folds are identity and the rows are bitwise
  what they always were.
* **KDMP checkpoints.** `dump.writePrim` and `dump.loadPrim` live in `koral/io/dump.zig`.
  Written collectively under MPI (`MPI_File_write_at_all`; each rank's φ-slab
  is one contiguous byte range of the file). The file is
  **byte-identical** to a serial run's checkpoint (gate 6 pins this), so
  `--restart` works at ANY rank count in both directions: a serial/1-rank
  checkpoint restarts at N ranks and vice versa, and the plain serial binary
  reads MPI-written files. A checkpoint interrupted mid-write (walltime kill)
  is rejected rather than loaded. The 44-byte header is written last, after a
  sync, so it doubles as a completion marker. An incomplete file
  fails with `BadMagic`, a short one with `Truncated`, on every rank. This
  matters because `--restart <dir>` picks the *newest* checkpoint, which is
  exactly the one a kill leaves partial.
* **Silo.** Silo output is serial-only at write time. Convert MPI-run checkpoints
  with `zig build kdmp2silo -Dsilo -Doptimize=ReleaseFast -- <params.toml>
  <prims#####.kdmp | dumps-dir> [out-dir]` (`puffy.setup` from the same params
  file as the run, so the mesh and derived fields are reconstructed identically). Native PMPIO
  dumps land in P4c.

The rank-0 heartbeat gains an `mpi=…%` field (share of stepping wall-clock in
halo exchange + collectives since the last output row), and the per-pass
timer table has `halo` and `collect` rows. `pin_threads = true` (Linux only)
binds each team thread to one CPU in the rank's affinity mask, including the
cgroup cpuset under Slurm. It prints a per-rank confirmation. Use it on clusters;
skip it on laptops.

### What the `puffy` executable prints

On startup it prints one line with NV, grid and ghost
depth, mass and the geometrized-unit time scale (`GM/c³` in seconds via
`Units.gmc3()`), spin, horizon radius, RMIN (with the count of cells inside the
horizon), RMAX, thread count, and build mode:

```
puffy: NV=13 grid 384×360×1 (+3 ghosts) | M=10 M☉ (GM/c³=4.9…e-05s) a=0 r_h=2 RMIN=1.85 (~… cells inside) RMAX=500 | threads=1 | build=ReleaseFast
```

Warnings follow when RMIN lies outside the horizon, spin is nonzero, or a Debug
build is used. When configured, the driver also reports the MESA table path. A
fresh run prints the beta-normalization factor; a restarted run reports its
checkpoint, time, step, and next frame number.

While stepping it interleaves three kinds of console output: a ~1 Hz C-style
heartbeat (`st #… t=… dt=… znps=… tgpd=… fail# … imp# …`, zone-steps/s,
target-days-per-day, fixup/implicit counters), one status line per output
cadence:

```
puffy: t=… nstep=… dt=… | Ṁ=… L=… H/R=… β⁻¹=… | nan=… hdfix=… radimpfail=…
```

and a per-pass wall-clock timer table (`sim/timers.zig`) at the same cadence.

If any primitive becomes non-finite, the run exits with `error.NanDetected`.
All quantities are in **geometrized/code
units** (GU); unit conversion to
CGS/Eddington is left to downstream analysis, matching KORAL's convention of
keeping `scalars[]` in GU.

---

## 3. The params (TOML) file

Runtime parameters live in a **flat TOML-subset** file parsed by
`koral/params.zig`. The parser is line-based: `key = value`, `#` comments (honored
outside quoted strings), blank lines and `[section]` headers ignored (sections are
purely cosmetic grouping). Keys must match `Params` field names exactly. An
unknown key returns `error.UnknownParamsKey` (typo protection, since the
params file is the run record). Values are typed by field: `f64` via `parseFloat`,
`usize` via `parseInt` base 10, `bool` as `true`/`false`, strings de-quoted and
duplicated.

These are *runtime* numbers a physicist tweaks between runs; anything that changes
the generated code (module set, reconstruction, coordinates) is a **comptime**
`Config` and lives in code, not here (§7).

Many fields are **optional** (`?f64`/`?bool`/`?usize`/string, default null/empty):
an absent key means "keep the compiled preset", a present key always overrides it.
`puffy.Physics.fromParams` copies every set override onto a **new** `Physics`
value, starting from `puffy.defaults`, before building the `Sim`. Params-file
values therefore affect the run rather than merely documenting it.
Tests and goldens never call `fromParams`, so they stay pinned to the validated
literals. `puffy.setup` / `puffy.load` wrap that plus the optional MESA table
and `Sim.Options` construction; the driver and the replay tools share them.

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
| `rmin`, `rmax` | `0.0`, `0.0` | Radial extent. `0` means *auto*: PUFFY derives `rmin = rminForSpin(bhspin)` (0.925·r_h, = 1.85 at a=0) and keeps the fiducial `rmax = 500`. A positive value is an explicit override. PUFFY derives `MINX = log(rmin − mksr0)` for MKS2. |
| `mksr0` | *null* | MKS2 radial offset `R0` in `r = R0 + e^{x1}` (`MKSR0`); overrides the problem's `mp.mksr0` when set (PUFFY default 0.1; the AGN preset uses −1.5). |
| `mksh0` | *null* | MKS2 polar squeeze `H0` (`MKSH0`); overrides `mp.mksh0` when set (PUFFY default 0.9). |
| `miny`, `maxy` | `0.0`, `1.0` | θ-direction internal bounds (PUFFY ignores them, problem constants). |
| `minz`, `maxz` | `0.0`, `1.0` | φ-direction internal bounds. PUFFY ignores them and uses the fixed `PHIWEDGE = π/2`. |

**Run control**

| Field | Default | Meaning |
|---|---|---|
| `tstart` | `0.0` | Start time. |
| `tmax` | `0.0` | Stop time (code units). |
| `nstep_max` | `maxInt(usize)` | Max number of steps. |
| `tsteplim` | `0.5` | CFL factor (`TSTEPLIM`). |
| `dtout1` | `0.0` | Output cadence (code time): scalars + KDMP checkpoint + silo, all on the same frame. |
| `dtout2` | `0.0` | **Reserved / unread** (C's avg-file cadence). KDMP dumps are written on *every* output frame, not gated by this. |
| `nout_step` | `0` | Output every N steps regardless of code time (`0` = disabled). This gives a predictable file count even when the CFL timestep changes. |
| `out_dir` | `"dumps"` | Output directory. |

**Floors & ceilings** (all `?f64 = null`, unset keeps the problem's compiled
preset, e.g. `FloorParams.puffy`; a set value overrides it)

| Field | PUFFY preset value | Meaning |
|---|---|---|
| `rhofloor` | `1.0e-30` | Minimum density. |
| `uurhoratiomin` / `uurhoratiomax` | `1.0e-8` / `1.0` | Bounds on `uint/rho`. |
| `eerhoratiomin` / `eerhoratiomax` | `1.0e-20` / `1.0e4` | Bounds on radiation energy / rho. |
| `eeuuratiomin` / `eeuuratiomax` | `1.0e-20` / `1.0e20` | Bounds on radiation energy / uint. |
| `b2rhoratiomax` | `50.0` | `b²/rho` ceiling. |
| `b2uuratiomax` | `50.0` | `b²/uint` ceiling. |
| `gammamaxhd` / `gammamaxrad` | `10.0` / `10.0` | Max gas / radiation Lorentz factor. |
| `zamo_floor_frame` | off | `true` = inject the `b²/ρ` floor mass in the ZAMO frame, isentropically (C `B2RHOFLOORFRAME`; disables the `b²/u` floor). Default drift frame. |
| `isentropic_b2rhofloors` | off | Drift path scales `uint` by the same factor as `rho`, injected mass carries the pre-floor `u/ρ` (koral_lite_puffy `ISENTROPIC_B2RHOFLOORS`; AGN: on). |
| `b2uufloor` | on | `false` disables the `b²/uint` ceiling entirely (koral_lite_puffy comments the trigger out; AGN: off). |
| `fluid_floor_inside_horizon` | off | Below `r_horizon` the drift-frame velocity algebra is skipped and floors act in the fluid frame, velocity untouched (koral_lite_puffy 2026-08-11; AGN: on). |

**Optional physics overrides** (same null-means-keep-preset semantics; consumed
by `puffy.Physics.fromParams`, these are how `puffy_agn.toml` retargets the
run without a recompile; see `docs/PUFFY_AGN_DIVERGENCES.md`)

| Field | Meaning (C name) |
|---|---|
| `radimpeps`, `radimpmaxiter` | Implicit-solver convergence tolerance / iteration cap (`RADIMP*`). |
| `opdamp_maxlevels`, `opdamp_factor` | Opacity-damping retry ladder for failed implicit solves (`OPDAMPINIMPLICIT`; AGN: 3 levels ×10). |
| `doradimpfixups` | Neighbour-average failed-implicit cells (`DORADIMPFIXUPS`; AGN: on). |
| `reduceorderatbh` | Drop one reconstruction order inside the BH horizon (`REDUCEORDERATBH`). |
| `reduceorderafterfixup` | Cells whose most recent u2p/implicit pass demanded a fixup reconstruct one order lower next sweep (koral_lite_puffy `REDUCEORDERAFTERFIXUP`; AGN: on). |
| `radimp_lag_opac` | Freeze opacities at the pre-solve state across the whole implicit Newton solve (koral_lite_puffy `copy_state_opac`; needed for non-monotonic `mesa_table` κ; AGN: on). |
| `scale_jacobian` | Row/column-scale the implicit Jacobian by the iterated energy (`SCALE_JACOBIAN`; validated: on, koral_lite_puffy 2026-08-11: off). |
| `radimp_max_en_change_down` / `radimp_max_en_change_up` | Per-trial-step limits on how far the iterated energy may drop/rise (`RADIMPLICITMAXENCHANGEDOWN/UP`). |
| `radimp_max_damping` | Smallest Newton damping factor before a rung gives up (`MAXRADIMPDAMPING`). |
| `alpharadvisc`, `maxradviscvel` | Radiative-viscosity ν = α·mfp coefficient and velocity-damping threshold (`ALPHARADVISC`/`MAXRADVISCVEL`; koral_lite_puffy: 0.1/0.3). |
| `expectedhr` | Dynamo's assumed disk H/R before `CALCHRONTHEGO` has a measurement (`EXPECTEDHR`; koral_lite_puffy: 0.7). |
| `dampradwavespeednearaxis` | Within N cells of each pole, keep the radiative wavespeed at the undamped 1/3 (`DAMPRADWAVESPEEDNEARAXIS`; AGN: 2). |
| `bremsstrahlung`, `kleinnishina` | Opacity-channel toggles. |
| `synchrotron_bridge` | Replace the Terelfactor NR suppression with the Ramesh NR bridge (`USE_SYNCHROTRON_BRIDGE_FUNCTIONS`). |
| `scattering` | `false` disables electron scattering *and* (∝ κ_es) Comptonization, matching C's AGN problem. |
| `mesa_table` | Path to a MESA Rosseland opacity table (`""` = off); replaces the free-free Rosseland channels with table lookups (`MESA_KAPPA`). Must match the composition (`hfrac` → X). |
| `hfrac`, `hefrac` | Gas composition; setting `hfrac` switches μ's to the full composition formulas. |
| `lt_kappa`, `maxbeta` | Torus entropy constant (PUFFY 60; AGN 8e-2) and β-normalization target (PUFFY 1/20; AGN 1/30). |
| `perturb` | Fractional internal-energy perturbation applied before the gas/radiation pressure split. The coordinate-hashed noise is deterministic across thread and MPI decompositions. Null or zero preserves the golden initial state. |
| `rhoatmmin`, `atm_tgas`, `atm_trad_init`, `atm_erad_factor` | Atmosphere density/temperature/radiation constants. |

**Execution**

| Field | Default | Meaning |
|---|---|---|
| `deterministic` | `false` | Reserved determinism flag (currently unused). |
| `nthreads` | `1` | Persistent worker team for all per-step passes; `1` is the bit-identical serial path (§8). |
| `ntz` | `0` | MPI φ-ring size (`-Dmpi` builds, 3D only; §2 "Running under MPI"). `0` = auto (the launched rank count); nonzero must match it. `nx/ny/nz` stay global; each rank owns `nz/ntz` φ-cells. |
| `pin_threads` | `false` | Bind each team thread to one cpu of the process affinity mask (the cgroup cpuset under Slurm, MPI plan P4b). Linux only, inert elsewhere; no effect on any FP result. Turn on for cluster runs. |

### PUFFY example (`koral/problems/puffy/puffy.toml`)

```toml
# PUFFY, radiative-MHD limotorus, 10 M☉ Schwarzschild, MKS2

[physical]
mass = 10.0        # MASS (solar masses)
bhspin = 0.0       # BHSPIN, Schwarzschild
gam = 1.6666666666666667   # GAMMA 5/3

[grid]
nx = 384           # TNX (radial)
ny = 360           # TNY (theta)
nz = 1             # TNZ, 1 = 2D axisymmetric; nz>1 subdivides the PHIWEDGE=π/2 wedge (3D)
rmin = 1.85        # RMIN override (= rminForSpin(0); < r_horizon=2, clean excision)
rmax = 500.0       # RMAX
mksr0 = 0.1        # MKSR0
mksh0 = 0.9        # MKSH0
miny = 0.001       # MINY  (PUFFY problem constant, not read)
maxy = 0.999       # MAXY  (PUFFY problem constant, not read)
minz = -0.7853981633974483   # -PHIWEDGE/2 (problem constant, not read)
maxz = 0.7853981633974483    # +PHIWEDGE/2 (problem constant, not read)

[run]
tstart = 0.0
tmax = 10000.0     # run duration in GM/c^3 (reduce for a quick test)
nstep_max = 100000000
tsteplim = 0.5     # TSTEPLIM (CFL factor)
dtout1 = 100.0     # output cadence (code time): scalars + KDMP checkpoint + silo
dtout2 = 500.0     # reserved (C's avg-file cadence), not read
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

(The checked-in `puffy.toml` may differ in run-control values, e.g. a small
`nstep_max` or `nthreads > 1` for local iteration; the physics values above are
the validated reference.)

Section headers such as `[physical]` and `[grid]` are cosmetic. The parser uses
field names, so all fields could sit in one section. The production PUFFY grid
is `TNX=384, TNY=360, TNZ=1, NG=3`; `nx`/`ny`/`nz` can be changed freely
(`makeGridNz` builds the grid), and `rmin`/`rmax`/`mksr0`/`mksh0` are honored as
overrides. Only the θ bounds and φ wedge are fixed problem constants. Note
the driver runs while `t < tmax` **and** `nstep < nstep_max`, so `tmax = 0.0`
would take zero steps.

---

## 4. Output formats

### `scalars.dat`, diagnostic time series

Written by `koral/io/dump.zig` (`appendScalarLine` / `scalar_header`). A
whitespace-separated text file, one row per output cadence, in **GU/code units**.
The header line is:

```
# t dt nstep mass mdot radlum totallum H/R maxPmag/Ptot n_hdfix n_radimpfail n_nan n_floorguard
```

The 13 columns of a `ScalarRow` (`koral/io/dump.zig`), and where each is computed
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
| `maxPmag/Ptot` | Max magnetization β⁻¹ = pmag/ptot over the domain (`pmag=b²/2`, `ptot=(Γ−1)u [+ Ê/3]`). | `maxPmagPtot` |
| `n_hdfix` | Count of cells that needed an HD inversion fixup. | `collectDiag` |
| `n_radimpfail` | Count of implicit radiative-source failures. | `Sim.n_radimp_failures` |
| `n_nan` | Count of cells with a non-finite primitive. | `collectDiag` |
| `n_floorguard` | Cumulative drift-floor guard recoveries (koral_lite_puffy guard ladder; 0 on a healthy run). | `invert.n_guard_recoveries` |

Print precision (from `appendScalarLine`): `t/mass/mdot/radlum/totallum` at
`{e:.10}`, `dt/H/R/β⁻¹` at `{e:.6}`, counters/`nstep` as integers.

Diagnostic radii used by the PUFFY driver: `r_horizon = rHorizonBL(a)` (2.0 at
a = 0), `r_lum = 5000.0` (> RMAX ⇒ the outermost shell), `r_scale = 15.0`.

> **Wedge quirk (faithful to C):** `totalMass` uses the *raw* cell dz (the φ-wedge
> width) and does **not** expand to 2π for a `TNZ==1` slice, whereas `mdot`/`lum`
> **do** hardcode `dφ = 2π`. So the PUFFY 2D `mass` is a wedge mass, not a full
> torus mass. This inconsistency is transcribed from `calc_totalmass`.

### `prims#####.kdmp`, binary primitive snapshot / restart checkpoint

Serialized by `koral/io/dump.zig`. `serializePrimDump` / `loadPrimDump` are the
pure in-memory path; `writePrim` / `loadPrim` / `resolveRestartPath` own the
file protocol for both serial and MPI (body first, 44-byte header last as a
completion marker so a killed write fails `parseDumpHeader` instead of looking
like a valid full-length file of zeros). The driver and `tools/mpi_gates.zig`
call those functions, they do not reimplement the write. Written on **every
output frame** (the `dtout1`/`nout_step` cadence, `dtout2` does not gate it),
and doubles as the `--restart` checkpoint. Little-endian throughout. Layout
(44-byte header):

```
"KDMP"              4 bytes magic
u32 version = 2
u32 nx, ny, nz, nv  (interior cell counts and NV)
f64 t
u64 nstep
u32 out_idx         (output frame counter, what makes a dump a restart point)
f64 p[nz][ny][nx][nv]   iv fastest, then ix, then iy, then iz (AoS, matches C get_u)
```

`primDumpSize` returns exactly the serialized byte count: `grid.nx*ny*nz` are
the *interior* (active) cell counts (the padded storage is `sx()=nx+2*ngx`).
The `iv`-fastest AoS order matches KORAL's `get_u`/`set_u` and the KSTP/KINI
golden byte order. A C-side restart (`res####.head/.dat`) can be converted to
KDMP with the `res2kdmp` tool (§2).

### `.silo`, VisIt field dumps *(optional, `-Dsilo`)*

Written by `koral/io/silo.zig` when the executable is built with `-Dsilo`
(otherwise `silo.write` is a comptime no-op and nothing Silo-related is compiled
or linked). One file per output frame at `{out_dir}/silo/puffyNNNN.silo`, in
Silo's `DB_PDB` format, openable directly in VisIt.

Each file holds a single non-collinear quad mesh `mesh1` (cell-boundary nodes
transformed to Cartesian via the BL spherical coordinates; a 2D `nz==1` run is
laid out in the meridional x-z plane, mirroring `silo.c`'s `SILO2D_XZPLANE`)
plus zone-centered fields: `rho`, `uint`, `entr`, `temp`, `gammagas`, the
four-velocity `u0..u3` / `lorentz`, the MHD set (`bsq`, `B1..B3`, `beta`,
`betainv`, `sigma`, `divB`), the radiation set (`ehat`, `erad`, `trad`), the
opacity channels (`kappa_*`, `tot_emissivity`), the per-cell fixup flags, and
the `velocity` / `magn_field` / `rad_flux` vectors. Field names and centering
mirror KORAL's `silo.c` so existing VisIt sessions and expressions carry over.
The frame's time/cycle are written as Silo `DTIME`/`TIME`/`CYCLE` metadata for
VisIt's time slider.

**No external dependency.** Silo is not linked from a system install or from
VisIt, it is compiled from source (LLNL Silo 4.12, PDB driver only, no HDF5)
and linked statically through the sibling [`silo-zig`](../../silo-zig) wrapper
package (`@import("silo")`, `silo.Writer`). The Silo source is a hash-pinned
`build.zig.zon` fetch, and the dependency is *lazy*, so a build without `-Dsilo`
never fetches or compiles it. This means `.silo` output works on machines with
no VisIt, e.g. HPC compute nodes, provided you build there (the build compiles
and runs Silo's `detect.c` to describe the host's native binary layout, which is
correct for a native build; a true cross-compile would need a target-generated
`pdform.h`). See `silo-zig/README.md` for the wrapper internals.

Inspect a `.silo` file without VisIt using the wrapper's bundled reader:

```sh
cd ../silo-zig && zig build readsilo -- /path/to/dumps/puffy/silo/puffy0000.silo
```

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

### Sadowski et al. (2013) reproduction gates

Two test files reproduce published, quantitative results using the algorithms
that exist in this port:

```sh
# Figure 1 at 500 cells and t=50, for linear theta=1 and theta=2
zig build test -Dtest-filter="Sadowski 2013 section 4.1"

# Figure 9 at its first published time, with the paper's grid and opacity
zig build test -Dtest-filter="thick pulse has the early Figure 9"

# All four Figure 9 times through t=100000
zig build test -Dslow-tests=true -Dtest-filter="full four-time Figure 9"
```

`paper2013_hydro_tests.zig` compares the paper's Table 1 shock directly with a
test-local exact relativistic Riemann solution. It checks both profile errors,
the shock position, and the reported result that theta=1 is more diffusive than
theta=2. MP5 is not available in this port and is not approximated by another
reconstruction method.

`paper2013_pulse_tests.zig` uses the Section 4.4 domain, 101 cells, temperature
profile, radiative constant, scattering optical depth, and output times. The
analytic diffusion curve is evaluated directly by expanding the fourth power
of the temperature profile into Gaussian terms. A mathematically exact
pure-scattering scaling multiplies rho, gas energy, and radiation energy by the
same factor while dividing the opacity per mass by that factor; this preserves
temperature, E/rho, chi, D, and the dimensionless solution while keeping the
implicit solve away from absolute tolerances at E of order 1e-40.

The Section 4.3 radiative shock tubes are already covered by
`radtube_tests.zig`: tubes 3a and 4a run normally, with tubes 1, 2, 4b and
higher-resolution checks under `-Dslow-tests=true`. Those runs validate the
published plateau states and the stationary radiation equations, but use the
current M1/PPM/RK2IMEX solver. The paper used Eddington closure for those plots,
so they are physics checks rather than pixel or raw-data reproductions.

The tests do not claim direct reproduction of results that require unavailable
features or inputs: MP5, the Eddington-closure comparison, the removed original
problem directories, or numerical data behind the published figures.

### Reading the results

`zig build test` compiles *one* test artifact from `koral/koral.zig`'s `test {}`
block, which `_ =` imports every unit module and every dedicated test file. If a
test fails, the Zig build runner prints a noisy `error: the following command
failed` / `run test` block around it. That framing is boilerplate. Read:

- The individual `FAIL`/`error.<Name>` lines with file:line and the printed
  deviation (golden tests print the observed max deviation via `DevTracker.check`,
  so you see *how far off* rather than dying on the first element).
- The process **exit code** (`0` = all passed/skipped).

**Skips are expected and not failures.** The golden readers
(`readGolden`/`readKstp`/`readKini`) return `error.SkipZigTest` when a golden file
is absent, so the suite is green on a machine that never ran `gen_golden.sh`. A
deleted golden shows as a skip, not a pass. Check for this when auditing
coverage. The **self**-golden reader deliberately does not do this: it fails with
`error.SelfGoldenMissing`, because those baselines need no C toolchain and are
regenerable from this repository alone, so an absent one means it was never
committed rather than never generated.

### Test-file inventory

Four complementary layers (there are deliberately **no** run-to-completion
end-to-end tests). Test files live apart from the code they gate, one folder per
family, with one naming rule each so a subsystem's files sort adjacently: theory
gates are `koral/tests/<subsystem>_tests.zig`, C-oracle goldens are
`koral/tests/golden/<subsystem>_golden_tests.zig` (the `.kgld`/`.kstp`/`.kini.gz`
data they read stays in the repo-root `tests/golden/`), and the Zig-generated
self-golden baseline is `koral/tests/selfgolden_tests.zig` over `.kslf` files in
`tests/selfgolden/`. The separate tree records provenance. `tests/golden` answers
"does this match KORAL C"; `tests/selfgolden` answers "did our own numbers move".
New test files must be
listed in `koral/koral.zig`'s `test` block, and `build.zig` enforces this at
configure time: any test file missing its `@import` fails the build with
`error.UnregisteredTestFile`, so a forgotten registration can't silently drop
coverage.

**1. Theory gates.** Mathematical identities and documented C quirks need no golden
data:

- `metric_tests.zig`, `evolution_tests.zig`, `mhd_evolution_tests.zig`,
  `radiation_tests.zig`, `opacity_tests.zig`, `implicit_tests.zig`,
  `state_tests.zig`, `flux_tests.zig`, `polaraxis_tests.zig`,
  `radstep_tests.zig`, `radtube_tests.zig`, `paper2013_hydro_tests.zig`,
  `paper2013_pulse_tests.zig`, `dynamo_tests.zig`,
  `radvisc_tests.zig`, `scalars_tests.zig`, `threading_tests.zig`,
  `puffy_tests.zig`, `sim_tests.zig` (`Sim.init` validation and initializer invariants),
  `restart_tests.zig` (KDMP round-trip), `simd_tests.zig` (scalar ↔ SIMD
  bit-identity), `ks_evolution_tests.zig` (2D Bondi/Michel through the polar axis
  in KS, and the Gammie et al. 2003 magnetized Bondi monopole: exact discrete
  monopole, divB at machine zero, stationarity converging), `timestep_tests.zig`
  (`Core.cflDt` against the CFL bound with the SR hydro eigenvalues), plus
  in-module `test` blocks in nearly every source file.

**2. Function-level C goldens.** `.kgld` records compare Zig and C at recorded input
points, with C's *own* geometry embedded per record (`geomFromRecord`) so only the
state algebra is compared:

- `metric_golden_tests.zig`, `state_golden_tests.zig`, `flux_golden_tests.zig`,
  `rad_golden_tests.zig`, `opac_golden_tests.zig`, `implicit_golden_tests.zig`.

**3. Forced-dt step and keystone goldens.** `.kstp` and `.kini.gz` records load C's post-init
state bit-for-bit, force C's recorded dt sequence, diff the whole domain each step:

- `step_golden_tests.zig` (the ZIG* tube step goldens),
  `puffystep_golden_tests.zig` (reduced-grid full PUFFY pipeline),
  `puffy_golden_tests.zig` (PUFFY t=0 keystone, full-grid only under
  `-Dslow-tests`), `visc_golden_tests.zig`, `dynamo_golden_tests.zig`.

Support files: `koral/testing/golden.zig` (`Golden`/`Kstp`/`Kini`/`DevTracker`
readers) and `koral/testing/tubes.zig` (the Farris/Sadowski radiative shock-tube
battery + the paper→KORAL unit bridge).

> **Expected non-zero floors.** Some goldens gate at `1e-8` because C's MKS2
> metric exports mix truncated `Pi` with exact π. The
> the PUFFY torus keystone at `~1e-3` (C's `gsl_qags` is only ~1e-3 accurate at
> the ℓ(λ) C¹ kink; Zig's GK21 is the *more* accurate side, proven by the
> `puffyeps12` variant). These bounds are deliberate; do not tighten them.

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
  `harness_visc.c` / `harness_dynamo.c`, function-level `.kgld` goldens for the
  metric, state algebra, flux vector, M1 radiation, opacities, implicit solver,
  scalar reductions, radiative viscosity, and dynamo.
- `harness_step.c`, the explicit/MHD/radiative **forced-dt step** oracle. Built 5×
  (PROBLEM 200-204 = ZIGSOD, ZIGOT, ZIGMHDTUBE, ZIGRADTUBE, ZIGRADPULSE). It
  replicates ko.c init and runs the RK2IMEX stage block verbatim from
  `problem.c:141-402`, dumping the whole domain each step as `.kstp`. PROBLEM 201
  also emits `ct.kgld` (flux-CT on PRNG-filled EMFs) and `bfroma.kgld`
  (`calc_BfromA` on PRNG vector potential), using a fixed-seed xorshift64* PRNG the
  Zig side replays.
- `harness_puffy_step.c`, the **full-PUFFY-pipeline** step oracle on a reduced
  64×60 grid (implicit rad source + radiative shear viscosity + dynamo), 4 steps as
  `.kstp` (gzipped).
- `harness_init.c`, the PUFFY **t=0 keystone**. Emits three `.kini` snapshots over
  the full 384×360 grid + ghosts: `puffy_t0_A` (A_φ still in the B slots),
  `puffy_t0_p` (all NV primitives after `calc_BfromA`, *the* keystone), and
  `puffy_t0_pfinal` (β-normalized B, ghosts deliberately stale). The `puffyeps12`
  variant tightens C's `qags` epsrel from 1e-8 to 1e-12 to attribute the torus
  keystone deviation to C's quadrature.

Large snapshots are `gzip -9`'d in the repo; the readers auto-inflate. A
`manifest.json` records the koral_lite SHA, PROBLEM list, compiler, and per-file
schema. After regenerating, commit the updated `tests/golden/**` and
`manifest.json`.

### Regenerating the self-goldens

```sh
zig build update-self-goldens    # rewrites tests/selfgolden/**, no C toolchain needed
```

The baseline comes from this repository, so regeneration needs no external
oracle. Run it only after reading what the failing test says moved and deciding
that the change is intended. The committed `.kslf`
files are the record of what this code used to compute; rewriting them without
looking silently destroys the only evidence that a refactor changed behaviour.

Regenerate with the same `-Doptimize` you test with (i.e. plain defaults): the
generator and the test share the `koral` module, which is what lets the comparison
gate on bit identity rather than a tolerance. A `DRIFT` verdict (bits moved, but by
less than 1e-12) means either the toolchain changed or the code changed by a very
small amount. The magnitude cannot distinguish those causes, so if you did not change
compilers, treat it as a code change.

Scenarios are defined in `koral/testing/selfscenarios.zig` and shared by the
generator and the checker; edit them there and regenerate, never one side alone.

---

## 7. Recipe: creating a new problem

A **problem** is: a comptime `Config`, a runtime `Params` file, an
initial-condition + boundary-condition module, and a one-line driver executable
that hands an `App` contract to `koral.driver.run`. PUFFY keeps its C `define.h`
knobs in a value type (`puffy.Physics`) so a preset is `fromParams`, not
process-wide mutation; a simple tube can skip that and build `Sim.Options`
directly (`Options.applyParams` maps the generic TOML keys). The library
(`koral`) provides everything else, including the pieces torus problems share
in `koral/problems/common/` (limotorus solver, floor atmospheres, β
normalization, the stock `bc.c` fragments). This section walks through a complete minimal example, a **relativistic
MHD shock tube in flat (Minkowski) space**, and then notes the extra pieces
PUFFY needs.

Directory layout for a new problem `mytube` (everything problem-specific lives
in one directory, like `koral/problems/puffy/`):

```
koral/config.zig                       # (optional) add a comptime Config const
koral/problems/mytube/mytube.zig       # init conditions + boundary conditions
koral/problems/mytube/main.zig         # the App contract + koral.driver.run (auto-discovered)
koral/problems/mytube/mytube.toml      # runtime params
koral/koral.zig                        # register mytube under `problems` + tests
```

### (a) Choose or define a comptime `Config`

A `Config` (`koral/config.zig`) selects the physics modules and algorithms.
Modules **must** be listed in canonical order, `hydro, electrons, relel, mhd,
forcefree, radiation`, or `validate()` `@compileError`s. `.hydro` is mandatory.

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
extend two things in lockstep (see §8, "Add a physics module"): the `VarTag`
enum (`koral/layout.zig`) and the owning module's tag list in `moduleTags`. `NV`
and all indices then regenerate automatically at comptime. For a standard
hydro/MHD/radiation tube you need none of this; the tags already exist.

### (c) Write `koral/problems/<name>/<name>.zig`, init + boundary conditions

Two responsibilities: build each domain cell's primitive vector, and supply the
`SpecificBc` callback for any axis using `.specific` boundaries.

**Boundary-condition callback signature** (`Sim(cfg).SpecificBc`, defined in
`koral/sim/options.zig`):

```zig
pub const SpecificBc = *const fn (
    ctx: ?*const anyopaque,      // the opaque context you register with the face
    core: *const Sim.CoreT,      // grid, metric cache, p, physics options — not the whole Sim
    ix: i64, iy: i64, iz: i64,   // ghost-cell indices (signed; ghosts are <0 or ≥n)
    t: f64,
    ifinit: bool,                // true during initial set_bc
    face: BcFace,                // .xlo/.xhi/.ylo/.yhi/.zlo/.zhi
) relele.Error![NV]f64;          // ghost-cell primitives, or a conversion error
```

A face is registered as `.bc = .{ .x = .{ .specific = .{ .f = &calc, .ctx = ... } } }`;
the callback travels with the face, so a `.specific` axis without a callback cannot
be expressed.

**Spherical grids.** Never put a grid face on the axis itself: `g^φφ = 1/sin²θ` diverges
there and the first sweep returns a flux NaN. PUFFY's `MINY = 0.001` (in MKS2 `x²`) is that
offset; the polar band (`num.polaraxis`) and `polarReflect` act on the first face, not on
θ = 0. C's 2D corner-ghost fill averages two ghosts at different radii, so the four
domain-corner cells cannot hold a field that varies in `r` (a `1/r²` monopole, say);
measure field diagnostics on an interior region. Both facts are exercised by the 2D Bondi
gates in `koral/tests/ks_evolution_tests.zig`, which is also the template for any new
curved-spacetime test: exact analytic prims on the radial faces, `polarReflect` on the
polar faces, a `phase` in the BC context if the B slots carry a vector potential before
`calcBfromA`.

The callback is fallible. Any frame or velocity conversion it performs
(`frames.transPmhdCoco`, `relele.convert`, …) returns `relele.Error`, so use
`try` and let it propagate. A boundary-adjacent cell can transiently reach a
spacelike velocity, and `setBc` already threads the error to the driver's
step-failure path (do **not** swallow it with `catch unreachable`). A BC that
does no conversions just returns the `[NV]f64` directly (the error union accepts
a plain value).

Because `Sim` is generic over `Config`, the callback lives inside a comptime
factory `Bc(SimT)` returning a struct with a `pub fn calc(...)`, exactly the
PUFFY pattern (`koral/problems/puffy/puffy.zig`,
`pub fn Bc(comptime SimT: type) type`).

A minimal `koral/problems/mytube/mytube.zig`:

```zig
const std = @import("std");
const config = @import("../../config.zig");
const layout = @import("../../layout.zig");
const grid_mod = @import("../../grid.zig");
const hydro = @import("../../physics/hydro.zig");
const relele = @import("../../relele.zig");   // for SpecificBc's error set
const sim_mod = @import("../../sim.zig");

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
        const x = sim.core.grid.xc(ix);      // cell-center internal coord
        const s = if (x < 0.0) left else right;
        try sim.initCell(ix, 0, 0, primsFor(s, sim.core.phys.gam)); // pp → p2u → store p,u
    }
    try sim.finishInit();                    // halo + set_bc(t) + initial dt guess
}

// --- boundary conditions (outflow / zero-gradient copy) -------------------
pub fn Bc(comptime SimT: type) type {
    return struct {
        pub fn calc(
            ctx: ?*const anyopaque,
            core: *const SimT.CoreT, ix: i64, iy: i64, iz: i64,
            t: f64, ifinit: bool, face: sim_mod.BcFace,
        ) relele.Error![SimT.nv]f64 {   // fallible: `try` any frame conversion
            _ = ctx; _ = t; _ = ifinit;
            // Copy the nearest domain cell (simple outflow). For a static tube
            // you could instead return the fixed left/right asymptotic state.
            const src: i64 = switch (face) {
                .xlo => 0,
                .xhi => core.nxi() - 1,
                else => unreachable,          // ny=nz=1: no y/z faces
            };
            var pp: [SimT.nv]f64 = undefined;
            core.p.load(src, iy, iz, &pp);
            return pp;   // no conversion here, so a plain return is fine
        }
    };
}
```

Key API points, all real:

- `sim.initCell(ix, iy, iz, pp)` stores the primitives and derives the conserveds
  via `p2u` (`koral/sim.zig`).
- `sim.finishInit()` runs `exchangeHalos()`, `setBc(self.t, true)`, then
  `initTimestepGuess()`, the serial/1-rank halo is a no-op. Seed the CFL
  denominator this way (or `s.step` will assert `tstepdenmax > 0` on the first step,
  turning a forgotten guess into a NaN run). A restart that has already adopted
  the checkpoint clock therefore fills ghosts at the resumed time.
- `sim.cflDt()` returns the CFL timestep `1/tstepdenmax` from the last wavespeeds,
  the one place the driver and `step()` share the dt expression.
- `sim.core.grid.xc(ix)` is the cell-center internal coordinate (the two-face average,
  `0.5*(xl(i)+xl(i+1))`, an intentional ulp-level match to C; do not "simplify").
  Everything a pass may read lives on `sim.core` (grid, cache, `p`, `u`, flags,
  `phys`, `num`); the Sim itself holds the scratch and the operators.
- `sim.nxi()/nyi()/nzi()` are the interior cell counts.
- `hydro.sFromU(rho, u, gam)` fills the entropy slot; the layout is indexed by
  `L.index(.rho)` etc., which `@compileError`s if the variable is absent for the
  active module set.

If your tube needs the vector-potential path (seed **B** from A rather than
directly), store A_φ in the B slots and call `sim.calcBfromA(true)` then
`setBc` before `finishInit`, exactly as PUFFY's `initAll` / `initAllWith` does
(which also runs β-normalization through `problems/common/magnetize.zig`).

### (d) Write `koral/problems/<name>/main.zig`, the driver

The driver is auto-discovered by `build.zig`, and it is one line: the generic
`koral.driver.run(App, init)` (`koral/driver.zig`) does the CLI, the params file,
the MPI world and φ-ring, the decomposition, `Sim.init`, restart or init, the
RK2IMEX loop with its dt guard, the heartbeat and every output. What you write
is the `App` contract. Use `koral/problems/puffy/main.zig` as the template; a
minimal version:

```zig
const std = @import("std");
const koral = @import("koral");

const mytube = koral.problems.mytube;      // after registering it (step e)

const App = struct {
    pub const name = "mytube";                       // log prefix, Silo file stem
    pub const cfg = koral.config.mytube;             // the comptime Config
    pub const default_params = "koral/problems/mytube/mytube.toml";
    pub const scalar_radii: koral.driver.ScalarRadii = .{ .lum = 1.0, .scale = 1.0 };
    pub const Physics = mytube.Physics;              // any value type; can be `struct {}`

    /// grid + Options (+ whatever the problem loads) from a params file.
    pub fn setup(comptime SimT: type, a: std.mem.Allocator, io: std.Io, p: *const koral.Params) !mytube.Setup(SimT) {
        var opt: SimT.Options = .{
            .phys = .{
                .coords = .mink,
                .floors = koral.solve.invert.FloorParams.cdefault,
                // opac = null  ⇒ no radiation source (SKIPRADSOURCE); fine for MHD-only.
            },
            .bc = .{ .x = .{ .specific = .{ .f = &mytube.Bc(SimT).calc } } },
        };
        opt.applyParams(p);   // gam, tsteplim, floors, implicit, threads … from the TOML
        _ = a; _ = io;
        return .{ .physics = .{}, .grid_global = mytube.makeGrid(p.nx), .options = opt };
    }

    /// Fill the domain; returns the β-normalization factor (1 for a tube).
    pub fn initAll(comptime SimT: type, s: *SimT, phys: *const Physics) !f64 {
        _ = phys;
        try mytube.initAll(SimT, s);
        return 1.0;
    }
};

pub fn main(init: std.process.Init) !void {
    return koral.driver.run(App, init);
}
```

`Setup(SimT)` is a struct with `.physics`, `.grid_global`, `.options` and a
`deinit(allocator)` (PUFFY's carries an optional MESA table; a tube's can be
empty). Optional `App` members: `banner(SimT, phys, p, grid_global)` for startup
notes, `afterInit(SimT, sim, phys, is_root)` for a post-init report, and
`reportSetupError(err, p)` for friendlier setup failures. The TOML may carry
`problem = "mytube"`; the driver refuses a file tagged for another problem.

`SimT.init` validates its preconditions through `Options.validate` and returns
`error.InvalidConfig` on a mismatch (ghost depth < reconstruction stencil,
`phys.coords ≠ cfg.coords`, `radvisc` with a null `opac`, `polaraxis` with
`ny ≤ 2·ncells` or on flat coords, an inconsistent decomposition).

Inside the driver's loop the core is `dt = s.cflDt()` (= `1/s.core.tstepdenmax`)
then `s.step(dt)`. `Sim.step` runs one full RK2IMEX step (`koral/sim/rk2imex.zig`),
alternating the implicit radiative-source operator with the explicit operator
across the stages. The explicit operator (`opExplicit`) itself is: wavespeeds →
reconstruction sweep → face fluxes → flux-CT for MHD → conserved update (the flux
divergence **and** the metric source term `ms[iv]*dt` are added together here) →
`calcU2p` inversion. Polar-axis correction (`doCorrect`) and the final entropy
update (`updateEntropy`) happen at the stage boundaries in `step`. `cfg.timestepping`
selects the integrator at comptime; only `.rk2imex` is implemented, the other
schemes fail to compile.

### (e) Register in `koral/koral.zig`

Add the problem module and (optionally) its tests to the `problems` namespace and
the `test {}` block:

```zig
pub const problems = struct {
    pub const puffy  = @import("problems/puffy/puffy.zig");
    pub const mytube = @import("problems/mytube/mytube.zig");   // add this
};

test {
    // ...
    _ = @import("problems/mytube/mytube.zig");                  // pull in its tests
}
```

### (f) The `.toml`

`koral/problems/mytube/mytube.toml`:

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
zig build run-mytube -Doptimize=ReleaseFast -- koral/problems/mytube/mytube.toml
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
`Sim.Options` (default `1.5`, between MinMod=1 and MC=2). The kernel lives in
`koral/fv/recon.zig`: the scheme is the tagged union
`Scheme = { donor, linear{theta}, ppm }` and the dispatch is
`reconstruct(comptime T, scheme, u: [5]T, dx: [5]T)` (with the NV wrapper
`reconstructN`); the kernel is generic over `f64` and `@Vector(W, f64)`. To add
a *new* scheme, extend the `Reconstruction` enum and its `ghostCells()` switch
in `config.zig`, add a variant to `Scheme` (with its `radius()`), and add its
branch in `reconstruct`.

### Switch the Riemann flux (LAXF / HLL)

Comptime field on `Config`:

```zig
.flux = .laxf,   // Lax-Friedrichs: F* = ½(fl+fr) − ½·ag·(uR−uL)
.flux = .hll,    // Harten-Lax-van Leer two-wave
```

`Sim.fluxesAtFaces` switches on `cfg.flux` into `koral/fv/laxf.zig`. Hydro and
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
gammamaxhd`, and `b2rhofloorframe: B2FloorFrame = { driftframe, zamoframe }`
(the magnetization-floor injection frame; the params key `zamo_floor_frame`
drives it); `RadParams` carries `gammamaxrad, eradfloor, eerhoratiomin/max,
eeuuratiomin/max`. Most of these are also params-file-overridable at runtime
(§3), a recompile is only needed for a new named preset. To add a new floor,
add a field (defaulting to the choices.h
value) and a clamp block in `checkFloorsMhd` / `checkFloorsRad`, keeping the
`ret<0 → sFromU` entropy re-sync at the end. To add a problem-specific set, add a
named `pub const` instance (like `.puffy`) and wire it through `options()`.

### Add a coordinate system

Contained in `koral/metric/`:

1. Add the tag to the `Coords` enum (`koral/config.zig`).
2. Write `gcov<New>(x: [4]Dual3, mp) [4][4]Dual3` in `koral/metric/forms.zig`
   returning only the **covariant** metric, the inverse, determinant,
   Christoffels, and `dlgdet` all derive automatically via `metric.compute`
   (dual-number AD).
3. Add the switch arm in `metric.gcovDual`, and a `gttpert` arm in
   `metric.gttpert`.
4. If it participates in coordinate transforms, add point maps and Jacobians in
   `koral/metric/coco.zig` and wire `cocoN`/`dxdx` (route through KS as the hub, as
   C does).

Nothing in `precompute` or downstream needs changing, `MetricCache` and
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
`KappaesMode` + `Params.kappaesAt` (`.none` disables scattering entirely, the
`scattering = false` params toggle). `Channels` also carries the
`synchrotron_bridge` toggle and a `mesa: ?*const MesaTable` pointer
(`koral/physics/mesa.zig`) that replaces the free-free Rosseland channels with
MESA table lookups. Note on π: C splits it here (truncated `Pi` in the emission
constants, exact `M_PI` in synchrotron's `bsqcgs`), but the port uses one exact
π for both, C's `fourpi`/`fourmpi` pair is a single `thermo.Consts.fourpi`.

### Add or modify a boundary-condition type

Per-axis handling is `bc.x/bc.y/bc.z: BcKind` on `Sim.Options`, where
`BcKind = union(enum) { periodic, copy, specific: { f, ctx } }` (defined in
`koral/sim/options.zig`, re-exported as `Sim.BcKind`). `.copy` clamps to the
domain edge; `.periodic` wraps (with the `NY<NG ⇒ pin to 0` C quirk);
`.specific` carries your callback and its opaque context (§7c). The stock
fragments a callback can compose are in `koral/problems/common/bcs.zig`
(outflow with r-rescaling, inner copy, polar reflection). To add a brand-new
*built-in* kind, extend the `BcKind` union and handle it in `setBcCell`
(`koral/sim/bc.zig`; and, if it needs corners, `fillCorners2d`/`fillCorners3d`). Arbitrary/boundary-varying conditions should go through the
`.specific` callback rather than a new enum value.

Note: both 2D x-y (`fillCorners2d`) and 3D (`fillCorners3d`) ghost-corner
filling are implemented; the only unimplemented layout is the x-z 2D case
(`ny==1, nz>1`), which `@panic`s.

### Enable & tune threading

Set `nthreads` in the params file (or directly on `Sim.Options`). `nthreads > 1`
makes `Sim.init` spawn a **persistent worker team** (`koral/threading.zig`,
P1 of the parallelization plan) that every per-step pass dispatches on,
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
MINK radiation box and a full PUFFY torus step). The 2D ghost-corner fill,
flux-CT region boundary, and dynamo curl remain serial and form
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
   `radiation.calcRij` + `relele.lowerSecond` for tensor fluxes rather than
   re-deriving the stress tensors.
2. Add a field to `dump.ScalarRow`.
3. Extend the `appendScalarLine` format string **and** its arg tuple (they must
   stay in sync, the `bufPrint` format has exactly 12 specifiers today), and add
   the column name to `scalar_header`.
4. Populate it in `dump.scalarRow` (or the problem's wrapper around it). The
   PUFFY driver calls `dump.scalarRow(SimT, &s, dt, r_lum, r_scale)`, the two
   diagnostic radii stay problem-side.

Keep reductions in **GU**, unit conversion is the driver's job at print time.

### Add a new physics module and wire it into the step

This is the large change. In canonical order:

1. **`koral/config.zig`.** Append a `Module` enum value in its canonical position.
   `validate()` enforces strictly increasing ordinals. Add any
   validation coupling rules.
2. **`koral/layout.zig`.** Add the module's `VarTag`s to the `VarTag` enum and its
   ordered tag slice to `moduleTags`. `NV`, `index()`, and `hasVar()` regenerate at
   comptime.
3. **Physics kernels.** Add the module's contributions to the stress-energy
   assembly (`calcTij` for a new fluid term), the flux vector (`fFluxPrime`, guarded
   by `L.hasVar(...)`), the wavespeeds, and the `p2u`/`u2p` conversions. Kernels
   dispatch on `cfg.has(m)` / `L.hasVar(tag)` at comptime, so inactive modules cost
   nothing.
4. **`koral/sim.zig`.** If the module needs an explicit source, add it to
   `opExplicit`'s conserved-update (alongside `metricSource*dt`); an implicit source
   goes in `opImplicit` / `solve/implicit.zig`.

Because everything is comptime-gated on the module set, enabling the module for a
problem is just a `Config` change; the layout and indices follow automatically.

---

## 9. Codebase map

Where the major pieces live (all under `koral/` unless noted):

| Area | Files |
|---|---|
| Comptime config / layout / units / grid / storage | `config.zig`, `layout.zig`, `units.zig`, `grid.zig`, `field.zig`, `params.zig`, `geometry.zig`, `koral.zig` (namespace root), `comm/comm.zig`, `comm/serial.zig`, `comm/mpi/` |
| Math utilities | `math/dual.zig`, `math/linalg.zig`, `math/simd.zig`, `math/misc.zig`, `math/quad.zig` |
| Metric (dual-AD, cache, coordinate transforms) | `metric/forms.zig`, `metric/metric.zig`, `metric/coco.zig`, `metric/precompute.zig` |
| Velocities / boosts / index gymnastics | `relele.zig`, `frames.zig` |
| Primitives ↔ conserved | `p2u.zig`, `solve/invert.zig`, `solve/invert_rad.zig` |
| Fluid & MHD physics | `physics/hydro.zig`, `physics/bfield.zig` (exported as `physics.mhd`) |
| M1 radiation | `physics/radiation.zig`, `solve/invert_rad.zig` |
| Microphysics (thermo, opacities, four-force) | `physics/thermo.zig`, `physics/opacities.zig`, `physics/mesa.zig` (+ `data/mesa_tables/`), `physics/radforce.zig`, `units.zig` |
| Implicit rad-gas source | `solve/implicit.zig` |
| Reconstruction / wavespeeds / flux / Riemann | `fv/recon.zig`, `physics/wavespeeds.zig`, `physics/flux.zig`, `fv/laxf.zig` |
| Evolution driver | `sim.zig`, `sim/storage.zig`, `sim/bc.zig`, `sim/timers.zig`, `threading.zig` |
| Constrained transport, dynamo, radiative viscosity | `sim/ct.zig`, `sim/dynamo.zig`, `physics/radvisc.zig` (pure kernels), `sim/rijvisc.zig` (gather + per-step pass) |
| PUFFY problem + quadrature | `problems/puffy/puffy.zig` (`Physics`, `setup`, `initAllWith`), `problems/puffy/main.zig`, `problems/puffy/*.toml`, `math/quad.zig` |
| Diagnostics / I/O | `io/scalars.zig`, `io/dump.zig`, `io/silo.zig` (+ `io/silo_disabled.zig`) |
| Build / tests / oracle / tools | `build.zig`, `koral.zig` (`test {}`), `tools/gen_golden.sh`, `tools/{res2kdmp,kdmp2silo,qmri,kdmp2png,kdmp2lc,goldtest,mpi_gates,bench_implicit}.zig`, `testing/golden.zig`, `testing/tubes.zig`, `oracle/harness_*.c`, `tests/golden/` |

The library imports almost nothing outward. The foundation (`config`/`layout`/
`grid`/`field`/`units`/`params`) is leaf infrastructure consumed by every physics
and solve module; `sim.zig` is the orchestrator that ties them together. Problems
sit on top and are the only executables.

---

*Related documents:* `ARCHITECTURE.md` (design rationale, storage-layout
insulation, C-diffability contract) and `PHYSICS.md` (GR rad-MHD equations, M1
closure, unit system).
