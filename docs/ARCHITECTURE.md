# koral-zig architecture

This guide traces the data structures and the exact path through one simulation
step. Use it when changing the solver or finding which module owns an operation.

- **PHYSICS.md** covers the equations and their physical meaning (EOS, stress-energy tensors,
  M1 closure, opacities, four-force, dynamo, viscosity).
- **USER_GUIDE.md** explains how to build, configure a run, and interpret output.

Where a formula appears here it is included only to make the *control flow* legible;
PHYSICS.md is the authority on the physics itself.

---

## Contents

1. [Design constraints](#1-design-constraints)
2. [Repository layout and dependencies](#2-repository-layout-and-dependencies)
3. [The core data model](#3-the-core-data-model)
4. [The metric / coordinate layer](#4-the-metric-coordinate-layer)
5. [The anatomy of one time step](#5-the-anatomy-of-one-time-step)
6. [The solver stack: p2u, u2p, the implicit ladder, fixups](#6-the-solver-stack-p2u-u2p-the-implicit-ladder-fixups)
7. [Reconstruction and Riemann flux](#7-reconstruction-and-riemann-flux)
8. [MHD specifics and the extra physics in the step](#8-mhd-specifics-and-the-extra-physics-in-the-step)
9. [Boundary conditions and polar-axis correction](#9-boundary-conditions-and-polar-axis-correction)
10. [The threading model](#10-the-threading-model)
11. [Testing and the oracle](#11-testing-and-the-oracle)
12. [Where things live](#12-where-things-live-navigation-quick-reference)

---

## 1. Design constraints

koral-zig is a Zig 0.16 reimplementation of the **KORAL** general-relativistic
radiation-MHD C code, targeting the **PUFFY** radiative-MHD accretion torus: a disk
around a black hole in modified Kerr-Schild (MKS2) coordinates, evolving ideal MHD
coupled to M1-closure radiation with thermal Comptonization, radiative shear viscosity,
and a mean-field "mimic" dynamo. The reference configuration is the 2D axisymmetric
10 M☉ Schwarzschild torus (`puffy.toml`), but the problem is runtime-retargetable:
`puffy.Physics` holds mass, spin, RMIN/RMAX, floors, opacities, and the rest of
the C `define.h` knobs as a **value** (`Physics.fromParams` applies a TOML preset;
`rminForSpin(a)` keeps RMIN inside the Kerr horizon). Tests and goldens use
`puffy.defaults` and never go through `fromParams`. Presets exist for 3D (`puffy3d.toml`), Sgr A*
(`puffy3d_sgra.toml`, mass = 4.297e6), a spinning Sgr A* (`puffy3d_sgra_spin.toml`,
bhspin = 0.9375), and a 10⁹ M☉ AGN with MESA opacities (`puffy_agn.toml`).

### The C-diffability contract

The port is designed for direct comparison with the reference C code in
`../koral_lite`. A Zig dump should match a C dump index by index within a few
ulps. That constraint explains several choices that would otherwise look odd:

- **Matching variable order.** The state vector uses KORAL's fixed module order
  (hydro → electrons → relel → mhd → forcefree → radiation) so that primitive slot *i*
  in Zig is primitive slot *i* in C. `@intFromEnum` on the `Module` enum gives the
  canonical ordinal, and `Config.validate()` `@compileError`s if modules are listed out
  of order (see [§3](#3-the-core-data-model)).
- **`f64` throughout.** All arithmetic is `f64`. Expression structure is preserved
  alongside mathematical value. Examples include `/CCC/CCC/CCC/CCC`, the nested
  `@sqrt(@sqrt(...))` in the LTE inverse, and the exact parenthesisation of the ko.h unit
  macros are all reproduced so results agree to a few ulps.
- **Transcribed quirks and bugs.** Where the C code has an inconsistency, koral-zig
  keeps it rather than "fixing" it, because fixing it would break the golden diff. A
  representative (non-exhaustive) list:
  - `Grid.xc` is the face average `0.5*(xl(i)+xl(i+1))`, *not* `minx+(i+0.5)*dx`, an
    ulp-level match to C `set_grid`. This behavior has been checked and must stay.
  - The radiation z-wavespeed limiter uses the *y* optical depth (`rv2z = rv2dim[1]`).
  - The tensor boosts have "dead-code" α-corrections in the ff→lab direction that C never
    actually applies, so lab↔ff boosts are not mutual inverses. Reproduced verbatim.
  - `totalMass` does *not* expand the φ-wedge to 2π for `TNZ==1`, while `mdot`/`lum` do.

  These are documented at each site in the code and in [§11](#11-testing-and-the-oracle).

### Serial-first, opt-in threading, opt-in φ-only MPI

- The serial path defines the golden result. `nthreads = 1` produces bit-identical results
  to the recorded C oracle. That is the configuration all golden tests run in.
- **Threading is opt-in and bit-identical.** Setting `nthreads > 1` parallelises the
  per-cell inversion and implicit passes over rows of the θ-index; because the rows are
  disjoint and each cell is computed locally, and only integer counters are summed after
  the join, the numeric result is identical to serial (see [§10](#10-the-threading-model)).
- **MPI is opt-in and φ-only.** `koral/comm/comm.zig` comptime-selects `Serial` or
  `comm/mpi/mpi.zig` from `-Dmpi`. The serial backend is a single-rank no-op
  (`exchangeHalos` does nothing, `allreduce` is the identity). The MPI backend is a
  Cart φ-ring with persistent zero-copy halo exchange, in-place collectives, and
  MPI-IO checkpoints (`dump.writePrim` / `loadPrim`). `nx/ny/nz` in the params file
  are global; each rank owns a z-slab. 2D (`nz = 1`) never decomposes. Numerics talk
  only to the `comm.Backend` interface.

### Compile-time configuration

Which physics runs, how wide the state vector is, and how deep the ghost zones are, are
all **comptime** facts derived from a single `Config` struct. The state-vector layout,
the ghost depth, and the kernel selection are generated at compile time, replacing
KORAL's hand-maintained `define.h`/`choices.h`/`mnemonics.h` macro layer. Change the
`Config` and the storage width `NV`, every index, and the ghost depth regenerate
automatically.

---

## 2. Repository layout and dependencies

The build produces two main kinds of artifact:

- **The `koral` library.** `koral/…` contains the physics and numerics and starts at
  `koral/koral.zig`.
- **Problem executables.** Each `koral/problems/<name>/main.zig` is one line:
  `koral.driver.run(App, init)`. The generic driver (`koral/driver.zig`) parses the
  CLI, loads `Params`, opens the MPI world and φ-ring, decomposes the grid, builds
  the `Sim`, restarts or initialises, runs the time loop and writes every output.
  `App` is the problem's comptime contract (name, `cfg`, `default_params`,
  `scalar_radii`, `Physics`, `setup`, `initAll`, optional `banner` / `afterInit` /
  `reportSetupError`). Currently the only problem is `koral/problems/puffy/` (with
  its `.toml` presets alongside); `koral/problems/common/` holds the library pieces
  torus problems share (limotorus solver, floor atmospheres, β normalization, the
  stock `bc.c` fragments, the seed-MRI diagnostic, the init perturbation).

`build.zig` discovers every subdirectory of `koral/problems/` that has a `main.zig` and,
per directory, builds one executable importing `koral`, plus `<name>` (build) and
`run-<name>` (build & run) steps (`problems/common/` has no `main.zig` and is skipped). A single `test` step (`zig build test`) runs one `addTest` over the whole `koral`
module; `-Dtest-filter` narrows it and `-Dslow-tests` enables the full-grid keystones.
A configure-time guard fails the build with `error.UnregisteredTestFile` if any
`koral/tests/*_tests.zig` or `koral/tests/golden/*_golden_tests.zig` is missing from
`koral.zig`'s `test {}` block. `-Dsilo` builds LLNL Silo from the lazy dependency;
without it, `koral/io/silo_disabled.zig` supplies the stub. Tool steps build the
implicit benchmark, restart and Silo converters, MRI diagnostic, GRRT image and
light-curve programs, and EHT verification runner. MPI builds also add `mpi-gates`.

### Namespace root

`koral/koral.zig` re-exports config, layout, storage, metrics, physics, solvers,
finite-volume methods, simulation, communication, I/O, rendering, problems, and
testing. Its `test {}` block imports every module and
every test file, so the single test artifact contains the entire suite. A problem driver
does `const koral = @import("koral");` and reaches everything through it.

### Dependency layers (bottom-up)

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │ koral/problems/puffy/main.zig = koral.driver.run(App, init)          │
  │   App: puffy.Physics / puffy.setup / puffy.initAllWith / banner      │
  │   (problems/common/: limotorus · atmosphere · magnetize · bcs · mri) │
  └───────────────┬──────────────────────────────────────────────────────┘
                  │  imports koral
  ┌───────────────▼──────────────────────────────────────────────────────┐
  │ sim.zig , Sim(cfg): composition root — embeds Core, owns the pass    │
  │   scratch + stage buffers, exposes the operators; sim/rk2imex.zig is │
  │   the integrator. Every pass takes *CoreT (sim/core.zig) plus its own│
  │   scratch: sim/{timestep,u2p,fixup,explicit,implicit_op,entropy,     │
  │   stage,bc,polaraxis,ct,dynamo,rijvisc}.zig; sim/options.zig is the  │
  │   Options a problem builds.                                          │
  └── calls ─┬───────────┬───────────┬──────────┬────────────────────────┘
             │           │           │          │
    ┌────────▼──┐  ┌─────▼────┐ ┌────▼─────┐ ┌──▼─────────┐
    │ fv/       │  │ physics/ │ │ solve/   │ │ metric/    │
    │ recon     │  │ hydro    │ │ invert   │ │ metric     │
    │ laxf      │  │ bfield   │ │ invert_  │ │ coco       │
    │           │  │ radiation│ │  rad     │ │ precompute │
    │           │  │ wavesp.  │ │ implicit │ │ (MetricC.) │
    │           │  │ flux     │ │          │ │            │
    │           │  │ thermo   │ │          │ │            │
    │           │  │ opacities│ │          │ │            │
    │           │  │ radforce │ │          │ │            │
    │           │  │ radvisc  │ │          │ │            │
    │           │  │ mesa     │ │          │ │            │
    └─────┬─────┘  └────┬─────┘ └────┬─────┘ └─────┬──────┘
          └─── relele / frames / p2u / threading ──┘
                                    │
  ┌─────────────────────────────────▼─────────────────────────────────────┐
  │ Foundation (leaf infra, only std + intra-foundation imports):         │
  │  config · layout · grid · field · units · params · geometry           │
  │  math/dual · math/linalg · math/misc · math/simd · comm/{serial,mpi}  │
  │  testing/*                                                            │
  └───────────────────────────────────────────────────────────────────────┘
```

`config.zig` is imported by `layout` and `geometry`; `grid.zig` by `field`; the
whole foundation is consumed by `sim.zig` and every physics/solve/flux module. The
foundation itself calls almost nothing outward. It is leaf infrastructure.

Note two intentional mutual dependencies you will meet: `physics/radiation.zig` ↔
`solve/invert_rad.zig` (radiation calls `u2pRad`; the inversion calls `calcFfRtt`), and
`physics/radforce.zig` ↔ `solve/invert_rad.zig` (via `RadParams`/`checkFloorsRad`).

---

## 3. The core data model

### 3.1 The comptime `Config`

`koral/config.zig`. Everything that changes *generated* code lives here.

```zig
pub const Config = struct {
    modules: []const Module,               // physics subsystems, in canonical order
    reconstruction: Reconstruction = .linear,   // donor_cell | linear | ppm
    flux: FluxMethod = .laxf,              // laxf | hll
    timestepping: TimeStepping = .rk2imex, // rk1 | rk2 | rk2heun | rk2imex
    coords: Coords = .mink,                // mink | bl | ks | mks2
    velprim: VelType = .velr,              // gas velocity primitive parametrisation
    velprim_rad: VelType = .velr,          // radiation-frame velocity primitive
    evolve_photon_number: bool = false,
    n_relel_bins: usize = 0,
};
```

Key methods:

```zig
pub fn ghostCells(comptime self: Config) usize;   // → reconstruction.ghostCells(); PUFFY(ppm)=3, linear/donor=2
pub fn has(comptime self: Config, comptime m: Module) bool;  // inline-for module gate
pub fn validate(comptime self: Config) void;      // C: am_i_sane(), @compileError on illegal combos
```

`validate()` enforces: hydro is mandatory; radiation requires hydro; relel requires
electrons; forcefree requires mhd; relel needs `n_relel_bins > 0` and vice versa;
photon-number needs radiation; and modules must be strictly increasing in enum order
without duplicates (this ordering is exactly what guarantees index-for-index C parity).

The PUFFY config is `koral.config.puffy`:
`modules = {hydro, mhd, radiation}`, `ppm`, `laxf`, `rk2imex`, `mks2`, giving **NV = 13,
NG = 3**.

### 3.2 `VarLayout(cfg)` and the variable tags, NV = 13

`koral/layout.zig`. `VarTag` is the enum of every variable the state vector can carry, in
C order. `moduleTags(cfg, m)` is the single source of variable ordering:

| module | tags contributed |
| --- | --- |
| hydro | `rho, uu, vx, vy, vz, entr` |
| electrons | `entre, entri` |
| relel | `{}` (bins appended positionally after the electrons block) |
| mhd | `b1, b2, b3` |
| forcefree | `uuff, vxff, vyff, vzff` |
| radiation | `ee, fx, fy, fz` (+ `nf` if `evolve_photon_number`) |

`VarLayout(cfg)` is a comptime-generated type replacing KORAL's `mnemonics.h` index
arithmetic. It exposes:

```zig
const count;                              // = C's NV
const tags;                               // []const VarTag in state order
pub fn index(comptime tag: VarTag) usize; // = C's RHO/B1/EE... or @compileError if absent
pub fn hasVar(comptime tag: VarTag) bool;
pub fn relelIndex(bin: usize) usize;      // = C's NEREL(i); RUNTIME (bins are positional)
```

For PUFFY the layout is exactly:

```
RHO=0  UU=1  VX=2  VY=3  VZ=4  ENTR=5  B1=6  B2=7  B3=8  EE=9  FX=10  FY=11  FZ=12
```

Two-temperature (adding `entre`/`entri`) would push `B1=8, EE=11, NV=15`; photon-number
appends `nf` after `fz` (NV=14). Relel bins are **not** `VarTag`s, they occupy
`n_relel_bins` positional slots inserted right after the electrons block but still inside
the hydro block (C `NEREL(i)=8+i`). `index()` is comptime and `@compileError`s if the
variable is absent. This catches module-gating bugs at compile time.

### 3.3 `Grid` and `Field`, AoS storage with ghosts

`koral/grid.zig`. The grid is uniform by construction. Curvature lives in the
metric. Fields include active cells `nx,ny,nz`; ghost depth `ng` from `Config.ghostCells`;
per-dim ghost depth after dimension collapse `ngx,ngy,ngz`; internal-coord bounds
`minx..maxz`; uniform spacing `dx,dy,dz`.

```zig
pub fn init(opts: struct { nx, ny=1, nz=1, ng, minx, maxx, miny=0, maxy=1, minz=0, maxz=1 }) Grid;
pub fn sx(g) usize;   // padded storage dim = nx + 2*ngx   (SX = NX + 2*NGCX)
pub fn cellCount(g) usize;   // sx*sy*sz, total stored cells incl. ghosts
pub fn xc(g, ix: i64) f64;   // cell CENTER = 0.5*(xl(ix)+xl(ix+1))   ← ulp-match, not min+(i+.5)dx
pub fn xl(g, ix: i64) f64;   // left FACE = minx + ix*dx   (yc/zc, yl/zl are the y/z analogues)
pub fn cellSize(g, i: i64, dim: usize) f64;  // per-cell size = xl(i+1)-xl(i)  (used everywhere in evolution)
```

**Dimension collapse:** a dimension of size 1 gets zero ghosts and `SX/SY/SZ = 1`; callers
must pin that dim's cell index to 0. PUFFY is 2D (`nz=1`).

`koral/field.zig`. `Field(NV)` is a comptime-parameterised, cell-centered, Array-of-Structs
store, **variable-fastest**, exactly C's `get_u`/`set_u` layout:

```
offset = (jx + jy*SX + jz*SY*SX)*NV + iv,   jx = ix + NGCX, ...
```

Signed `i64` cell indices address ghosts (negative or ≥ n). The **kernel-facing API is
`load`/`store`**, which `@memcpy` a whole cell's `NV` variables to/from a `[NV]f64` stack
buffer, kernels never touch `data` directly. This insulation is what lets the storage
layout flip to AoSoA later without touching a single kernel.

### 3.4 `Geometry` and the `MetricCache`

`koral/geometry.zig`. `Geometry` is the per-point bundle handed to every kernel (C:
`struct geometry`):

```zig
coords: Coords,                                   // which coordinate system this point is in
xxvec: [4]f64,                                    // point coords, xxvec[0]=0 (stationary metric)
gg: [4][5]f64,   GG: [4][5]f64,                   // g_μν (covariant), g^μν (contravariant)
gdet: f64,  alpha: f64,  gttpert: f64,            // √-g, lapse, perturbed g_tt
```

(The former `ix/iy/iz/ifacedim` cell-identity fields were write-only. The viscous-flux
face average `rijvisc.faceAvg` takes its `dim` explicitly rather than from geometry, so they
were dropped in the P5 pass.)

The **4×5** layout is C's: columns 0..3 are the metric/inverse, column 4 packs extras
(`gg[i][4]=dlgdet` for i<3, `gg[3][4]=gdet`; `GG[3][4]=gttpert`, other `GG[i][4]=0`).
Reading the wrong column silently gives garbage, index only `[0..4][0..4]` for the tensor
itself. `alpha = sqrt(-1/GG[0][0])`.

`Geometry.cov()` and `.con()` return `MetricCovOf(f64)` / `MetricConOf(f64)`, one-pointer
views of `gg` / `GG` that are nominally distinct types, so a typed view of one block cannot
be passed where the other is expected. Scalar helpers (`relele.lowerVec`, `frames.boost*`,
`wavespeeds.lrCore`, …) take the whole `*const Geometry` and select the block themselves;
the lane-generic `<name>G` helpers take the two views, since a `@Vector` batch has no
`Geometry` to select from. Build views with the accessors: an anonymous `.{ .m = … }`
literal coerces to *either* view type and so silently defeats the distinction.

The `MetricCache` (`koral/metric/precompute.zig`) is built once per run (in `sim.zig`) and
stores, over all cells including ghosts: cell-center metric blocks, Christoffels, Jacobians
to/from output coordinates, and per-face metric blocks. The solver reads geometry through
`cache.fillGeometry` / `cache.fillGeometryFace` and Christoffels through `cache.kr`.
See [§4](#4-the-metric-coordinate-layer).

### 3.5 `Units`

`koral/units.zig`. All CGS ↔ geometrized-unit (GU) conversions for one BH mass. Geometrized
units set `G = c = 1` and use the BH mass as the length scale; temperatures stay in Kelvin
in both systems.

```zig
pub fn init(mass_msun: f64) Units;   // .mass = mass_msun, .masscm = mass_msun * MSUNCM
```

`masscm = mass * MSUNCM` (`MSUNCM = 147700.0` cm = GM☉/c²). PUFFY `mass = 10` →
`masscm = 1.477e6` cm. A full set of `<quantity>Cgs2Gu`/`Gu2Cgs` pairs (len, time, vel,
rho, numdens, surfdens, mass, kappa, enden, flux, heatcool, erg, charge, cross) mirror the
ko.h macro parenthesisation. GU physical constants (`kBoltz`, `mProton`, `mElectr`,
`sigmaRad`, `aRad = 4*sigmaRad`, `kappaEsCoeff`, `sigmaThompson`, `gmc2 = masscm`,
`gmc3 = gmc2/CCC0`) live here too, along with the LTE blackbody relations
`lteEfromT (E = 4σ T⁴)` / `lteTfromE`.

Two things to watch: the **opacity (kappa) conversion is the inverse of the density
conversion** (`kappaCgs2Gu = x*(CCC²/(GGG*masscm))`, multiply where rho divides); and the
Params **default mass is `1/147700`**, which makes `MASSCM = 1` and matches C's
length identity. It is not the PUFFY mass; the PUFFY params file must set `mass = 10`.

### 3.6 `Params`

`koral/params.zig`. The runtime knobs a physicist tweaks between runs (numbers that do
*not* change generated code). Loaded from a flat TOML-subset file, which is the *run
record*: unknown keys are a hard error (`UnknownParamsKey`) as typo protection.

```zig
pub fn load(a, io, path) !Params;   // readFileAlloc (1 MiB cap), then parse()
pub fn parse(a, text) !Params;      // line-based TOML subset: strip comments, skip [sections], split on first '='
```

Fields are grouped: physical (`mass, bhspin, gam`), grid (`nx, ny, nz`), domain
(`rmin, rmax, mksr0, mksh0, miny/maxy/minz/maxz`), run control (`tstart, tmax, nstep_max,
tsteplim, dtout1, dtout2, nout_step, out_dir`), floors/ceilings, and execution
(`deterministic, nthreads`). `parseValue` dispatches on the field type
(`f64`/`usize`/`bool`/`[]const u8`).

On top of these sits a large group of **optional physics overrides** (all `?T = null`,
"unset" means "keep the compiled preset"), added for the AGN / Sgr A* presets: implicit
tolerances (`radimpeps`, `radimpmaxiter`), opacity damping (`opdamp_maxlevels`,
`opdamp_factor`), fixup/order toggles (`doradimpfixups`, `reduceorderatbh`,
`dampradwavespeednearaxis`), opacity channels (`bremsstrahlung`, `kleinnishina`,
`synchrotron_bridge`, `scattering`, `mesa_table`), floor knobs (`zamo_floor_frame`,
`eerhoratiomin/max`, `eeuuratiomin/max`, `gammamaxrad`, `rhoatmmin`), composition
(`hfrac`, `hefrac`), and torus/atmosphere constants (`lt_kappa`, `maxbeta`, `atm_tgas`,
`atm_trad_init`, `atm_erad_factor`). `puffy.Physics.fromParams` folds non-null values
onto `puffy.defaults` and returns a new `Physics` value. Nothing mutates process
state. `puffy.setup` and `puffy.load` also attach an optional heap MESA table and
build `Sim.Options` (`Physics.toOptions`). The driver and the replay tools
(`kdmp2silo`, `kdmp2png`, `kdmp2lc`, `qmri`) share that path.

### 3.7 Per-cell derived state, `radforce.RadState`

The C `struct_of_state` bundle. An early `state.zig` `State(cfg)` stub was intended to
grow into a comptime composition of each module's fields, but that design was never built
and the type had zero users, so it was **deleted in the P5 pass**. The live per-cell state
bundle used by the source solver is `radforce.RadState`
(see [§6](#6-the-solver-stack-p2u-u2p-the-implicit-ladder-fixups)).

---

## 4. The metric / coordinate layer

`koral/metric/*` and `koral/math/dual.zig`. This layer assembles the GR metric (g_μν,
g^μν, √-g, Christoffels, Jacobians) at every cell center and face, and provides coordinate
transforms.

### 4.1 Dual-number automatic differentiation

`math/dual.zig` defines the generic `Dual(comptime N)`, a forward-AD scalar carrying a
value plus `N` partials, with `Dual3 = Dual(3)` as the metric workhorse, three spatial
partials ∂/∂(x1,x2,x3), because the metric is **stationary** (∂_t g = 0). All arithmetic
and transcendentals (`sin, cos, tan, exp, log, sqrt, atan`) propagate exact derivatives by
the chain rule. This replaces KORAL's giant Mathematica-exported closed-form derivative
expressions: you write each covariant metric once, evaluate it in `Dual3`, and get both g
and ∂g/∂xⁱ for free. The 4×4 determinant/inverse live in `math/linalg.zig` (`det4`,
`inv4`), generic over any `Dual(N)`, `Dual(0)` instantiates a plain-f64 version with a
bit-identical value chain, so the dual and scalar metric paths share one implementation.

### 4.2 Metric forms and assembly

`metric/forms.zig` writes each covariant metric `gcovMink/gcovBl/gcovKs/gcovMks2` as a
closed-form `[4][4]Dual3` expression, and defines `MetricParams` (re-exported by
`metric.zig`). `metric/metric.zig::compute` is the workhorse (C `calc_metric`):
`gcovDual` seeds x1,x2,x3 as independent dual variables (x0=t is constant), dispatches to
the right form, computes the determinant and inverse via `linalg.det4`/`linalg.inv4` in
dual arithmetic, then fills:

```zig
gcov, gcon, gdet = @sqrt(-det.v),
dlgdet[i] = 0.5*det.d[i]/det.v,
kris[i][j][k] = ½ g^il (∂_j g_lk + ∂_k g_lj − ∂_l g_jk),   // ∂_0 ≡ 0 (stationary)
gttpert.
```

`MetricParams { a, mksr0, mksh0 }` carries BH spin and the MKS2 offsets (PUFFY: a=0,
mksr0=0.1, mksh0=0.9). MKS2 is the exact pushforward of Kerr-Schild through the diagonal
map `r = R0 + e^{x1}`, `θ = θ(x2)`, with the derivatives coming from `Dual3` rather than a
transcribed Mathematica export. `metric.zig` also hosts the Kerr radii helpers
`rHorizonBL(a)` and `rIscoBL(a)`, used by the PUFFY driver and `rminForSpin`.

### 4.3 The MetricCache: centers, faces, and the Christoffel correction

`metric/precompute.zig::MetricCache` walks every cell, including ghosts. For each
cell it calls `metric.compute` at the center, then `applyKrisCorrection`. This is the `GDETIN==1`/
`MODYFIKUJKRZYSIE` branch that rewrites the Christoffel trace Γ^μ_{κμ} so it equals the
*finite-difference* √-g gradient across the cell. This makes the geometric source terms
telescope against the discrete flux divergence so uniform states stay uniform. It needs √-g at
the two faces per direction, but only √-g. It calls `metric.gdetAt` (just `gcovDual` +
`det4` + `@sqrt(-det.v)`, bitwise-identical to `compute().gdet`) rather than a full `metric.compute`,
skipping the dual inverse and the 64-entry Christoffel assembly at those samples. The cache build is
still center-`compute`-dominated but no longer pays two extra full inversions per cell.

Faces store metric blocks at all x/y/z **left** faces. Read accessors:

```zig
pub fn fillGeometry(self, ix, iy, iz) Geometry;         // C: fill_geometry
pub fn fillGeometryFace(self, ix, iy, iz, dim) Geometry;// C: fill_geometry_face (left face in `dim`)
pub fn kr(self, i, j, k, ix, iy, iz) f64;               // cached Γ^i_jk
```

`geometryAt(coords, mp, x)` gives an on-the-fly `Geometry` at an arbitrary off-grid point
(no Christoffel-trace correction, Christoffels are not part of `Geometry` anyway).

### 4.4 Coordinate transforms (coco)

`metric/coco.zig` provides point transforms `cocoN(x, from, to, mp)` and analytic Jacobians
`dxdx(x, from, to, mp)` between BL / KS / MKS2, **all routed through KS as the hub** exactly
like C's `coco_N`. `mink` is never a transform endpoint (unsupported pairs panic). This is
what `frames.transPallCoco` and the I/O diagnostics use to move states between the
evolution coordinates (MKS2) and Boyer-Lindquist (for problem setup and diagnostics).

> **The two-π quirk (not preserved).** C's MKS2 metric-flavour θ uses the truncated
> `Pi = 3.141592654` while its coco point-transform θ uses exact `M_PI`, so C's two θ
> flavours disagree at ~1e-9. Since 2026-07-06 the port uses one exact `std.math.pi` for
> both (`metric/forms.zig`) and the flavours agree exactly. C's internal spread survives
> only as the 1e-8 gates on MKS2-*derived* golden comparisons
> (`koral/tests/golden/metric_golden_tests.zig`). That tolerance comes from C.

---

## 5. The anatomy of one time step

This is the centerpiece. One time step is `Sim.step(forced_dt)` in `koral/sim.zig`, an
`RK2IMEX` IMEX integrator (Pareschi-Russo SSP2(2,2,2)) with implicit coefficient
`γ = 1 − 1/√2 ≈ 0.29289`. The implicit operator handles the stiff radiation-gas four-force
exchange; the explicit operator handles the hyperbolic MHD/radiation transport.

### 5.1 The `Sim(cfg)` object and its `Core`

`Sim(comptime cfg)` is a factory returning the composition root specialised to the
config. It embeds a `Core(cfg)` (`koral/sim/core.zig`) as `sim.core`: the grid, the
`MetricCache`, the primitive field `p` and conserved field `u`, the per-cell integer
`flags` and wavespeed/timestep scalars (`scal`), the `Physics` and `Numerics` option
groups, the worker team, the decomposition and comm backend, the pass timers, and
the CFL / order-reduction scalars the passes write. The Core also carries the small
accessors (`nxi/nyi/nzi`, `getFlag/setFlag`, `scGet/scSet`, `isCorner`,
`isCellCorrectedPolaraxis`, `initCell`, `cflDt`) and the MPI seam (`exchangeHalos`,
`globalMax/Sum`). `wide = cfg.has(.mhd)` (mirrors C's `MPI4CORNERS`) makes MHD sweeps
reach ±1 ghost row and fill 2D corners; `base_order` (0/1/2) comes from
`cfg.reconstruction`; both are `pub const`s of the Core.

**Every pass takes `*CoreT` plus its own scratch as explicit arguments**, so a pass
cannot reach the integrator's stage buffers, the face stores or another pass's
scratch — ownership is enforced by the type. The Sim owns what is not core state:

| `Sim` field | owner | holds |
| --- | --- | --- |
| `integ` | `sim/rk2imex.zig` `Integrator(NV)` | the eight RK2IMEX stage buffers ([§5.2](#52-the-stage-buffers-what-each-one-holds)) |
| `faces` | `sim/explicit.zig` `Faces(NV)` | `pb_l/pb_r`, `fl_l/fl_r`, `flb`; flux-CT rewrites `flb`'s B rows |
| `bak` | `sim/fixup.zig` `Backups(NV)` | the whole-grid u/p fixup backups |
| `ct` | `sim/ct.zig` `Scratch` | corner EMFs + the vector-potential work field; `void` without MHD |
| `visc` | `sim/rijvisc.zig` `State` | `?`: the viscosity parameters + per-cell R^i_j, only when `radvisc` is set |
| `dynamo` | `sim/dynamo.zig` `State` | `?`: the dynamo parameters + ΔA_φ / scale-height scratch, only when `dynamo` is set |
| `bc`, `parallel` | `sim/options.zig` | the per-axis boundary kinds and the team knobs |
| `t`, `dt`, `nstep`, `n_radimp_*` | `Sim` | the clock and the implicit-solver diagnostics |

The pass modules are `sim/timestep.zig` (wavespeeds + CFL), `sim/u2p.zig`,
`sim/fixup.zig`, `sim/explicit.zig` (sweeps, flux combination, metric source,
conserved update, driven through its `Ctx`), `sim/implicit_op.zig`,
`sim/entropy.zig`, `sim/stage.zig` (the exact-shape stage arithmetic), `sim/bc.zig`
(all boundary-condition logic, [§9](#9-boundary-conditions-and-polar-axis-correction)),
`sim/polaraxis.zig` (the polar-axis band, same section), `sim/ct.zig` (flux-CT + the
vector-potential machinery, [§8.1](#81-constrained-transport-flux-ct)),
`sim/dynamo.zig` (the mimic dynamo, [§8.4](#84-the-mimic-dynamo-run-after-each-explicit-sub-step);
`applyDynamo` is operator-level and takes the Sim because it composes `setBc` and
`calcU2p`), and `sim/rijvisc.zig` (the sim-coupled half of the radiative shear
viscosity, [§8.3](#83-radiative-shear-viscosity-filled-oncestep-added-at-faces); its
pure per-cell kernels live in `physics/radvisc.zig`). `sim/storage.zig` holds
`FaceStore`, the `Flag`/`Scal` enums and slot tables; `sim/timers.zig` the always-on
per-pass wall-clock instrumentation (`Pass`, `PassTimers`) that the operators feed and
the driver prints at scalar cadence. The worker team ([§10](#10-the-threading-model))
lives in the root-level `threading.zig`. The Sim's methods are the operators
(`calcWavespeeds`, `calcU2p`, `cellFixup`, `opExplicit`, `opImplicit`, `doCorrect`,
`fluxCt`, `calcBfromA`, `calcRijViscTotal`, `applyDynamo`, `updateEntropy`, `setBc`,
`step`): thin, they own the timers and hand each pass its banded worker.

`Sim.Options` (`koral/sim/options.zig`, generic over the Core) is what a problem
builds, grouped by owner: `phys: Physics` (`coords`, metric params `mp`, `gam`,
`floors` (`FloorParams`), `rad` (`RadParams`), `opac` (`?radforce.Params`; **null ≡
SKIPRADSOURCE**, no implicit operator), `implicit` (`ImplicitParams`)); `num: Numerics`
(`tsteplim`, `minmod_theta`, the `fixups` switches, `reduceorderatbh` (C
`REDUCEORDERATBH`: drop one reconstruction order inside the BL horizon),
`reduceorderafterfixup`, `dampradwavespeednearaxis`, `polaraxis: ?PolarAxis` with
`ncells`); `bc` (per-axis `x/y/z: BcKind`, where `.specific` carries the callback and
its opaque context, so "specific without a callback" cannot be expressed);
`radvisc: ?radvisc.Params`; `dynamo: ?dynamo.Params`; `parallel` (`nthreads`,
`pin_threads`); `comm`; `decomp`. The three optional passes follow the `opac` idiom —
null means off — so the old bool-plus-params pairs are gone. `Options.applyParams(p)`
is the generic TOML mapping every problem shares (floors, radiative floors, implicit
solver, numerics, threads); `Options.validate(cfg, g)` holds the init-time checks.

`Sim.init` calls `validate` and returns `error.InvalidConfig` in every build mode. It
checks `g.ng ≥ cfg.ghostCells()` because PPM's `i−2` load would otherwise address a
negative padded index; `phys.coords == cfg.coords` (the runtime metric coords must match
the comptime coords the physics layer reads via `Cfg.coords`); `radvisc` requires a
non-null `opac` (ν = α·mfp needs opacities, otherwise C's SKIPRADSOURCE keeps viscosity
active but the Zig path would silently store ν = 0); `polaraxis` requires
`ny > 2·ncells` and spherical coords (on `.mink`, the overwrite returns early but
`isCellCorrectedPolaraxis` does not, so three evolution passes would skip polar rows
without another pass supplying them); and the MPI consistency of grid, decomposition and
backend. See [§9](#9-boundary-conditions-and-polar-axis-correction). Every allocation in
`init` carries an `errdefer`, so a failed init leaks nothing.

### 5.2 The stage buffers (what each one holds)

The IMEX arithmetic shuffles the conserved vector `u` through several whole-grid snapshot
buffers. Understanding them is the key to reading `step()`:

| buffer | holds |
| --- | --- |
| `u` | the live conserved vector being advanced |
| `p` | the live primitive vector (kept in sync by `calcU2p`) |
| `ut0` | `u` at the **start of the step** (`U^n`); the base every "together" combine builds on |
| `ut1` | `u` just **before the 1st explicit** operator (post-1st-implicit) |
| `ut2` | `u` just **before the 2nd explicit** operator |
| `dut1` | 1st explicit **stage derivative** `F(U^{(1)}) = (u − ut1)/dt` |
| `dut2` | 2nd explicit stage derivative `(u − ut2)/dt` |
| `drt1` | 1st implicit **stage derivative** `(u − ut0)/(dt·γ)`; the radiative-source increment rate |
| `drt2` | 2nd implicit stage derivative `(u − uforget)/(dt·γ)` |
| `uforget` | `u` just **before the 2nd implicit** operator (the base for `drt2`) |
| `u_bak`, `p_bak` | full-grid backups used by `cellFixup` while it averages flagged cells from their neighbours |

(C also keeps `ptm1` and `ppostimplicit` snapshots around the implicit operator; both are
write-only on this path and had no Zig consumer, so they were dropped.)

`stageDeriv(dst, a, b, δ)` computes `dst = (1/δ)·a + (−1/δ)·b`; `stageCombine(dst, a, f1,
b, f2, c)` computes `dst = a + f1·b + f2·c`. Both run over the domain in the exact C
`a*x+b*y` expression shape (so forced-dt tests gate at 1e-13).

### 5.3 `step()`, the exact sequence

Read this alongside `step()` in `sim.zig`. `own_dt = cflDt() = 1/tstepdenmax` is the CFL dt
that the *previous* step accumulated; `dt = forced_dt orelse own_dt`. `step()` first asserts
the denominator was seeded (`forced_dt != null or tstepdenmax > 0`) so a forgotten
`initTimestepGuess` fails immediately instead of stepping with `dt = +inf`; the driver calls
the same `cflDt()` so the two cannot drift (the bound itself, per-dimension speeds summed
over `Δx_d`, is gated against the SR eigenvalues in `tests/timestep_tests.zig`). The
wavespeed accumulators are then reset
(`tstepdenmax = -1`), the once-per-step viscous stress is filled (if `radviscosity`),
`γ = 1 − 1/√2`, and `saveTimesteps()` records each cell's `dt`. Then:

```
step(dt):
  ── 1st implicit ─────────────────────────────────────────────
  ut0  ← u                       (snapshot U^n)
  opImplicit(t, dt·γ)            → advances u,p by the radiative source  [U^(1)]
  drt1 ← (u − ut0)/(dt·γ)        (implicit stage derivative)

  ── 1st explicit ─────────────────────────────────────────────
  ut1  ← u
  calcU2p(t);  doCorrect();  setBc(t,false)
  opExplicit(t, dt)             → writes F(U^(1)) into u via flux divergence + metric source
  if dynamo: applyDynamo(t, dt)
  copyEntropyCount()
  dut1 ← (u − ut1)/dt            (explicit stage derivative)

  ── 1st together ─────────────────────────────────────────────
  u = ut0 + dt·dut1 + dt·(1 − 2γ)·drt1

  ── 2nd implicit ─────────────────────────────────────────────
  uforget ← u
  calcU2p(t)
  doCorrect()
  opImplicit(t, γ·dt)           [U^(2)]
  drt2 ← (u − uforget)/(dt·γ)

  ── 2nd explicit ─────────────────────────────────────────────
  ut2 ← u
  calcU2p(t);  doCorrect();  setBc(t,false)
  opExplicit(t, dt)             → F(U^(2))
  if dynamo: applyDynamo(t, dt)
  dut2 ← (u − ut2)/dt

  ── together ─────────────────────────────────────────────────
  u = ut0 + (dt/2)·(dut1 + dut2)      (explicit average)
  u = u   + (dt/2)·(drt1 + drt2)      (add implicit average)

  ── finalise ─────────────────────────────────────────────────
  calcU2p(t);  doCorrect();  setBc(t,false)
  t += dt;  updateEntropy();  nstep += 1
```

(C also copies `p → ptm1` before each implicit and `p → ppostimplicit` after,
write-only on this path, dropped; see the buffer table above.)

`opImplicit` is a structural no-op when the config has no radiation or `opt.opac == null`
(SKIPRADSOURCE). The copies still happen, but the source solve is skipped.

### 5.4 Inside `opExplicit`, the explicit-operator pipeline

`opExplicit(t, dt)` (C `op_explicit`) is the finite-volume transport update. Its `t`
argument is unused (the metric source terms are cell-local and time-independent). The
pipeline:

```
opExplicit(dt):
  1. calcWavespeeds()        per cell (+1 ghost layer): hydro speeds (ahd*) via
                             wavespeeds.gasWavespeedsLr, τ-limited radiation speeds (arad*)
                             via radiation.calcRadWavespeeds, and the CFL denominator
                             tstepden.  Fills scal[].

  2. sweep(0), sweep(1), sweep(2)   per direction (skipped if nDim ≤ 1):
       for each face:
         load 3- or 5-point primitive stencil along dim (±2 only if base_order>1)
         pick the recon.Scheme from base_order minus any local order reduction
           (boundary reduction; REDUCEORDERATBH inside the BL horizon)
         recon.reconstructN(NV, scheme, stencil, dx5) → left/right face states (pl, pr)
         invert.checkFloorsMhd on each reconstructed state
         flux_mod.fFluxPrime(pl, dim, geom, gam) → ffl        one-sided flux vectors
         flux_mod.fFluxPrime(pr, dim, geom, gam) → ffr
         if radviscosity: rijvisc_mod.addRadViscFlux (sim/rijvisc.zig) adds the M12 shear term
         store pb_r[dim]@i ← pl,  fl_r[dim]@i ← ffl
         store pb_l[dim]@(i+1) ← pr,  fl_l[dim]@(i+1) ← ffr

  3. fluxesAtFaces()         combine one-sided fluxes into the numerical flux flb[dim]:
                             per face, hydro rows (iv < EE) use ahd speeds, radiation rows
                             (iv ≥ EE) use arad speeds, TWO independent hyperbolic systems.
                             LAXF or HLL selected by cfg.flux.
                             uLl/uRl come from p2u of the stored face primitives.

  4. if mhd: ct.fluxCt(...)   Tóth flux-CT correction: rebuild the B-rows of the face
                             fluxes from corner-EMF averages so div(B)=0 to machine
                             precision.

  5. conserved update        per domain cell:
       du = −Σ_d (flb_r − flb_l)·dt/dx_d
       u  = u + du + metricSource(cell)·dt

  6. calcU2p()               invert u → p, run fixups, refresh ghosts.
```

`sweep(dim)` and `fluxesAtFaces` are band-parallel over `cross[0]` and keep the
contiguous x index innermost. For `dim==0`, x is the sweep direction. For `dim!=0`,
x is `cross[0]`, the band range each worker owns. The worker keeps x as its band dimension but
iterates it innermost, with the sweep direction in the middle (`c1 outer → i middle → x inner`),
turning the five-point stencil into contiguous x streams instead of `iy`/`iz` strides. The per-face
work lives in `inline fn sweepFace`/`fluxFace`; `dx5` (function of `(i,dim)`) is hoisted out of the
inner loop. Because every `(i,c0,c1)` writes only its own face slots, the loop order is free and
bit-identical. The per-cell inner loops (`updateRows`, the stage arithmetic, `fluxFace`'s combine)
`load` each accessed cell/face into an `[NV]f64` buffer once and index the buffer per component,
rather than recomputing the flat offset `NV` times.

The **metric source** (`metricSource`, C `f_metric_source_term_arb`, GDETIN==1) only the
Christoffel-contraction rows survive: it builds T^μν via `hydro.calcTij`, lowers one index
via `relele.lowerSecond`, and accumulates `ss[row] += gdet · T^k_l · Γ^l_{ν k}` (and the
analogous `R^k_l` term for radiation rows). This is the geometric momentum/energy source
S_ν = √-g T^k_l Γ^l_{νk}.

### 5.5 Inside `opImplicit`

`opImplicit(t, dtin)` is a no-op unless the layout has `.ee` and `opt.opac != null`.
Otherwise it sets `impl_dt = dtin` and dispatches `implicitRowsWorker` over rows (via
`parallelRange` on the persistent team, [§10](#10-the-threading-model)), each cell calling
`ImplT.solveImplicitLab` (the 6-rung Newton ladder,
[§6](#6-the-solver-stack-p2u-u2p-the-implicit-ladder-fixups)). It sets the
`radimp_fixup` flag (0 ok / −1 fail), tallies iteration/failure counters, then runs
`cellFixup(.radimp_fixup)`.

### 5.6 `calcU2p`, the inversion sweep

`calcU2p` (C `calc_u2p`) runs the per-cell conserved-to-primitive inversion through
`parallelRange(u2pRowsWorker)`. Each cell calls `invert.u2pMhd`, then
`invert_rad.u2pRad` when radiation is active, followed by the floor checks. The
sweep then calls `cellFixup(.hd_fixup)`, `cellFixup(.rad_fixup)` when radiation is active, and
`setBc(time, false)`. All three also run on the team ([§10](#10-the-threading-model)): the
fixup averaging reads the *frozen* pre-pass `p` and writes only `_bak`, and `setBc` fills
disjoint ghost faces. Both stay bit-identical to serial. `cellFixup` first scans the flag
column (`anyFlagSet`) and returns before any copy when nothing is flagged, which is the common
case.

---

## 6. The solver stack: p2u, u2p, the implicit ladder, fixups

### 6.1 Forward map p2u

`koral/p2u.zig`. `p2u(cfg, pp, geom, gamma)` produces the full conserved vector: zero-init,
`p2uMhd`, then `p2uRad` if the layout has radiation.

```zig
pub fn p2u(comptime cfg, pp, geom, gamma) relele.Error![count]f64;
pub fn p2uMhd(comptime cfg, pp, uu, geom, gamma) relele.Error!void;
pub fn p2uRad(comptime cfg, pp, uu, geom) relele.Error!void;   // M1: R^t_μ into EE..EE+3
```

`p2uMhd` fills `√-g·(ρu^t)` into RHO, the energy `√-g·(T^t_t + ρu^t)` into UU, momenta
`√-g·T^t_i` into VX/VY/VZ, entropy `√-g·(s u^t)` into ENTR, and `√-g·B^i` into B1/B2/B3.
The energy slot is assembled **cancellation-free** via `calcUtp1` (which returns `1 + u_t`
without catastrophic cancellation against the rest mass), so `uu[UU]` is C's "energy minus
rest mass". `GDETIN==1` so `gdetu = geom.gdet` everywhere.

### 6.2 The u2p cascade and floors

`koral/solve/invert.zig`. `u2pMhd` is the MHD conserved→primitive cascade:

```zig
pub fn u2pMhd(comptime cfg, uu_in, pp, geom, pgamma, floors: FloorParams) U2pResult;
```

It floors negative conserved density (`ret = -2`), tries the **hot** energy inversion
(`u2pSolverW` with `Etype.hot`), and on failure (`ALLOWENTROPYU2P`) tries the **entropy**
inversion (`ret = -1`); commits primitives on success or restores `ppbak` (`ret = -3`).

`u2pSolverW` is a 1-D Newton solve in `W = wγ²` (the Noble W-solver), with `fU2pHot` /
`fU2pEntropy` giving residual + analytic derivative, `wOutOfBounds` guarding v²<1, a
W-increase init loop, and W-halving damping. Return codes are C's exact golden-compared
values (0, −101..−105, −120, −150).

`checkFloorsMhd` enforces `RHOFLOOR`, the `UURHORATIO` min/max, the B²/ρ and B²/uint
ceilings via the **drift-frame** (Ressler+2017) mass/energy addition, and the γ≤GAMMAMAXHD
velocity ceiling; it re-syncs ENTR from (ρ,uint) if anything floored. `FloorParams` has
`.cdefault` (choices.h defaults) and `.puffy` (define.h overrides: rhofloor 1e-30,
uurhoratio 1e-8..1e0, b2rhoratio/b2uuratio 50, gammamaxhd 10).

**Radiation inversion** lives in `koral/solve/invert_rad.zig`. `u2pRad` is the closed-form
M1 inversion (`m1GammaRel2` → `m1Erf` recovers γ_rel², Ê; reconstruct the relative
4-velocity), with a two-branch **cold fallback** (slow γ≈1, fast γ=γmax; keep whichever
reproduces the original R^t_t) and `checkFloorsRad` for the radiative floors. `RadParams`
has `.cdefault` and `.puffy` presets.

### 6.3 The implicit Newton ladder

`koral/solve/implicit.zig`. `Impl(cfg)` is a comptime factory; `solveImplicitLab` is the
top-level entry called from `opImplicit`. It regenerates `uu00 = p2u(pp00)`
(`FORCEUEQPINIMPLICIT`), picks whether to start from the radiation or MHD primitives based
on `ehat < threshold·u`, and tries a **fixed 6-rung ladder** of 4-primitive Newton solves,
returning on the first success:

```
rung 0:  [startwith, energy,  lab]
rung 1:  [startwith, energy,  ff ]
rung 2:  [swapped,   energy,  lab]
rung 3:  [swapped,   energy,  ff ]
rung 4:  [startwith, entropy, ff ]
rung 5:  [swapped,   entropy, ff ]
```

Each rung is `solve4dPrim`: an optional 1-D bisection warm start (`solve1dPrim`), then a
Newton loop that at each iteration calls `applyConstraints` (the conservation glue that
reconstructs the *other* fluid and inverts it back. This is the most common failure
site. The rung then evaluates `residual`, builds a **one-sided finite-difference Jacobian**
(sign starts −1, flips to +1 on constraint failure), optionally scales it
(`SCALE_JACOBIAN`), inverts it (`invert4`, GSL-style LU with no singular guard), and applies
a damped Newton step under change limiters (energy, gas T, rad T). Two independent success
gates run each iteration: the absolute residual `errbase < conv` at the top, and the
relative-change gate at the bottom. `WhichEq` (energy/entropy), `WhichPrim` (rad/mhd), and
`WhichFrame` (lab/ff) select the residual; lab+entropy is impossible by construction and
asserted out. `ImplicitParams` has `.cdefault` and `.puffy` presets plus four PUFFY
switches (`start_with_bisect`, `scale_jacobian`, `allow_rad_ceiling`,
`allow_entr_in_4dprim`).

Two later additions: `ImplicitParams.simd_jacobian` (default **on**) batches the four
perturbed residual evaluations of the FD Jacobian through the `@Vector(4, f64)` twin
`residualG`, gated bit-for-bit against the scalar path by `koral/tests/simd_tests.zig`, and
timed by the `bench-implicit` build step. And the whole 6-rung ladder can be wrapped in
an outer **opacity-damping loop** (`opdamp_maxlevels`/`opdamp_factor`, C
`OPDAMPINIMPLICIT`): each level retries the ladder with the four-force scaled by
`opdamp_factor^-level`; level 0 (the default `opdamp_maxlevels = 0`) is bit-identical to
the plain single pass.

### 6.4 Fixups

`cellFixup(which)` (C `cell_fixup`) averages cells flagged by the inversion/implicit passes
(`hd_fixup`, `rad_fixup`, `radimp_fixup`) from their non-flagged in-domain neighbours, using
the whole-grid `u_bak`/`p_bak` backups. `hd_fixup` never averages ρ or B; `rad_fixup`
averages only the radiation slots; `radimp_fixup` averages both fluids but never ρ/B. The
"enough neighbours" threshold scales with dimensionality (1D≥1, 2D≥2, 3D≥3). Corrected
polar-axis cells are skipped. It first scans the flag column (`anyFlagSet`) and returns
immediately when nothing carries `which`. This common case skips the four full-grid
`u`↔`u_bak`/`p`↔`p_bak` copies that would otherwise run every call.

---

## 7. Reconstruction and Riemann flux

`koral/fv/recon.zig`, `koral/physics/wavespeeds.zig`, `koral/physics/flux.zig`,
`koral/fv/laxf.zig`.

**Reconstruction** turns cell averages into left/right face states. The scheme is a
tagged union `recon.Scheme = { donor, linear{theta}, ppm }` with `Scheme.radius()`
(0/1/2) giving the stencil depth; `reconstruct(comptime T, scheme, u: [5]T, dx: [5]T)`
produces the face pair for one variable and `reconstructN(NV, ...)` loops it over the
state vector. The sweep picks the scheme per face from `base_order` minus any local
order reduction (boundary reduction, and `REDUCEORDERATBH` inside the BL horizon,
PPM → linear → donor). Order 1 is minmod-θ (PUFFY θ=1.5); order 2 is PPM (Colella &
Woodward, non-uniform widths); order 0 is donor cell. Both linear and PPM silently fall
back to donor cell at local extrema, so reconstruction order is not globally uniform.
The kernel is `comptime T`-generic (`f64` or `@Vector(W, f64)`, limiter branches via
`simd.select`) and lane-for-lane bit-identical to scalar.

**Wavespeeds.** `lrCore(ucon, geom, wspeed2s, dims)` is the shared HARM characteristic-speed
core (used by *both* gas and radiation): given a fluid-frame wavespeed² per direction, it
solves the boosted characteristic quadratic for lab-frame dxⁱ/dt. `gasWspeed2` gives the
relativistic magnetosonic speed² `cs² + vA² − cs²·vA²` (cs² ceiling 0.95, vA² floored at 0).
`gasWavespeedsLr` builds `ucon`, computes bsq, calls `lrCore`, and clamps co-going speeds
(left ≤ 0 ≤ right). Radiation speeds come from `radiation.calcRadWavespeeds`, which calls
the same `lrCore` with the M1 rest-frame v²=1/3 (unlimited, for the timestep) and a
τ-limited v² (for the fluxes).

**Flux.** `fFluxPrime(cfg, pp, idim, geom, gamma)` is the advective flux of *all* conserved
rows across a face: density/entropy advection, the cancellation-free energy row (same
`calcUtp1` assembly as p2u), the momentum rows (`T^{idim}_i` via `hydro.calcTij` +
`lowerSecond`), the antisymmetric induction fluxes `b^k u^d − b^d u^k`, and the pure-M1
radiation rows `R^{idim}_ν`. Every row is multiplied by `gdetu = geom.gdet`. The PUFFY
radiative shear-viscosity term is added separately at faces in M12.

**Riemann combination.** `laxf(fl, fr, ul, ur, ag)` is Lax-Friedrichs `½(fr + fl − ag(ur −
ul))`; `hll(fl, fr, ul, ur, al, ar)` is the two-wave HLL solver. `fluxesAtFaces` selects
per row: hydro rows use `ag = max` cell speeds; radiation rows (`iv ≥ index(.ee)`) use the
`arad` speeds. Hydro and radiation are treated as two independent hyperbolic systems and
must never share a wavespeed.

---

## 8. MHD specifics and the extra physics in the step

`koral/sim/ct.zig`, `koral/sim/dynamo.zig`, `koral/physics/radvisc.zig` (pure
radviscosity kernels), and `koral/sim/rijvisc.zig` (the sim-coupled radviscosity
gather + per-step pass).

### 8.1 Constrained transport (flux-CT)

`ct.fluxCt(SimT, sim)` applies Tóth flux-CT after the Riemann fluxes are computed: it
builds corner EMFs by averaging the B-rows of the face fluxes `flb[0/1/2]`, stores them,
then rebuilds those B-rows from 0.5-averages of the neighbouring corner EMFs, enforcing
discrete div(B)=0. The face-normal B self-flux is explicitly zeroed. Requires `GDETIN==1`.

### 8.2 Vector potential → B

`ct.calcBfromA(SimT, sim, ifoverwrite)` converts a cell-centered vector potential A
(carried transiently in the b1..b3 primitive slots) into a divergence-free B:
`cornerAverageA` interpolates A to corners, `calcBfromACore` takes the discrete curl
`B = curl(A)/√-g`, and if `ifoverwrite` writes B back into `p` and recomputes `u` via p2u.
Ghost cells get stale scratch, so the caller must `setBc` afterward. This is used at
initialisation and after the dynamo. `curlFromA` is the reusable (average + curl) entry the
dynamo calls.

### 8.3 Radiative shear viscosity, filled once/step, added at faces

`calcRijViscTotal(sim, dt)` (`koral/sim/rijvisc.zig`) runs **once per step** (from
`step()`, before the RK stages) with `global_dt = this step's dt`, populating `sim.rijvisc`
(the viscous R^i_j over domain + one ghost ring). The stress is `R^{ij}_visc = −2ν Ê σ^{ij}`,
with the shear tensor σ from the radiation-frame velocity field (`calcShearLab` gathers the
FD gradients, the pure `shearFromGradients` builds σ) and the viscosity coefficient
`ν = α·mfp` (the pure `viscCoeff`: mfp = 1/χ, capped by the local BL radius and by the max
stable explicit diffusion coefficient `mindx²/(2·global_dt·2)`). During each direction's
flux sweep, `addRadViscFlux` face-averages the stored R^i_j and hands it to the pure
`addViscFlux`, which damps it by the characteristic viscous velocity (`MAXRADVISCVEL = 0.1`)
and adds `gdet·dampfac·R^i_j` into the M1 radiation flux rows. The pure kernels live in
`koral/physics/radvisc.zig`.

### 8.4 The mimic dynamo, run after each explicit sub-step

`dynamo.applyDynamo(SimT, sim, t, dt)` runs after *each* explicit sub-step (see the two
`applyDynamo` calls in `step()`). The sequence is `calcScaleHeight` (density-weighted RMS
angular scale height per radius, into `sim.scaleth`) → `setBc` → `mimicDynamo` → `calcU2p`.
`mimicDynamo` builds a toroidal δA_φ (an α-Ω prescription gated by field pitch angle,
plasma β, radius, and scale height, with `ALPHAFLIPSSIGN` making α antisymmetric about the
midplane), applies `DAMPBETA` azimuthal-field damping to `p[b3]`, curls δA_φ into a poloidal
B via `curlFromA`, superimposes it on the domain, and recomputes `u` via p2u. `Params`
carries the PUFFY dynamo tunables.

---

## 9. Boundary conditions and polar-axis correction

`setBc(t, ifinit)` (C `set_bc`) fills the x/y/z ghost columns (no corners) via `setBcCell`
for each ghost depth. Per-axis behaviour is `BcKind`:

- **`.periodic`**, wraps the index (plus the C quirk: if `NY < NG` pin the index to 0).
- **`.copy`**, clamps to the domain edge (outflow copy).
- **`.specific`.** Carries the user-supplied `SpecificBc` function pointer and its
  opaque context (PUFFY uses this for both radial and both polar faces); the callback
  receives `*const CoreT`, so it sees the grid, the cache, `p` and the physics options
  and nothing else of the Sim. The callback is **fallible**
  (`relele.Error![NV]f64`): a boundary-adjacent cell can transiently reach a spacelike
  velocity in a frame conversion, and `setBcCell` propagates that with `try` rather than
  letting the BC swallow it (`catch unreachable` → panic in safe builds, UB in ReleaseFast).

After filling a ghost cell's primitives, `setBcCell` runs p2u to derive its conserveds.
If `wide` (MHD), `setBc` fills the ghost corners: `fillCorners2d` for the 2D x-y case
(`ny>1, nz==1`) and `fillCorners3d` for 3D (`ny>1, nz>1`), copying one-cell surfaces
from adjacent domain rows/columns and averaging diagonal cells. The only unimplemented
case is the x-z 2D layout (`ny==1, nz>1`), which `@panic`s (no target needs it). All of
this lives in `koral/sim/bc.zig`, re-exported by `sim.zig`.

The diagonal average is C's rule (`finite.c:3203`), not an interpolation: a corner
ghost is the mean of two ghosts at *different* radii, so for any field that varies
in `r` the four domain-corner cells never see a consistent neighbour. On a monopole
`B^r ∝ 1/r²` this seeds a small `B^θ` at the corners that the edge rows then carry a
few cells inward; the magnetized-Bondi gate measures its field diagnostics on a fixed
physical interior region for exactly this reason. It is a boundary property, not a
solver one, and PUFFY inherits it unchanged.

**Polar-axis correction.** Lives in `koral/sim/polaraxis.zig`, fronted by `sim.zig`'s
`doCorrect` and `isCellCorrectedPolaraxis`. `polaraxis.correct` overwrites the
`nccorrectpolar` most-polar rows at each θ-edge from a source row, scaling the θ-velocity
components by `fac = |θ − θ_axis|/|θ_src − θ_axis|` (so they ramp to zero at the pole)
while copying other scalars/velocities verbatim; p2u at the *target* geometry rewrites the
conserveds. B is untouched.

Two practical facts about the grid the band sits on. The grid must never put a face on
the axis itself, `g^φφ = 1/sin²θ` diverges there; PUFFY's `MINY = 0.001` (in MKS2 `x²`)
is that offset, and the band and the reflection BC act on the *first face*, not on
θ = 0. And the physics anchor for the whole arrangement is the 2D Bondi/Michel flow in
`koral/tests/ks_evolution_tests.zig`: driven through the band and the stock reflection
on a full-θ KS grid, the drift converges at second order and the profile stays θ-uniform
to roundoff (the Christoffel-trace correction makes the discrete θ-balance of a uniform
pressure exact).

The overwrite and predicate form one contract. Corrected polar cells are
special-cased in four places, `u2pRows` (B-only inversion), `cellFixup` (skipped),
`implicitRowsWorker` (skipped), and the overwrite itself, which is what supplies the
values the other three decline to compute. Both halves derive their row set from a single
`polaraxis.band()`, which returns null (treatment inactive) unless `num.polaraxis`
is set *and* the coords are spherical; `Options.validate` rejects the two
configs that would make the contract unsatisfiable. The polar-axis EMF zeroing
(`adjust_fluxcttoth_emfs`) belongs in this module when it is transcribed, even though C
keeps it in `magn.c`.

The PUFFY driver's `specific_bc` (fragments in `koral/problems/common/bcs.zig`) implements: xhi outflow with radial rescaling and a
no-inflow clamp; xlo plain copy (no inflow check because `RMIN = 1.85 < r_horizon = 2`, so
material only leaves the grid); and ylo/yhi polar reflection with VY/B2/FY sign flips.
The z faces are `unreachable` (TNZ=1).

---

## 10. The threading model

Threading is a persistent **worker team** (root-level `threading.zig::Team`), created once in
`Sim.init` and owned for the run. `Team.init(allocator, nthreads)` spawns `nthreads−1` helper OS threads that
park on a hand-rolled futex. Zig 0.16 moved `Mutex` and `Condition` behind the
`std.Io` event loop, so the team uses the primitive that `std.Io.Threaded` uses internally.
the main thread participates in every region. `nthreads ≤ 1` leaves `team = null` and every
dispatch runs inline as the bit-identical serial path. Spawn failure yields a narrower team
rather than an error.

`parallelRange(Ctx, ctx, team, lo, hi, worker)` splits one region `[lo, hi)` into
about eight tiles per worker. Workers claim contiguous tiles through one atomic ticket counter.
The implicit solver's per-cell cost varies by orders of magnitude between the disk body and the
polar funnel, so this balances expensive cells better than a static split. Each worker accumulates a stack-local `ChunkResult`
(`{ err, n_fail, n_iters, tsd_max, tsd_min }`) across its tiles; after the region these merge
order-insensitively, integer counters sum, `tsd_max`/`tsd_min` reduce by max/min, `err` takes the
first. The merged result does not depend on how tiles fell to workers.

Effectively the whole step runs on the team: `calcWavespeeds`, the three `sweep`s and
`fluxesAtFaces`, the conserved update (`updateRows`), the RK stage arithmetic
(`stageDeriv`/`stageCombine`, with `parallelCopy` for whole-field copies and `parallelZero` for the
flux clears), the `u2p` inversion (`u2pRows`), the implicit source solve (`implicitRows`), the
`cellFixup` neighbour-averaging (flags frozen during the pass; reads the pre-pass `p`, writes only
the `_bak` slots), the boundary fills (`setBc`), and the polar-axis correction (`doCorrect`). Every
one writes only its own cell/row/band's output slots (disjoint), and the only cross-tile reductions
are the order-insensitive `ChunkResult` merges. The parallel result is bit-identical to
serial at any thread count**. That bit-identity is what lets the goldens be diffed bit-for-bit
regardless of `nthreads`.

`threading_tests.zig` pins this bit-identity across thread counts.

---

## 11. Testing and the oracle

koral-zig's correctness rests on four complementary layers, and, importantly, there are
**no end-to-end run-to-completion tests**. Confidence in the full production run comes from
the t=0 init keystone plus per-function goldens plus a reduced-grid step test plus the
threading bit-identity, not from a single long integration.

### 11.1 The test layers

1. **Theory gates** (`*_tests.zig`), mathematical identities and documented quirks, no
   golden data: conservation, symmetry, the M1 closure trace `R^μ_μ = 0`, round-trips
   (e.g. `sFromU`/`uFromS` to 1e-12), reconstruction convergence orders, IMEX L-stability,
   div(B), floor properties, dynamo saturation, and threading bit-identity; analytic
   solutions evolved through the full step: Sod, acoustic and Alfvén waves, Bondi/Michel
   in 1D and, through the polar band on a KS grid, in 2D (`ks_evolution_tests.zig`, with
   the Gammie et al. 2003 magnetized-Bondi monopole for flux-CT in curved spacetime); and
   the CFL bound against the SR hydro eigenvalues (`timestep_tests.zig`).
2. **Function-level C goldens** (`*_golden_tests.zig` reading `.kgld` files), diff Zig
   against C at recorded input points. State/flux/rad records embed **C's own geometry**
   (`geomFromRecord`) so only the state *algebra* is compared, keeping gates tight (1e-13
   for closed-form, 1e-8 for iterative solvers).
3. **Forced-dt step tests** (`step_golden_tests`, `puffystep_golden_tests` reading `.kstp`),
   the Zig side loads C's post-init state bit-for-bit, forces C's recorded dt sequence, and
   diffs the whole domain each step against a growing budget. This isolates step arithmetic
   from init-quadrature discrepancies and from CFL choices. The PUFFY t=0 keystone
   (`puffy_golden_tests` reading `.kini.gz`) is the cell-by-cell init comparison
   (full-grid only under `-Dslow-tests`).
4. **Self-goldens** (`selfgolden_tests.zig` reading `.kslf` from `tests/selfgolden/`),
   the assembled PUFFY pipeline re-run and compared against a baseline **this repository
   generated**, not the C oracle. Layers 2-3 answer *"is the transcription faithful?"*;
   this one answers *"did our own numbers move?"* The theory gates cannot
   ask, since they check identities and known solutions rather than the composition of
   init → CFL → RK2IMEX → BCs → fixups. It exists because layers 2-3 are committed but
   not *regenerable* once the Zig side deliberately stops matching koral_lite:
   `tools/gen_golden.sh` then describes code that no longer exists, and the self-goldens
   are what remains of end-to-end coverage.

   Two granularities per scenario, both including ghosts: per-step **scalars** (t, dt, the
   CFL denominators, the implicit counters, seven f64, so every step keeps them and they
   localize *when* a run diverged) and full **field snapshots** at the endpoints only
   (~350 KiB gzipped each, and a perturbation at step k is still present at step n). The
   gate is **bit identity**, the same standard the threading determinism test holds to,
   a tolerance gate would let a real sub-1e-12 change through, which is exactly what a
   regression baseline exists to catch. The 1e-12 threshold is a *diagnosis* line only,
   separating a reported `REGRESSION` from a bit-level `DRIFT`.

   `koral/testing/selfscenarios.zig` defines the scenarios **once** and is shared by the
   checker and by `zig build update-self-goldens`, so a baseline cannot describe something
   the test does not re-run. The generator builds against the same `koral` module at the
   same `-Doptimize` as the test artifact, which is what makes bit identity achievable;
   regenerate with plain defaults. Regenerating is a deliberate act, the committed files
   are the record of what this code used to compute.

All golden readers return `error.SkipZigTest` when the file is absent, so the suite is
green on a machine that never ran the oracle. `DevTracker` accumulates the worst normalised
deviation `dev = |c−z|/max(1, |c|, |z|)` per quantity class and reports the observed max.

Registration of test files in `koral.zig`'s `test {}` block is no longer purely by
convention: `build.zig` scans `koral/tests/` (theory gates) and `koral/tests/golden/`
(C-oracle goldens) at configure time and fails with `error.UnregisteredTestFile` if any
is missing an `@import`. The suite now spans ~30
files, beyond the ones discussed above it includes `sim_tests.zig` (`Sim.init`
precondition rejection), `restart_tests.zig` (KDMP round-trip), `simd_tests.zig`
(scalar↔SIMD bit-identity), `dynamo_tests.zig`/`radvisc_tests.zig` (M12 dynamo-law and
shear-algebra invariants), `scalars_tests.zig`, `polaraxis_tests.zig`, and golden suites
per subsystem (`metric`, `state`, `flux`, `rad`, `opac`, `implicit`, `visc`, `dynamo`,
`step`, `puffy`).

### 11.2 The file formats

- **KGLD**, function-level golden: header `"KGLD" | version | nrec | nin | nout`, then
  `nrec × (nin + nout)` little-endian f64. Each record is `{in, out}`.
- **KSTP**, forced-dt step file: header `"KSTP" | version | nx,ny,nz,nv | nrec`; per
  record `t, dt, u[], p[], flags[]` (iv fastest), `nflags` (2 or 4) inferred from length.
  Record 0 is the post-init state.
- **KINI**, PUFFY t=0 keystone grid snapshot (always gzipped): header with ghost layers,
  stride, and which primitive slots are sampled; data iz-slowest, vars-fastest.

### 11.3 The oracle and `gen_golden.sh`

The oracle harnesses (`oracle/harness_*.c`) **are** koral_lite: `tools/gen_golden.sh`
compiles the unmodified `../koral_lite` sources and links each harness against them, so the
goldens are literally the reference C output. `harness_step.c` and `harness_puffy_step.c`
transcribe the RK2IMEX stage block **verbatim** from `problem.c:141-402` (serial path);
`harness_init.c` transcribes the `ko.c:140-263` init order. The script copies koral_lite
into a scratch tree (never mutating the original), overlays the koral-zig-specific C test
problems (`ZIGSOD/ZIGOT/ZIGMHDTUBE/ZIGRADTUBE/ZIGRADPULSE`, PROBLEM 200-204), registers them
in `problem.h`, and compiles serial clang + GSL objects. A `manifest.json` records the
koral_lite SHA, problem list, compiler, and per-file schema.

Two attribution details worth knowing: every RADIATION harness calls
`gsl_set_error_handler_off()` so singular FD Jacobians yield a silent inf (matching Zig's LU
with the handler off) instead of aborting; and a `puffyeps12` variant tightens C's
quadrature `epsrel` from 1e-8 to 1e-12 to prove that the ~1e-3 PUFFY torus keystone
deviation is *C's* `gsl_qags` error near the ℓ(λ) kink, not a Zig bug. Those field-scale
deviations are expected rather than regressions. A real bug lands far above that floor.

---

## 12. Where things live, navigation quick-reference

| Subsystem | Primary files |
| --- | --- |
| **Foundation** (config, layout, storage, units, params, geometry, comm) | `koral/config.zig`, `koral/layout.zig`, `koral/grid.zig`, `koral/field.zig`, `koral/units.zig`, `koral/params.zig`, `koral/geometry.zig`, `koral/koral.zig`, `koral/comm/comm.zig` (Serial vs MPI), `koral/comm/serial.zig`, `koral/comm/mpi/` |
| **Math utilities** | `koral/math/dual.zig`, `koral/math/linalg.zig`, `koral/math/simd.zig`, `koral/math/misc.zig`, `koral/math/quad.zig` |
| **Metric / coordinates** | `koral/metric/forms.zig`, `koral/metric/metric.zig`, `koral/metric/coco.zig`, `koral/metric/precompute.zig` |
| **Velocity / index / boost algebra** | `koral/relele.zig`, `koral/frames.zig` |
| **Primitive↔conserved (forward + inverse)** | `koral/p2u.zig`, `koral/solve/invert.zig` |
| **Gas thermodynamics + MHD stress-energy** | `koral/physics/hydro.zig`, `koral/physics/bfield.zig` (exported as `koral.physics.mhd`) |
| **M1 radiation core + inversion** | `koral/physics/radiation.zig`, `koral/solve/invert_rad.zig` |
| **Opacities / thermodynamics / four-force** | `koral/physics/thermo.zig`, `koral/physics/opacities.zig`, `koral/physics/mesa.zig`, `koral/physics/radforce.zig`, `koral/units.zig` |
| **Implicit radiation-gas source solver** | `koral/solve/implicit.zig` |
| **Reconstruction + wavespeeds + flux + Riemann** | `koral/fv/recon.zig`, `koral/physics/wavespeeds.zig`, `koral/physics/flux.zig`, `koral/fv/laxf.zig` |
| **Run driver + problem contract** | `koral/driver.zig` (`run(App, init)`), `koral/problems/puffy/main.zig` (the `App`) |
| **Evolution driver** | `koral/sim.zig` (composition root + operators), `koral/sim/core.zig` (`Core`: the state every pass sees), `koral/sim/options.zig` (`Options`, `applyParams`, `validate`), `koral/sim/rk2imex.zig` (integrator + stage buffers), `koral/sim/{timestep,u2p,fixup,explicit,implicit_op,entropy,stage}.zig` (the passes), `koral/sim/storage.zig`, `koral/sim/bc.zig`, `koral/sim/polaraxis.zig`, `koral/sim/timers.zig`, `koral/threading.zig` |
| **Constrained transport + dynamo + radiative viscosity** | `koral/sim/ct.zig`, `koral/sim/dynamo.zig`, `koral/physics/radvisc.zig` (pure kernels), `koral/sim/rijvisc.zig` (gather + per-step pass) |
| **PUFFY problem (Physics, IC, presets)** | `koral/problems/puffy/puffy.zig` (`Physics`, `setup`, `initAllWith`, `Bc` dispatch), `koral/problems/puffy/*.toml` |
| **Shared torus-problem library** | `koral/problems/common/limotorus.zig` (tools.c solver, `Params`), `atmosphere.zig` (floor atmospheres, LTE split), `magnetize.zig` (β normalization), `bcs.zig` (bc.c fragments), `mri.zig` (seed quality), `perturb.zig`, `koral/math/quad.zig`, `koral/math/misc.zig` (`rtbis`) |
| **Diagnostics + output** | `koral/io/scalars.zig`, `koral/io/dump.zig`, `koral/io/silo.zig` (+ `silo_disabled.zig` stub) |
| **Build + oracle + goldens + tools** | `build.zig`, `tools/gen_golden.sh`, `tools/{res2kdmp,kdmp2silo,qmri,kdmp2png,kdmp2lc,goldtest,mpi_gates,bench_implicit}.zig`, `koral/testing/golden.zig`, `koral/testing/tubes.zig`, `oracle/harness_*.c`, `tests/golden/**` |
| **Self-goldens (Zig-generated baseline)** | `koral/testing/selfgolden.zig`, `koral/testing/selfscenarios.zig`, `koral/tests/selfgolden_tests.zig`, `tools/gen_self_golden.zig`, `tests/selfgolden/**` |

### Starting points for common tasks

- **Understand a step** → `koral/sim.zig::step` ([§5](#5-the-anatomy-of-one-time-step)).
- **Add a physics module** → append a `Module` in canonical order, add its `moduleTags`
  slice and `VarTag`s in `layout.zig`; NV and all indices regenerate at comptime.
- **Add a runtime tunable** → add a field with a default to `Params` (parser already
  handles f64/usize/bool/string). For a PUFFY/AGN knob, also fold it in
  `Physics.fromParams`.
- **Add a reconstruction / flux / integrator / coordinate system** → extend the relevant
  `Config` enum; kernels dispatch on it at comptime.
- **Change floors or implicit tolerances** → the `FloorParams` / `RadParams` /
  `ImplicitParams` presets in `solve/invert*.zig` and `solve/implicit.zig`; most are also
  runtime-overridable from the params file via `Physics.fromParams` (the optional-override
  group in §3.6).
- **Instantiate a new problem** → create `koral/problems/<name>/` with a value-typed
  physics bundle (see `puffy.Physics`), init/BC hooks composed from `problems/common/`,
  and a `main.zig` whose `App` satisfies the `koral/driver.zig` contract and whose `main`
  is `koral.driver.run(App, init)`; `build.zig` auto-discovers it (USER_GUIDE §7).
- **Regenerate goldens after a koral_lite change** → run `tools/gen_golden.sh` and commit
  `tests/golden/**` + `manifest.json` ([§11](#11-testing-and-the-oracle)).
- **Deliberately change the numerics** → review what the self-golden reports moved, then
  `zig build update-self-goldens` and commit `tests/selfgolden/**` ([§11](#11-testing-and-the-oracle)).
