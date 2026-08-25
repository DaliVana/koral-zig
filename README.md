# koral-zig

A Zig reimplementation of [KORAL](https://github.com/achael/koral_lite), the
general-relativistic radiation-magnetohydrodynamics (GR-RMHD) code. This port
focuses on the PUFFY radiation-supported torus around a 10 M☉ Schwarzschild black
hole from Lančová et al. (2019).

![Zig 0.16.0](https://img.shields.io/badge/Zig-0.16.0-f7a41d)
![status](https://img.shields.io/badge/status-2D%20%26%203D%20validated-brightgreen)
![license](https://img.shields.io/badge/license-GPLv3-blue)

The implementation preserves KORAL's variable order and floating-point expression
shape. Its comparison tests can therefore check individual cells against the C
reference at machine precision, including deliberate C quirks. Zig adds memory
safety and a small dependency footprint without hiding the numerical lineage.

- **What it solves.** The GRMHD conservation laws coupled to M1 radiation
  moments on a black-hole spacetime in modified Kerr-Schild (MKS2) coordinates.
  The model includes an implicit radiation-gas four-force, bremsstrahlung,
  synchrotron, Klein-Nishina scattering, Comptonization, radiative shear
  viscosity, constrained transport, and a mean-field dynamo.
- **How it runs.** Single binary per problem, one TOML config, no runtime
  dependencies beyond the Zig standard library. Serial by default with **opt-in
  shared-memory threading**; opt-in **φ-only MPI** (`-Dmpi`) splits a 3D wedge
  across ranks. Runs on a laptop, a workstation node, or a multi-node φ-ring.
- **How it's checked.** Analytic tests and C-generated golden files cover the
  physics kernels and assembled step. The 2D and 3D PUFFY comparison cases match
  the C reference bit for bit. See [Testing](#testing).

The [documentation index](docs/README.md) points to separate guides for running the
code, understanding its structure, and checking the physics.

---

## Quick start

### 1. Install Zig 0.16.0

koral-zig requires **exactly Zig 0.16.0** (it uses the 0.16 `std.Io` API; earlier
releases will not compile it). Pick one:

**Official binary.** This is the safest way to pin the required version.
Download the `0.16.0` archive for your OS/arch from
**[ziglang.org/download](https://ziglang.org/download/)**, unpack it, and put the
folder on your `PATH`. For example, on Apple Silicon macOS:

```sh
# grab the exact filename shown on the download page for your platform
curl -LO https://ziglang.org/download/0.16.0/zig-aarch64-macos-0.16.0.tar.xz
tar xf zig-aarch64-macos-0.16.0.tar.xz
export PATH="$PWD/zig-aarch64-macos-0.16.0:$PATH"   # add to your shell profile to persist
```

Linux (`zig-x86_64-linux-0.16.0.tar.xz`) and Windows (`zig-x86_64-windows-0.16.0.zip`)
work the same way.

**Version manager.** Use this if you switch between Zig versions:

```sh
# zvm, https://github.com/tristanisham/zvm
zvm install 0.16.0 && zvm use 0.16.0
```

**Package manager.** A command such as `brew install zig` works only when the
package is Zig 0.16.0.

Verify:

```sh
zig version      # must print 0.16.0
```

### 2. Get the code

```sh
git clone https://github.com/<user>/koral-zig.git
cd koral-zig
```

### 3. Build and test

```sh
zig build test      # run the library test battery (fast; no external deps)
zig build           # build every problem executable into zig-out/bin/
```

### 4. Run the PUFFY simulation

The default `puffy.toml` runs a short, laptop-sized 2D slice. Use a release build
for anything beyond a smoke test:

```sh
# build & run in one step (recommended optimization for real runs)
zig build run-puffy -Doptimize=ReleaseFast -- koral/problems/puffy/puffy.toml

# ...or run the installed binary directly
./zig-out/bin/puffy koral/problems/puffy/puffy.toml
```

On startup it echoes the derived configuration, then prints a throttled progress
"heartbeat" and writes output frames into the run's `out_dir` (default
`dumps/puffy/`):

```
puffy: NV=13 grid 384×360×1 (+3 ghosts) | M=10 M☉ (GM/c³=4.9e-05s) | threads=12
puffy: init done — β-normalization fac = …
puffy: t=… nstep=… dt=… | Ṁ=… L=… H/R=… β⁻¹=… | nan=0 hdfix=… radimpfail=…
puffy: done (t=…, … steps)
```

Every quantity is in **geometrized/code units (GU)**, matching KORAL's convention;
convert to CGS/Eddington in downstream analysis. To run longer or larger, edit
`tmax`, `nstep_max`, `nx`, `ny`, and `nthreads` in the TOML.

---

## Configuration

Runtime parameters live in a small flat-TOML file passed on the command line.
Ready-made configs for the PUFFY problem are in
[`koral/problems/puffy/`](koral/problems/puffy/):

| File | Run |
|---|---|
| [`puffy.toml`](koral/problems/puffy/puffy.toml) | 2D axisymmetric (384×360), the validated production slice |
| [`puffy3d.toml`](koral/problems/puffy/puffy3d.toml) | 3D wedge (`nz > 1`, periodic φ), workstation-watchable |
| `puffy3d_sgra*.toml` | Sgr A*-flavored 3D variants |

Common knobs: `mass`, `bhspin`, `gam` (physics); `nx`/`ny`/`nz` (resolution);
`tmax`, `nstep_max`, `tsteplim` (run control); `dtout1`/`dtout2`/`nout_step`
(output cadence); `nthreads` (1 = serial, >1 = shared-memory worker team);
floors/ceilings. The full field-by-field reference is in the
[User Guide](docs/USER_GUIDE.md#3-the-params-toml-file).

Anything that changes generated code, such as the module set, reconstruction
scheme, or coordinate system, belongs in the problem's compile-time `Config`.
See the guide's [new-problem recipe](docs/USER_GUIDE.md#7-recipe-creating-a-new-problem).

---

## Output and visualization

Written into the config's `out_dir`:

- **`scalars.dat`.** A whitespace-separated time series of global diagnostics
  (mass, accretion rate Ṁ, radiative/total luminosity, scale height H/R,
  peak magnetization, fixup/NaN counters). One row per output frame.
- **`prims#####.kdmp`.** Little-endian snapshots of every primitive, written on
  each `dtout1` or `nout_step` output frame. Each file is a restart checkpoint.
  `dtout2` is reserved and does not control these files.
- **`.silo`** *(optional).* VisIt-readable field dumps, enabled with
  `-Dsilo`. Silo is compiled from source (LLNL Silo 4.12, PDB driver, no HDF5)
  and linked statically through the sibling [`silo-zig`](../silo-zig) wrapper.
  VisIt and a system Silo installation are not build dependencies.

See the [User Guide](docs/USER_GUIDE.md#4-output-formats) for exact byte layouts
and column definitions.

### Restarting a run

Every output frame writes a checkpoint, so a run is always resumable:

```sh
# continue from a specific checkpoint, or from the newest one in a directory
./zig-out/bin/puffy koral/problems/puffy/puffy.toml --restart dumps/puffy
```

A C KORAL serial restart (`res####.head` + `res####.dat`) can be bridged into a
KDMP checkpoint with `zig build res2kdmp -- <path/res####.head> [out.kdmp]`.

---

## Documentation

The [`docs/`](docs/README.md) directory has focused guides for each kind of work:

| Document | For | Covers |
|---|---|---|
| **[User & Developer Guide](docs/USER_GUIDE.md)** | anyone building, running, or modifying it | prerequisites, build/run, the TOML reference, output formats, running tests, and recipes for adding a new problem or changing the physics |
| **[Architecture](docs/ARCHITECTURE.md)** | programmers getting into the code | design philosophy and the C-diffability contract, the module graph, the core data model, and the anatomy of one time step |
| **[Physics & Numerics](docs/PHYSICS.md)** | physicists validating or extending the science | the governing equations, metric/coordinates, stress tensors, opacities, viscosity, dynamo, the numerical scheme, and the PUFFY initial conditions |

The [validation log](docs/MILESTONES.md) records each milestone (M0-M14) and how it
was checked against C.

---

## Testing

Long turbulent runs diverge after tiny floating-point changes, so the test suite
checks smaller deterministic pieces:

- **Theory tests.** Pure Zig unit and property tests cover analytic identities
  (`g·G = δ`, Christoffel symmetry, p2u↔u2p round-trips, M1 closure invariants,
  `E = aT⁴`, conservation in the implicit solve, …) and known solutions
  (Sod tube, Bondi/Michel stationarity, radiative-shock battery).
- **C-comparison goldens.** Small C harnesses in
  [`oracle/`](oracle/) (compiled against the C KORAL) plus short forced-`dt` step
  comparisons on tiny grids, committed under `tests/golden/`. The 2D and 3D PUFFY
  problems match C bit-for-bit.
- **Self-goldens.** The assembled PUFFY pipeline (init → CFL → RK2IMEX → BCs →
  fixups) run for a few steps and pinned against a baseline this repository
  generated itself, under `tests/selfgolden/`. These detect movement in the
  composed Zig pipeline. They do not establish physical correctness; the theory
  and C-comparison tests do that. Self-goldens remain useful if the Zig and C
  implementations intentionally stop matching because they are the
  end-to-end net that survives if the Zig side ever stops matching koral_lite
  bit-for-bit, since at that point the C baseline can no longer be regenerated.

```sh
zig build test                    # the fast battery
zig build test -Dslow-tests       # + convergence studies, soaks, full-grid PUFFY t=0 keystone
zig build update-self-goldens     # rewrite the self-golden baseline; review first
```

Regenerating the C goldens needs `clang`, GSL (`brew install gsl`), and a sibling
checkout of [koral_lite](https://github.com/achael/koral_lite); the committed goldens
mean you only need the Zig toolchain to build, run, and test. The self-goldens need
nothing but the Zig toolchain, but regenerate them only after reviewing what moved:
the committed files *are* the record of what this code used to compute, and rewriting
them without looking discards it. Details in the
[User Guide](docs/USER_GUIDE.md#5-running-the-tests) and the
[validation log](docs/MILESTONES.md).

---

## Status

All planned milestones (M0-M14) are complete: the metric layer, frames/velocities,
p2u/u2p inversions with floors, reconstruction + wavespeeds + fluxes, hydro/MHD
evolution with constrained transport, M1 radiation, opacities + four-force, the
implicit radiation-gas solver, the full RK2-IMEX pipeline, the PUFFY problem
(2D **and** 3D, both validated bit-for-bit against C), the dynamo + radiative
viscosity + Comptonization, scalar diagnostics, restart checkpoints, and opt-in
threading.

**MPI** is opt-in (`-Dmpi`): a φ-only ring (`comm/mpi/`, one rank per node,
`nthreads` = cores on the node). 2D (`nz = 1`) never decomposes. HDF5 output is
deferred in favor of the KDMP + optional Silo paths.

See the [validation log](docs/MILESTONES.md) for the per-milestone detail and
measured tolerances.

---

## License

koral-zig is a derivative work of [KORAL / koral_lite](https://github.com/achael/koral_lite),
which is released under the **GNU General Public License v3.0**. koral-zig is
therefore distributed under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html)
as well.

## Credits and references

- **KORAL.** The original GR radiation-MHD code, by Aleksander Sądowski and
  collaborators (Sądowski, Narayan, Tchekhovskoy, Zhu, et al.).
- **koral_lite.** The lighter reference implementation this port follows, by
  Andrew Chael: <https://github.com/achael/koral_lite>.
- **PUFFY.** The radiation-supported thick-torus problem from Lančová et al. (2019),
  *"Puffy Accretion Disks: Sustaining 3D Dynamics of Radiation-supported Thick Tori,"*
  ApJ Letters 884, L37.

If you use this code, please also cite the original KORAL papers.
