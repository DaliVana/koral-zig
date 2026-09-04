# koral-zig documentation

koral-zig ports [KORAL](https://github.com/achael/koral_lite) to Zig 0.16. The
reference case is a 2D PUFFY torus around a 10 M☉ Schwarzschild black hole. The
repository also ships 3D, Sgr A*, and 10⁹ M☉ AGN presets. Runtime parameters
change the mass, spin, grid, and selected microphysics without changing the
golden-test defaults.

Choose the document by task:

| Document | Audience | What it covers |
|----------|----------|----------------|
| [Architecture](ARCHITECTURE.md) | Programmers changing the solver | Data layout, module dependencies, the exact `Sim.step` sequence, solver layers, boundaries, threading, and test infrastructure. |
| [Physics and numerics](PHYSICS.md) | Physicists checking or extending the model | GRMHD and M1 equations, coordinates, opacities, viscosity, dynamo, finite-volume methods, floors, and PUFFY initial conditions. |
| [User and developer guide](USER_GUIDE.md) | Anyone building, running, or modifying the code | Build commands, TOML fields, output formats, tests, new-problem setup, and common modifications. |
| [PUFFY AGN differences](PUFFY_AGN_DIVERGENCES.md) | Anyone using `puffy_agn.toml` | Settings and physics that differ from the `koral_lite_puffy` reference run. |
| [GRRT renderer](RENDER.md) | Anyone imaging checkpoints | Fast and slow light, adaptive refinement, FITS and light-curve export, command reference, and validation results. |
| [Validation log](MILESTONES.md) | Contributors checking project history | The M0-M14 checklist and later validation, performance, and cleanup work. |

### Project status

M0-M14 are complete. Later work added MESA Rosseland opacity tables, runtime
physics overrides for the AGN and Sgr A* presets, opt-in φ-only MPI, a SIMD
implicit-solver Jacobian, GRRT tools, the `koral/fv/` finite-volume
reorganization, and the 2026-09 `sim.zig` redesign (a `Core` every pass sees,
pass-owned scratch, a separate integrator, a generic run driver and the shared
`problems/common/` library). See
[`MILESTONES.md`](MILESTONES.md) for the milestone checklist and each milestone's
validation summary, or the top-level [`README.md`](../README.md) for the project
overview and quickstart.

### Source checks

The guides name files and functions instead of line numbers, which drift. When a
guide and the implementation disagree, treat the implementation as authoritative
and update the guide in the same change.
