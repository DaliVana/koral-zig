# koral-zig documentation

**koral-zig** is a Zig 0.16 reimplementation of [KORAL](https://github.com/achael/koral_lite)
(a general-relativistic radiation-magnetohydrodynamics code), targeting the
**PUFFY** radiation-supported accretion torus around a 10 M☉ Schwarzschild black
hole. It is a faithful, bit-comparable transcription of the C code — matching
variable order, `f64 == long double`, and even deliberate C quirks — with a
serial-first design and opt-in threading.

These three documents are written from the actual source and cross-reference each
other. Start with the one that matches what you need:

| Document | Audience | What it covers |
|----------|----------|----------------|
| [**ARCHITECTURE.md**](ARCHITECTURE.md) | Programmers getting into the code | The design philosophy and C-diffability contract, the module graph, the core data model (`Config`, `VarLayout`, `Grid`/`Field`, `MetricCache`), and — the centerpiece — the **anatomy of one time step** through `Sim.step` (the RK2IMEX-IMEX stages, `opExplicit`/`opImplicit`, `calcU2p`, the stage buffers), the solver stack, boundary conditions, the threading model, and the testing/oracle infrastructure. |
| [**PHYSICS.md**](PHYSICS.md) | Physicists validating or extending the science | The governing equations — GRMHD conservation laws + the M1 radiation moments with the four-force coupling — the coordinate systems and metric, the ideal-MHD and M1 stress tensors, opacities/Comptonization, radiative shear viscosity, the mean-field dynamo, the numerical scheme (PPM, LAXF/HLL, constrained transport, RK2-IMEX), floors/ceilings, and the PUFFY limotorus initial conditions. |
| [**USER_GUIDE.md**](USER_GUIDE.md) | Anyone building, running, or modifying the code | Prerequisites, build & run, the TOML config reference, output formats, running/regenerating tests and goldens, the complete **"how to create a new problem"** recipe, and a **"how to change things"** cookbook (swap reconstruction/flux/coordinates, adjust floors, add an opacity channel or boundary condition, tune threading, add a diagnostic). |

### Project status

The implementation follows a milestone plan (M0–M14) — all milestones are
complete: metric layer, relele/frames, p2u/u2p + floors, reconstruction +
wavespeeds + fluxes, hydro/MHD evolution with constrained transport, M1 radiation,
opacities + four-force, the implicit radiation–gas solver, the full RK2IMEX
pipeline, the PUFFY problem (limotorus init, GK21 quadrature, β-normalization,
2D and 3D), the dynamo + radiative viscosity + Comptonization, and the full driver
with scalar diagnostics, restart checkpoints, and opt-in threading. See
[`MILESTONES.md`](MILESTONES.md) for the milestone checklist and each milestone's
validation summary, or the top-level [`README.md`](../README.md) for the project
overview and quickstart.

### How the docs were kept honest

Every subsystem was mapped directly from the code, the documents were composed from
those maps, and each document was then adversarially cross-checked against the
source (signatures, file references, equations, and the step control flow). Line
numbers are avoided in favor of file + function names, since line numbers drift;
when in doubt, the code is the source of truth.
