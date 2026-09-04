//! Runtime configuration of a Sim, grouped by owner (redesign step 6,
//! 2026-09-04). `Options(CoreT)` is what a problem builds and hands to
//! `Sim.init`; `Physics` and `Numerics` then live on the Core every pass
//! reads, the boundary block and the parallel knobs stay on the Sim, and
//! the three optional passes (radiative viscosity, dynamo, polar-axis
//! correction) follow the `opac: ?Params` idiom: null means off, so the
//! old bool-plus-params pairs and their cross-field rules are gone.
//!
//! `applyParams` is the generic TOML → Options mapping every problem
//! shares (floors, radiation, implicit-solver and numerics knobs, threads);
//! a problem's own keys are its own business (puffy.Physics.fromParams).
//! `validate` holds the init-time precondition checks that the types cannot
//! express.

const std = @import("std");
const config = @import("../config.zig");
const metric = @import("../metric/metric.zig");
const invert = @import("../solve/invert.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const implicit = @import("../solve/implicit.zig");
const radforce = @import("../physics/radforce.zig");
const radvisc = @import("../physics/radvisc.zig");
const dynamo = @import("dynamo.zig");
const comm_mod = @import("../comm/comm.zig");
const bc_mod = @import("bc.zig");
const relele = @import("../relele.zig");
const params_mod = @import("../params.zig");
const Grid = @import("../grid.zig").Grid;

/// The physics every kernel reads (C: define.h / choices.h constants).
pub const Physics = struct {
    coords: config.Coords,
    mp: metric.MetricParams = .{},
    gam: f64 = 5.0 / 3.0, // C: GAMMA
    floors: invert.FloorParams = invert.FloorParams.cdefault,
    rad: invert_rad.RadParams = invert_rad.RadParams.cdefault,
    /// Opacities for the rad wavespeed τ-limiter (C: calc_chi) AND the
    /// implicit radiative source coupling (op_implicit). null ≡ C's
    /// SKIPRADSOURCE: optically thin wavespeeds and no implicit operator.
    opac: ?radforce.Params = null,
    implicit: implicit.ImplicitParams = implicit.ImplicitParams.cdefault,
};

/// C: DOFIXUPS && DOU2PMHDFIXUPS / DOU2PRADFIXUPS / DORADIMPFIXUPS.
/// `enabled` is the master switch; `u2prad`/`radimp` additionally gate their
/// pass (both off in the validated build; radimp is on in koral_lite_puffy).
pub const Fixups = struct {
    enabled: bool = true,
    u2prad: bool = false,
    radimp: bool = false,
};

/// C: CORRECT_POLARAXIS + NCCORRECTPOLAR (PUFFY define.h:107; 2). Do_correct
/// overwrites the most-polar `ncells` rows per side and u2p/implicit/fixups
/// skip them. Only meaningful on spherical-like coordinates.
pub const PolarAxis = struct { ncells: i64 = 2 };

/// Algorithm knobs that change no generated code.
pub const Numerics = struct {
    tsteplim: f64 = 0.5, // C: TSTEPLIM
    minmod_theta: f64 = 1.5, // C: MINMOD_THETA
    fixups: Fixups = .{},
    /// C: REDUCEORDERATBH. Drop the reconstruction order by one for
    /// cells whose center is inside the BL horizon (PPM→linear there).
    reduceorderatbh: bool = false,
    /// koral_lite_puffy REDUCEORDERAFTERFIXUP (finite.c:26-40, 2026-08-11):
    /// a cell whose most recent u2p / rad / implicit pass demanded a fixup
    /// reconstructs one order lower for one sweep.
    reduceorderafterfixup: bool = false,
    /// C: DAMPRADWAVESPEEDNEARAXIS + …NCELLS. Within this many cells of
    /// each pole, force the radiative wavespeed to the undamped 1/3. 0 = off.
    dampradwavespeednearaxis: usize = 0,
    /// null ≡ no polar-axis correction.
    polaraxis: ?PolarAxis = null,
};

/// Worker-team configuration (P1 threading, P4b pinning).
pub const Parallel = struct {
    /// Worker threads for the per-step passes. 1 ≡ the serial path
    /// (bit-identical; the golden tests run here).
    nthreads: usize = 1,
    /// Bind each team thread (main included) to one cpu of the process
    /// affinity mask (the cgroup cpuset under Slurm). Linux only; inert
    /// elsewhere. No FP effect.
    pin_threads: bool = false,
};

/// A problem's boundary-condition hook. Returns the ghost-cell primitives,
/// or an Error; the fallible frame/velocity conversions a boundary-adjacent
/// cell can reach propagate instead of being swallowed. It sees the Core
/// (grid, cache, p, physics) and nothing else of the Sim.
pub fn SpecificBc(comptime CoreT: type) type {
    return *const fn (
        ctx: ?*const anyopaque,
        core: *const CoreT,
        ix: i64,
        iy: i64,
        iz: i64,
        t: f64,
        ifinit: bool,
        face: bc_mod.BcFace,
    ) relele.Error![CoreT.nv]f64;
}

/// A `.specific` boundary: the callback plus an opaque context threaded
/// verbatim to it on every ghost fill (e.g. which shock tube / which Michel
/// solution the BC samples), so a runtime-parameterized BC needs no
/// module-level state. The comptime-bound BCs (puffy) ignore it.
pub fn Specific(comptime CoreT: type) type {
    return struct { f: SpecificBc(CoreT), ctx: ?*const anyopaque = null };
}

/// Boundary handling per axis (C: PERIODIC_?BC / COPY_?BC / SPECIFIC_BC).
/// A `.specific` face carries its callback, so "specific without a
/// callback" cannot be expressed.
pub fn BcKind(comptime CoreT: type) type {
    return union(enum) {
        periodic,
        copy,
        specific: Specific(CoreT),
    };
}

pub fn Bc(comptime CoreT: type) type {
    return struct {
        x: BcKind(CoreT) = .copy,
        y: BcKind(CoreT) = .copy,
        z: BcKind(CoreT) = .copy,
    };
}

pub fn Options(comptime CoreT: type) type {
    return struct {
        const Self = @This();

        phys: Physics,
        num: Numerics = .{},
        bc: Bc(CoreT) = .{},
        /// C: RADVISCOSITY==SHEARVISCOSITY (PUFFY define.h:96). The radiative
        /// shear-viscosity contribution to R^ij, computed once per step and
        /// added at the faces. null ≡ off.
        radvisc: ?radvisc.Params = null,
        /// C: MIMICDYNAMO (PUFFY define.h:66). The mean-field dynamo run after
        /// each explicit sub-step (problem.c:210,326). null ≡ off.
        dynamo: ?dynamo.Params = null,
        parallel: Parallel = .{},
        /// MPI plan P4a: the communication backend (comm/comm.zig; Serial
        /// no-ops by default, Mpi under -Dmpi). null ≡ serial semantics: when
        /// set, the Core binds the zero-copy exchange channels into `p` at
        /// init and runs the exchange episodes + end-of-step collective. Must
        /// outlive the Sim.
        comm: ?*comm_mod.Backend = null,
        /// The φ-slab decomposition this Sim's grid is the local piece of.
        /// null ≡ trivial (grid is global, all faces physical).
        decomp: ?comm_mod.Decomp = null,

        /// The generic TOML overrides every problem honours: floors, the
        /// radiative floors, the implicit solver, the numerics knobs and
        /// the worker team. Problem-specific keys (mass, torus, atmosphere,
        /// opacity channels) are the problem's to map.
        pub fn applyParams(self: *Self, p: *const params_mod.Params) void {
            self.phys.gam = p.gam;
            applyFloors(&self.phys.floors, p);
            applyRad(&self.phys.rad, p);
            applyImplicit(&self.phys.implicit, p);
            applyNumerics(&self.num, p);
            self.parallel = .{ .nthreads = p.nthreads, .pin_threads = p.pin_threads };
        }

        /// Runtime precondition checks (P1 correctness). Each unchecked
        /// invariant otherwise fails far from its cause; error.InvalidConfig
        /// rather than std.debug.assert so ReleaseFast builds are covered
        /// too. Cost is negligible (once/run).
        pub fn validate(self: Self, comptime cfg: config.Config, g: Grid) error{InvalidConfig}!void {
            const L = CoreT.Layout;
            // (1) Ghost depth must cover the reconstruction stencil. PPM's
            //     unconditional i−2 load (with the ±1 MHD cross range) drives
            //     Field.cellOffset's @intCast negative when g.ng is too small —
            //     a safe-build panic, OOB UB in ReleaseFast — deep in the sweep.
            if (g.ng < cfg.ghostCells()) return error.InvalidConfig;
            // (2) The runtime metric coords (drives the MetricCache) must match
            //     the comptime coords every other consumer reads via Cfg.coords
            //     (dynamo/scalars/puffy coco transforms); divergence silently
            //     mixes two coordinate systems — wrong physics, no diagnostic.
            if (self.phys.coords != cfg.coords) return error.InvalidConfig;
            // (3) Radiative shear viscosity needs opacities: ν = α·mfp, mfp =
            //     1/χ, and χ = calc_chi is undefined without opac. C's
            //     SKIPRADSOURCE keeps viscosity active via mfp=1e50, but the Zig
            //     path would silently store ν=0 with the sweep still running.
            if (comptime L.hasVar(.ee)) {
                if (self.radvisc != null and self.phys.opac == null) return error.InvalidConfig;
            }
            if (self.num.polaraxis) |pa| {
                // (4) The overwrite sources must lie outside the overwritten
                //     band: with ny ≤ 2·ncells every row is "corrected" —
                //     order-dependent garbage (reduced ny is reachable in tests).
                if (@as(i64, @intCast(g.ny)) <= 2 * pa.ncells) return error.InvalidConfig;
                // (5) Spherical-only, and only ONE half of it knows that: the
                //     overwrite returns early on .mink while the row predicate
                //     does not, so u2p/fixups/implicit would skip rows nobody
                //     supplies. Reject outright.
                if (cfg.coords == .mink) return error.InvalidConfig;
            }
            // (6) MPI consistency: the grid must be the decomposition's local
            //     slab, the backend's ring must match its ntz, and a real
            //     split needs a backend (skipped z-face BCs would otherwise
            //     never be filled by anyone).
            const dc = self.decomp orelse comm_mod.Decomp.serial(g);
            if (dc.local.nz != g.nz or dc.local.izoff != g.izoff) return error.InvalidConfig;
            if (dc.ntz > 1 and self.comm == null) return error.InvalidConfig;
            if (self.comm) |c| {
                if (c.size() != dc.ntz) return error.InvalidConfig;
            }
        }
    };
}

/// The floor overrides (params.zig [floors]).
pub fn applyFloors(f: *invert.FloorParams, p: *const params_mod.Params) void {
    if (p.zamo_floor_frame) |z| f.b2rhofloorframe = if (z) .zamoframe else .driftframe;
    if (p.isentropic_b2rhofloors) |b| f.isentropic_b2rhofloors = b;
    if (p.b2uufloor) |b| f.b2uufloor = b;
    if (p.rhofloor) |v| f.rhofloor = v;
    if (p.uurhoratiomin) |v| f.uurhoratiomin = v;
    if (p.uurhoratiomax) |v| f.uurhoratiomax = v;
    if (p.b2rhoratiomax) |v| f.b2rhoratiomax = v;
    if (p.b2uuratiomax) |v| f.b2uuratiomax = v;
    if (p.gammamaxhd) |v| f.gammamaxhd = v;
}

/// The radiative-floor overrides.
pub fn applyRad(r: *invert_rad.RadParams, p: *const params_mod.Params) void {
    if (p.gammamaxrad) |v| r.gammamaxrad = v;
    if (p.eerhoratiomin) |v| r.eerhoratiomin = v;
    if (p.eerhoratiomax) |v| r.eerhoratiomax = v;
    if (p.eeuuratiomin) |v| r.eeuuratiomin = v;
    if (p.eeuuratiomax) |v| r.eeuuratiomax = v;
}

/// The implicit-solver overrides.
pub fn applyImplicit(i: *implicit.ImplicitParams, p: *const params_mod.Params) void {
    if (p.opdamp_maxlevels) |n| i.opdamp_maxlevels = n;
    if (p.opdamp_factor) |v| i.opdamp_factor = v;
    if (p.radimp_lag_opac) |b| i.lag_opac = b;
    if (p.scale_jacobian) |b| i.scale_jacobian = b;
    if (p.radimp_max_en_change_down) |v| i.max_en_change_down = v;
    if (p.radimp_max_en_change_up) |v| i.max_en_change_up = v;
    if (p.radimp_max_damping) |v| i.max_damping = v;
    if (p.radimpeps) |v| i.eps = v;
    if (p.radimpmaxiter) |v| i.maxiter = v;
}

/// The numerics overrides (unset keys keep the compiled preset).
pub fn applyNumerics(n: *Numerics, p: *const params_mod.Params) void {
    n.tsteplim = p.tsteplim;
    if (p.doradimpfixups) |b| n.fixups.radimp = b;
    if (p.reduceorderatbh) |b| n.reduceorderatbh = b;
    if (p.reduceorderafterfixup) |b| n.reduceorderafterfixup = b;
    if (p.dampradwavespeednearaxis) |v| n.dampradwavespeednearaxis = v;
}
