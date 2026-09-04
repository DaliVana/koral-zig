//! The evolution driver: `Sim(cfg)` is the composition root problems, tests
//! and the run driver hold. It embeds the `Core` every pass reads
//! (sim/core.zig: grid, cache, p/u, flags, scalars, physics + numerics
//! options, team, comm), owns the pass-owned scratch and the integrator's
//! stage buffers, and exposes the operators — thin methods that own the
//! timers and hand each pass its banded worker. Transcribed from KORAL's
//! serial path:
//!
//!   problem.c:141-402   RK2IMEX stage arithmetic (γ = 1 − 1/√2)  → sim/rk2imex.zig
//!   finite.c:633        op_explicit (fused sweep: reconstruct → floors →
//!                       f_flux_prime per face side)              → sim/explicit.zig
//!   finite.c:1461       f_calc_fluxes_at_faces (LAXF / HLL)     → sim/explicit.zig
//!   finite.c:356/394    calc_wavespeeds / save_wavespeeds        → sim/timestep.zig
//!   finite.c:546        calc_u2p (+ cell_fixup finite.c:5012)    → sim/u2p.zig, sim/fixup.zig
//!   finite.c:1400       op_implicit                              → sim/implicit_op.zig
//!   finite.c:2805       set_bc incl. the 2D corner filling       → sim/bc.zig
//!   physics.c:789       f_metric_source_term_arb (GDETIN == 1)   → sim/explicit.zig
//!   u2p.c:2257          update_entropy                           → sim/entropy.zig
//!   magn.c              flux_ct / calc_BfromA                    → sim/ct.zig
//!   rad.c:4628          calc_Rij_visc_total                      → sim/rijvisc.zig
//!   finite.c:1370       apply_dynamo                             → sim/dynamo.zig
//!   finite.c:594        do_correct (CORRECT_POLARAXIS)           → sim/polaraxis.zig
//!
//! Every pass takes `*CoreT` (plus its own scratch as explicit arguments),
//! so a pass cannot reach the stage buffers, the face stores or another
//! pass's scratch: ownership is enforced by the type (redesign step 6,
//! 2026-09-04). The operators below are the only code that sees the whole
//! Sim. `Sim.Options` (sim/options.zig) is what a problem builds.
//!
//! C-fidelity notes:
//!  * "wide" sweep bounds mirror MPI4CORNERS, which choices.h:790 force-
//!    defines whenever MAGNFIELD is on; even in serial builds. So MHD
//!    configs sweep ±1 ghost rows and fill 2D ghost corners; hydro configs
//!    skip corners entirely (`if_outsidegc` continue).
//!  * All cell sizes/centers use C's per-cell FP forms (Grid.cellSize/xc), and
//!    stage updates use C's exact `a*x + b*y` expression shapes so that
//!    forced-dt step tests can gate at 1e-13.
//!  * copyi_u (domain+ghosts, no corners) is replaced by full-array copies:
//!    the slots that differ (ghost corners of stage buffers) are never read.
//!  * op_implicit (M9) is a structural no-op when phys.opac is null
//!    (≡ SKIPRADSOURCE); do_correct → correct_polaraxis (M10) is enabled
//!    by num.polaraxis (CORRECT_POLARAXIS, on for PUFFY in M11).
//!  * upreexplicit/ppreexplicit copies (finite.c:644) are skipped; nothing
//!    reads them before M12 (radviscosity / entropy mixing).

const std = @import("std");
const config = @import("config.zig");
const grid_mod = @import("grid.zig");
const relele = @import("relele.zig");
const threading = @import("threading.zig");
const comm_mod = @import("comm/comm.zig");
const core_mod = @import("sim/core.zig");
const options = @import("sim/options.zig");
const storage = @import("sim/storage.zig");
const bc = @import("sim/bc.zig");
const polaraxis = @import("sim/polaraxis.zig");
const timers_mod = @import("sim/timers.zig");
const timestep = @import("sim/timestep.zig");
const u2p_pass = @import("sim/u2p.zig");
const fixup_pass = @import("sim/fixup.zig");
const explicit_pass = @import("sim/explicit.zig");
const implicit_op = @import("sim/implicit_op.zig");
const entropy_pass = @import("sim/entropy.zig");
const rk2imex = @import("sim/rk2imex.zig");
const ct = @import("sim/ct.zig");
const rijvisc_mod = @import("sim/rijvisc.zig");
const dynamo_mod = @import("sim/dynamo.zig");

const Grid = grid_mod.Grid;

pub const Error = core_mod.Error;

// Re-exported so external code keeps referencing sim.BcFace / sim.Flag /
// sim.FaceStore / sim.PassTimers / sim.Pass / sim.nowNs.
pub const BcFace = bc.BcFace;
pub const FaceStore = storage.FaceStore;
pub const Flag = storage.Flag;
pub const PassTimers = timers_mod.PassTimers;
pub const Pass = timers_mod.Pass;
/// Monotonic wall clock in ns (sim/timers.zig); re-exported so the run
/// driver can time steps / throttle the heartbeat with the same clock.
pub const nowNs = timers_mod.nowNs;
pub const Core = core_mod.Core;

pub fn Sim(comptime cfg: config.Config) type {
    const CoreType = core_mod.Core(cfg);
    const L = CoreType.Layout;
    const NV = CoreType.nv;

    return struct {
        const Self = @This();
        pub const Cfg = cfg;
        pub const Layout = L;
        pub const nv = NV;
        pub const CoreT = CoreType;
        pub const FieldT = CoreType.FieldT;
        pub const FaceT = FaceStore(NV);
        pub const wide = CoreType.wide;
        pub const base_order = CoreType.base_order;
        pub const Options = options.Options(CoreType);
        pub const SpecificBc = options.SpecificBc(CoreType);
        pub const BcKind = options.BcKind(CoreType);
        /// sim/ct.zig's scratch type when the layout has B, else void.
        pub const CtScratch = if (L.hasVar(.b1)) ct.Scratch else void;

        /// The state every pass reads (sim/core.zig).
        core: CoreType,
        /// Boundary handling per axis (sim/bc.zig reads it through setBc).
        bc: options.Bc(CoreType),
        parallel: options.Parallel,
        /// The integrator's stage buffers (sim/rk2imex.zig, problem.c).
        integ: rk2imex.Integrator(NV),
        /// The explicit operator's face stores (sim/explicit.zig): pbL/pbR,
        /// flL/flR per side and the combined flb. Flux-CT rewrites flb's B rows.
        faces: explicit_pass.Faces(NV),
        /// The fixup pass's whole-grid u/p backups (sim/fixup.zig, finite.c:5030).
        bak: fixup_pass.Backups(NV),
        /// Constrained-transport scratch (sim/ct.zig: corner EMFs + the
        /// vector-potential work field). Present iff the config has MHD;
        /// `void` otherwise so a hydro-only Sim carries nothing.
        ct: CtScratch,
        /// Radiative-viscosity parameters + R^i_j scratch (sim/rijvisc.zig);
        /// null ≡ off.
        visc: ?rijvisc_mod.State,
        /// Dynamo parameters + ΔA_φ / scale-height scratch (sim/dynamo.zig);
        /// null ≡ off.
        dynamo: ?dynamo_mod.State,

        t: f64 = 0,
        dt: f64 = 0,
        /// dt the CFL logic would have chosen this step (before forcing).
        own_dt: f64 = 0,
        nstep: u64 = 0,
        /// implicit-solver diagnostics (C: global_int_slot counters), all
        /// run-cumulative; the driver deltas them for the per-step heartbeat.
        n_radimp_failures: u64 = 0,
        /// This step's implicit-failure count, MAXed across ranks by the
        /// end-of-step collective (the C abort path's load-bearing signal;
        /// == the local per-step delta on serial/1-rank runs).
        n_radimp_fail_step: u64 = 0,
        n_radimp_iters: u64 = 0,
        n_radimp_solves: u64 = 0,

        pub fn init(allocator: std.mem.Allocator, g: Grid, opt: Options) !Self {
            try opt.validate(cfg, g);
            const dc = opt.decomp orelse comm_mod.Decomp.serial(g);

            const team: ?*threading.Team = if (opt.parallel.nthreads > 1)
                try threading.Team.init(allocator, opt.parallel.nthreads, opt.parallel.pin_threads)
            else
                null;
            errdefer if (team) |tm| tm.deinit();

            var core = try CoreType.init(allocator, g, .{
                .phys = opt.phys,
                .num = opt.num,
                .team = team,
                .decomp = dc,
                .comm = opt.comm,
                // Kerr coords are guaranteed here — both switches imply a
                // Kerr problem.
                .bl_cache = opt.dynamo != null or opt.radvisc != null,
            });
            errdefer core.deinit();

            // Every allocation carries an errdefer, so a failed init leaks
            // nothing. Zero-init through the team like the Core's stores.
            var integ = try rk2imex.Integrator(NV).init(allocator, g, team);
            errdefer integ.deinit();
            var faces = try explicit_pass.Faces(NV).init(allocator, g, team);
            errdefer faces.deinit();
            var bak = try fixup_pass.Backups(NV).init(allocator, g, team);
            errdefer bak.deinit();
            var ct_scratch: CtScratch = undefined;
            if (comptime L.hasVar(.b1)) {
                ct_scratch = try ct.Scratch.init(allocator, g, team);
            } else {
                ct_scratch = {};
            }
            errdefer if (comptime L.hasVar(.b1)) ct_scratch.deinit();
            var visc: ?rijvisc_mod.State = if (opt.radvisc) |rp| try rijvisc_mod.State.init(allocator, g, team, rp) else null;
            errdefer if (visc) |*v| v.deinit();
            const dyn: ?dynamo_mod.State = if (opt.dynamo) |dp| try dynamo_mod.State.init(allocator, g, team, dp) else null;

            return .{
                .core = core,
                .bc = opt.bc,
                .parallel = opt.parallel,
                .integ = integ,
                .faces = faces,
                .bak = bak,
                .ct = ct_scratch,
                .visc = visc,
                .dynamo = dyn,
            };
        }

        pub fn deinit(self: *Self) void {
            const team = self.core.team;
            if (self.dynamo) |*d| d.deinit();
            if (self.visc) |*v| v.deinit();
            if (comptime L.hasVar(.b1)) self.ct.deinit();
            self.bak.deinit();
            self.faces.deinit();
            self.integ.deinit();
            self.core.deinit();
            if (team) |tm| tm.deinit();
            self.* = undefined;
        }

        // ---- forwards to the Core (the entry points callers already use) ----

        pub fn nxi(self: *const Self) i64 {
            return self.core.nxi();
        }
        pub fn nyi(self: *const Self) i64 {
            return self.core.nyi();
        }
        pub fn nzi(self: *const Self) i64 {
            return self.core.nzi();
        }
        pub inline fn getFlag(self: *const Self, f: Flag, ix: i64, iy: i64, iz: i64) i32 {
            return self.core.getFlag(f, ix, iy, iz);
        }
        pub fn isPhysicalBoundary(self: *const Self, face: BcFace) bool {
            return self.core.isPhysicalBoundary(face);
        }
        pub fn exchangeHalos(self: *Self) void {
            self.core.exchangeHalos();
        }
        pub fn globalMax(self: *Self, v: f64) f64 {
            return self.core.globalMax(v);
        }
        pub fn globalSum(self: *Self, v: f64) f64 {
            return self.core.globalSum(v);
        }
        pub fn cflDt(self: *const Self) f64 {
            return self.core.cflDt();
        }
        /// Set one domain cell's primitives and derived conserveds
        /// (C: PR_INIT body; pp then p2u).
        pub fn initCell(self: *Self, ix: i64, iy: i64, iz: i64, pp: [NV]f64) Error!void {
            return self.core.initCell(ix, iy, iz, pp);
        }

        /// The ONE per-step collective (plan §7.1), at the END of step() so
        /// the driver's next cflDt() reads global values (§1.1-6): a folded
        /// Allreduce(MAX) of [tstepdenmax, −tstepdenmin, this step's local
        /// implicit-failure count]. MAX is exactly associative → bitwise
        /// reproducible at any rank count.
        pub fn reduceStepGlobals(self: *Self, fail_before: u64) void {
            const local_fail = self.n_radimp_failures - fail_before;
            self.n_radimp_fail_step = local_fail;
            if (self.core.comm) |c| {
                self.core.timers.begin(.collect);
                defer self.core.timers.end();
                var buf = [3]f64{ self.core.tstepdenmax, -self.core.tstepdenmin, @floatFromInt(local_fail) };
                c.allreduceMax(buf[0..]);
                self.core.tstepdenmax = buf[0];
                self.core.tstepdenmin = -buf[1];
                self.n_radimp_fail_step = @intFromFloat(buf[2]);
            }
        }

        // ---- initialization -------------------------------------------------

        /// ko.c init tail + problem.c:59-82: halo exchange, ghost fill, and
        /// the initial timestep guess. Call after initCell over the domain
        /// (and after calcBfromA for vector-potential problems). Uses
        /// `self.t` so a restart that has already adopted the checkpoint
        /// clock fills ghosts at the resumed time. `exchangeHalos` is a
        /// no-op serially / at 1 rank.
        pub fn finishInit(self: *Self) Error!void {
            self.core.exchangeHalos();
            try self.setBc(self.t, true);
            self.initTimestepGuess();
        }

        /// problem.c:59-82; initial dt guess from max_ws = 10⁴ (sim/timestep.zig).
        pub fn initTimestepGuess(self: *Self) void {
            timestep.initTimestepGuess(CoreType, &self.core);
        }

        /// C: calc_BfromA (magn.c:462): cell-centered A_i in the B slots →
        /// B, optionally overwriting p/u everywhere (sim/ct.zig). Call setBc
        /// afterwards, as ko.c does.
        pub fn calcBfromA(self: *Self, ifoverwrite: bool) relele.Error!void {
            if (comptime !L.hasVar(.b1)) return;
            return ct.calcBfromA(CoreType, &self.core, &self.ct, ifoverwrite);
        }

        /// C: calc_divB (magn.c:693) at one cell corner; 0 outside the domain.
        pub fn calcDivB(self: *const Self, ix: i64, iy: i64, iz: i64) f64 {
            return ct.calcDivB(CoreType, &self.core, ix, iy, iz);
        }

        // ---- boundary conditions --------------------------------------------

        /// C: set_bc (finite.c:2805). The implementation (ghost fill + the 2D
        /// corner filling, and the MPI-growth seam) lives in sim/bc.zig; this
        /// is the stable method entry point that problems/tests call.
        pub fn setBc(self: *Self, t: f64, ifinit: bool) Error!void {
            self.core.timers.begin(.bc);
            defer self.core.timers.end();
            return bc.setBc(CoreType, &self.core, &self.bc, t, ifinit);
        }

        // ---- the operators --------------------------------------------------
        // The persistent team + dynamic-tile mechanism (parallelRange +
        // ChunkResult) lives in threading.zig; the per-cell pass bodies live
        // in sim/*.zig, each taking the Core plus its own scratch. The
        // operators below own the timers and hand the pass's banded worker to
        // the team. Most passes only ever report an error, so they go through
        // `parallelRangeErr`; the two that *reduce* — wavespeeds (tsd_max/min)
        // and the implicit operator (iteration/solve/failure counters) —
        // consume the merged `ChunkResult` themselves.

        /// C: calc_wavespeeds (finite.c:356) over domain + 1 ghost layer.
        pub fn calcWavespeeds(self: *Self) Error!void {
            self.core.timers.begin(.wavespeeds);
            defer self.core.timers.end();
            return timestep.calcWavespeeds(CoreType, &self.core);
        }

        /// C: calc_u2p (finite.c:546). Per-cell inversion + floors, then
        /// fixup averaging and a boundary refresh. `t` is the simulation time
        /// the refreshed ghosts are evaluated at (C: global_time).
        pub fn calcU2p(self: *Self, t: f64) Error!void {
            self.core.timers.begin(.u2p);
            defer self.core.timers.end();
            try threading.parallelRangeErr(CoreType, &self.core, self.core.team, 0, self.core.nyi(), u2p_pass.rowsFn(CoreType));

            try self.cellFixup(.hd_fixup);
            if (comptime L.hasVar(.ee)) {
                try self.cellFixup(.rad_fixup);
            }
            try self.setBc(t, false);
        }

        /// C: cell_fixup (finite.c:5012), types FIXUP_U2PMHD / FIXUP_U2PRAD /
        /// FIXUP_RADIMP (sim/fixup.zig). This is the stable entry point: the
        /// enable switches and the timer; the flag scan, the u/p staging and
        /// the neighbour averaging live in the pass.
        pub fn cellFixup(self: *Self, comptime which: Flag) Error!void {
            const fx = self.core.num.fixups;
            const enabled = switch (which) {
                // C: DOFIXUPS && DOU2PMHDFIXUPS / DOU2PRADFIXUPS / DORADIMPFIXUPS
                .hd_fixup => fx.enabled,
                .rad_fixup => fx.enabled and fx.u2prad,
                .radimp_fixup => fx.enabled and fx.radimp,
                else => @compileError("cellFixup: bad type"),
            };
            if (!enabled) return;
            self.core.timers.begin(.fixup);
            defer self.core.timers.end();
            return fixup_pass.cellFixup(CoreType, &self.core, &self.bak, which);
        }

        /// C: flux_ct (magn.c:240): rebuild the B rows of the face fluxes from
        /// corner-EMF averages (sim/ct.zig). No-op without a magnetic field.
        pub fn fluxCt(self: *Self) void {
            if (comptime !L.hasVar(.b1)) return;
            ct.fluxCt(CoreType, &self.core, &self.ct, &self.faces.flb);
        }

        /// C: calc_Rij_visc_total (rad.c:4628): the once-per-step viscous
        /// R^i_j fill (sim/rijvisc.zig). No-op when radiative viscosity is off.
        pub fn calcRijViscTotal(self: *Self, global_dt: f64) Error!void {
            if (comptime !L.hasVar(.ee)) return;
            const visc = if (self.visc) |*v| v else return;
            self.core.timers.begin(.radvisc);
            defer self.core.timers.end();
            return rijvisc_mod.calcRijViscTotal(CoreType, &self.core, visc, global_dt);
        }

        /// C: op_explicit (finite.c:633). The transport passes (sweep, flux
        /// combination, conserved update + metric source) live in
        /// sim/explicit.zig; this composes them in C's order.
        pub fn opExplicit(self: *Self, t: f64, dtin: f64) Error!void {
            // (upreexplicit/ppreexplicit copies skipped — see header)
            // calc_avgs_throughout: CALCHRONTHEGO only (M12)

            try self.calcWavespeeds();

            var ctx = explicit_pass.Ctx(CoreType){
                .core = &self.core,
                .faces = &self.faces,
                .visc = if (self.visc) |*v| v else null,
                .dt = dtin,
            };

            {
                self.core.timers.begin(.sweep);
                defer self.core.timers.end();
                inline for (0..3) |dim| try explicit_pass.sweep(CoreType, &ctx, dim);
            }

            {
                self.core.timers.begin(.fluxes);
                defer self.core.timers.end();
                try explicit_pass.fluxesAtFaces(CoreType, &ctx);
            }

            if (comptime cfg.has(.mhd)) {
                self.core.timers.begin(.fluxct);
                defer self.core.timers.end();
                self.fluxCt();
            }

            // conserved update: du from flux divergence + source terms
            {
                self.core.timers.begin(.update);
                defer self.core.timers.end();
                try explicit_pass.update(CoreType, &ctx);
            }

            // (EXPLICIT_LAB_RAD_SOURCE not used — PUFFY couples implicitly)

            try self.calcU2p(t);
            // (MIXENTROPIESPROPERLY off)
        }

        /// C: update_entropy (u2p.c:2257). Recompute S(ρ,u) and refresh the
        /// MHD conserveds. Band-parallel over iy (cell-local).
        pub fn updateEntropy(self: *Self) Error!void {
            self.core.timers.begin(.entropy);
            defer self.core.timers.end();
            try threading.parallelRangeErr(CoreType, &self.core, self.core.team, 0, self.core.nyi(), entropy_pass.rowsFn(CoreType));
        }

        /// C: op_implicit (finite.c:1400). The implicit radiative source
        /// operator: per-cell solve_implicit_lab, RADIMPFIXUPFLAG, then
        /// cell_fixup(FIXUP_RADIMP). Structural no-op without radiation or
        /// when phys.opac is null (≡ SKIPRADSOURCE).
        pub fn opImplicit(self: *Self, dtin: f64) Error!void {
            if (comptime !L.hasVar(.ee)) return;
            _ = self.core.phys.opac orelse return;

            self.core.timers.begin(.implicit);
            defer self.core.timers.end();
            var ctx = implicit_op.Ctx(CoreType){ .core = &self.core, .dt = dtin };
            const res = threading.parallelRange(implicit_op.Ctx(CoreType), &ctx, self.core.team, 0, self.core.nyi(), implicit_op.rowsWorker(CoreType));
            self.n_radimp_failures += res.n_fail;
            self.n_radimp_iters += res.n_iters;
            self.n_radimp_solves += res.n_solves;
            if (res.err) |e| return e;

            try self.cellFixup(.radimp_fixup);
        }

        /// C: do_correct (finite.c:594); CORRECT_POLARAXIS only; the 3D /
        /// smoothing / NS-surface variants are not PUFFY machinery
        /// (sim/polaraxis.zig).
        pub fn doCorrect(self: *Self) Error!void {
            if (polaraxis.band(CoreType, &self.core) == null) return;
            self.core.timers.begin(.correct);
            defer self.core.timers.end();
            return polaraxis.correct(CoreType, &self.core);
        }

        /// C: apply_dynamo (finite.c:1370) after an explicit sub-step
        /// (sim/dynamo.zig). No-op when the dynamo is off.
        pub fn applyDynamo(self: *Self, t: f64, dt: f64) Error!void {
            if (comptime !L.hasVar(.b1)) return;
            if (self.dynamo == null) return;
            self.core.timers.begin(.dynamo);
            defer self.core.timers.end();
            return dynamo_mod.applyDynamo(Self, self, t, dt);
        }

        /// C: calc_avgs_throughout (mpi.c:3168): the per-radius scale height
        /// into the dynamo's scaleth (sim/dynamo.zig). Requires the dynamo.
        pub fn calcScaleHeight(self: *Self) void {
            dynamo_mod.calcScaleHeight(CoreType, &self.core, &self.dynamo.?);
        }

        /// C: mimic_dynamo (magn.c:1003) for one sub-step (sim/dynamo.zig).
        /// Requires the dynamo and a magnetic field.
        pub fn mimicDynamo(self: *Self, dt: f64) Error!void {
            if (comptime !L.hasVar(.b1)) return;
            return dynamo_mod.mimicDynamo(CoreType, &self.core, &self.dynamo.?, &self.ct, dt);
        }

        // ---- one full RK2IMEX step (problem.c:141-402; sim/rk2imex.zig) ---------

        /// Advance one step. The integrator is selected from cfg.timestepping
        /// at comptime; only RK2IMEX is implemented, the other schemes the
        /// Config enum names fail here at compile time.
        pub fn step(self: *Self, forced_dt: ?f64) Error!void {
            return switch (comptime cfg.timestepping) {
                .rk2imex => rk2imex.step(Self, self, forced_dt),
                else => @compileError("Sim.step: only .rk2imex is implemented (cfg.timestepping = " ++ @tagName(cfg.timestepping) ++ ")"),
            };
        }
    };
}

test {
    _ = FaceStore(5);
    _ = timers_mod;
    _ = options;
}
