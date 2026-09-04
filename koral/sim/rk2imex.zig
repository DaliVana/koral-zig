//! The RK2IMEX integrator (problem.c:141-402): Pareschi–Russo SSP2(2,2,2)
//! with implicit coefficient γ = 1 − 1/√2. `Integrator(NV)` owns the eight
//! whole-grid stage buffers; `step` is the exact per-step sequence, moved
//! verbatim out of sim.zig (redesign step 4, 2026-09-04) — every copy,
//! stage derivative and combination keeps C's expression shape
//! (sim/stage.zig) so the forced-dt step tests gate at 1e-13.
//!
//! `Sim.step` selects this file from `cfg.timestepping` at comptime; the
//! other schemes the Config enum names are not implemented and fail to
//! compile there rather than at runtime.
//!
//! What each buffer holds:
//!   ut0      u at the start of the step (U^n)
//!   ut1/ut2  u just before the 1st / 2nd explicit operator
//!   dut1/2   explicit stage derivatives F(U^(k)) = (u − utk)/dt
//!   drt1/2   implicit stage derivatives (u − base)/(dt·γ)
//!   uforget  u just before the 2nd implicit operator (base for drt2)

const std = @import("std");
const relele = @import("../relele.zig");
const field_mod = @import("../field.zig");
const Grid = @import("../grid.zig").Grid;
const threading = @import("../threading.zig");
const storage = @import("storage.zig");
const stage = @import("stage.zig");
const timestep = @import("timestep.zig");
const entropy_pass = @import("entropy.zig");

const Error = relele.Error || error{OutOfMemory};

/// The stage buffers, allocated raw and zeroed through the team (NUMA
/// first-touch) like every other whole-grid store.
pub fn Integrator(comptime NV: usize) type {
    return struct {
        const Self = @This();
        pub const FieldT = field_mod.Field(NV);
        pub const names = .{ "ut0", "ut1", "ut2", "dut1", "dut2", "drt1", "drt2", "uforget" };

        ut0: FieldT,
        ut1: FieldT,
        ut2: FieldT,
        dut1: FieldT,
        dut2: FieldT,
        drt1: FieldT,
        drt2: FieldT,
        uforget: FieldT,

        pub fn init(allocator: std.mem.Allocator, g: Grid, team: ?*threading.Team) !Self {
            var self: Self = undefined;
            var n: usize = 0;
            errdefer {
                inline for (names, 0..) |name, k| {
                    if (k < n) @field(self, name).deinit();
                }
            }
            inline for (names) |name| {
                @field(self, name) = try FieldT.initUninitialized(allocator, g);
                n += 1;
                threading.parallelZero(team, @field(self, name).data);
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            inline for (names) |name| @field(self, name).deinit();
            self.* = undefined;
        }
    };
}

// ---- stage arithmetic (exact C expression shapes; sim/stage.zig) ----------

fn copyFull(comptime SimT: type, self: *SimT, dst: *SimT.FieldT, src: *const SimT.FieldT) void {
    self.core.timers.begin(.stage);
    defer self.core.timers.end();
    stage.copyFull(SimT.CoreT, &self.core, dst, src);
}

/// dst = (1/Δ)·a + (−1/Δ)·b over the domain (problem.c:181/230/…).
fn stageDeriv(comptime SimT: type, self: *SimT, dst: *SimT.FieldT, a_f: *const SimT.FieldT, b_f: *const SimT.FieldT, delta: f64) void {
    self.core.timers.begin(.stage);
    defer self.core.timers.end();
    stage.deriv(SimT.CoreT, &self.core, dst, a_f, b_f, delta);
}

/// dst = a + f1·b + f2·c over the domain (problem.c:243/357).
fn stageCombine(comptime SimT: type, self: *SimT, dst: *SimT.FieldT, a_f: *const SimT.FieldT, f1: f64, b_f: *const SimT.FieldT, f2: f64, c_f: *const SimT.FieldT) void {
    self.core.timers.begin(.stage);
    defer self.core.timers.end();
    stage.combine(SimT.CoreT, &self.core, dst, a_f, f1, b_f, f2, c_f);
}

pub fn step(comptime SimT: type, self: *SimT, forced_dt: ?f64) Error!void {
    self.core.timers.begin(.step);
    defer self.core.timers.end();

    const t = self.t;

    // The CFL denominator must have been seeded (initTimestepGuess, part
    // of C's solve preamble) or be forced — otherwise tstepdenmax is 0
    // and own_dt is +inf → NaNs on the first step, diagnosed only far
    // downstream. Turn that ordering bug into an immediate failure.
    std.debug.assert(forced_dt != null or self.core.tstepdenmax > 0);
    self.own_dt = self.cflDt();
    const dt = forced_dt orelse self.own_dt;
    self.dt = dt;
    // Baseline for the end-of-step failure fold (§7.1): the counter
    // is run-cumulative, the collective reduces this step's delta.
    const radimp_fail_at_step_start = self.n_radimp_failures;

    // reset wavespeed accumulators for this step
    self.core.tstepdenmax = -1;
    self.core.tstepdenmin = storage.big;

    // C: calc_Rij_visc_total (problem.c:127) — once per step, with
    // global_dt = this step's dt, feeding the RADVISCNUDAMP cap.
    try self.calcRijViscTotal(dt);

    const gam_imex = 1.0 - 1.0 / @sqrt(2.0);

    // my_finger: PR_FINGER not defined for any target problem
    timestep.saveTimesteps(SimT.CoreT, &self.core);
    // set_gammagas: CONSISTENTGAMMA off

    // ---- 1st implicit ----
    copyFull(SimT, self, &self.integ.ut0, &self.core.u);
    // (C copies p→ptm1 and, post-implicit, p→ppostimplicit here; both
    //  are write-only on this path — no Zig consumer — so dropped.)
    try self.opImplicit(dt * gam_imex); // U(1)
    stageDeriv(SimT, self, &self.integ.drt1, &self.core.u, &self.integ.ut0, dt * gam_imex);

    // ---- 1st explicit ----
    copyFull(SimT, self, &self.integ.ut1, &self.core.u);
    try self.calcU2p(t);
    try self.doCorrect();
    self.exchangeHalos(); // C: mpi_exchangedata BEFORE set_bc (problem.c:198-204)
    try self.setBc(t, false);
    try self.opExplicit(t, dt); // F(U(1))
    try self.applyDynamo(t, dt);
    // op_intermediate (electrons) — no-op
    entropy_pass.copyEntropyCount(SimT.CoreT, &self.core);
    stageDeriv(SimT, self, &self.integ.dut1, &self.core.u, &self.integ.ut1, dt);

    // ---- 1st together: u = ut0 + dt·dut1 + dt(1−2γ)·drt1 ----
    stageCombine(SimT, self, &self.core.u, &self.integ.ut0, dt, &self.integ.dut1, dt * (1.0 - 2.0 * gam_imex), &self.integ.drt1);

    // ---- 2nd implicit ----
    copyFull(SimT, self, &self.integ.uforget, &self.core.u);
    try self.calcU2p(t);
    // (C copies p→ptm1 / p→ppostimplicit here; write-only, dropped.)
    try self.doCorrect();
    try self.opImplicit(gam_imex * dt); // U(2)
    stageDeriv(SimT, self, &self.integ.drt2, &self.core.u, &self.integ.uforget, dt * gam_imex);

    // ---- 2nd explicit ----
    copyFull(SimT, self, &self.integ.ut2, &self.core.u);
    try self.calcU2p(t);
    try self.doCorrect();
    self.exchangeHalos(); // C: problem.c:314-320
    try self.setBc(t, false);
    try self.opExplicit(t, dt); // F(U(2))
    try self.applyDynamo(t, dt);
    stageDeriv(SimT, self, &self.integ.dut2, &self.core.u, &self.integ.ut2, dt);

    // ---- explicit together: u = ut0 + dt/2·(dut1 + dut2) ----
    stageCombine(SimT, self, &self.core.u, &self.integ.ut0, dt / 2.0, &self.integ.dut1, dt / 2.0, &self.integ.dut2);
    // ---- implicit together: u += dt/2·(drt1 + drt2) ----
    stageCombine(SimT, self, &self.core.u, &self.core.u, dt / 2.0, &self.integ.drt1, dt / 2.0, &self.integ.drt2);

    // ---- final inversion & bookkeeping ----
    try self.calcU2p(t);
    try self.doCorrect();
    // C: problem.c:386-392 — the THIRD per-step exchange site, and
    // not optional. `calcRijViscTotal` runs at the TOP of the next
    // step (before that step's first exchange) and reads z-ghost
    // primitives over iz ∈ [−1, nz+1); without this the radiative
    // shear viscosity would consume ghosts a full RK stage stale.
    self.exchangeHalos();
    try self.setBc(t, false);

    self.t = t + dt;
    try self.updateEntropy();
    self.reduceStepGlobals(radimp_fail_at_step_start);
    self.nstep += 1;
}
