//! M5/M6/M10 C-comparison: forced-dt step tests + flux_ct / calc_BfromA
//! goldens.
//!
//! Step tests: the C oracle (oracle/harness_step.c) runs 10 RK2IMEX steps
//! of each tiny problem, recording t, dt and the full domain u/p each step.
//! The Zig side loads C's post-init state bit-for-bit, forces C's recorded
//! dt sequence, and diffs the domain each step against a growing Budget:
//! explicit-only problems (1e-13 → 1e-10 over 10 steps), radiative problems
//! (2e-10 → 8e-7; the implicit Newton stops at RADIMPCONV = 1e-10
//! residuals, so razor-edge iteration differences legitimately move
//! converged cells at the ~1e-10 level; measured plateaus ~2e-9). Zig's own
//! CFL dt must match C's within the same budget (it reflects the previous
//! step's state deviation).
//!
//! Deviations are normalized per variable by the max |value| over the C
//! record (momentum rows pass through zero; cf. the geometry-in-record
//! golden notes in README).

const std = @import("std");
const config = @import("../../config.zig");
const grid_mod = @import("../../grid.zig");
const sim_mod = @import("../../sim.zig");
const relele = @import("../../relele.zig");
const ct = @import("../../sim/ct.zig");
const hydro = @import("../../physics/hydro.zig");
const radforce = @import("../../physics/radforce.zig");
const implicit = @import("../../solve/implicit.zig");
const units_mod = @import("../../units.zig");
const golden = @import("../../testing/golden.zig");
const tubes = @import("../../testing/tubes.zig");

const Grid = grid_mod.Grid;

const cfg_hydro = config.Config{
    .modules = &.{.hydro},
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const cfg_mhd = config.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};

/// Per-step deviation budget: `base` at step 1, growing one decade per
/// `steps_per_decade` steps. Explicit-only problems gate at (1e-13, 3);
/// radiative problems at (1e-11, 2); the implicit Newton stops at
/// RADIMPCONV = 1e-10 residuals, so a razor-edge extra iteration on either
/// side legitimately moves converged cells at the ~1e-10 level (plan M10:
/// 1e-11 → 1e-6 over 10 steps).
const Budget = struct {
    base: f64 = 1.0e-13,
    steps_per_decade: f64 = 3.0,

    fn at(b: Budget, istep: usize) f64 {
        const k = @as(f64, @floatFromInt(istep - 1)) / b.steps_per_decade;
        return b.base * std.math.pow(f64, 10.0, k);
    }
};

fn runStepTest(
    comptime SimT: type,
    comptime relpath: []const u8,
    comptime what: []const u8,
    g: Grid,
    budget_spec: Budget,
    opt: SimT.Options,
) !void {
    const a = std.testing.allocator;
    var k = try golden.readKstp(a, relpath);
    defer k.deinit();

    try std.testing.expectEqual(k.nx, g.nx);
    try std.testing.expectEqual(k.ny, g.ny);
    try std.testing.expectEqual(k.nz, g.nz);
    try std.testing.expectEqual(k.nv, SimT.nv);

    var s = try SimT.init(a, g, opt);
    defer s.deinit();

    // load C's post-init state bit-for-bit (restart-read prototype),
    // then rebuild ghosts and the initial dt guess exactly like C
    const r0 = k.rec(0);
    for (0..k.nz) |iz| {
        for (0..k.ny) |iy| {
            for (0..k.nx) |ix| {
                for (0..k.nv) |iv| {
                    const v_u = r0.u[k.idx(iv, ix, iy, iz)];
                    const v_p = r0.p[k.idx(iv, ix, iy, iz)];
                    s.u.set(iv, @intCast(ix), @intCast(iy), @intCast(iz), v_u);
                    s.p.set(iv, @intCast(ix), @intCast(iy), @intCast(iz), v_p);
                }
            }
        }
    }
    try s.setBc(0.0, true);
    s.initTimestepGuess();

    var dev_u = golden.DevTracker{};
    var dev_p = golden.DevTracker{};
    var worst_budget: f64 = 0;
    var per_step: [64]f64 = @splat(0);

    for (1..k.nrec) |istep| {
        const r = k.rec(istep);
        try s.step(r.dt);

        // C's dt is what Zig's CFL logic would also have chosen. Its
        // deviation tracks the previous step's state deviation, so the
        // tolerance follows the state budget.
        const dt_tol = @max(1e-12, budget_spec.at(istep));
        try std.testing.expectApproxEqRel(r.dt, s.own_dt, dt_tol);
        try std.testing.expectApproxEqRel(r.t, s.t, 1e-12);

        // per-variable scales from the C record
        var scale: [SimT.nv]f64 = @splat(1e-300);
        var scale_p: [SimT.nv]f64 = @splat(1e-300);
        for (0..k.nz) |iz| {
            for (0..k.ny) |iy| {
                for (0..k.nx) |ix| {
                    for (0..k.nv) |iv| {
                        scale[iv] = @max(scale[iv], @abs(r.u[k.idx(iv, ix, iy, iz)]));
                        scale_p[iv] = @max(scale_p[iv], @abs(r.p[k.idx(iv, ix, iy, iz)]));
                    }
                }
            }
        }

        const budget = budget_spec.at(istep);
        var step_max: f64 = 0;
        var worst_iv: usize = 0;
        var worst_ix: usize = 0;
        var worst_scale: f64 = 0;
        var ncell: usize = 0;
        for (0..k.nz) |iz| {
            for (0..k.ny) |iy| {
                for (0..k.nx) |ix| {
                    const zx: i64 = @intCast(ix);
                    const zy: i64 = @intCast(iy);
                    const zz: i64 = @intCast(iz);
                    for (0..k.nv) |iv| {
                        const cu = r.u[k.idx(iv, ix, iy, iz)];
                        const zu = s.u.get(iv, zx, zy, zz);
                        dev_u.add(cu / scale[iv], zu / scale[iv], istep);
                        const du = @abs(cu - zu) / scale[iv];
                        const cp = r.p[k.idx(iv, ix, iy, iz)];
                        const zp = s.p.get(iv, zx, zy, zz);
                        dev_p.add(cp / scale_p[iv], zp / scale_p[iv], istep);
                        const dp = @abs(cp - zp) / scale_p[iv];
                        if (@max(du, dp) > step_max) {
                            step_max = @max(du, dp);
                            worst_iv = iv;
                            worst_ix = ix;
                            worst_scale = if (du > dp) scale[iv] else scale_p[iv];
                        }
                    }
                    // flags must agree exactly: ENTROPYFLAG, HDFIXUPFLAG,
                    // and for RADIATION builds RADFIXUPFLAG, RADIMPFIXUPFLAG
                    const fb = ncell * k.nflags;
                    const cfe: i32 = @intFromFloat(r.flags[fb]);
                    const cff: i32 = @intFromFloat(r.flags[fb + 1]);
                    try std.testing.expectEqual(cfe, s.getFlag(.entropy, zx, zy, zz));
                    try std.testing.expectEqual(cff, s.getFlag(.hd_fixup, zx, zy, zz));
                    if (k.nflags == 4) {
                        const cfr: i32 = @intFromFloat(r.flags[fb + 2]);
                        const cfi: i32 = @intFromFloat(r.flags[fb + 3]);
                        try std.testing.expectEqual(cfr, s.getFlag(.rad_fixup, zx, zy, zz));
                        try std.testing.expectEqual(cfi, s.getFlag(.radimp_fixup, zx, zy, zz));
                    }
                    ncell += 1;
                }
            }
        }
        per_step[istep] = step_max;
        worst_budget = @max(worst_budget, step_max / budget);
        if (step_max > budget) {
            const wx: i64 = @intCast(worst_ix);
            std.debug.print("[{s}] per-step max devs: ", .{what});
            for (1..istep + 1) |i| std.debug.print("{e:.2} ", .{per_step[i]});
            std.debug.print(
                "\nstep {d}: max dev {e:.3} exceeds budget {e:.3} at iv={d} ix={d} (scale {e:.2}) " ++
                    "cu={e:.6} zu={e:.6} cp={e:.6} zp={e:.6}\n",
                .{
                    istep,                                step_max,
                    budget,                               worst_iv,
                    worst_ix,                             worst_scale,
                    r.u[k.idx(worst_iv, worst_ix, 0, 0)], s.u.get(worst_iv, wx, 0, 0),
                    r.p[k.idx(worst_iv, worst_ix, 0, 0)], s.p.get(worst_iv, wx, 0, 0),
                },
            );
            return error.GoldenMismatch;
        }
    }

    const final_bound = budget_spec.at(k.nrec - 1);
    try dev_u.check(final_bound, what ++ " conserveds");
    try dev_p.check(final_bound, what ++ " primitives");
    std.debug.print("golden [{s}]: worst dev/budget ratio {d:.3}; per-step max devs: ", .{ what, worst_budget });
    for (1..k.nrec) |i| std.debug.print("{e:.2} ", .{per_step[i]});
    std.debug.print("\n", .{});
}

test "golden step: sod64 (SR Sod, 10 forced-dt RK2IMEX steps) vs C" {
    const SimT = sim_mod.Sim(cfg_hydro);
    const g = Grid.init(.{ .nx = 64, .ng = 3, .minx = 0, .maxx = 1 });
    try runStepTest(SimT, "step/sod64.kstp", "step sod64", g, .{}, .{
        .coords = .mink,
        .gam = 5.0 / 3.0,
        .bc_x = .copy,
    });
}

test "golden step: mhdtube64 (Balsara-1, Γ=2, 10 forced-dt steps) vs C" {
    const SimT = sim_mod.Sim(cfg_mhd);
    const g = Grid.init(.{ .nx = 64, .ng = 3, .minx = 0, .maxx = 1 });
    try runStepTest(SimT, "step/mhdtube64.kstp", "step mhdtube64", g, .{}, .{
        .coords = .mink,
        .gam = 2.0,
        .bc_x = .copy,
    });
}

test "golden step: ot32 (SR Orszag-Tang 32², periodic, 10 forced-dt steps) vs C" {
    const SimT = sim_mod.Sim(cfg_mhd);
    const tp = 2.0 * std.math.pi;
    const g = Grid.init(.{ .nx = 32, .ny = 32, .ng = 3, .minx = 0, .maxx = tp, .miny = 0, .maxy = tp });
    try runStepTest(SimT, "step/ot32.kstp", "step ot32", g, .{}, .{
        .coords = .mink,
        .gam = 5.0 / 3.0,
        .bc_x = .periodic,
        .bc_y = .periodic,
    });
}

//
// ---- M10: radiative step tests --------------------------------------------------
//

const SimRad = sim_mod.Sim(tubes.cfg_rad);

/// ZIGRADTUBE/bc.c: Dirichlet: ghosts pinned to the tube-2 states.
fn radtubeBc(
    ctx: ?*const anyopaque,
    s: *const SimRad,
    ix: i64,
    iy: i64,
    iz: i64,
    t: f64,
    ifinit: bool,
    face: sim_mod.BcFace,
) relele.Error![SimRad.nv]f64 {
    _ = ctx;
    _ = s;
    _ = iy;
    _ = iz;
    _ = t;
    _ = ifinit;
    _ = face;
    const tube = &tubes.tubes[1];
    return tubes.ppSide(tubes.cfg_rad, if (ix < 0) tube.left else tube.right, tube.gam);
}

test "golden step: radtube64 (Farris tube 2, grey κ, 10 forced-dt steps) vs C" {
    const tube = &tubes.tubes[1];
    const g = Grid.init(.{ .nx = 64, .ng = 3, .minx = -20, .maxx = 20 });
    // base bumped 2.0e-10 -> 5.0e-10: the koral_lite entropy-derivative fix
    // (dwmrho0dW = 1/gamma -> 1/gamma^2) shifts a shock-adjacent cell's
    // rung/branch selection at step 3, jumping the deviation to ~1.6e-9
    // (then plateauing ~1e-9) instead of the old smooth decade growth.
    try runStepTest(SimRad, "step/radtube64.kstp", "step radtube64", g, .{ .base = 5.0e-10, .steps_per_decade = 2.5 }, .{
        .coords = .mink,
        .gam = tube.gam,
        .bc_x = .specific,
        .specific_bc = &radtubeBc,
        .opac = tube.params(),
        .implicit = implicit.ImplicitParams.puffy,
    });
}

test "golden step: radpulse64 (τ-thick LTE pulse, 10 forced-dt steps) vs C" {
    const g = Grid.init(.{ .nx = 64, .ng = 3, .minx = -50, .maxx = 50 });
    try runStepTest(SimRad, "step/radpulse64.kstp", "step radpulse64", g, .{ .base = 2.0e-10, .steps_per_decade = 2.5 }, .{
        .coords = .mink,
        .gam = 5.0 / 3.0,
        .bc_x = .copy,
        .opac = tubes.pulse.params(),
        .implicit = implicit.ImplicitParams.puffy,
    });
}

//
// ---- flux_ct golden -----------------------------------------------------------
//

test "golden: flux_ct (Tóth EMF averaging) vs C" {
    const a = std.testing.allocator;
    var gld = try golden.readGolden(a, "flux/ct.kgld", 6, 3);
    defer gld.deinit();

    const SimT = sim_mod.Sim(cfg_mhd);
    const L = SimT.Layout;
    const tp = 2.0 * std.math.pi;
    const g = Grid.init(.{ .nx = 32, .ny = 32, .ng = 3, .minx = 0, .maxx = tp, .miny = 0, .maxy = tp });
    var s = try SimT.init(a, g, .{ .coords = .mink, .bc_x = .periodic, .bc_y = .periodic });
    defer s.deinit();

    for (0..gld.nrec) |i| {
        const r = gld.rec(i);
        const dim: usize = @intFromFloat(r.in[0]);
        const ix: i64 = @intFromFloat(r.in[1]);
        const iy: i64 = @intFromFloat(r.in[2]);
        s.flb[dim].set(L.index(.b1), ix, iy, 0, r.in[3]);
        s.flb[dim].set(L.index(.b2), ix, iy, 0, r.in[4]);
        s.flb[dim].set(L.index(.b3), ix, iy, 0, r.in[5]);
    }

    ct.fluxCt(SimT, &s);

    var dev = golden.DevTracker{};
    for (0..gld.nrec) |i| {
        const r = gld.rec(i);
        const dim: usize = @intFromFloat(r.in[0]);
        const ix: i64 = @intFromFloat(r.in[1]);
        const iy: i64 = @intFromFloat(r.in[2]);
        dev.add(r.out[0], s.flb[dim].get(L.index(.b1), ix, iy, 0), i);
        dev.add(r.out[1], s.flb[dim].get(L.index(.b2), ix, iy, 0), i);
        dev.add(r.out[2], s.flb[dim].get(L.index(.b3), ix, iy, 0), i);
    }
    try dev.check(1.0e-15, "flux_ct");
}

//
// ---- calc_BfromA golden ---------------------------------------------------------
//

test "golden: calc_BfromA (corner average + curl, overwrite) vs C" {
    const a = std.testing.allocator;
    var gld = try golden.readGolden(a, "flux/bfroma.kgld", 5, 3);
    defer gld.deinit();

    const SimT = sim_mod.Sim(cfg_mhd);
    const L = SimT.Layout;
    const gam: f64 = 5.0 / 3.0;
    const tp = 2.0 * std.math.pi;
    const g = Grid.init(.{ .nx = 32, .ny = 32, .ng = 3, .minx = 0, .maxx = tp, .miny = 0, .maxy = tp });
    var s = try SimT.init(a, g, .{ .coords = .mink, .gam = gam, .bc_x = .periodic, .bc_y = .periodic });
    defer s.deinit();

    // uniform hydro background (B result is independent of it), A from record
    for (0..gld.nrec) |i| {
        const r = gld.rec(i);
        const ix: i64 = @intFromFloat(r.in[0]);
        const iy: i64 = @intFromFloat(r.in[1]);
        var pp: [SimT.nv]f64 = @splat(0);
        pp[L.index(.rho)] = 1.0;
        pp[L.index(.uu)] = 0.1;
        pp[L.index(.entr)] = hydro.sFromU(1.0, 0.1, gam);
        pp[L.index(.b1)] = r.in[2];
        pp[L.index(.b2)] = r.in[3];
        pp[L.index(.b3)] = r.in[4];
        try s.initCell(ix, iy, 0, pp);
    }
    try s.setBc(0.0, true);
    try ct.calcBfromA(SimT, &s, true);

    var dev = golden.DevTracker{};
    for (0..gld.nrec) |i| {
        const r = gld.rec(i);
        const ix: i64 = @intFromFloat(r.in[0]);
        const iy: i64 = @intFromFloat(r.in[1]);
        dev.add(r.out[0], s.p.get(L.index(.b1), ix, iy, 0), i);
        dev.add(r.out[1], s.p.get(L.index(.b2), ix, iy, 0), i);
        dev.add(r.out[2], s.p.get(L.index(.b3), ix, iy, 0), i);
    }
    try dev.check(1.0e-14, "calc_BfromA");
}
