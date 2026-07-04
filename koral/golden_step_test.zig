//! M5/M6 C-comparison: forced-dt step tests + flux_ct / calc_BfromA goldens.
//!
//! Step tests (plan M5/M6): the C oracle (oracle/harness_step.c) runs 10
//! RK2IMEX steps of each tiny problem, recording t, dt and the full domain
//! u/p each step. The Zig side loads C's post-init state bit-for-bit,
//! forces C's recorded dt sequence, and diffs the domain each step against
//! a budget growing from 1e-13 (step 1) to 1e-10 (step 10). Zig's own CFL
//! dt is separately required to match C's at 1e-12.
//!
//! Deviations are normalized per variable by the max |value| over the C
//! record (momentum rows pass through zero; cf. the geometry-in-record
//! golden notes in README).

const std = @import("std");
const config = @import("config.zig");
const grid_mod = @import("grid.zig");
const sim_mod = @import("sim.zig");
const ct = @import("magn/ct.zig");
const hydro = @import("physics/hydro.zig");
const golden = @import("testing/golden.zig");

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

fn runStepTest(
    comptime SimT: type,
    comptime relpath: []const u8,
    comptime what: []const u8,
    g: Grid,
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

    for (1..k.nrec) |istep| {
        const r = k.rec(istep);
        try s.step(r.dt);

        // C's dt is what Zig's CFL logic would also have chosen
        try std.testing.expectApproxEqRel(r.dt, s.own_dt, 1e-12);
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

        const budget = 1.0e-13 * std.math.pow(f64, 10.0, @as(f64, @floatFromInt(istep - 1)) / 3.0);
        var step_max: f64 = 0;
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
                        step_max = @max(step_max, @abs(cu - zu) / scale[iv]);
                        const cp = r.p[k.idx(iv, ix, iy, iz)];
                        const zp = s.p.get(iv, zx, zy, zz);
                        dev_p.add(cp / scale_p[iv], zp / scale_p[iv], istep);
                        step_max = @max(step_max, @abs(cp - zp) / scale_p[iv]);
                    }
                    // flags: ENTROPYFLAG, HDFIXUPFLAG must agree exactly
                    const cfe: i32 = @intFromFloat(r.flags[ncell * 2]);
                    const cff: i32 = @intFromFloat(r.flags[ncell * 2 + 1]);
                    try std.testing.expectEqual(cfe, s.getFlag(.entropy, zx, zy, zz));
                    try std.testing.expectEqual(cff, s.getFlag(.hd_fixup, zx, zy, zz));
                    ncell += 1;
                }
            }
        }
        worst_budget = @max(worst_budget, step_max / budget);
        if (step_max > budget) {
            std.debug.print("step {d}: max dev {e:.3} exceeds budget {e:.3}\n", .{ istep, step_max, budget });
            return error.GoldenMismatch;
        }
    }

    try dev_u.check(1.0e-10, what ++ " conserveds");
    try dev_p.check(1.0e-10, what ++ " primitives");
    std.debug.print("golden [{s}]: worst dev/budget ratio {d:.3}\n", .{ what, worst_budget });
}

test "golden step: sod64 (SR Sod, 10 forced-dt RK2IMEX steps) vs C" {
    const SimT = sim_mod.Sim(cfg_hydro);
    const g = Grid.init(.{ .nx = 64, .ng = 3, .minx = 0, .maxx = 1 });
    try runStepTest(SimT, "step/sod64.kstp", "step sod64", g, .{
        .coords = .mink,
        .gam = 5.0 / 3.0,
        .bc_x = .copy,
    });
}

test "golden step: mhdtube64 (Balsara-1, Γ=2, 10 forced-dt steps) vs C" {
    const SimT = sim_mod.Sim(cfg_mhd);
    const g = Grid.init(.{ .nx = 64, .ng = 3, .minx = 0, .maxx = 1 });
    try runStepTest(SimT, "step/mhdtube64.kstp", "step mhdtube64", g, .{
        .coords = .mink,
        .gam = 2.0,
        .bc_x = .copy,
    });
}

test "golden step: ot32 (SR Orszag-Tang 32², periodic, 10 forced-dt steps) vs C" {
    const SimT = sim_mod.Sim(cfg_mhd);
    const tp = 2.0 * std.math.pi;
    const g = Grid.init(.{ .nx = 32, .ny = 32, .ng = 3, .minx = 0, .maxx = tp, .miny = 0, .maxy = tp });
    try runStepTest(SimT, "step/ot32.kstp", "step ot32", g, .{
        .coords = .mink,
        .gam = 5.0 / 3.0,
        .bc_x = .periodic,
        .bc_y = .periodic,
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
