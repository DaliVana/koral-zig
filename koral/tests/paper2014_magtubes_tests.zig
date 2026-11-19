//! Reproduction gate for Sadowski et al. (2014, MNRAS 439, 503), section 4.2,
//! Table 3 and Figures 4-5: the Ryu & Jones (1995) magnetic shock tubes 2a
//! and 5a, scaled into the non-relativistic regime of a relativistic code by
//! a velocity factor C = 100 (v -> v/C, p -> p/C^2, B -> B/C, t -> t C).
//!
//! The paper compares by eye with published plots. Here the evolved state
//! is checked against the exact structure of the solution instead:
//!
//!  * RANKINE-HUGONIOT. Across each isolated discontinuity the seven
//!    non-relativistic ideal-MHD jump conditions F(U_a) - s U_a =
//!    F(U_b) - s U_b must hold with ONE speed s. s is taken from the mass
//!    flux and the other six conditions are the test. For 2a this is done
//!    for both fast shocks (upstream = the exact initial state, downstream
//!    = the computed plateau); for 5a for the slow shock, and the contact is
//!    checked for continuity of v, p and B.
//!  * the normal field B^x stays exactly constant (flux-CT in 1D),
//!  * the total mass in the tube changes by exactly the boundary mass flux
//!    of the undisturbed states,
//!  * self-convergence of the density in L1 (first order at shocks).
//!
//! Units. Ryu & Jones write B in units with magnetic pressure B^2 / 2, i.e.
//! their "B / sqrt(4 pi)" numbers are this code's B directly; Table 3's
//! "/ 4 pi C" is that 1 / sqrt(4 pi) / C.
//!
//! One correction to Table 3: it lists B^z = 4 for the right state of 2a.
//! Ryu & Jones (1995) have B^z = 2 on both sides, and the paper's own Figure
//! 4 shows B^z ~ 0.56 = 2 / sqrt(4 pi) at both ends, so B^z_R = 2 is used.
//!
//! Final times. Figure 4 (2a) shows the fast shocks at x ~ 0.30 and 0.95,
//! the Ryu & Jones state at their t = 0.2, i.e. t = 20 here; Figure 5 (5a)
//! is their t = 0.1, t = 10 here. (The text quotes t = 10 for both.)
//!
//! Numerics as in the paper: 512 cells, linear reconstruction with minmod
//! theta = 2, Lax-Friedrichs, RK2.

const std = @import("std");
const figdata = @import("../testing/figdata.zig");
const build_options = @import("build_options");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const hydro = @import("../physics/hydro.zig");

const Grid = grid_mod.Grid;
const expect = std.testing.expect;

const cfg = config.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .linear,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;
const NV = SimT.nv;

const gam: f64 = 5.0 / 3.0;
const cscale: f64 = 100.0;
const s4pi: f64 = 3.5449077018110318; // sqrt(4 pi)
/// C's U2PCONV, the production inversion tolerance.
const c_u2pconv: f64 = 1.0e-10;

/// One side in Ryu & Jones units (B already divided by sqrt(4 pi)).
const State = struct {
    rho: f64,
    p: f64,
    v: [3]f64,
    b: [3]f64,
};

const Tube = struct {
    name: []const u8,
    left: State,
    right: State,
    /// final time in Ryu & Jones units
    t_rj: f64,
};

const tube2a = Tube{
    .name = "2a",
    .left = .{ .rho = 1.08, .p = 0.95, .v = .{ 1.2, 0.01, 0.5 }, .b = .{ 2.0 / s4pi, 3.6 / s4pi, 2.0 / s4pi } },
    .right = .{ .rho = 1.0, .p = 1.0, .v = .{ 0, 0, 0 }, .b = .{ 2.0 / s4pi, 4.0 / s4pi, 2.0 / s4pi } },
    .t_rj = 0.2,
};
const tube5a = Tube{
    .name = "5a",
    .left = .{ .rho = 1.0, .p = 1.0, .v = .{ 0, 0, 0 }, .b = .{ 0.75, 1.0, 0 } },
    .right = .{ .rho = 0.125, .p = 0.1, .v = .{ 0, 0, 0 }, .b = .{ 0.75, -1.0, 0 } },
    .t_rj = 0.1,
};

/// Evolved profile, scaled back to Ryu & Jones units.
const Profile = struct {
    a: std.mem.Allocator,
    x: []f64,
    s: []State,
    dx: f64,
    /// total mass in the tube at t = 0 and at the end
    mass0: f64,
    mass1: f64,

    fn deinit(p: *Profile) void {
        p.a.free(p.x);
        p.a.free(p.s);
    }

    /// Mean state over [x0, x1] (a plateau).
    fn plateau(p: *const Profile, x0: f64, x1: f64) State {
        var m = State{ .rho = 0, .p = 0, .v = .{ 0, 0, 0 }, .b = .{ 0, 0, 0 } };
        var n: f64 = 0;
        for (p.x, p.s) |x, st| {
            if (x < x0 or x > x1) continue;
            m.rho += st.rho;
            m.p += st.p;
            for (0..3) |k| {
                m.v[k] += st.v[k];
                m.b[k] += st.b[k];
            }
            n += 1;
        }
        m.rho /= n;
        m.p /= n;
        for (0..3) |k| {
            m.v[k] /= n;
            m.b[k] /= n;
        }
        return m;
    }

    /// Largest relative spread of any variable over [x0, x1]; small on a
    /// genuine plateau.
    fn flatness(p: *const Profile, x0: f64, x1: f64) f64 {
        const m = p.plateau(x0, x1);
        var worst: f64 = 0;
        for (p.x, p.s) |x, st| {
            if (x < x0 or x > x1) continue;
            worst = @max(worst, @abs(st.rho - m.rho) / m.rho);
            worst = @max(worst, @abs(st.p - m.p) / m.p);
            for (0..3) |k| {
                worst = @max(worst, @abs(st.v[k] - m.v[k]));
                worst = @max(worst, @abs(st.b[k] - m.b[k]));
            }
        }
        return worst;
    }
};

fn run(a: std.mem.Allocator, t: *const Tube, nx: usize, u2pconv: f64) !Profile {
    const g = Grid.init(.{ .nx = nx, .ng = 2, .minx = 0, .maxx = 1 });
    var s = try SimT.init(a, g, .{
        .phys = .{ .coords = .mink, .gam = gam, .floors = .{ .u2pconv = u2pconv } },
        .num = .{ .minmod_theta = 2.0 },
        .bc = .{ .x = .copy },
    });
    defer s.deinit();

    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const st = if (g.xc(ix) < 0.5) t.left else t.right;
        var pp: [NV]f64 = @splat(0);
        const uint = st.p / (cscale * cscale) / (gam - 1.0);
        pp[L.index(.rho)] = st.rho;
        pp[L.index(.uu)] = uint;
        pp[L.index(.entr)] = hydro.sFromU(st.rho, uint, gam);
        // VELR primitive = u^i in Minkowski
        var vsq: f64 = 0;
        for (0..3) |k| vsq += (st.v[k] / cscale) * (st.v[k] / cscale);
        const lor = 1.0 / @sqrt(1.0 - vsq);
        for (0..3) |k| {
            pp[L.index(.vx) + k] = lor * st.v[k] / cscale;
            pp[L.index(.b1) + k] = st.b[k] / cscale;
        }
        try s.initCell(ix, 0, 0, pp);
    }
    try s.finishInit();

    var mass0: f64 = 0;
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) mass0 += lorentzRho(&s, ix) * g.dx;

    const tend = t.t_rj * cscale;
    var n_fix: usize = 0;
    var n_entr: usize = 0;
    while (s.t < tend) {
        var dt: ?f64 = null;
        if (s.t + 1.0 / s.core.tstepdenmax > tend) dt = tend - s.t;
        try s.step(dt);
        var jx: i64 = 0;
        while (jx < s.nxi()) : (jx += 1) {
            if (s.getFlag(.hd_fixup, jx, 0, 0) != 0) n_fix += 1;
            if (s.getFlag(.entropy, jx, 0, 0) != 0) n_entr += 1;
        }
    }
    // neither tube needs a fixup or the entropy fallback anywhere
    try std.testing.expectEqual(@as(usize, 0), n_fix);
    try std.testing.expectEqual(@as(usize, 0), n_entr);

    var prof = Profile{
        .a = a,
        .x = try a.alloc(f64, nx),
        .s = try a.alloc(State, nx),
        .dx = g.dx,
        .mass0 = mass0,
        .mass1 = 0,
    };
    errdefer prof.deinit();
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const j: usize = @intCast(ix);
        prof.x[j] = g.xc(ix);
        var st: State = undefined;
        st.rho = s.core.p.get(L.index(.rho), ix, 0, 0);
        st.p = (gam - 1.0) * s.core.p.get(L.index(.uu), ix, 0, 0) * cscale * cscale;
        var usq: f64 = 0;
        for (0..3) |k| {
            const u = s.core.p.get(L.index(.vx) + k, ix, 0, 0);
            usq += u * u;
        }
        const lor = @sqrt(1.0 + usq);
        for (0..3) |k| {
            st.v[k] = s.core.p.get(L.index(.vx) + k, ix, 0, 0) / lor * cscale;
            st.b[k] = s.core.p.get(L.index(.b1) + k, ix, 0, 0) * cscale;
        }
        prof.s[j] = st;
        prof.mass1 += lorentzRho(&s, ix) * g.dx;
        // flux-CT in 1D: the normal field never changes, bit for bit
        try std.testing.expectEqual(t.left.b[0] / cscale, s.core.p.get(L.index(.b1), ix, 0, 0));
    }
    return prof;
}

/// The conserved rest-mass density rho u^t, read from the conserved field
/// itself (the primitives only reproduce it to the inversion tolerance).
fn lorentzRho(s: *const SimT, ix: i64) f64 {
    return s.core.u.get(L.index(.rho), ix, 0, 0);
}

/// Conserved variables and x-fluxes of non-relativistic ideal MHD
/// (magnetic pressure B^2 / 2): rho, momentum x y z, B^y, B^z, energy.
fn consFlux(st: State) struct { u: [7]f64, f: [7]f64 } {
    const bsq = st.b[0] * st.b[0] + st.b[1] * st.b[1] + st.b[2] * st.b[2];
    const vsq = st.v[0] * st.v[0] + st.v[1] * st.v[1] + st.v[2] * st.v[2];
    const vb = st.v[0] * st.b[0] + st.v[1] * st.b[1] + st.v[2] * st.b[2];
    const e = st.p / (gam - 1.0) + 0.5 * st.rho * vsq + 0.5 * bsq;
    const ptot = st.p + 0.5 * bsq;
    return .{
        .u = .{ st.rho, st.rho * st.v[0], st.rho * st.v[1], st.rho * st.v[2], st.b[1], st.b[2], e },
        .f = .{
            st.rho * st.v[0],
            st.rho * st.v[0] * st.v[0] + ptot - st.b[0] * st.b[0],
            st.rho * st.v[0] * st.v[1] - st.b[0] * st.b[1],
            st.rho * st.v[0] * st.v[2] - st.b[0] * st.b[2],
            st.v[0] * st.b[1] - st.v[1] * st.b[0],
            st.v[0] * st.b[2] - st.v[2] * st.b[0],
            (e + ptot) * st.v[0] - st.b[0] * vb,
        },
    };
}

const Jump = struct { speed: f64, worst: f64 };

/// Shock speed from the mass flux, then the worst residual of the other six
/// jump conditions, each normalized by the size of its own two terms.
fn jump(a: State, b: State) Jump {
    const ca = consFlux(a);
    const cb = consFlux(b);
    const s = (ca.f[0] - cb.f[0]) / (ca.u[0] - cb.u[0]);
    var worst: f64 = 0;
    for (1..7) |k| {
        const lhs = ca.f[k] - s * ca.u[k];
        const rhs = cb.f[k] - s * cb.u[k];
        const scale = @max(@max(@abs(ca.f[k]), @abs(cb.f[k])), @max(@abs(s * ca.u[k]), @abs(s * cb.u[k])));
        if (scale > 0) worst = @max(worst, @abs(lhs - rhs) / scale);
    }
    return .{ .speed = s, .worst = worst };
}

/// Figure data: the whole profile in Ryu & Jones units.
fn dumpProfile(comptime name: []const u8, t: *const Tube, p: *const Profile) void {
    var fig = figdata.Table.init(name);
    defer fig.finish();
    fig.meta("source", "Sadowski et al. 2014, section 4.2: Ryu & Jones tube {s} at t = {d} (their units), {d} cells", .{ t.name, t.t_rj, p.x.len });
    fig.meta("columns", "x rho p vx vy vz By Bz", .{});
    for (p.x, p.s) |x, st| fig.row(&.{ x, st.rho, st.p, st.v[0], st.v[1], st.v[2], st.b[1], st.b[2] });
}

fn maxAbs3(a: [3]f64, b: [3]f64) f64 {
    return @max(@abs(a[0] - b[0]), @max(@abs(a[1] - b[1]), @abs(a[2] - b[2])));
}

/// L1 difference of the density between a coarse run and a finer reference,
/// the reference averaged onto the coarse cells.
fn l1Rho(coarse: *const Profile, fine: *const Profile) f64 {
    const ratio = fine.x.len / coarse.x.len;
    var sum: f64 = 0;
    for (coarse.s, 0..) |st, j| {
        var avg: f64 = 0;
        for (0..ratio) |k| avg += fine.s[j * ratio + k].rho;
        sum += @abs(st.rho - avg / @as(f64, @floatFromInt(ratio)));
    }
    return sum / @as(f64, @floatFromInt(coarse.x.len));
}

test "Sadowski 2014 section 4.2: Ryu-Jones 2a, fast shocks satisfy the MHD jump conditions" {
    const a = std.testing.allocator;
    var p = try run(a, &tube2a, 512, c_u2pconv);
    defer p.deinit();
    dumpProfile("k14_magtube_2a", &tube2a, &p);

    // Wave pattern at t = 0.2 (Ryu & Jones units): fast shock at 0.30,
    // rotational discontinuity 0.52, slow shock 0.55, contact 0.61, slow
    // shock 0.68, rotational discontinuity 0.71, fast shock 0.95.
    const post_left = p.plateau(0.42, 0.50);
    const post_right = p.plateau(0.76, 0.92);
    const flat_l = p.flatness(0.42, 0.50);
    const flat_r = p.flatness(0.76, 0.92);

    // left-going fast shock: exact left state -> computed plateau
    const jl = jump(tube2a.left, post_left);
    // right-going fast shock: computed plateau -> exact right state
    const jr = jump(post_right, tube2a.right);
    std.debug.print(
        "paper 2014 tube 2a: left fast shock s={d:.4} RH residual {e:.2}; right fast shock s={d:.4} RH residual {e:.2}; plateau flatness {e:.1} {e:.1}\n",
        .{ jl.speed, jl.worst, jr.speed, jr.worst, flat_l, flat_r },
    );
    std.debug.print(
        "paper 2014 tube 2a: post-shock state rho {d:.4} p {d:.4} v ({d:.5}, {d:.5}, {d:.5}) By {d:.4} Bz {d:.4}\n",
        .{ post_left.rho, post_left.p, post_left.v[0], post_left.v[1], post_left.v[2], post_left.b[1], post_left.b[2] },
    );
    try expect(flat_l <= 2e-3 and flat_r <= 2e-3);
    try expect(jl.worst <= 1.5e-3);
    try expect(jr.worst <= 1.5e-3);
    // shock positions 0.5 + s t agree with where the profile jumps
    try expect(@abs(0.5 + jl.speed * tube2a.t_rj - 0.305) <= 0.01);
    try expect(@abs(0.5 + jr.speed * tube2a.t_rj - 0.95) <= 0.01);

    // contact discontinuity: only the density jumps
    const cl = p.plateau(0.57, 0.595);
    const cr = p.plateau(0.635, 0.665);
    try expect(@abs(cl.rho - cr.rho) >= 0.1);
    try expect(@abs(cl.p - cr.p) / cl.p <= 2e-3);
    try expect(maxAbs3(cl.v, cr.v) <= 2e-3);
    try expect(maxAbs3(cl.b, cr.b) <= 2e-3);
}

test "Sadowski 2014 section 4.2: Ryu-Jones 5a, slow shock and contact" {
    const a = std.testing.allocator;
    var p = try run(a, &tube5a, 512, c_u2pconv);
    defer p.deinit();
    dumpProfile("k14_magtube_5a", &tube5a, &p);

    // Wave pattern at t = 0.1: fast rarefaction 0.32-0.44, slow compound
    // wave 0.47, contact 0.56, slow shock 0.64, fast rarefaction 0.82-0.87.
    const pre = p.plateau(0.59, 0.63);
    const post = p.plateau(0.66, 0.80);
    const js = jump(pre, post);
    std.debug.print(
        "paper 2014 tube 5a: slow shock s={d:.4} RH residual {e:.2}; post-shock rho {d:.5} vx {d:.5} (plateau spread {e:.1})\n",
        .{ js.speed, js.worst, post.rho, post.v[0], p.flatness(0.66, 0.80) },
    );
    // The theta = 2 limiter leaves a few-per-cent ripple behind this shock
    // (the paper's Figure 5 has it too); the plateau MEAN obeys the jump
    // conditions.
    try expect(js.worst <= 2e-2);
    try expect(@abs(0.5 + js.speed * tube5a.t_rj - 0.64) <= 0.01);

    const cl = p.plateau(0.50, 0.55);
    const cr = p.plateau(0.585, 0.63);
    try expect(cl.rho / cr.rho >= 2.0);
    try expect(@abs(cl.p - cr.p) / cl.p <= 1e-2);
    try expect(maxAbs3(cl.v, cr.v) <= 1e-2);
    try expect(maxAbs3(cl.b, cr.b) <= 1e-2);

    // the transverse field never leaves the x-y plane
    for (p.s) |st| try expect(st.b[2] == 0 and st.v[2] == 0);
}

test "Sadowski 2014 section 4.2: Ryu-Jones tubes are self-convergent in L1" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;
    for ([_]*const Tube{ &tube2a, &tube5a }) |t| {
        var r128 = try run(a, t, 128, c_u2pconv);
        defer r128.deinit();
        var r256 = try run(a, t, 256, c_u2pconv);
        defer r256.deinit();
        var r1024 = try run(a, t, 1024, c_u2pconv);
        defer r1024.deinit();
        const d1 = l1Rho(&r128, &r1024);
        const d2 = l1Rho(&r256, &r1024);
        const order = std.math.log2(d1 / d2);
        std.debug.print("paper 2014 tube {s}: L1(rho) 128 vs 1024 {e:.3}, 256 vs 1024 {e:.3}, order {d:.2}\n", .{ t.name, d1, d2, order });
        try expect(order >= 0.6); // first order at the discontinuities
    }
}

test "Sadowski 2014 section 4.2: tube mass balance is exact up to the inversion tolerance" {
    // The update is in flux form, so the mass in the tube can change only
    // through the boundary flux of the undisturbed end states: rho u^x t on
    // the inflow side of 2a, nothing for 5a. That holds to roundoff until
    // step ~130, when the first numerical signal (2 cells per step, far ahead
    // of any physical wave) reaches the open ends. What arrives is roundoff,
    // but the inversion turns it into a perturbation of order U2PCONV in
    // W = rho h gamma^2, which in this non-relativistic scaling (u / rho ~
    // 1e-4) is 1e-6 in the pressure and 1e-7 in the density; that then leaves
    // through the copy boundaries. Measured mass error: 1e-7 relative at C's
    // tolerance, 2.5e-11 at 1e-14. No fixup and no entropy fallback fires
    // (asserted in run), so this is the whole budget.
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;
    var usq: f64 = 0;
    for (0..3) |k| usq += (tube2a.left.v[k] / cscale) * (tube2a.left.v[k] / cscale);
    const inflow = tube2a.left.rho * (tube2a.left.v[0] / cscale) / @sqrt(1.0 - usq) * tube2a.t_rj * cscale;
    for ([_]f64{ c_u2pconv, 1.0e-14 }) |conv| {
        var p2a = try run(a, &tube2a, 512, conv);
        defer p2a.deinit();
        var p5a = try run(a, &tube5a, 512, conv);
        defer p5a.deinit();
        const e2a = @abs((p2a.mass1 - p2a.mass0) / inflow - 1.0);
        const e5a = @abs(p5a.mass1 / p5a.mass0 - 1.0);
        std.debug.print("paper 2014 tubes: mass balance error at u2pconv {e:.0}: 2a {e:.2}, 5a {e:.2}\n", .{ conv, e2a, e5a });
        const bound: f64 = if (conv == c_u2pconv) 1.0e-6 else 1.0e-9;
        try expect(e2a <= bound and e5a <= bound);
    }
}
