//! M10 theory gates for the full radiative RK2IMEX step (Sim.step with
//! op_implicit active):
//!
//!  * κ = 0 ⇒ RK2IMEX ≡ its explicit RK2 part. With G ≡ 0 the implicit
//!    stage degenerates to the FORCEUEQPINIMPLICIT p2u∘u2p projection plus
//!    Newton exits by relative change (RADIMPCONVREL = 1e-8), each call
//!    legitimately moving a converged cell by up to ~1e-8; in C exactly
//!    as here. 2 calls × 5 steps ⇒ agreement at ~1e-7 (measured 1.2e-7),
//!    gated at 1e-6.
//!  * temporal order ≈ 2 on a smooth 0-D gas-radiation relaxation driven
//!    through the full step machinery (uniform periodic grid: the explicit
//!    operator telescopes to zero and only the IMEX source arithmetic acts).
//!  * L-stability through the step: κρΔt = 1e4 relaxation is monotone with
//!    no oscillation around the equilibrium (Pareschi-Russo SSP2(2,2,2)
//!    with γ = 1 − 1/√2 has R(∞) = 0).
//!  * optically thin streaming pulse propagates at the M1 streaming speed.
//!  * optically thick scattering pulse spreads diffusively, σ²(t) growing
//!    at 2D with D = 1/(3χ).

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const units_mod = @import("../units.zig");
const hydro = @import("../physics/hydro.zig");
const radforce = @import("../physics/radforce.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const implicit = @import("../solve/implicit.zig");
const tubes = @import("../testing/tubes.zig");

const cfg = tubes.cfg_rad;
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;
const NV = SimT.nv;
const Grid = grid_mod.Grid;

const gam: f64 = 5.0 / 3.0;

fn simOptions(opac: ?radforce.Params) SimT.Options {
    return .{
        .coords = .mink,
        .gam = gam,
        .rad = invert_rad.RadParams.cdefault,
        .opac = opac,
        .implicit = implicit.ImplicitParams.puffy,
        .bc_x = .periodic,
    };
}

// ---- κ = 0: IMEX ≡ explicit RK2 up to inversion tolerance -----------------

test "kappa=0: RK2IMEX with implicit active ≡ SKIPRADSOURCE run" {
    const a = std.testing.allocator;
    const mass = tubes.tubes[1].mass();
    const g = Grid.init(.{ .nx = 32, .ng = 3, .minx = 0, .maxx = 1 });

    var s_null = try SimT.init(a, g, simOptions(null));
    defer s_null.deinit();
    var s_zero = try SimT.init(a, g, simOptions(
        radforce.Params.grey(units_mod.Units.init(mass), tubes.comp, 0.0, 0.0),
    ));
    defer s_zero.deinit();

    // smooth periodic state with gradients in every family
    var ix: i64 = 0;
    while (ix < g.nx) : (ix += 1) {
        const x = g.xc(ix);
        const sx = @sin(2.0 * std.math.pi * x);
        var pp: [NV]f64 = @splat(0);
        pp[L.index(.rho)] = 1.0 + 0.1 * sx;
        pp[L.index(.uu)] = 0.01 * (1.0 + 0.05 * sx);
        pp[L.index(.vx)] = 0.02 * sx;
        pp[L.index(.entr)] = hydro.sFromU(pp[L.index(.rho)], pp[L.index(.uu)], gam);
        pp[L.index(.ee)] = 0.005 * (1.0 + 0.1 * @cos(2.0 * std.math.pi * x));
        pp[L.index(.fx)] = 0.01 * sx;
        try s_null.initCell(ix, 0, 0, pp);
        try s_zero.initCell(ix, 0, 0, pp);
    }
    try s_null.finishInit();
    try s_zero.finishInit();

    for (0..5) |_| {
        try s_null.step(null);
        try s_zero.step(null);
    }

    // per-variable row-max normalization (velocity rows cross zero)
    var scale: [NV]f64 = @splat(1e-10);
    ix = 0;
    while (ix < g.nx) : (ix += 1) {
        var pa: [NV]f64 = undefined;
        s_null.p.load(ix, 0, 0, &pa);
        for (0..NV) |iv| scale[iv] = @max(scale[iv], @abs(pa[iv]));
    }
    var worst: f64 = 0;
    ix = 0;
    while (ix < g.nx) : (ix += 1) {
        var pa: [NV]f64 = undefined;
        var pb: [NV]f64 = undefined;
        s_null.p.load(ix, 0, 0, &pa);
        s_zero.p.load(ix, 0, 0, &pb);
        for (0..NV) |iv| {
            worst = @max(worst, @abs(pa[iv] - pb[iv]) / scale[iv]);
        }
    }
    std.debug.print("kappa=0 IMEX vs SKIPRADSOURCE: worst rel dev {e:.3}\n", .{worst});
    try std.testing.expect(worst <= 1.0e-6);
}

// ---- 0-D relaxation through the full step ----------------------------------

/// uniform gas + comoving radiation off LTE; paper units (T = p/ρ),
/// a = tubes.aCode(mass); returns the sim ready to step
fn relaxSim(a: std.mem.Allocator, kabs: f64, mass: f64) !SimT {
    var s = try SimT.init(
        a,
        Grid.init(.{ .nx = 4, .ng = 3, .minx = 0, .maxx = 1 }),
        simOptions(radforce.Params.grey(units_mod.Units.init(mass), tubes.comp, kabs, 0.0)),
    );
    errdefer s.deinit();
    const t_pap: f64 = 4.0e-3; // gas temperature p/ρ
    const ee0 = 3.0 * tubes.aCode(mass) * t_pap * t_pap * t_pap * t_pap; // 3× LTE
    var pp: [NV]f64 = @splat(0);
    pp[L.index(.rho)] = 1.0;
    pp[L.index(.uu)] = t_pap / (gam - 1.0);
    pp[L.index(.entr)] = hydro.sFromU(1.0, pp[L.index(.uu)], gam);
    pp[L.index(.ee)] = ee0;
    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) try s.initCell(ix, 0, 0, pp);
    try s.finishInit();
    return s;
}

fn cellEeUu(s: *const SimT) struct { ee: f64, uu: f64 } {
    var pp: [NV]f64 = undefined;
    s.p.load(0, 0, 0, &pp);
    return .{ .ee = pp[L.index(.ee)], .uu = pp[L.index(.uu)] };
}

test "temporal order ≈ 2 on smooth relaxation (full step machinery)" {
    const a = std.testing.allocator;
    const mass = tubes.tubes[1].mass();
    const kabs: f64 = 2.0;
    const t_end: f64 = 1.5;

    // reference: N = 128
    var e_ref: f64 = undefined;
    var u_ref: f64 = undefined;
    {
        var s = try relaxSim(a, kabs, mass);
        defer s.deinit();
        const n = 128;
        for (0..n) |_| try s.step(t_end / @as(f64, n));
        const c = cellEeUu(&s);
        e_ref = c.ee;
        u_ref = c.uu;
    }

    var errs: [3]f64 = undefined;
    inline for ([_]usize{ 8, 16, 32 }, 0..) |n, i| {
        var s = try relaxSim(a, kabs, mass);
        defer s.deinit();
        for (0..n) |_| try s.step(t_end / @as(f64, n));
        const c = cellEeUu(&s);
        errs[i] = @max(@abs(c.ee - e_ref) / e_ref, @abs(c.uu - u_ref) / u_ref);
    }
    const o1 = std.math.log2(errs[0] / errs[1]);
    const o2 = std.math.log2(errs[1] / errs[2]);
    std.debug.print("relaxation temporal order: errs {e:.3} {e:.3} {e:.3}; orders {d:.2} {d:.2}\n", .{ errs[0], errs[1], errs[2], o1, o2 });
    try std.testing.expect(o1 > 1.7 and o1 < 2.4);
    try std.testing.expect(o2 > 1.7 and o2 < 2.4);
}

test "stiff relaxation (κρΔt = 1e4) is monotone with no oscillation" {
    const a = std.testing.allocator;
    const mass = tubes.tubes[1].mass();
    const kabs: f64 = 100.0;
    var s = try relaxSim(a, kabs, mass);
    defer s.deinit();

    // equilibrium from total-energy conservation: a·((γ−1)u)⁴ = e_tot − u
    const c0 = cellEeUu(&s);
    const e_tot = c0.ee + c0.uu;
    const acode = tubes.aCode(mass);
    var lo: f64 = 0;
    var hi: f64 = e_tot;
    for (0..200) |_| {
        const mid = 0.5 * (lo + hi);
        const tp = (gam - 1.0) * mid; // ρ = 1
        const f = acode * tp * tp * tp * tp - (e_tot - mid);
        if (f > 0) hi = mid else lo = mid;
    }
    const ee_eq = e_tot - 0.5 * (lo + hi);

    // The first huge step lands essentially AT equilibrium (R(∞) = 0); the
    // nonlinear remainder may put it slightly past (measured ~9e-4 of eq) —
    // a single crossing, not oscillation. After that: monotone decay with
    // no re-crossing.
    try s.step(100.0); // κρΔt = 1e4
    const c1 = cellEeUu(&s);
    const d1 = c1.ee - ee_eq;
    std.debug.print("stiff: first-step d/eq {e:.3}\n", .{d1 / ee_eq});
    try std.testing.expect(@abs(d1) <= 1e-2 * ee_eq);

    var prev = @abs(d1);
    const side1: f64 = if (d1 > 0) 1 else -1;
    for (0..5) |_| {
        try s.step(100.0);
        const c = cellEeUu(&s);
        const d = c.ee - ee_eq;
        try std.testing.expect(@abs(d) <= prev * (1.0 + 1e-12) + 1e-12 * ee_eq);
        if (side1 * d < 0) {
            try std.testing.expect(@abs(d) <= 1e-6 * ee_eq);
        }
        prev = @abs(d);
    }
    const cf = cellEeUu(&s);
    try std.testing.expectApproxEqRel(ee_eq, cf.ee, 1e-3);
}

// ---- radiative pulses -------------------------------------------------------

/// centroid and variance of (|u[EE]| − background) over the domain
fn pulseMoments(s: *const SimT, ee_bg: f64) struct { xc: f64, var_: f64 } {
    var m0: f64 = 0;
    var m1: f64 = 0;
    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const e = @abs(s.u.get(L.index(.ee), ix, 0, 0)) - ee_bg;
        const x = s.grid.xc(ix);
        m0 += e;
        m1 += e * x;
    }
    const xc = m1 / m0;
    var m2: f64 = 0;
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const e = @abs(s.u.get(L.index(.ee), ix, 0, 0)) - ee_bg;
        const x = s.grid.xc(ix) - xc;
        m2 += e * x * x;
    }
    return .{ .xc = xc, .var_ = m2 / m0 };
}

test "optically thin pulse streams at the M1 streaming speed (5%)" {
    const a = std.testing.allocator;
    const g = Grid.init(.{ .nx = 128, .ng = 3, .minx = -50, .maxx = 50 });
    var opt = simOptions(null); // κ = 0: SKIPRADSOURCE
    opt.bc_x = .copy;
    var s = try SimT.init(a, g, opt);
    defer s.deinit();

    const urad: f64 = 20.0; // γ ≈ 20 ⇒ v = u/γ ≈ 0.99875
    const v_expect = urad / @sqrt(1.0 + urad * urad);
    const e_bg: f64 = 1.0e-4;
    var ix: i64 = 0;
    while (ix < g.nx) : (ix += 1) {
        const x = g.xc(ix);
        var pp: [NV]f64 = @splat(0);
        pp[L.index(.rho)] = 1.0;
        pp[L.index(.uu)] = 0.01 / (gam - 1.0);
        pp[L.index(.entr)] = hydro.sFromU(1.0, pp[L.index(.uu)], gam);
        pp[L.index(.ee)] = e_bg * (1.0 + 50.0 * @exp(-(x + 25.0) * (x + 25.0) / 25.0));
        pp[L.index(.fx)] = urad;
        try s.initCell(ix, 0, 0, pp);
    }
    try s.finishInit();

    // background R^t_t magnitude for subtraction (streaming: 4/3 γ² − 1/3)
    const ee_bg_lab = e_bg * (4.0 / 3.0 * (1.0 + urad * urad) - 1.0 / 3.0);

    const x0 = pulseMoments(&s, ee_bg_lab).xc;
    const t_end: f64 = 30.0;
    while (s.t < t_end) try s.step(null);
    const x1 = pulseMoments(&s, ee_bg_lab).xc;

    const v = (x1 - x0) / s.t;
    std.debug.print("thin pulse: centroid speed {d:.4} (expect {d:.4})\n", .{ v, v_expect });
    try std.testing.expect(@abs(v - v_expect) <= 0.05);
}

test "optically thick scattering pulse diffuses at D = 1/(3χ)" {
    const a = std.testing.allocator;
    const mass = tubes.tubes[1].mass();
    const kes: f64 = 10.0; // χ = κ_es·ρ = 10, τ_cell ≈ 7.8
    const g = Grid.init(.{ .nx = 256, .ng = 3, .minx = -50, .maxx = 50 });
    var opt = simOptions(radforce.Params.grey(units_mod.Units.init(mass), tubes.comp, 0.0, kes));
    opt.bc_x = .copy;
    var s = try SimT.init(a, g, opt);
    defer s.deinit();

    const e_bg: f64 = 1.0e-6;
    var ix: i64 = 0;
    while (ix < g.nx) : (ix += 1) {
        const x = g.xc(ix);
        var pp: [NV]f64 = @splat(0);
        pp[L.index(.rho)] = 1.0;
        pp[L.index(.uu)] = 0.01 / (gam - 1.0);
        pp[L.index(.entr)] = hydro.sFromU(1.0, pp[L.index(.uu)], gam);
        pp[L.index(.ee)] = e_bg * (1.0 + 3.0 * @exp(-x * x / 25.0));
        try s.initCell(ix, 0, 0, pp);
    }
    try s.finishInit();

    const m0 = pulseMoments(&s, e_bg);
    const t_end: f64 = 90.0;
    while (s.t < t_end) try s.step(null);
    const m1 = pulseMoments(&s, e_bg);

    const d_meas = (m1.var_ - m0.var_) / (2.0 * s.t);
    const d_expect = 1.0 / (3.0 * kes);
    std.debug.print("thick pulse: D {e:.4} (expect {e:.4}, ratio {d:.3})\n", .{ d_meas, d_expect, d_meas / d_expect });
    try std.testing.expect(@abs(d_meas / d_expect - 1.0) <= 0.05);
}
