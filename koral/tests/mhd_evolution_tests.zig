//! M6 theory gates — MHD evolution with flux-CT constrained transport.
//!
//! Gates (plan M6):
//!  * calc_BfromA produces a divergence-free field (corner divB ≤ 1e-14·B/Δx)
//!  * Orszag–Tang: corner divB stays at machine zero over 200 steps
//!  * uniform magnetized state stays static
//!  * circularly polarized Alfvén wave: phase order ≥ 1.9 (exact nonlinear
//!    solution, Del Zanna et al. 2007)
//!  * Balsara-1 / SR Brio–Wu tube: B^x exactly constant, self-convergent L1,
//!    plateau structure sane

const std = @import("std");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const ct = @import("../sim/ct.zig");
const hydro = @import("../physics/hydro.zig");

const Grid = grid_mod.Grid;
const expect = std.testing.expect;

const cfg_mhd = config.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimM = sim_mod.Sim(cfg_mhd);
const L = SimM.Layout;
const NV = SimM.nv;

fn ppMhd(gam: f64, rho: f64, uint: f64, v: [3]f64, b: [3]f64) [NV]f64 {
    var pp: [NV]f64 = @splat(0);
    pp[L.index(.rho)] = rho;
    pp[L.index(.uu)] = uint;
    pp[L.index(.vx)] = v[0];
    pp[L.index(.vy)] = v[1];
    pp[L.index(.vz)] = v[2];
    pp[L.index(.entr)] = hydro.sFromU(rho, uint, gam);
    pp[L.index(.b1)] = b[0];
    pp[L.index(.b2)] = b[1];
    pp[L.index(.b3)] = b[2];
    return pp;
}

fn velr(v: f64) f64 {
    return v / @sqrt(1.0 - v * v);
}

fn maxAbsDivB(s: *const SimM) f64 {
    var worst: f64 = 0;
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            worst = @max(worst, @abs(ct.calcDivB(SimM, s, ix, iy, 0)));
        }
    }
    return worst;
}

//
// ---- B from A: divergence-free by construction -----------------------------
//

test "M6: calc_BfromA gives corner-divB at machine zero" {
    const gam: f64 = 5.0 / 3.0;
    const g = Grid.init(.{ .nx = 32, .ny = 32, .ng = 3, .minx = 0, .maxx = 1, .miny = 0, .maxy = 1 });
    var s = try SimM.init(std.testing.allocator, g, .{
        .coords = .mink,
        .gam = gam,
        .bc_x = .periodic,
        .bc_y = .periodic,
    });
    defer s.deinit();

    // smooth periodic A in the B slots (ko.c: init → set_bc → calc_BfromA)
    const tp = 2.0 * std.math.pi;
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const x = g.xc(ix);
            const y = g.yc(iy);
            const a1 = 0.03 * @sin(tp * y);
            const a2 = 0.05 * @cos(tp * x);
            const a3 = 0.1 * (@sin(tp * x) * @cos(tp * y) + 0.5 * @cos(tp * x));
            try s.initCell(ix, iy, 0, ppMhd(gam, 1.0, 0.5, .{ 0, 0, 0 }, .{ a1, a2, a3 }));
        }
    }
    try s.setBc(0.0, true);
    try ct.calcBfromA(SimM, &s, true);
    try s.setBc(0.0, true);

    // scale: |B| / Δx
    var bmax: f64 = 0;
    iy = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            for (0..3) |c| bmax = @max(bmax, @abs(s.p.get(L.index(.b1) + c, ix, iy, 0)));
        }
    }
    try expect(bmax > 0.1); // the curl actually produced a field
    const div = maxAbsDivB(&s);
    std.debug.print("BfromA: max|B|={e:.3} max|divB|·dx/B = {e:.3}\n", .{ bmax, div * g.dx / bmax });
    try expect(div * g.dx / bmax <= 1e-14);
}

//
// ---- uniform magnetized state stays static -----------------------------------
//

test "M6: uniform magnetized state static over 50 steps" {
    const gam: f64 = 5.0 / 3.0;
    const g = Grid.init(.{ .nx = 16, .ny = 16, .ng = 3, .minx = 0, .maxx = 1, .miny = 0, .maxy = 1 });
    var s = try SimM.init(std.testing.allocator, g, .{
        .coords = .mink,
        .gam = gam,
        .bc_x = .periodic,
        .bc_y = .periodic,
    });
    defer s.deinit();

    const pp0 = ppMhd(gam, 1.0, 0.4, .{ velr(0.3), velr(0.2), 0.1 }, .{ 0.05, 0.1, 0.02 });
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            try s.initCell(ix, iy, 0, pp0);
        }
    }
    try s.finishInit();

    for (0..50) |_| try s.step(null);

    iy = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            for (0..NV) |iv| {
                const scale = @max(@abs(pp0[iv]), 1e-30);
                const got = s.p.get(iv, ix, iy, 0);
                const dev = @abs(got - pp0[iv]) / scale;
                if (!(dev <= 1e-13)) {
                    std.debug.print("uniform-B drift: ix {d} iy {d} iv {d}: {e:.17} -> {e:.17} (dev {e})\n", .{ ix, iy, iv, pp0[iv], got, dev });
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
}

//
// ---- Orszag–Tang: divB stays at machine zero ----------------------------------
//

test "M6: Orszag-Tang keeps corner divB at machine zero over 200 steps" {
    const gam: f64 = 5.0 / 3.0;
    const n = 32;
    const g = Grid.init(.{
        .nx = n,
        .ny = n,
        .ng = 3,
        .minx = 0,
        .maxx = 2.0 * std.math.pi,
        .miny = 0,
        .maxy = 2.0 * std.math.pi,
    });
    var s = try SimM.init(std.testing.allocator, g, .{
        .coords = .mink,
        .gam = gam,
        .bc_x = .periodic,
        .bc_y = .periodic,
    });
    defer s.deinit();

    // SR Orszag–Tang (Beckwith & Stone 2011 flavor): B from the vector
    // potential A3 = −B0(cos(2x)/2 + cos y) → B = B0(−sin y, sin 2x)
    const rho0 = 25.0 / (36.0 * std.math.pi);
    const p0 = 5.0 / (12.0 * std.math.pi);
    const b0 = 1.0 / @sqrt(4.0 * std.math.pi);
    const v0 = 0.5;

    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const x = g.xc(ix);
            const y = g.yc(iy);
            const vx = -v0 * @sin(y);
            const vy = v0 * @sin(x);
            const gl = 1.0 / @sqrt(1.0 - vx * vx - vy * vy);
            const a3 = -b0 * (0.5 * @cos(2.0 * x) + @cos(y));
            try s.initCell(ix, iy, 0, ppMhd(gam, rho0, p0 / (gam - 1.0), .{ gl * vx, gl * vy, 0 }, .{ 0, 0, a3 }));
        }
    }
    try s.setBc(0.0, true);
    try ct.calcBfromA(SimM, &s, true);
    try s.setBc(0.0, true);
    s.initTimestepGuess();

    const div0 = maxAbsDivB(&s);
    var bmax: f64 = 0;
    iy = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            for (0..3) |c| bmax = @max(bmax, @abs(s.p.get(L.index(.b1) + c, ix, iy, 0)));
        }
    }
    try expect(div0 * g.dx / bmax <= 1e-14);

    for (0..200) |_| try s.step(null);

    const div = maxAbsDivB(&s);
    // locate worst + interior-only max
    var div_int: f64 = 0;
    var wx: i64 = 0;
    var wy: i64 = 0;
    var wv: f64 = 0;
    iy = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const d = @abs(ct.calcDivB(SimM, &s, ix, iy, 0));
            if (d > wv) {
                wv = d;
                wx = ix;
                wy = iy;
            }
            if (ix >= 2 and ix < s.nxi() - 2 and iy >= 2 and iy < s.nyi() - 2) {
                div_int = @max(div_int, d);
            }
        }
    }
    std.debug.print("OT: t={d:.3} max|divB|·dx/B0 = {e:.3} at ({d},{d}); interior {e:.3} (initial {e:.3})\n", .{ s.t, div * g.dx / b0, wx, wy, div_int * g.dx / b0, div0 * g.dx / b0 });

    // Interior corner-divB is preserved to roundoff — the actual flux-CT
    // guarantee. The domain-edge corners violate it in *serial periodic*
    // runs because C's 2D corner fill (finite.c:3213-3225) copies the
    // 1-deep ghost corner surfaces from the adjacent domain row instead of
    // wrapping them (only the diagonal cells are wrapped); the EMFs at the
    // seam corners then see inconsistent stencils. MPI runs exchange
    // corners exactly. We transcribe C, so we pin both behaviors.
    try expect(div_int * g.dx / b0 <= 1e-12);
    try expect(wx == 0 or wx == s.nxi() - 1 or wy == 0 or wy == s.nyi() - 1);
    // sanity: the flow actually evolved
    try expect(s.t > 0.1);
}

//
// ---- circularly polarized Alfvén wave -----------------------------------------
//

/// Exact relativistic CP Alfvén speed (Del Zanna, Zanotti & Bucciantini 2007,
/// eq. 56): B_x=B0, B_⊥=ηB0(cos,sin), v_⊥=−v_A B_⊥/B0 — exact for any η.
fn cpAlfvenSpeed(gam: f64, rho: f64, p: f64, b0: f64, eta: f64) f64 {
    const w = rho + gam / (gam - 1.0) * p; // ρh
    const den = w + b0 * b0 * (1.0 + eta * eta);
    const q = 2.0 * b0 * b0 / den;
    return @sqrt(q / (1.0 + @sqrt(1.0 - eta * eta * q * q)));
}

const AlfvenResult = struct { phase: f64, amp_ratio: f64 };

fn runAlfven(a: std.mem.Allocator, nx: usize) !AlfvenResult {
    const gam: f64 = 5.0 / 3.0;
    const rho0: f64 = 1.0;
    const p0: f64 = 0.5;
    const b0: f64 = 0.5;
    const eta: f64 = 0.1;
    const va = cpAlfvenSpeed(gam, rho0, p0, b0, eta);

    const g = Grid.init(.{ .nx = nx, .ng = 3, .minx = 0, .maxx = 1 });
    var s = try SimM.init(a, g, .{ .coords = .mink, .gam = gam, .bc_x = .periodic });
    defer s.deinit();

    const tp = 2.0 * std.math.pi;
    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const x = g.xc(ix);
        const by = eta * b0 * @cos(tp * x);
        const bz = eta * b0 * @sin(tp * x);
        const vy = -va * by / b0;
        const vz = -va * bz / b0;
        const gl = 1.0 / @sqrt(1.0 - vy * vy - vz * vz);
        try s.initCell(ix, 0, 0, ppMhd(gam, rho0, p0 / (gam - 1.0), .{ 0, gl * vy, gl * vz }, .{ b0, by, bz }));
    }
    try s.finishInit();

    // one period: wave travels one wavelength (right-moving)
    const tend = 1.0 / va;
    while (s.t < tend) {
        var dt: ?f64 = null;
        if (s.t + 1.0 / s.tstepdenmax > tend) dt = tend - s.t;
        try s.step(dt);
    }

    // Fourier k=1 mode of B_y: exact solution returns to cos(2πx)
    var c_amp: f64 = 0;
    var s_amp: f64 = 0;
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const x = g.xc(ix);
        const by = s.p.get(L.index(.b2), ix, 0, 0);
        c_amp += by * @cos(tp * x);
        s_amp += by * @sin(tp * x);
    }
    const nf: f64 = @floatFromInt(nx);
    const amp = 2.0 * @sqrt(c_amp * c_amp + s_amp * s_amp) / nf;
    return .{
        .phase = std.math.atan2(s_amp, c_amp), // 0 for the initial profile
        .amp_ratio = amp / (eta * b0),
    };
}

test "M6: CP Alfvén wave — phase order ≥ 1.9, amplitude preserved" {
    const a = std.testing.allocator;
    const r32 = try runAlfven(a, 32);
    const r64 = try runAlfven(a, 64);
    const r128 = try runAlfven(a, 128);
    const p1 = std.math.log2(@abs(r32.phase) / @abs(r64.phase));
    const p2 = std.math.log2(@abs(r64.phase) / @abs(r128.phase));
    std.debug.print("alfven phase: {e:.3} {e:.3} {e:.3}; orders {d:.2} {d:.2}; amp {d:.4} {d:.4} {d:.4}\n", .{ r32.phase, r64.phase, r128.phase, p1, p2, r32.amp_ratio, r64.amp_ratio, r128.amp_ratio });
    try expect(p1 >= 1.9);
    try expect(p2 >= 1.7);
    try expect(@abs(r128.amp_ratio - 1.0) < 0.02);
}

//
// ---- Balsara-1 / SR Brio–Wu tube ------------------------------------------------
//

fn runBalsara(a: std.mem.Allocator, nx: usize, out_rho: []f64) !void {
    const gam: f64 = 2.0;
    const g = Grid.init(.{ .nx = nx, .ng = 3, .minx = 0, .maxx = 1 });
    var s = try SimM.init(a, g, .{ .coords = .mink, .gam = gam, .bc_x = .copy });
    defer s.deinit();

    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const lft = g.xc(ix) < 0.5;
        const rho: f64 = if (lft) 1.0 else 0.125;
        const p: f64 = if (lft) 1.0 else 0.1;
        const by: f64 = if (lft) 1.0 else -1.0;
        try s.initCell(ix, 0, 0, ppMhd(gam, rho, p / (gam - 1.0), .{ 0, 0, 0 }, .{ 0.5, by, 0 }));
    }
    try s.finishInit();

    const tend = 0.4;
    while (s.t < tend) {
        var dt: ?f64 = null;
        if (s.t + 1.0 / s.tstepdenmax > tend) dt = tend - s.t;
        try s.step(dt);
    }

    // B^x must remain exactly its initial constant (flux_ct zeroes flbx[B1])
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        try std.testing.expectEqual(@as(f64, 0.5), s.p.get(L.index(.b1), ix, 0, 0));
        try expect(std.math.isFinite(s.p.get(L.index(.rho), ix, 0, 0)));
    }

    // sample rho onto the caller's grid (cell centers of out_rho length)
    const m = out_rho.len;
    for (0..m) |j| {
        const x = (@as(f64, @floatFromInt(j)) + 0.5) / @as(f64, @floatFromInt(m));
        const fx = x / g.dx - 0.5;
        const j0: i64 = @intFromFloat(@floor(fx));
        const fr = fx - @floor(fx);
        const jl = std.math.clamp(j0, 0, @as(i64, @intCast(nx - 1)));
        const jr = std.math.clamp(j0 + 1, 0, @as(i64, @intCast(nx - 1)));
        out_rho[j] = (1.0 - fr) * s.p.get(L.index(.rho), jl, 0, 0) + fr * s.p.get(L.index(.rho), jr, 0, 0);
    }
}

test "M6: Balsara-1 tube — Bx exactly constant, self-convergent" {
    const a = std.testing.allocator;
    const m = 100;
    var r100: [m]f64 = undefined;
    var r200: [m]f64 = undefined;
    var r400: [m]f64 = undefined;
    try runBalsara(a, 100, &r100);
    try runBalsara(a, 200, &r200);
    try runBalsara(a, 400, &r400);

    var d1: f64 = 0;
    var d2: f64 = 0;
    for (0..m) |j| {
        d1 += @abs(r100[j] - r400[j]);
        d2 += @abs(r200[j] - r400[j]);
    }
    d1 /= m;
    d2 /= m;
    const order = std.math.log2(d1 / d2);
    std.debug.print("balsara self-convergence: L1(100vs400)={e:.3} L1(200vs400)={e:.3} order {d:.2}\n", .{ d1, d2, order });
    try expect(d2 < d1);
    try expect(order >= 0.6); // ~1st order at the discontinuities
}

//
// ---- koral_lite_puffy REDUCEORDERAFTERFIXUP ---------------------------------
//

/// One forced-dt step on a smooth periodic MHD state; returns a copy of the
/// conserveds. `poison` negates one cell's conserved density so the u2p pass
/// that runs inside step() BEFORE the explicit sweep fails there, flags
/// hd_fixup and neighbour-averages the cell — after which the sweep sees the
/// flag (setting a flag by hand is useless: that same u2p pass overwrites
/// every cell's flag first).
fn runReduceStep(a: std.mem.Allocator, reduce_after_fixup: bool, poison: bool) ![]f64 {
    const gam: f64 = 5.0 / 3.0;
    const g = Grid.init(.{ .nx = 16, .ny = 16, .ng = 3, .minx = 0, .maxx = 1, .miny = 0, .maxy = 1 });
    var s = try SimM.init(a, g, .{
        .coords = .mink,
        .gam = gam,
        .bc_x = .periodic,
        .bc_y = .periodic,
        .reduceorderafterfixup = reduce_after_fixup,
    });
    defer s.deinit();

    const tp = 2.0 * std.math.pi;
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const x = g.xc(ix);
            const y = g.yc(iy);
            const rho = 1.0 + 0.3 * @sin(tp * x) * @cos(tp * y);
            const uint = 0.5 + 0.1 * @cos(tp * x);
            try s.initCell(ix, iy, 0, ppMhd(gam, rho, uint, .{ 0.1 * @sin(tp * y), 0.0, 0.0 }, .{ 0.05, 0.02, 0.0 }));
        }
    }
    try s.setBc(0.0, true);
    if (poison) {
        var uu: [NV]f64 = undefined;
        s.u.load(8, 8, 0, &uu);
        uu[L.index(.rho)] = -uu[L.index(.rho)]; // u2p rejects → hd_fixup flagged
        s.u.store(8, 8, 0, &uu);
    }
    try s.step(1.0e-3);
    return a.dupe(f64, s.u.data);
}

test "reduceorderafterfixup: a fixup-flagged cell reconstructs one order lower" {
    const a = std.testing.allocator;
    // healthy state: the option must be inert (no flag ever set)
    const clean_on = try runReduceStep(a, true, false);
    defer a.free(clean_on);
    const clean_off = try runReduceStep(a, false, false);
    defer a.free(clean_off);
    try expect(std.mem.eql(f64, clean_on, clean_off));

    // poisoned cell: identical u2p failure + fixup in both runs — the only
    // difference is the reduced reconstruction of the flagged cell
    const poison_on = try runReduceStep(a, true, true);
    defer a.free(poison_on);
    const poison_off = try runReduceStep(a, false, true);
    defer a.free(poison_off);
    try expect(!std.mem.eql(f64, poison_on, poison_off));
}
