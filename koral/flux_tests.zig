//! M4 theory gates: reconstruction (donor/minmod-θ/PPM), characteristic
//! wavespeeds, flux assembly, LAXF/HLL face combination.

const std = @import("std");
const config = @import("config.zig");
const layout = @import("layout.zig");
const relele = @import("relele.zig");
const hydro = @import("physics/hydro.zig");
const recon = @import("fv/recon.zig");
const wavespeeds = @import("physics/wavespeeds.zig");
const fluxmod = @import("physics/flux.zig");
const laxf_mod = @import("fv/laxf.zig");
const precompute = @import("metric/precompute.zig");
const metric = @import("metric/metric.zig");

const geometryAt = precompute.geometryAt;
const pgamma: f64 = 5.0 / 3.0;
const puffy_mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };

fn expectClose(a: f64, b: f64, tol: f64) !void {
    const dev = @abs(a - b) / @max(1.0, @max(@abs(a), @abs(b)));
    if (!(dev <= tol)) {
        std.debug.print("expectClose: {e:.17} vs {e:.17} (dev {e}, tol {e})\n", .{ a, b, dev, tol });
        return error.TestExpectedApproxEq;
    }
}

/// Stencil of cell values of f at the 5 cell centers implied by dx,
/// plus the positions of the center cell's faces.
fn linStencil(a: f64, b: f64, dx: [5]f64) struct { u: [5]f64, xl: f64, xr: f64 } {
    var left: f64 = 0;
    var u: [5]f64 = undefined;
    var xl: f64 = 0;
    var xr: f64 = 0;
    for (0..5) |i| {
        if (i == 2) {
            xl = left;
            xr = left + dx[i];
        }
        u[i] = a + b * (left + 0.5 * dx[i]);
        left += dx[i];
    }
    return .{ .u = u, .xl = xl, .xr = xr };
}

test "recon: linear data is reconstructed exactly" {
    // uniform grid: both the θ-limited linear scheme and PPM are exact
    const dx_u: [5]f64 = @splat(1.0);
    const s = linStencil(0.3, -1.7, dx_u);
    for ([_]recon.Scheme{ .{ .linear = .{ .theta = 1.5 } }, .ppm }) |scheme| {
        const f = recon.reconstruct(f64, scheme, s.u, dx_u);
        try expectClose(f.left, 0.3 + -1.7 * s.xl, 1e-15);
        try expectClose(f.right, 0.3 + -1.7 * s.xr, 1e-15);
    }
    // mildly non-uniform grid: PPM stays exact on linear data
    const dx_n = [5]f64{ 1.0, 1.2, 0.9, 1.1, 0.8 };
    const sn = linStencil(-0.2, 0.9, dx_n);
    const fn2 = recon.reconstruct(f64, .ppm, sn.u, dx_n);
    try expectClose(fn2.left, -0.2 + 0.9 * sn.xl, 1e-14);
    try expectClose(fn2.right, -0.2 + 0.9 * sn.xr, 1e-14);
}

test "recon: PPM reproduces a monotone parabola exactly from cell averages" {
    // u(x) = x² on cells [i, i+1], i = 1..5: averages ((i+1)³−i³)/3
    var u: [5]f64 = undefined;
    for (0..5) |i| {
        const a = 1.0 + @as(f64, @floatFromInt(i));
        u[i] = ((a + 1.0) * (a + 1.0) * (a + 1.0) - a * a * a) / 3.0;
    }
    const f = recon.reconstruct(f64, .ppm, u, @splat(1.0));
    try expectClose(f.left, 9.0, 1e-13); // left face of cell [3,4] at x=3
    try expectClose(f.right, 16.0, 1e-13); // right face at x=4
}

test "recon: face values never leave the stencil range (fuzz)" {
    var prng = std.Random.DefaultPrng.init(0x5eefca7);
    const rng = prng.random();
    for (0..5000) |_| {
        var u: [5]f64 = undefined;
        for (0..5) |i| u[i] = 2.0 * rng.float(f64) - 1.0;
        var dx: [5]f64 = undefined;
        for (0..5) |i| dx[i] = 0.5 + 1.5 * rng.float(f64);
        var lo = u[0];
        var hi = u[0];
        for (u) |v| {
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        for ([_]recon.Scheme{ .donor, .{ .linear = .{ .theta = 1.5 } }, .ppm }) |scheme| {
            const f = recon.reconstruct(f64, scheme, u, dx);
            try std.testing.expect(f.left >= lo - 1e-14 and f.left <= hi + 1e-14);
            try std.testing.expect(f.right >= lo - 1e-14 and f.right <= hi + 1e-14);
            // limited parabola keeps u0 between its own face values
            try std.testing.expect((f.right - u[2]) * (u[2] - f.left) >= -1e-15);
        }
    }
}

test "recon: mirror symmetry (linear bitwise; PPM to a few ulp)" {
    // The PPM left/right face formulas are algebraic mirrors written around
    // different base cells (u0 + dx0·(um1−u0)/Σ vs um1 + dxm1·(u0−um1)/Σ),
    // so even C's version is only ulp-symmetric, not bitwise.
    var prng = std.Random.DefaultPrng.init(0x3122023);
    const rng = prng.random();
    for (0..2000) |_| {
        var u: [5]f64 = undefined;
        var dx: [5]f64 = undefined;
        for (0..5) |i| {
            u[i] = 2.0 * rng.float(f64) - 1.0;
            dx[i] = 0.5 + 1.5 * rng.float(f64);
        }
        const um = [5]f64{ u[4], u[3], u[2], u[1], u[0] };
        const dxm = [5]f64{ dx[4], dx[3], dx[2], dx[1], dx[0] };

        const lin = recon.Scheme{ .linear = .{ .theta = 1.5 } };
        const fl = recon.reconstruct(f64, lin, u, dx);
        const flm = recon.reconstruct(f64, lin, um, dxm);
        try std.testing.expectEqual(fl.left, flm.right);
        try std.testing.expectEqual(fl.right, flm.left);

        const fp = recon.reconstruct(f64, .ppm, u, dx);
        const fpm = recon.reconstruct(f64, .ppm, um, dxm);
        try expectClose(fp.left, fpm.right, 1e-13);
        try expectClose(fp.right, fpm.left, 1e-13);
    }
}

test "LAXF and HLL are consistent: F*(U,U) = F(U)" {
    const f: f64 = 1.7;
    const u: f64 = -0.4;
    try std.testing.expectEqual(f, laxf_mod.laxf(f, f, u, u, 0.83));
    try expectClose(laxf_mod.hll(f, f, u, u, -0.6, 0.4), f, 1e-15);
    // upwind branches
    try std.testing.expectEqual(@as(f64, 2.0), laxf_mod.hll(2.0, 3.0, u, u, 0.1, 0.5));
    try std.testing.expectEqual(@as(f64, 3.0), laxf_mod.hll(2.0, 3.0, u, u, -0.5, -0.1));
}

test "wavespeeds: sound speed at rest in Minkowski" {
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });

    var pp: [L.count]f64 = @splat(0);
    pp[L.index(.rho)] = 1.0;
    pp[L.index(.uu)] = 0.05;
    pp[L.index(.entr)] = hydro.sFromU(1.0, 0.05, pgamma);

    const a = try wavespeeds.gasWavespeedsLr(cfg, pp, &geo, pgamma, .{ true, true, true });
    // c_s² = Γ(Γ−1)u / (ρ + Γu)
    const cs = @sqrt(pgamma * (pgamma - 1.0) * 0.05 / (1.0 + pgamma * 0.05));
    for (0..3) |dim| {
        try expectClose(a[2 * dim], -cs, 1e-12);
        try expectClose(a[2 * dim + 1], cs, 1e-12);
    }
}

test "wavespeeds: p→0 fast speed approaches the relativistic Alfvén speed" {
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });

    var pp: [L.count]f64 = @splat(0);
    pp[L.index(.rho)] = 1.0;
    pp[L.index(.uu)] = 1e-12;
    pp[L.index(.entr)] = hydro.sFromU(1.0, 1e-12, pgamma);
    pp[L.index(.b1)] = 2.0; // bsq = 4 at rest

    const a = try wavespeeds.gasWavespeedsLr(cfg, pp, &geo, pgamma, .{ true, false, false });
    const bsq = 4.0;
    const va = @sqrt(bsq / (bsq + 1.0)); // relativistic v_A, p → 0
    try expectClose(a[1], va, 1e-10);
    try expectClose(a[0], -va, 1e-10);
}

test "wavespeeds: boost along x is relativistic velocity addition" {
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });
    const wspeed2: f64 = 0.17; // fluid-frame speed² of the mode
    const c0 = @sqrt(wspeed2);
    const v: f64 = 0.6;
    const gam = 1.0 / @sqrt(1.0 - v * v);
    const ucon = [4]f64{ gam, gam * v, 0, 0 };

    const a = wavespeeds.lrCore(ucon, &geo, .{ wspeed2, wspeed2, wspeed2 }, .{ true, false, false });
    try expectClose(a[0], (v - c0) / (1.0 - v * c0), 1e-10);
    try expectClose(a[1], (v + c0) / (1.0 + v * c0), 1e-10);
}

test "flux: at rest in Minkowski the flux is pure total pressure" {
    const cfg = config.puffy;
    const L = layout.VarLayout(cfg);
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });

    var pp: [L.count]f64 = @splat(0);
    pp[L.index(.rho)] = 2.0;
    pp[L.index(.uu)] = 0.3;
    pp[L.index(.entr)] = hydro.sFromU(2.0, 0.3, pgamma);
    pp[L.index(.b1)] = 0.4;
    pp[L.index(.b2)] = -0.1;
    pp[L.index(.ee)] = 0.7;

    const ff = try fluxmod.fFluxPrime(cfg, pp, 0, &geo, pgamma);
    const bsq = 0.4 * 0.4 + 0.1 * 0.1;
    const ptot = (pgamma - 1.0) * 0.3 + 0.5 * bsq;

    try expectClose(ff[L.index(.rho)], 0.0, 1e-15);
    try expectClose(ff[L.index(.uu)], 0.0, 1e-15);
    try expectClose(ff[L.index(.entr)], 0.0, 1e-15);
    // T^1_1 = ptot − b¹b_1 at rest
    try expectClose(ff[L.index(.vx)], ptot - 0.4 * 0.4, 1e-14);
    try expectClose(ff[L.index(.vy)], -(0.4 * -0.1), 1e-14); // −b¹b_2
    try expectClose(ff[L.index(.b1)], 0.0, 1e-15);
    try expectClose(ff[L.index(.b2)], 0.0, 1e-15);
    try expectClose(ff[L.index(.b3)], 0.0, 1e-15);
    // isotropic radiation: R^1_0 = 0, R^1_1 = Ê/3
    try expectClose(ff[L.index(.ee)], 0.0, 1e-15);
    try expectClose(ff[L.index(.fx)], 0.7 / 3.0, 1e-14);
    try expectClose(ff[L.index(.fy)], 0.0, 1e-15);
}

test "flux: p2u time-row consistency — F with u^i ∝ conserved density rows" {
    // For any state, the flux of RHO along dim equals uu[RHO]·(u^d/u^t):
    // both are gdet·rho·u^μ components. Cross-checks flux vs p2u assembly.
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    const p2u_mod = @import("p2u.zig");
    var prng = std.Random.DefaultPrng.init(0xf1a5);
    const rng = prng.random();

    const geo = geometryAt(.ks, puffy_mp, .{ 0, 4.2, 1.1, 0.2 });
    for (0..50) |_| {
        var pp: [L.count]f64 = @splat(0);
        pp[L.index(.rho)] = 0.5 + rng.float(f64);
        pp[L.index(.uu)] = 0.1 * pp[L.index(.rho)];
        for (0..3) |i| pp[L.index(.vx) + i] = 0.3 * (2.0 * rng.float(f64) - 1.0);
        pp[L.index(.entr)] = hydro.sFromU(pp[0], pp[1], pgamma);
        for (0..3) |i| pp[L.index(.b1) + i] = 0.2 * (2.0 * rng.float(f64) - 1.0);

        var uu: [L.count]f64 = @splat(0);
        try p2u_mod.p2uMhd(cfg, pp, &uu, &geo, pgamma);
        const u = try relele.convertBoth(
            .{ 0, pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
            .velr,
            &geo,
        );
        for (0..2) |idim| {
            const ff = try fluxmod.fFluxPrime(cfg, pp, idim, &geo, pgamma);
            try expectClose(ff[L.index(.rho)], uu[L.index(.rho)] * u.con[idim + 1] / u.con[0], 1e-13);
            try expectClose(ff[L.index(.entr)], uu[L.index(.entr)] * u.con[idim + 1] / u.con[0], 1e-13);
        }
    }
}
