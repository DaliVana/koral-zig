//! M7 theory gates — M1 radiation basics.
//!
//! Gates (plan M7):
//!  * isotropic radiation at rest → Eddington tensor diag(Ê, Ê/3, …) @1e-14
//!  * M1 closure trace identity R^μ_μ = 0 @1e-13 (MINK and KS)
//!  * streaming limit: R^0x, R^xx → R^00 as γ → ∞ @1e-10
//!  * u2p_rad round-trips p2u_rad @1e-12 over MINK/KS/MKS2 state sweeps
//!  * closed-form γ² agrees with a test-local bisection of the M1
//!    consistency equation @1e-12
//!  * over-cap / under-floor conserveds take the cold branch (corrected
//!    set, γ pinned to γmax or 1) — M1's |F̂| ≤ Ê ceiling in VELR form
//!  * comoving observer: Ê(u_gas = u_rad) = Êrf @1e-13 (frame covariance
//!    of Ê under coordinate transforms is an M2 gate)
//!  * rad wavespeeds ±1/√3 at rest as τ → 0 @1e-12; thick limit 4/(3τ);
//!    damping monotone in τ; the rv2z = rv2dim[1] C quirk is pinned
//!  * check_floors_rad floor/ratio properties (PUFFY parameters)
//!  * uniform MINK state with radiation static over 50 steps @1e-14
//!    (κ = 0 ⇒ the explicit operator alone must preserve it)

const std = @import("std");
const config = @import("config.zig");
const layout = @import("layout.zig");
const grid_mod = @import("grid.zig");
const sim_mod = @import("sim.zig");
const relele = @import("relele.zig");
const hydro = @import("physics/hydro.zig");
const radiation = @import("physics/radiation.zig");
const invert_rad = @import("solve/invert_rad.zig");
const p2u_mod = @import("p2u.zig");
const precompute = @import("metric/precompute.zig");
const metric = @import("metric/metric.zig");
const Geometry = @import("geometry.zig").Geometry;

const Grid = grid_mod.Grid;
const expect = std.testing.expect;
const geometryAt = precompute.geometryAt;

const puffy_mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
const gam: f64 = 5.0 / 3.0;

const cfg = config.Config{
    .modules = &.{ .hydro, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const L = layout.VarLayout(cfg);
const NV = L.count;

fn expectClose(a: f64, b: f64, tol: f64) !void {
    const dev = @abs(a - b) / @max(1.0, @max(@abs(a), @abs(b)));
    if (!(dev <= tol)) {
        std.debug.print("expectClose: {e:.17} vs {e:.17} (dev {e}, tol {e})\n", .{ a, b, dev, tol });
        return error.TestExpectedApproxEq;
    }
}

fn ppRad(rho: f64, uint: f64, v: [3]f64, erf: f64, f: [3]f64) [NV]f64 {
    var pp: [NV]f64 = @splat(0);
    pp[L.index(.rho)] = rho;
    pp[L.index(.uu)] = uint;
    pp[L.index(.vx)] = v[0];
    pp[L.index(.vy)] = v[1];
    pp[L.index(.vz)] = v[2];
    pp[L.index(.entr)] = hydro.sFromU(rho, uint, gam);
    pp[L.index(.ee)] = erf;
    pp[L.index(.fx)] = f[0];
    pp[L.index(.fy)] = f[1];
    pp[L.index(.fz)] = f[2];
    return pp;
}

/// Test geometries: MINK origin, KS mid-radius, MKS2 mid-torus.
fn testGeoms() [3]Geometry {
    return .{
        geometryAt(.mink, puffy_mp, .{ 0, 0.3, 0.7, 0.1 }),
        geometryAt(.ks, puffy_mp, .{ 0, 7.3, 1.1, 0.4 }),
        geometryAt(.mks2, puffy_mp, .{ 0, 2.2, 0.45, 0.4 }),
    };
}

/// VELR components with a prescribed Lorentz factor wrt the normal observer
/// (γ² = 1 + g_ij ũ^i ũ^j) — coordinate-basis components in curved metrics
/// carry the metric scale, so raw component sampling gives huge γ.
fn velrGamma(geo: *const Geometry, gamma: f64, rng: std.Random) [3]f64 {
    var n: [3]f64 = undefined;
    for (0..3) |i| n[i] = 2.0 * rng.float(f64) - 1.0;
    var qsq: f64 = 0;
    for (0..3) |i| {
        for (0..3) |j| qsq += n[i] * n[j] * geo.gg[i + 1][j + 1];
    }
    const s = @sqrt((gamma * gamma - 1.0) / qsq);
    return .{ n[0] * s, n[1] * s, n[2] * s };
}

//
// ---- M1 closure tensor -------------------------------------------------------
//

test "M7: isotropic radiation at rest is an Eddington tensor" {
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });
    const erf: f64 = 3.7;
    const pp = ppRad(1.0, 0.1, .{ 0, 0, 0 }, erf, .{ 0, 0, 0 });
    const rij = try radiation.calcRij(cfg, pp, &geo);

    for (0..4) |i| {
        for (0..4) |j| {
            const want: f64 = if (i != j) 0.0 else if (i == 0) erf else erf / 3.0;
            try expectClose(rij[i][j], want, 1e-14);
        }
    }
}

test "M7: M1 closure is traceless (R^mu_mu = 0), MINK and KS" {
    var prng = std.Random.DefaultPrng.init(0x4d37726164);
    const rng = prng.random();

    for (testGeoms()) |geo| {
        for (0..200) |_| {
            const erf = std.math.pow(f64, 10.0, -8.0 + 9.0 * rng.float(f64));
            const f = velrGamma(&geo, 1.0 + 7.0 * rng.float(f64), rng);
            const pp = ppRad(1.0, 0.1, .{ 0, 0, 0 }, erf, f);

            const rij_up = try radiation.calcRij(cfg, pp, &geo);
            const rij = relele.indices2221(rij_up, &geo.gg);
            var trace: f64 = 0;
            var scale: f64 = 0;
            for (0..4) |i| {
                trace += rij[i][i];
                scale += @abs(rij[i][i]);
            }
            if (!(@abs(trace) / scale <= 1e-13)) {
                std.debug.print("trace {e} scale {e} coords {any} f {any}\n", .{ trace, scale, geo.coords, f });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "M7: streaming limit R^0x, R^xx -> R^00 as gamma -> inf" {
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });
    const ut: f64 = 1.0e5; // VELR ũ^x, γ ≈ 1e5 → corrections O(γ⁻²) ≈ 4e-11
    const pp = ppRad(1.0, 0.1, .{ 0, 0, 0 }, 2.0, .{ ut, 0, 0 });
    const rij = try radiation.calcRij(cfg, pp, &geo);

    try expectClose(rij[0][1] / rij[0][0], 1.0, 1e-10);
    try expectClose(rij[1][1] / rij[0][0], 1.0, 1e-10);
}

test "M7: comoving observer sees Ehat = Erf" {
    var prng = std.Random.DefaultPrng.init(0x4d37636f6d);
    const rng = prng.random();

    for (testGeoms()) |geo| {
        for (0..50) |_| {
            const v = velrGamma(&geo, 1.0 + 3.0 * rng.float(f64), rng);
            const erf = std.math.pow(f64, 10.0, -6.0 + 7.0 * rng.float(f64));
            // same relative velocity for gas and radiation frame; the double
            // contraction cancels at O(γ⁴·ε) so the gate is 1e-12
            const pp = ppRad(1.0, 0.1, v, erf, v);
            const ehat = try radiation.calcFfEhat(cfg, pp, &geo);
            if (!(@abs(ehat - erf) / erf <= 1e-12)) {
                std.debug.print("ehat {e} erf {e} coords {any} v {any}\n", .{ ehat, erf, geo.coords, v });
                return error.TestUnexpectedResult;
            }
        }
    }
}

//
// ---- u2p_rad ---------------------------------------------------------------
//

test "M7: u2p_rad round-trips p2u_rad (MINK/KS/MKS2 sweep)" {
    var prng = std.Random.DefaultPrng.init(0x4d37753270);
    const rng = prng.random();
    const rp = invert_rad.RadParams.puffy;

    var worst: f64 = 0;
    for (testGeoms()) |geo| {
        for (0..300) |_| {
            const erf = std.math.pow(f64, 10.0, -10.0 + 11.0 * rng.float(f64));
            // γ up to 6, safely below GAMMAMAXRAD = 10
            const f = velrGamma(&geo, 1.0 + 5.0 * rng.float(f64), rng);
            const pp0 = ppRad(1.0, 0.1, .{ 0, 0, 0 }, erf, f);

            var uu: [NV]f64 = @splat(0);
            try p2u_mod.p2uRad(cfg, pp0, &uu, &geo);

            var pp = pp0;
            pp[L.index(.ee)] = 1.0; // wreck the guess: inversion must not read it
            pp[L.index(.fx)] = 0.0;
            pp[L.index(.fy)] = 0.0;
            pp[L.index(.fz)] = 0.0;
            const res = invert_rad.u2pRad(cfg, uu, &pp, &geo, rp);

            if (res.ret != 0 or res.corrected) {
                std.debug.print("u2p_rad ret {d} cor {any} coords {any} erf {e} f {any}\n", .{ res.ret, res.corrected, geo.coords, erf, f });
                return error.TestUnexpectedResult;
            }
            inline for ([_]layout.VarTag{ .ee, .fx, .fy, .fz }) |tag| {
                const iv = L.index(tag);
                const dev = @abs(pp[iv] - pp0[iv]) / @max(@abs(pp0[iv]), erf);
                worst = @max(worst, dev);
            }
        }
    }
    std.debug.print("u2p_rad round-trip: worst dev {e:.3}\n", .{worst});
    try expect(worst <= 1e-12);
}

test "M7: closed-form gamma^2 agrees with bisection of the M1 consistency equation" {
    var prng = std.Random.DefaultPrng.init(0x4d37676d32);
    const rng = prng.random();
    const rp = invert_rad.RadParams.cdefault;

    for (testGeoms()) |geo| {
        for (0..40) |_| {
            const erf = std.math.pow(f64, 10.0, -4.0 + 5.0 * rng.float(f64));
            // γ ≤ 20: the consistency equation's root conditioning
            // degrades as γ², keep it well posed for a 1e-12 gate
            const f = velrGamma(&geo, 1.0 + 19.0 * rng.float(f64), rng);
            const pp0 = ppRad(1.0, 0.1, .{ 0, 0, 0 }, erf, f);

            var uu: [NV]f64 = @splat(0);
            try p2u_mod.p2uRad(cfg, pp0, &uu, &geo);
            const gdetu = geo.gdet;
            const avcov = [4]f64{
                uu[L.index(.ee)] / gdetu,
                uu[L.index(.fx)] / gdetu,
                uu[L.index(.fy)] / gdetu,
                uu[L.index(.fz)] / gdetu,
            };
            const avcon = relele.indices12(avcov, &geo.GG);

            // residual h(γ²) = γ² − (1 + q(ũ(γ²))) with Ê(γ²) and ũ(γ²)
            // from the M1 relations — an independent formulation of the
            // closed-form solution's defining equation
            const H = struct {
                fn h(g2: f64, av_con: [4]f64, geom: *const Geometry) f64 {
                    const a = geom.alpha;
                    const e = 3.0 * av_con[0] * a * a / (4.0 * g2 - 1.0);
                    const gr = @sqrt(g2);
                    var q: f64 = 0;
                    var ur: [4]f64 = undefined;
                    for (1..4) |i| {
                        ur[i] = a * (av_con[i] + 1.0 / 3.0 * e * geom.GG[0][i] * (4.0 * g2 - 1.0)) /
                            (4.0 / 3.0 * e * gr);
                    }
                    for (1..4) |i| {
                        for (1..4) |j| q += ur[i] * ur[j] * geom.gg[i][j];
                    }
                    return g2 - (1.0 + q);
                }
            };

            // bracket and bisect
            var lo: f64 = 1.0;
            var hi: f64 = 1.0e12;
            if (!(H.h(lo, avcon, &geo) <= 0.0)) lo = 1.0 - 1e-12;
            try expect(H.h(hi, avcon, &geo) > 0.0);
            for (0..200) |_| {
                const mid = 0.5 * (lo + hi);
                if (H.h(mid, avcon, &geo) > 0.0) hi = mid else lo = mid;
            }
            const g2_bisect = 0.5 * (lo + hi);

            // recovered γ² from the inversion's primitives
            var pp = pp0;
            const res = invert_rad.u2pRad(cfg, uu, &pp, &geo, rp);
            try expect(res.ret == 0);
            var q: f64 = 0;
            const fr = [3]f64{ pp[L.index(.fx)], pp[L.index(.fy)], pp[L.index(.fz)] };
            for (0..3) |i| {
                for (0..3) |j| q += fr[i] * fr[j] * geo.gg[i + 1][j + 1];
            }
            const g2_closed = 1.0 + q;

            if (!(@abs(g2_closed - g2_bisect) / g2_bisect <= 1e-12)) {
                std.debug.print("g2 closed {e:.17} bisect {e:.17} coords {any}\n", .{ g2_closed, g2_bisect, geo.coords });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "M7: over-cap and under-floor conserveds take the cold branch (VELR flux ceiling)" {
    const rp = invert_rad.RadParams.puffy; // GAMMAMAXRAD = 10
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });

    // γ = 30 > γmax: the fast (γ = γmax) cold solver must win
    {
        const ut = @sqrt(30.0 * 30.0 - 1.0);
        const pp0 = ppRad(1.0, 0.1, .{ 0, 0, 0 }, 1.0, .{ ut, 0, 0 });
        var uu: [NV]f64 = @splat(0);
        try p2u_mod.p2uRad(cfg, pp0, &uu, &geo);

        var pp = pp0;
        const res = invert_rad.u2pRad(cfg, uu, &pp, &geo, rp);
        try expect(res.ret == 0);
        try expect(res.corrected);
        const fx = pp[L.index(.fx)];
        const g2 = 1.0 + fx * fx;
        // γ pinned to γmax (the cold solver reconstructs ũ consistently)
        try expectClose(g2, rp.gammamaxrad * rp.gammamaxrad, 1e-10);
        try expect(pp[L.index(.ee)] > 0.0 and std.math.isFinite(pp[L.index(.ee)]));
    }

    // Ê far below the floor: corrected, γ pinned near 1 (slow branch)
    {
        const pp0 = ppRad(1.0, 0.1, .{ 0, 0, 0 }, 1.0e-85, .{ 0.5, 0, 0 });
        var uu: [NV]f64 = @splat(0);
        try p2u_mod.p2uRad(cfg, pp0, &uu, &geo);

        var pp = pp0;
        const res = invert_rad.u2pRad(cfg, uu, &pp, &geo, rp);
        try expect(res.corrected);
        if (res.ret == 0) {
            const fx = pp[L.index(.fx)];
            const g2 = 1.0 + fx * fx;
            try expect(g2 <= rp.gammamaxrad * rp.gammamaxrad * 1.0001);
        }
    }
}

//
// ---- rad wavespeeds ----------------------------------------------------------
//

test "M7: rad wavespeeds at rest are +-1/sqrt(3) when optically thin" {
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });
    const pp = ppRad(1.0, 0.1, .{ 0, 0, 0 }, 1.0, .{ 0, 0, 0 });
    const dims = [3]bool{ true, true, true };

    const aval = try radiation.calcRadWavespeeds(cfg, pp, &geo, .{ 0, 0, 0 }, dims);
    const c13 = 1.0 / @sqrt(3.0);
    for (0..3) |d| {
        try expectClose(aval[2 * d], -c13, 1e-12);
        try expectClose(aval[2 * d + 1], c13, 1e-12);
        // τ = 0: the limited speeds equal the unlimited ones
        try std.testing.expectEqual(aval[2 * d], aval[6 + 2 * d]);
        try std.testing.expectEqual(aval[2 * d + 1], aval[6 + 2 * d + 1]);
    }
}

test "M7: tau-damping — thick limit 4/(3 tau), monotone, thin unchanged" {
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });
    const pp = ppRad(1.0, 0.1, .{ 0, 0, 0 }, 1.0, .{ 0, 0, 0 });
    const dims = [3]bool{ true, false, false };

    // thick limit: |a| → 4/(3τ)
    {
        const tau: f64 = 1.0e4;
        const aval = try radiation.calcRadWavespeeds(cfg, pp, &geo, .{ tau, 0, 0 }, dims);
        try expectClose(@abs(aval[6]), 4.0 / (3.0 * tau), 1e-12);
        try expectClose(aval[7], 4.0 / (3.0 * tau), 1e-12);
        // unlimited slots keep 1/√3 (they set the timestep)
        try expectClose(aval[1], 1.0 / @sqrt(3.0), 1e-12);
    }

    // monotone: |a_limited| non-increasing with τ; thin (τ ≲ 4/√3) undamped
    var prev: f64 = 1.0;
    var k: f64 = -6.0;
    while (k <= 6.0) : (k += 0.5) {
        const tau = std.math.pow(f64, 10.0, k);
        const aval = try radiation.calcRadWavespeeds(cfg, pp, &geo, .{ tau, 0, 0 }, dims);
        const a = @abs(aval[6]);
        try expect(a <= prev + 1e-15);
        prev = a;
        if (tau <= 4.0 / (3.0 * @sqrt(1.0 / 3.0)) - 0.1) {
            try expectClose(a, 1.0 / @sqrt(3.0), 1e-12);
        }
    }
}

test "M7: C quirk — the z rad-speed limiter uses the y optical depth (rad.c:3702)" {
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });
    const pp = ppRad(1.0, 0.1, .{ 0, 0, 0 }, 1.0, .{ 0, 0, 0 });
    const dims = [3]bool{ true, true, true };

    const aval = try radiation.calcRadWavespeeds(cfg, pp, &geo, .{ 5.0, 7.0, 1000.0 }, dims);
    // at rest in MINK the quadratic is identical per dim → bitwise equality
    try std.testing.expectEqual(aval[6 + 2], aval[6 + 4]); // zl == yl
    try std.testing.expectEqual(aval[6 + 3], aval[6 + 5]); // zr == yr
    // and the z speed is NOT what τ_z = 1000 would give
    try expect(@abs(@abs(aval[6 + 4]) - 4.0 / (3.0 * 1000.0)) > 1e-6);
    try expectClose(@abs(aval[6 + 4]), 4.0 / (3.0 * 7.0), 1e-12);
}

//
// ---- rad floors ---------------------------------------------------------------
//

test "M7: check_floors_rad — E floor and Ehat/rho, Ehat/u ratio floors (PUFFY)" {
    const rp = invert_rad.RadParams.puffy;
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });

    // absolute rad-frame floor (ρ, u chosen so no ratio floor also fires:
    // the ratio checks use Ehat of the *original* state)
    {
        var pp = ppRad(1.0e-70, 1.0e-71, .{ 0, 0, 0 }, 1.0e-85, .{ 0, 0, 0 });
        const ret = try invert_rad.checkFloorsRad(cfg, &pp, &geo, rp);
        try expect(ret == -1);
        try std.testing.expectEqual(rp.eradfloor, pp[L.index(.ee)]);
    }

    // E below the absolute floor with ordinary ρ, u: the Ehat/ρ and Ehat/u
    // MIN floors fire *after* the absolute floor and overwrite it (the C
    // cascade re-uses the original-state Ehat and Eratio)
    {
        var pp = ppRad(1.0, 0.1, .{ 0, 0, 0 }, 1.0e-85, .{ 0, 0, 0 });
        const ret = try invert_rad.checkFloorsRad(cfg, &pp, &geo, rp);
        try expect(ret == -1);
        // final value from the last firing floor: Eratio·EEUURATIOMIN·u
        try expect(pp[L.index(.ee)] > rp.eradfloor);
        try expectClose(pp[L.index(.ee)], rp.eeuuratiomin * 0.1, 1e-10);
    }

    // Ehat > EERHORATIOMAX·ρ raises ρ to Ehat/EERHORATIOMAX
    // (at rest with F̂ = 0, Ehat = Ê exactly)
    {
        var pp = ppRad(1.0e-6, 1.0e-7, .{ 0, 0, 0 }, 1.0, .{ 0, 0, 0 });
        const ehat = try radiation.calcFfEhat(cfg, pp, &geo);
        const ret = try invert_rad.checkFloorsRad(cfg, &pp, &geo, rp);
        try expect(ret == -1);
        try std.testing.expectEqual(1.0 / rp.eerhoratiomax * ehat, pp[L.index(.rho)]);
    }

    // in-range state is untouched
    {
        var pp = ppRad(1.0, 0.1, .{ 0.1, 0, 0 }, 0.5, .{ 0.2, 0, 0 });
        const pp0 = pp;
        const ret = try invert_rad.checkFloorsRad(cfg, &pp, &geo, rp);
        try expect(ret == 0);
        for (0..NV) |iv| try std.testing.expectEqual(pp0[iv], pp[iv]);
    }
}

//
// ---- evolution: uniform state with radiation stays static ----------------------
//

test "M7: uniform MINK state with radiation static over 50 steps (rest and boosted)" {
    const SimT = sim_mod.Sim(cfg);
    // {v, ũ_rad}: at rest, and gas+radiation both boosted
    const cases = [_][2]f64{ .{ 0.0, 0.0 }, .{ 0.25, 0.4 } };

    for (cases) |c| {
        const g = Grid.init(.{ .nx = 24, .ng = 3, .minx = 0, .maxx = 1 });
        var s = try SimT.init(std.testing.allocator, g, .{
            .coords = .mink,
            .gam = gam,
            .rad = invert_rad.RadParams.puffy,
            .bc_x = .periodic,
        });
        defer s.deinit();

        const ut = c[0] / @sqrt(1.0 - c[0] * c[0]);
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            try s.initCell(ix, 0, 0, ppRad(1.0, 0.1, .{ ut, 0, 0 }, 0.3, .{ c[1], 0, 0 }));
        }
        try s.finishInit();

        var pp0: [NV]f64 = undefined;
        s.p.load(10, 0, 0, &pp0);

        for (0..50) |_| try s.step(null);

        ix = 0;
        while (ix < s.nxi()) : (ix += 1) {
            var pp: [NV]f64 = undefined;
            s.p.load(ix, 0, 0, &pp);
            for (0..NV) |iv| {
                const scale = @max(@abs(pp0[iv]), 1e-30);
                const dev = @abs(pp[iv] - pp0[iv]) / scale;
                if (!(dev <= 1e-14)) {
                    std.debug.print("rad static drift: ix {d} iv {d}: {e:.17} -> {e:.17} (dev {e})\n", .{ ix, iv, pp0[iv], pp[iv], dev });
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
}
