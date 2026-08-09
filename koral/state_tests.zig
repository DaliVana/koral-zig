//! M2+M3 theory gates: velocity conversions, Lorentz boosts, coordinate
//! transforms of primitives, p2u/u2p round trips, floors.
//!
//! Everything here checks *mathematical identities* (or documented C
//! quirks) — no golden data needed. Tolerances follow the plan:
//! algebraic identities 1e-13, boost/transform identities 1e-12,
//! iterative inversion round trips 1e-8 (bounded by U2PCONV = 1e-10).

const std = @import("std");
const config = @import("config.zig");
const layout = @import("layout.zig");
const relele = @import("relele.zig");
const frames = @import("frames.zig");
const mhd = @import("physics/bfield.zig");
const hydro = @import("physics/hydro.zig");
const p2u_mod = @import("p2u.zig");
const invert = @import("solve/invert.zig");
const metric = @import("metric/metric.zig");
const precompute = @import("metric/precompute.zig");
const coco = @import("metric/coco.zig");
const Geometry = @import("geometry.zig").Geometry;

const Coords = config.Coords;
const MetricParams = metric.MetricParams;
const geometryAt = precompute.geometryAt;

const puffy_mp = MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
const spin_mp = MetricParams{ .a = 0.9, .mksr0 = 0.1, .mksh0 = 0.9 };

const pgamma: f64 = 5.0 / 3.0; // PUFFY GAMMA

fn expectClose(a: f64, b: f64, tol: f64) !void {
    const dev = @abs(a - b) / @max(1.0, @max(@abs(a), @abs(b)));
    if (!(dev <= tol)) {
        std.debug.print("expectClose: {e:.17} vs {e:.17} (dev {e}, tol {e})\n", .{ a, b, dev, tol });
        return error.TestExpectedApproxEq;
    }
}

/// The geometry battery all M2/M3 tests run over: flat space, Schwarzschild
/// KS near and far, spinning KS inside the ergosphere, BL, and PUFFY MKS2.
const test_geoms = [_]struct { coords: Coords, mp: MetricParams, x: [4]f64 }{
    .{ .coords = .mink, .mp = puffy_mp, .x = .{ 0, 1.0, 2.0, 3.0 } },
    .{ .coords = .ks, .mp = puffy_mp, .x = .{ 0, 8.0, 1.1, 0.3 } },
    .{ .coords = .ks, .mp = puffy_mp, .x = .{ 0, 2.05, 0.8, 0.0 } },
    .{ .coords = .ks, .mp = spin_mp, .x = .{ 0, 1.6, 1.4, 0.2 } }, // ergoregion
    .{ .coords = .bl, .mp = spin_mp, .x = .{ 0, 2.2, 1.0, 0.1 } },
    .{ .coords = .mks2, .mp = puffy_mp, .x = .{ 0, @log(3.0 - 0.1), 0.3, 0.0 } },
    .{ .coords = .mks2, .mp = puffy_mp, .x = .{ 0, @log(1.9 - 0.1), 0.62, 0.5 } },
};

/// VELR components with an exact target Lorentz factor γ w.r.t. the normal
/// observer: random direction, scaled so 1 + q² = γ².
fn velrWithGamma(rng: std.Random, gg: *const [4][5]f64, gamma_target: f64) [3]f64 {
    var n: [3]f64 = undefined;
    for (0..3) |i| n[i] = 2.0 * rng.float(f64) - 1.0;
    var qsq: f64 = 0;
    for (0..3) |i| {
        for (0..3) |j| {
            qsq += n[i] * n[j] * gg[i + 1][j + 1];
        }
    }
    const scale = @sqrt((gamma_target * gamma_target - 1.0) / qsq);
    return .{ n[0] * scale, n[1] * scale, n[2] * scale };
}

fn logUniform(rng: std.Random, lo_exp: f64, hi_exp: f64) f64 {
    return std.math.pow(f64, 10.0, lo_exp + (hi_exp - lo_exp) * rng.float(f64));
}

//
// ---- M2: velocity conversions ---------------------------------------------
//

test "conv_vels: u·u = -1 after VELR→VEL4 and VEL3→VEL4 everywhere" {
    var prng = std.Random.DefaultPrng.init(0x4b52414c);
    const rng = prng.random();
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        for (0..20) |_| {
            const v = velrWithGamma(rng, &geo.gg, 1.0 + 8.0 * rng.float(f64));
            // 1e-12: the roundoff of u·u grows like γ² (γ up to 9 here)
            const uc4 = try relele.convert(.{ 0, v[0], v[1], v[2] }, .velr, .vel4, &geo, .recompute_ut);
            const uc4cov = relele.lowerVec(uc4, &geo);
            try expectClose(relele.dot(uc4, uc4cov), -1.0, 1e-12);

            // through VEL3 and back up to VEL4
            const uc3 = try relele.convert(uc4, .vel4, .vel3, &geo, .recompute_ut);
            const uc4b = try relele.convert(uc3, .vel3, .vel4, &geo, .recompute_ut);
            const uc4bcov = relele.lowerVec(uc4b, &geo);
            try expectClose(relele.dot(uc4b, uc4bcov), -1.0, 1e-12);
        }
    }
}

test "conv_vels: all round trips return the input" {
    var prng = std.Random.DefaultPrng.init(0x600d5eed);
    const rng = prng.random();
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        for (0..10) |_| {
            const v = velrWithGamma(rng, &geo.gg, 1.0 + 5.0 * rng.float(f64));
            const vr = [4]f64{ 0, v[0], v[1], v[2] };

            const via4 = try relele.convert(try relele.convert(vr, .velr, .vel4, &geo, .recompute_ut), .vel4, .velr, &geo, .recompute_ut);
            for (1..4) |i| try expectClose(via4[i], vr[i], 1e-13);

            const via3 = try relele.convert(try relele.convert(vr, .velr, .vel3, &geo, .recompute_ut), .vel3, .velr, &geo, .recompute_ut);
            for (1..4) |i| try expectClose(via3[i], vr[i], 1e-13);

            const uc4 = try relele.convert(vr, .velr, .vel4, &geo, .recompute_ut);
            const back4 = try relele.convert(try relele.convert(uc4, .vel4, .vel3, &geo, .recompute_ut), .vel3, .vel4, &geo, .recompute_ut);
            for (0..4) |i| try expectClose(back4[i], uc4[i], 1e-13);
        }
    }
}

test "conv_vels: superluminal VEL3 is rejected" {
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });
    const bad = [4]f64{ 0, 1.5, 0, 0 };
    try std.testing.expectError(
        relele.Error.VelocityConversionFailed,
        relele.convert(bad, .vel3, .vel4, &geo, .recompute_ut),
    );
}

test "normal observer: n·n = -1 and its VELR vanishes" {
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        const ncon = relele.normalObs4vel(&geo);
        const ncov = relele.lowerVec(ncon, &geo);
        try expectClose(relele.dot(ncon, ncov), -1.0, 1e-13);

        const nrel = relele.normalObsVelr(&geo);
        for (1..4) |i| try expectClose(nrel[i], 0.0, 1e-13);
    }
}

test "indices: raise∘lower is the identity" {
    var prng = std.Random.DefaultPrng.init(0x1dece5);
    const rng = prng.random();
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        var a: [4]f64 = undefined;
        for (0..4) |i| a[i] = 2.0 * rng.float(f64) - 1.0;
        const back = relele.raiseVec(relele.lowerVec(a, &geo), &geo);
        for (0..4) |i| try expectClose(back[i], a[i], 1e-12);

        var t: [4][4]f64 = undefined;
        for (0..4) |i| {
            for (0..4) |j| t[i][j] = 2.0 * rng.float(f64) - 1.0;
        }
        const tback = relele.raiseBoth(relele.lowerBoth(t, &geo), &geo);
        for (0..4) |i| {
            for (0..4) |j| try expectClose(tback[i][j], t[i][j], 1e-12);
        }
    }
}

//
// ---- M2: Lorentz boosts ----------------------------------------------------
//

test "Lorentz matrix preserves the coordinate metric: g(Lx,Ly) = g(x,y)" {
    var prng = std.Random.DefaultPrng.init(0x10e247);
    const rng = prng.random();
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        const v = velrWithGamma(rng, &geo.gg, 3.0);
        const l = try frames.lorentzLab2Ff(v, &geo);

        // (L^T g L)_κλ = g_κλ
        for (0..4) |kk| {
            for (0..4) |ll| {
                var s: f64 = 0;
                for (0..4) |m| {
                    for (0..4) |n| {
                        s += l[m][kk] * l[n][ll] * geo.gg[m][n];
                    }
                }
                try expectClose(s, geo.gg[kk][ll], 1e-12);
            }
        }
    }
}

test "Lorentz lab2ff maps the fluid velocity to α·(normal observer)" {
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        const v = [3]f64{ 0.2, -0.1, 0.05 };
        const l = try frames.lorentzLab2Ff(v, &geo);
        const u = try relele.convert(.{ 0, v[0], v[1], v[2] }, .velr, .vel4, &geo, .recompute_ut);
        const lu = relele.transformVec(u, l);
        const w = relele.normalObs4vel(&geo);
        for (0..4) |i| try expectClose(lu[i], w[i], 1e-12);
    }
}

test "Lorentz lab2ff and ff2lab matrices are mutual inverses" {
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        const v = [3]f64{ -0.15, 0.3, 0.08 };
        const l1 = try frames.lorentzLab2Ff(v, &geo);
        const l2 = try frames.lorentzFf2Lab(v, &geo);
        for (0..4) |i| {
            for (0..4) |j| {
                var s: f64 = 0;
                for (0..4) |k| s += l2[i][k] * l1[k][j];
                try expectClose(s, relele.kron(i, j), 1e-12);
            }
        }
    }
}

test "vector boosts round-trip exactly (α cancels between directions)" {
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        const v = [3]f64{ 0.25, 0.1, -0.2 };
        const a = [4]f64{ 0.7, -1.3, 0.4, 2.1 };
        const there = try frames.boost2Lab2Ff(a, v, &geo);
        const back = try frames.boost2Ff2Lab(there, v, &geo);
        for (0..4) |i| try expectClose(back[i], a[i], 1e-12);
    }
}

test "tensor boost round-trip pins the C α-quirk (ff2lab applies no α)" {
    // C's boost22_ff2lab α-division is dead code (frames.c:435-445): the
    // output is the bare L L T. So undoing lab2ff's α scaling by hand and
    // then applying ff2lab must recover T exactly. If ff2lab ever gained a
    // real α correction (the quirk "fixed"), this fails.
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        const v = [3]f64{ 0.12, -0.22, 0.05 };
        var t: [4][4]f64 = undefined;
        for (0..4) |i| {
            for (0..4) |j| t[i][j] = 0.1 + @as(f64, @floatFromInt(i * 4 + j));
        }
        var there = try frames.boost22Lab2Ff(t, v, &geo);
        for (0..4) |i| {
            there[i][0] /= geo.alpha;
            there[0][i] /= geo.alpha;
        }
        const back = try frames.boost22Ff2Lab(there, v, &geo);
        for (0..4) |i| {
            for (0..4) |j| {
                try expectClose(back[i][j], t[i][j], 1e-11);
            }
        }
    }
}

//
// ---- M2: coordinate transforms of vectors and primitives -------------------
//

test "trans2_coco preserves u·u between coordinate systems" {
    const pairs = [_]struct { from: Coords, to: Coords, x: [4]f64, mp: MetricParams, tol: f64 }{
        .{ .from = .bl, .to = .ks, .x = .{ 0, 4.0, 1.1, 0.3 }, .mp = spin_mp, .tol = 1e-12 },
        .{ .from = .ks, .to = .bl, .x = .{ 0, 3.0, 0.9, 0.1 }, .mp = spin_mp, .tol = 1e-12 },
        .{ .from = .mks2, .to = .ks, .x = .{ 0, @log(4.0 - 0.1), 0.4, 0.2 }, .mp = puffy_mp, .tol = 1e-12 },
        // ks→mks2 and the bl↔mks2 composites cross C's two-π flavor mix
        .{ .from = .ks, .to = .mks2, .x = .{ 0, 4.0, 1.2, 0.2 }, .mp = puffy_mp, .tol = 1e-8 },
        .{ .from = .bl, .to = .mks2, .x = .{ 0, 5.0, 1.3, 0.4 }, .mp = puffy_mp, .tol = 1e-8 },
        .{ .from = .mks2, .to = .bl, .x = .{ 0, @log(6.0 - 0.1), 0.55, 0.0 }, .mp = puffy_mp, .tol = 1e-8 },
    };
    var prng = std.Random.DefaultPrng.init(0xc0c0);
    const rng = prng.random();
    for (pairs) |pc| {
        const g1 = geometryAt(pc.from, pc.mp, pc.x);
        const x2 = coco.cocoN(pc.x, pc.from, pc.to, pc.mp);
        const g2 = geometryAt(pc.to, pc.mp, x2);

        const v = velrWithGamma(rng, &g1.gg, 2.0);
        const u = try relele.convert(.{ 0, v[0], v[1], v[2] }, .velr, .vel4, &g1, .recompute_ut);
        const uc2 = frames.trans2Coco(pc.x, u, pc.from, pc.to, pc.mp);

        const s1 = relele.dot(u, relele.lowerVec(u, &g1));
        const s2 = relele.dot(uc2, relele.lowerVec(uc2, &g2));
        try expectClose(s2, s1, pc.tol);
    }
}

test "trans_pall_coco: BL→MKS2→BL round-trips all PUFFY primitives" {
    const cfg = config.puffy;
    const L = layout.VarLayout(cfg);

    const x_bl = [4]f64{ 0, 5.0, 1.2, 0.3 };
    const g_bl = geometryAt(.bl, puffy_mp, x_bl);
    const x_mks2 = coco.cocoN(x_bl, .bl, .mks2, puffy_mp);
    const g_mks2 = geometryAt(.mks2, puffy_mp, x_mks2);

    var pp: [L.count]f64 = @splat(0);
    pp[L.index(.rho)] = 1.3;
    pp[L.index(.uu)] = 0.05;
    pp[L.index(.vx)] = -0.05;
    pp[L.index(.vy)] = 0.01;
    pp[L.index(.vz)] = 0.15;
    pp[L.index(.entr)] = hydro.sFromU(1.3, 0.05, pgamma);
    pp[L.index(.b1)] = 0.02;
    pp[L.index(.b2)] = -0.005;
    pp[L.index(.b3)] = 0.04;
    pp[L.index(.ee)] = 0.01;
    pp[L.index(.fx)] = 0.02;
    pp[L.index(.fy)] = -0.01;
    pp[L.index(.fz)] = 0.03;

    const pp_m = try frames.transPallCoco(cfg, pp, &g_bl, &g_mks2, puffy_mp);
    // note: geom1 for the way back must carry the MKS2 position
    const pp_b = try frames.transPallCoco(cfg, pp_m, &g_mks2, &g_bl, puffy_mp);

    // scalars ride through untouched
    try std.testing.expectEqual(pp[L.index(.rho)], pp_b[L.index(.rho)]);
    try std.testing.expectEqual(pp[L.index(.uu)], pp_b[L.index(.uu)]);
    try std.testing.expectEqual(pp[L.index(.ee)], pp_b[L.index(.ee)]);
    // vectors cross the two-π flavor mix twice
    for ([_]usize{ L.index(.vx), L.index(.vy), L.index(.vz), L.index(.b1), L.index(.b2), L.index(.b3), L.index(.fx), L.index(.fy), L.index(.fz) }) |iv| {
        try expectClose(pp_b[iv], pp[iv], 1e-8);
    }
}

//
// ---- M3: entropy, Tij, b·u -------------------------------------------------
//

test "entropy: uFromS inverts sFromU" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rng = prng.random();
    for (0..200) |_| {
        const rho = logUniform(rng, -20, 2);
        const u = rho * logUniform(rng, -8, 2);
        const s = hydro.sFromU(rho, u, pgamma);
        try expectClose(hydro.uFromS(s, rho, pgamma), u, 1e-12);
    }
}

test "b·u = 0 and bsq = b·b for the magnetic four-vector" {
    var prng = std.Random.DefaultPrng.init(0xb0b);
    const rng = prng.random();
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        const v = velrWithGamma(rng, &geo.gg, 1.0 + 4.0 * rng.float(f64));
        const u = try relele.convertBoth(.{ 0, v[0], v[1], v[2] }, .velr, &geo);
        const B = [3]f64{ 0.3, -0.7, 0.2 };
        const b = mhd.bconBcovBsqFrom4vel(B, u.con, u.cov, &geo);
        try expectClose(relele.dot(b.bcon, u.cov), 0.0, 1e-13);
        try expectClose(relele.dot(b.bcon, b.bcov), b.bsq, 1e-13);
    }
}

test "p2u momentum rows equal gdet·T^t_i from calc_Tij" {
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        var pp: [L.count]f64 = .{ 1.0, 0.2, 0.1, -0.05, 0.2, 0, 0.15, 0.05, -0.1 };
        pp[L.index(.entr)] = hydro.sFromU(pp[0], pp[1], pgamma);

        var uu: [L.count]f64 = @splat(0);
        try p2u_mod.p2uMhd(cfg, pp, &uu, &geo, pgamma);

        const t = try hydro.calcTij(cfg, pp, &geo, pgamma);
        const tmix = relele.lowerSecond(t, &geo); // T^μ_ν
        for (1..4) |i| {
            try expectClose(uu[L.index(.vx) + i - 1], geo.gdet * tmix[0][i], 1e-12);
        }
        // energy row: uu[UU] = gdet·(T^t_t + rho·u^t), assembled via utp1
        const u = try relele.convertBoth(.{ 0, pp[2], pp[3], pp[4] }, .velr, &geo);
        try expectClose(uu[L.index(.uu)], geo.gdet * (tmix[0][0] + pp[0] * u.con[0]), 1e-10);
    }
}

//
// ---- M3: p2u → u2p round trips ---------------------------------------------
//

fn randomState(
    comptime cfg: config.Config,
    rng: std.Random,
    geo: *const Geometry,
    max_gamma: f64,
    max_b2orho: f64,
) [layout.VarLayout(cfg).count]f64 {
    const L = layout.VarLayout(cfg);
    var pp: [L.count]f64 = @splat(0);
    const rho = logUniform(rng, -10, 2);
    const uint = rho * logUniform(rng, -6, 2);
    pp[L.index(.rho)] = rho;
    pp[L.index(.uu)] = uint;
    const v = velrWithGamma(rng, &geo.gg, 1.0 + (max_gamma - 1.0) * rng.float(f64));
    pp[L.index(.vx)] = v[0];
    pp[L.index(.vy)] = v[1];
    pp[L.index(.vz)] = v[2];
    pp[L.index(.entr)] = hydro.sFromU(rho, uint, pgamma);
    if (comptime L.hasVar(.b1)) {
        if (rng.float(f64) < 0.8) {
            // scale a random direction so gg_ij B^i B^j = f·ρ (b² is the
            // same ballpark; exact value is irrelevant for coverage)
            const f = logUniform(rng, -8, std.math.log10(max_b2orho));
            const bdir = velrWithGamma(rng, &geo.gg, @sqrt(1.0 + f * rho));
            pp[L.index(.b1)] = bdir[0];
            pp[L.index(.b2)] = bdir[1];
            pp[L.index(.b3)] = bdir[2];
        }
    }
    return pp;
}

test "u2p_hot recovers p2u output across the state sweep (≤1e-8)" {
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    var prng = std.Random.DefaultPrng.init(0x2b2b);
    const rng = prng.random();

    var worst: f64 = 0;
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        for (0..300) |_| {
            const pp0 = randomState(cfg, rng, &geo, 5.0, 100.0);
            var uu: [L.count]f64 = @splat(0);
            try p2u_mod.p2uMhd(cfg, pp0, &uu, &geo, pgamma);

            // perturbed initial guess, as the evolution would hand it in
            var pp = pp0;
            pp[L.index(.rho)] *= 1.1;
            pp[L.index(.uu)] *= 0.9;

            const ret = invert.u2pSolverW(cfg, uu, &pp, &geo, .hot, pgamma);
            try std.testing.expectEqual(@as(i32, 0), ret);
            // Physical error scales: the solver converges W = (ρ+Γu)γ² to
            // U2PCONV, so |Δu| ~ conv·W-scale (not conv·u — u can be ≪ ρ),
            // and VELR errors are absolute at the γ scale (order 1).
            const wscale = pp0[L.index(.rho)] + pgamma * pp0[L.index(.uu)];
            for (0..L.count) |iv| {
                if (iv == L.index(.entr)) continue; // ENTR is Sc/γ, not the input entropy
                const denom = if (iv == L.index(.uu))
                    wscale
                else if (iv >= L.index(.vx) and iv <= L.index(.vz))
                    @max(1.0, @abs(pp0[iv]))
                else
                    @max(1e-30, @abs(pp0[iv]));
                const dev = @abs(pp[iv] - pp0[iv]) / denom;
                if (dev > worst) worst = dev;
                // 1e-7: U2PCONV=1e-10 on the residual, amplified by the
                // root-to-primitive condition number (~γ², b²/ρ up to 100)
                if (dev > 1e-7) {
                    std.debug.print("u2p roundtrip var {d}: {e} vs {e} (dev {e})\n", .{ iv, pp0[iv], pp[iv], dev });
                    return error.TestExpectedApproxEq;
                }
            }
        }
    }
    std.debug.print("u2p hot roundtrip worst dev: {e:.3}\n", .{worst});
}

test "u2p scale invariance: U → λU (B → √λ B) rescales (ρ,u) by λ, v unchanged" {
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    const geo = geometryAt(.ks, puffy_mp, .{ 0, 6.0, 1.0, 0.0 });

    var pp0: [L.count]f64 = .{ 1.0, 0.1, 0.2, -0.1, 0.05, 0, 0.1, 0.02, -0.03 };
    pp0[L.index(.entr)] = hydro.sFromU(pp0[0], pp0[1], pgamma);
    var uu: [L.count]f64 = @splat(0);
    try p2u_mod.p2uMhd(cfg, pp0, &uu, &geo, pgamma);

    const lam: f64 = 1.0e-6;
    var uu_s = uu;
    for (0..L.count) |iv| uu_s[iv] *= lam;
    for ([_]usize{ L.index(.b1), L.index(.b2), L.index(.b3) }) |iv| {
        uu_s[iv] = uu[iv] * @sqrt(lam);
    }
    // conserved entropy scales like ρ·log(...): not a pure λ scaling — rebuild
    // it from the scaled primitive state instead.
    var pp_s = pp0;
    pp_s[L.index(.rho)] *= lam;
    pp_s[L.index(.uu)] *= lam;
    pp_s[L.index(.b1)] *= @sqrt(lam);
    pp_s[L.index(.b2)] *= @sqrt(lam);
    pp_s[L.index(.b3)] *= @sqrt(lam);
    pp_s[L.index(.entr)] = hydro.sFromU(pp_s[0], pp_s[1], pgamma);
    var uu_ref: [L.count]f64 = @splat(0);
    try p2u_mod.p2uMhd(cfg, pp_s, &uu_ref, &geo, pgamma);
    for ([_]usize{ 0, 1, 2, 3, 4, 6, 7, 8 }) |iv| {
        try expectClose(uu_ref[iv], uu_s[iv], 1e-13);
    }

    var pp = pp_s;
    pp[L.index(.rho)] *= 1.2;
    const ret = invert.u2pSolverW(cfg, uu_ref, &pp, &geo, .hot, pgamma);
    try std.testing.expectEqual(@as(i32, 0), ret);
    try expectClose(pp[L.index(.rho)] / lam, pp0[L.index(.rho)], 1e-10);
    try expectClose(pp[L.index(.uu)] / lam, pp0[L.index(.uu)], 1e-10);
    for ([_]usize{ 2, 3, 4 }) |iv| try expectClose(pp[iv], pp0[iv], 1e-10);
}

test "entropy branch agrees with energy branch on smooth states (≤1e-6)" {
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    var prng = std.Random.DefaultPrng.init(0xe27);
    const rng = prng.random();

    const geo = geometryAt(.ks, puffy_mp, .{ 0, 5.0, 1.2, 0.0 });
    for (0..200) |_| {
        var pp0 = randomState(cfg, rng, &geo, 2.0, 1.0);
        // smooth: moderate internal energy
        pp0[L.index(.uu)] = pp0[L.index(.rho)] * 0.1;
        pp0[L.index(.entr)] = hydro.sFromU(pp0[L.index(.rho)], pp0[L.index(.uu)], pgamma);
        var uu: [L.count]f64 = @splat(0);
        try p2u_mod.p2uMhd(cfg, pp0, &uu, &geo, pgamma);

        var pp_hot = pp0;
        var pp_ent = pp0;
        try std.testing.expectEqual(@as(i32, 0), invert.u2pSolverW(cfg, uu, &pp_hot, &geo, .hot, pgamma));
        try std.testing.expectEqual(@as(i32, 0), invert.u2pSolverW(cfg, uu, &pp_ent, &geo, .entropy, pgamma));

        for ([_]usize{ 0, 1, 2, 3, 4 }) |iv| {
            const dev = @abs(pp_hot[iv] - pp_ent[iv]) /
                @max(1e-30, @max(@abs(pp_hot[iv]), @abs(pp_ent[iv])));
            if (@abs(pp_hot[iv]) > 1e-290) {
                try std.testing.expect(dev <= 1e-6);
            }
        }
    }
}

test "u2p fuzz: random conserved states never produce NaN primitives" {
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    var prng = std.Random.DefaultPrng.init(0xf022);
    const rng = prng.random();

    var nok: usize = 0;
    var nfail: usize = 0;
    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        for (0..2000) |_| {
            var uu: [L.count]f64 = undefined;
            for (0..L.count) |iv| {
                uu[iv] = (2.0 * rng.float(f64) - 1.0) * logUniform(rng, -10, 2);
            }
            var pp: [L.count]f64 = .{ 1.0, 0.1, 0, 0, 0, 0, 0, 0, 0 };
            pp[L.index(.entr)] = hydro.sFromU(1.0, 0.1, pgamma);
            const ret = invert.u2pSolverW(cfg, uu, &pp, &geo, .hot, pgamma);
            if (ret == 0) {
                nok += 1;
                for (0..L.count) |iv| {
                    try std.testing.expect(!std.math.isNan(pp[iv]));
                }
            } else {
                nfail += 1;
            }
        }
    }
    std.debug.print("u2p fuzz: {d} inverted, {d} rejected (flagged, no NaN)\n", .{ nok, nfail });
}

//
// ---- M3: floors -------------------------------------------------------------
//

test "floors: postconditions and near-idempotence" {
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    const fl = invert.FloorParams.puffy;
    var prng = std.Random.DefaultPrng.init(0xf10025);
    const rng = prng.random();

    for (test_geoms) |tg| {
        const geo = geometryAt(tg.coords, tg.mp, tg.x);
        for (0..200) |_| {
            var pp = randomState(cfg, rng, &geo, 12.0, 1.0);
            // push states across every floor with some probability
            if (rng.float(f64) < 0.3) pp[L.index(.rho)] = 1e-35;
            if (rng.float(f64) < 0.3) pp[L.index(.uu)] = pp[L.index(.rho)] * 1e-14;
            if (rng.float(f64) < 0.3) pp[L.index(.uu)] = pp[L.index(.rho)] * 1e4;
            if (rng.float(f64) < 0.3) {
                pp[L.index(.b1)] = 30.0 * @sqrt(pp[L.index(.rho)]);
            }

            // For γ ≫ GAMMAMAXHD combined with a strong field, the
            // drift-frame reset can yield a superluminal 3-velocity. C
            // ignores conv_vels' error there and reads an *uninitialized*
            // ucont (u2p.c:688 + relele.c:199) — undefined behavior we
            // refuse to mirror; such states are rejected with an error.
            _ = invert.checkFloorsMhd(cfg, &pp, &geo, pgamma, fl) catch continue;

            // postconditions
            try std.testing.expect(pp[L.index(.rho)] >= fl.rhofloor);
            try std.testing.expect(pp[L.index(.uu)] >= fl.uurhoratiomin * pp[L.index(.rho)] * (1.0 - 1e-12));
            try std.testing.expect(pp[L.index(.uu)] <= fl.uurhoratiomax * pp[L.index(.rho)] * (1.0 + 1e-12));
            var qsq: f64 = 0;
            for (1..4) |i| {
                for (1..4) |j| {
                    qsq += pp[L.index(.vx) - 1 + i] * pp[L.index(.vx) - 1 + j] * geo.gg[i][j];
                }
            }
            try std.testing.expect(1.0 + qsq <= fl.gammamaxhd * fl.gammamaxhd * (1.0 + 1e-10));

            // Idempotence holds only when the magnetic floor did not fire:
            // the drift-frame reset changes the velocity, which changes b²,
            // so a second pass can re-floor by an O(Δv) factor (C behaves
            // identically). For non-magnetic floors the caps are exact.
            const u = try relele.uconUcovFromPrims(
                .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
                &geo,
            );
            const b = mhd.bconBcovBsqFrom4vel(
                .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
                u.con,
                u.cov,
                &geo,
            );
            if (b.bsq < 0.9 * fl.b2rhoratiomax * pp[L.index(.rho)] and
                b.bsq < 0.9 * fl.b2uuratiomax * pp[L.index(.uu)])
            {
                var pp2 = pp;
                _ = try invert.checkFloorsMhd(cfg, &pp2, &geo, pgamma, fl);
                for (0..L.count) |iv| {
                    const dev = @abs(pp2[iv] - pp[iv]) / @max(1e-30, @abs(pp[iv]));
                    try std.testing.expect(dev <= 1e-13);
                }
            }
        }
    }
}

test "u2p cascade: clean state uses the hot branch; garbage requests fixup" {
    const cfg = config.Config{ .modules = &.{ .hydro, .mhd } };
    const L = layout.VarLayout(cfg);
    const geo = geometryAt(.ks, puffy_mp, .{ 0, 6.0, 1.0, 0.0 });

    var pp0: [L.count]f64 = .{ 1.0, 0.1, 0.1, -0.02, 0.05, 0, 0.1, 0.0, 0.02 };
    pp0[L.index(.entr)] = hydro.sFromU(pp0[0], pp0[1], pgamma);
    var uu: [L.count]f64 = @splat(0);
    try p2u_mod.p2uMhd(cfg, pp0, &uu, &geo, pgamma);

    var pp = pp0;
    const res = invert.u2pMhd(cfg, uu, &pp, &geo, pgamma, .puffy);
    try std.testing.expectEqual(@as(i32, 0), res.ret);
    try std.testing.expect(!res.entropy_used);
    try std.testing.expect(!res.fixup);

    // negative conserved density → floor + fixup demanded
    var uu_bad = uu;
    uu_bad[L.index(.rho)] = -uu[L.index(.rho)];
    var pp_b = pp0;
    const res_b = invert.u2pMhd(cfg, uu_bad, &pp_b, &geo, pgamma, .puffy);
    try std.testing.expect(res_b.fixup);
}
