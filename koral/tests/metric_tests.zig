//! M1 theory gates — analytic identities for the metric layer.
//! No C comparison here (that's tests/golden); everything below is checked
//! against mathematics: inverse identity, zero/symmetric Christoffels,
//! metric compatibility, known determinants, transform round-trips,
//! Jacobian consistency, horizon finiteness.

const std = @import("std");
const Dual3 = @import("../math/dual.zig").Dual3;
const metric = @import("../metric/metric.zig");
const forms = @import("../metric/forms.zig");
const coco = @import("../metric/coco.zig");
const precompute = @import("../metric/precompute.zig");
const Grid = @import("../grid.zig").Grid;

const Coords = metric.Coords;
const MetricParams = metric.MetricParams;
const pi = std.math.pi;

const puffy_mp = MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
const spin_mp = MetricParams{ .a = 0.9, .mksr0 = 0.1, .mksh0 = 0.9 };

/// Sample points per coordinate system (r spans horizon→outer PUFFY domain,
/// θ from near-axis to near-equator; angles avoid exact axis).
fn samplePoints(coords: Coords, mp: MetricParams, buf: *[64][4]f64) [][4]f64 {
    var n: usize = 0;
    switch (coords) {
        .mink => {
            const vals = [_]f64{ -3.4, 0.2, 5.7 };
            for (vals) |x| for (vals) |y| for (vals) |z| {
                buf[n] = .{ 0, x, y, z };
                n += 1;
            };
        },
        .bl, .ks => {
            const rh = 1.0 + @sqrt(1.0 - mp.a * mp.a);
            const rs = [_]f64{ rh + 0.05, rh + 0.5, 5.0, 21.7, 103.0, 450.0 };
            const ths = [_]f64{ 0.005, 0.3, 0.5 * pi, 2.7, pi - 0.005 };
            for (rs) |r| for (ths) |th| {
                buf[n] = .{ 0, r, th, 0.3 };
                n += 1;
            };
        },
        .mks2 => {
            const rs = [_]f64{ 1.85, 2.6, 5.0, 21.7, 103.0, 450.0 };
            const x2s = [_]f64{ 0.001, 0.1, 0.5, 0.9, 0.999 };
            for (rs) |r| for (x2s) |x2| {
                buf[n] = .{ 0, @log(r - mp.mksr0), x2, 0.3 };
                n += 1;
            };
        },
    }
    return buf[0..n];
}

fn expectClose(expected: f64, actual: f64, rtol: f64, atol: f64) !void {
    const err = @abs(expected - actual);
    if (err <= atol) return;
    const denom = @max(@abs(expected), @abs(actual));
    if (err / denom > rtol) {
        std.debug.print("expected {e}, got {e} (rel {e}, abs {e})\n", .{ expected, actual, err / denom, err });
        return error.ToleranceExceeded;
    }
}

const all_coords = [_]Coords{ .mink, .bl, .ks, .mks2 };
const all_mps = [_]MetricParams{ puffy_mp, spin_mp };

test "metric: g·G = identity at 1e-13 (1e-11 near horizon)" {
    for (all_mps) |mp| {
        for (all_coords) |coords| {
            var buf: [64][4]f64 = undefined;
            for (samplePoints(coords, mp, &buf)) |x| {
                const d = metric.compute(coords, mp, x);
                const near_horizon = coords != .mink and physR(coords, mp, x) < 2.2;
                const tol: f64 = if (near_horizon) 1e-11 else 1e-13;
                for (0..4) |i| {
                    for (0..4) |j| {
                        var s: f64 = 0;
                        for (0..4) |k| s += d.gcov[i][k] * d.gcon[k][j];
                        const want: f64 = if (i == j) 1.0 else 0.0;
                        try expectClose(want, s, tol, tol);
                    }
                }
            }
        }
    }
}

fn physR(coords: Coords, mp: MetricParams, x: [4]f64) f64 {
    return switch (coords) {
        .bl, .ks => x[1],
        .mks2 => @exp(x[1]) + mp.mksr0,
        .mink => 0,
    };
}

test "metric: Minkowski Christoffels are exactly zero" {
    var buf: [64][4]f64 = undefined;
    for (samplePoints(.mink, puffy_mp, &buf)) |x| {
        const d = metric.compute(.mink, puffy_mp, x);
        for (0..4) |i| for (0..4) |j| for (0..4) |k| {
            try std.testing.expectEqual(@as(f64, 0), d.kris[i][j][k]);
        };
        try std.testing.expectEqual(@as(f64, 1), d.gdet);
        for (d.dlgdet) |v| try std.testing.expectEqual(@as(f64, 0), v);
    }
}

test "metric: Christoffel symmetry Γ^i_jk = Γ^i_kj (exact)" {
    for (all_coords) |coords| {
        var buf: [64][4]f64 = undefined;
        for (samplePoints(coords, spin_mp, &buf)) |x| {
            const d = metric.compute(coords, spin_mp, x);
            for (0..4) |i| for (0..4) |j| for (0..4) |k| {
                try std.testing.expectEqual(d.kris[i][j][k], d.kris[i][k][j]);
            };
        }
    }
}

test "metric: dual-number ∂g matches 4th-order Richardson finite differences" {
    // Independent validation of the AD derivatives that feed Christoffels.
    for (all_mps) |mp| {
        for ([_]Coords{ .bl, .ks, .mks2 }) |coords| {
            var buf: [64][4]f64 = undefined;
            for (samplePoints(coords, mp, &buf)) |x| {
                // FD needs smooth headroom: skip points too close to axis/horizon.
                const r = physR(coords, mp, x);
                if (r < 2.3) continue;
                const th = switch (coords) {
                    .mks2 => forms.mks2Theta(Dual3.constant(x[2]), mp.mksh0).v,
                    else => x[2],
                };
                if (th < 0.2 or th > pi - 0.2) continue;

                const g = metric.gcovDual(coords, mp, x);
                for (1..4) |m| {
                    const h = 1e-4 * @max(1.0, @abs(x[m]));
                    var xp = x;
                    var xm = x;
                    for (0..4) |i| {
                        for (i..4) |j| {
                            xp[m] = x[m] + h;
                            xm[m] = x[m] - h;
                            const d1 = (metric.compute(coords, mp, xp).gcov[i][j] -
                                metric.compute(coords, mp, xm).gcov[i][j]) / (2 * h);
                            xp[m] = x[m] + h / 2;
                            xm[m] = x[m] - h / 2;
                            const d2 = (metric.compute(coords, mp, xp).gcov[i][j] -
                                metric.compute(coords, mp, xm).gcov[i][j]) / h;
                            const fd = (4.0 * d2 - d1) / 3.0;
                            try expectClose(fd, g[i][j].d[m - 1], 1e-7, 1e-9 * @max(1.0, @abs(g[i][j].v)));
                        }
                    }
                }
            }
        }
    }
}

test "metric: compatibility ∂_c g_ab = Γ^d_ca g_db + Γ^d_cb g_ad" {
    // Validates Christoffel assembly + inverse against the exact AD ∂g.
    for (all_mps) |mp| {
        for ([_]Coords{ .bl, .ks, .mks2 }) |coords| {
            var buf: [64][4]f64 = undefined;
            for (samplePoints(coords, mp, &buf)) |x| {
                const d = metric.compute(coords, mp, x);
                const g = metric.gcovDual(coords, mp, x);
                for (1..4) |c| {
                    for (0..4) |a_i| {
                        for (a_i..4) |b| {
                            var rhs: f64 = 0;
                            for (0..4) |dd| {
                                rhs += d.kris[dd][c][a_i] * d.gcov[dd][b] + d.kris[dd][c][b] * d.gcov[a_i][dd];
                            }
                            const lhs = g[a_i][b].d[c - 1];
                            // scale by the largest term entering the sum
                            var scale: f64 = 1.0;
                            for (0..4) |dd| {
                                scale = @max(scale, @abs(d.kris[dd][c][a_i] * d.gcov[dd][b]));
                                scale = @max(scale, @abs(d.kris[dd][c][b] * d.gcov[a_i][dd]));
                            }
                            try expectClose(lhs, rhs, 1e-11, 1e-11 * scale);
                        }
                    }
                }
            }
        }
    }
}

test "metric: MKS2 metric and coco flavors share one π" {
    const x2 = 0.37;
    const h0 = 0.9;
    const metric_theta = forms.mks2ThetaMetric(Dual3.constant(x2), h0).v;
    const coco_theta = forms.mks2Theta(Dual3.constant(x2), h0).v;
    try expectClose(metric_theta, coco_theta, 1e-15, 1e-15);
}

test "metric: √−g analytic — KS(a=0) = r² sinθ; MKS2 = KS × e^{x1} dθ/dx2; BL ≡ KS" {
    // Schwarzschild KS
    var buf: [64][4]f64 = undefined;
    for (samplePoints(.ks, puffy_mp, &buf)) |x| {
        const d = metric.compute(.ks, puffy_mp, x);
        try expectClose(x[1] * x[1] * @sin(x[2]), d.gdet, 1e-13, 1e-300);
    }
    // Kerr: BL and KS gdet coincide (same r,θ)
    for (samplePoints(.ks, spin_mp, &buf)) |x| {
        const dks = metric.compute(.ks, spin_mp, x);
        const dbl = metric.compute(.bl, spin_mp, x);
        try expectClose(dks.gdet, dbl.gdet, 1e-13, 1e-300);
        // C closed form: Σ·sinθ
        const sigma = x[1] * x[1] + spin_mp.a * spin_mp.a * @cos(x[2]) * @cos(x[2]);
        try expectClose(sigma * @sin(x[2]), dks.gdet, 1e-13, 1e-300);
    }
    // MKS2 = pushforward: gdet_KS(r,θ) · |J| with |J| = e^{x1} dθ/dx2.
    // Metric-flavor θ map throughout (same exact π as coco — see forms.zig).
    for (all_mps) |mp| {
        for (samplePoints(.mks2, mp, &buf)) |x| {
            const d = metric.compute(.mks2, mp, x);
            const xks = [4]f64{ x[0], @exp(x[1]) + mp.mksr0, forms.mks2ThetaMetric(Dual3.constant(x[2]), mp.mksh0).v, x[3] };
            const dks = metric.compute(.ks, mp, xks);
            const jac = @exp(x[1]) * forms.mks2DThetaDx2Metric(Dual3.constant(x[2]), mp.mksh0).v;
            try expectClose(dks.gdet * jac, d.gdet, 1e-13, 1e-300);
        }
    }
}

test "metric: dlgdet matches C's closed forms for KS and BL" {
    // C: calc_dlgdet_arb (metric.c:395) — KS: 2r/Σ, (−a²+2r²+3a²cos2θ)cotθ/(2Σ)
    for (all_mps) |mp| {
        var buf: [64][4]f64 = undefined;
        for (samplePoints(.ks, mp, &buf)) |x| {
            const d = metric.compute(.ks, mp, x);
            const a = mp.a;
            const sigma = x[1] * x[1] + a * a * @cos(x[2]) * @cos(x[2]);
            const dl0 = 2.0 * x[1] / sigma;
            const dl1 = (-a * a + 2.0 * x[1] * x[1] + 3.0 * a * a * @cos(2.0 * x[2])) / @tan(x[2]) / (2.0 * sigma);
            try expectClose(dl0, d.dlgdet[0], 1e-12, 1e-14);
            try expectClose(dl1, d.dlgdet[1], 1e-12, 1e-11 * @abs(dl1) + 1e-10);
            try expectClose(0.0, d.dlgdet[2], 0, 1e-14);
        }
    }
}

test "coco: BL↔KS↔MKS2 round-trips at 1e-12" {
    for (all_mps) |mp| {
        const rh = 1.0 + @sqrt(1.0 - mp.a * mp.a);
        var buf: [64][4]f64 = undefined;
        // start from MKS2 points (the production grid)
        for (samplePoints(.mks2, mp, &buf)) |x| {
            const r = physR(.mks2, mp, x);
            // KS↔MKS2 round trip (valid at all radii incl. inside horizon)
            {
                const y = coco.cocoN(coco.cocoN(x, .mks2, .ks, mp), .ks, .mks2, mp);
                for (0..4) |i| try expectClose(x[i], y[i], 1e-12, 1e-13);
            }
            // full chain MKS2→BL→MKS2 (t-shift defined only outside horizon)
            if (r > rh + 0.05) {
                const y = coco.cocoN(coco.cocoN(x, .mks2, .bl, mp), .bl, .mks2, mp);
                for (0..4) |i| try expectClose(x[i], y[i], 1e-12, 1e-12);
            }
        }
        // BL→KS→BL
        for (samplePoints(.bl, mp, &buf)) |x| {
            const y = coco.cocoN(coco.cocoN(x, .bl, .ks, mp), .ks, .bl, mp);
            for (0..4) |i| try expectClose(x[i], y[i], 1e-12, 1e-12);
        }
    }
}

test "coco: analytic Jacobians match FD of the point transform" {
    // Note: dxdx_BL2KS's φ-column (a/Δ) is NOT the derivative of our coco
    // (C omits the φ point-shift; the Jacobian carries the true transform),
    // so BL↔KS checks skip [3][1] — a C quirk preserved deliberately.
    for (all_mps) |mp| {
        const rh = 1.0 + @sqrt(1.0 - mp.a * mp.a);
        const pairs = [_][2]Coords{ .{ .mks2, .ks }, .{ .ks, .mks2 }, .{ .bl, .ks }, .{ .ks, .bl } };
        for (pairs) |pair| {
            var buf: [64][4]f64 = undefined;
            for (samplePoints(pair[0], mp, &buf)) |x| {
                if (physR(pair[0], mp, x) < rh + 0.2) continue;
                const jac = coco.dxdx(x, pair[0], pair[1], mp);
                // One exact π for metric and coco flavors (the C two-π split
                // is not preserved), so MKS2 gets no extra slack: the analytic
                // Jacobian IS the derivative of cocoN, FD noise only.
                const rt: f64 = 1e-9;
                for (0..4) |i| {
                    for (0..4) |m| {
                        if (i == 3 and m == 1 and (pair[0] == .bl or pair[1] == .bl)) continue;
                        const h = 1e-5 * @max(1.0, @abs(x[m]));
                        var xp = x;
                        var xm = x;
                        xp[m] += h;
                        xm[m] -= h;
                        const d1 = (coco.cocoN(xp, pair[0], pair[1], mp)[i] - coco.cocoN(xm, pair[0], pair[1], mp)[i]) / (2 * h);
                        xp[m] = x[m] + h / 2;
                        xm[m] = x[m] - h / 2;
                        const d2 = (coco.cocoN(xp, pair[0], pair[1], mp)[i] - coco.cocoN(xm, pair[0], pair[1], mp)[i]) / h;
                        const fd = (4.0 * d2 - d1) / 3.0;
                        try expectClose(fd, jac[i][m], rt, rt);
                    }
                }
            }
        }
    }
}

test "coco: J_my2out · J_out2my = identity at 1e-12" {
    for (all_mps) |mp| {
        const rh = 1.0 + @sqrt(1.0 - mp.a * mp.a);
        const pairs = [_][2]Coords{ .{ .mks2, .ks }, .{ .mks2, .bl }, .{ .bl, .ks } };
        for (pairs) |pair| {
            // With one exact π everywhere (C's two-π split not preserved),
            // dxdx pairs are true mutual inverses down to rounding — C's own
            // product deviated at ~1e-9 from its mixed-π flavors.
            const rt: f64 = 1e-12;
            var buf: [64][4]f64 = undefined;
            for (samplePoints(pair[0], mp, &buf)) |x| {
                if (physR(pair[0], mp, x) < rh + 0.05) continue;
                const j1 = coco.dxdx(x, pair[0], pair[1], mp);
                const y = coco.cocoN(x, pair[0], pair[1], mp);
                const j2 = coco.dxdx(y, pair[1], pair[0], mp);
                for (0..4) |i| {
                    for (0..4) |j| {
                        var s: f64 = 0;
                        for (0..4) |k| s += j2[i][k] * j1[k][j];
                        const want: f64 = if (i == j) 1.0 else 0.0;
                        try expectClose(want, s, rt, rt);
                    }
                }
            }
        }
    }
}

test "metric: g(BL) pushed through Jacobians equals g(MKS2) directly (r > 2.1)" {
    // Non-circular end-to-end check: our MKS2 metric comes from KS, here we
    // rebuild it from the *BL* form through the composite Jacobian.
    for (all_mps) |mp| {
        const rh = 1.0 + @sqrt(1.0 - mp.a * mp.a);
        var buf: [64][4]f64 = undefined;
        for (samplePoints(.mks2, mp, &buf)) |x| {
            if (physR(.mks2, mp, x) < rh + 0.7) continue;
            const d = metric.compute(.mks2, mp, x);
            // Evaluate g_BL at the *metric-flavor* θ — the flavor the MKS2
            // metric itself uses (coco's exact-π θ differs at ~1e-9).
            const xbl = [4]f64{ 0, @exp(x[1]) + mp.mksr0, forms.mks2ThetaMetric(Dual3.constant(x[2]), mp.mksh0).v, x[3] };
            const dbl = metric.compute(.bl, mp, xbl);
            const jac = coco.dxdx(x, .mks2, .bl, mp); // dx_BL/dx_MKS2
            var scale: f64 = 0;
            for (0..4) |i| for (0..4) |j| {
                scale = @max(scale, @abs(d.gcov[i][j]));
            };
            for (0..4) |i| {
                for (0..4) |j| {
                    var s: f64 = 0;
                    for (0..4) |k| {
                        for (0..4) |l| s += jac[k][i] * jac[l][j] * dbl.gcov[k][l];
                    }
                    try expectClose(d.gcov[i][j], s, 1e-11, 1e-11 * scale);
                }
            }
        }
    }
}

test "metric: finite at the horizon in horizon-penetrating coordinates" {
    for (all_mps) |mp| {
        const rh = 1.0 + @sqrt(1.0 - mp.a * mp.a);
        const xks = [4]f64{ 0, rh, 1.1, 0.3 };
        const dks = metric.compute(.ks, mp, xks);
        const xm = [4]f64{ 0, @log(rh - mp.mksr0), 0.4, 0.3 };
        const dm = metric.compute(.mks2, mp, xm);
        for ([_]metric.CoordData{ dks, dm }) |d| {
            for (0..4) |i| for (0..4) |j| {
                try std.testing.expect(std.math.isFinite(d.gcov[i][j]));
                try std.testing.expect(std.math.isFinite(d.gcon[i][j]));
                for (0..4) |k| try std.testing.expect(std.math.isFinite(d.kris[i][j][k]));
            };
            try std.testing.expect(std.math.isFinite(d.gdet) and d.gdet > 0);
        }
    }
}

test "metric: gttpert — g_tt = −1 + gttpert for BL/KS/MKS2" {
    for (all_mps) |mp| {
        for ([_]Coords{ .bl, .ks }) |coords| {
            var buf: [64][4]f64 = undefined;
            for (samplePoints(coords, mp, &buf)) |x| {
                const d = metric.compute(coords, mp, x);
                try expectClose(d.gcov[0][0], -1.0 + d.gttpert, 1e-13, 1e-15);
            }
        }
        // MKS2: time column of the map is identity ⇒ same relation holds —
        // but only at ~1e-9 for a≠0, because C's gttpert routes through the
        // exact-π coco θ while g_tt uses the metric-flavor θ (mirrored).
        var buf: [64][4]f64 = undefined;
        for (samplePoints(.mks2, mp, &buf)) |x| {
            const d = metric.compute(.mks2, mp, x);
            const rt: f64 = if (mp.a == 0) 1e-13 else 5e-9;
            try expectClose(d.gcov[0][0], -1.0 + d.gttpert, rt, 1e-15);
        }
    }
}

test "precompute: Christoffel trace equals the FD gdet gradient (the correction's defining property)" {
    const grid = Grid.init(.{
        .nx = 12,
        .ny = 10,
        .nz = 1,
        .ng = 3,
        .minx = @log(1.85 - 0.1),
        .maxx = @log(500.0 - 0.1),
        .miny = 0.001,
        .maxy = 0.999,
        .minz = -0.25 * pi,
        .maxz = 0.25 * pi,
    });
    var cache = try precompute.MetricCache.init(std.testing.allocator, grid, .{
        .coords = .mks2,
        .out_coords = .bl,
        .mp = puffy_mp,
    });
    defer cache.deinit();

    var iy: i64 = 0;
    while (iy < 10) : (iy += 3) {
        var ix: i64 = 0;
        while (ix < 12) : (ix += 4) {
            const gdet_c = cache.fillGeometry(ix, iy, 0).gdet;
            // κ = 1 (x-direction)
            {
                const hi = metric.compute(.mks2, puffy_mp, .{ 0, grid.xl(ix + 1), grid.yc(iy), grid.zc(0) }).gdet;
                const lo = metric.compute(.mks2, puffy_mp, .{ 0, grid.xl(ix), grid.yc(iy), grid.zc(0) }).gdet;
                const want = (hi - lo) / (grid.dx * gdet_c);
                var trace: f64 = 0;
                for (0..4) |mu| trace += cache.kr(mu, 1, mu, ix, iy, 0);
                try expectClose(want, trace, 1e-12, 1e-14);
            }
            // κ = 2 (y-direction)
            {
                const hi = metric.compute(.mks2, puffy_mp, .{ 0, grid.xc(ix), grid.yl(iy + 1), grid.zc(0) }).gdet;
                const lo = metric.compute(.mks2, puffy_mp, .{ 0, grid.xc(ix), grid.yl(iy), grid.zc(0) }).gdet;
                const want = (hi - lo) / (grid.dy * gdet_c);
                var trace: f64 = 0;
                for (0..4) |mu| trace += cache.kr(mu, 2, mu, ix, iy, 0);
                try expectClose(want, trace, 1e-12, 1e-14);
            }
        }
    }
}

test "precompute: fillGeometry/fillGeometryFace agree with direct computation" {
    const grid = Grid.init(.{
        .nx = 8,
        .ny = 6,
        .nz = 1,
        .ng = 3,
        .minx = @log(1.85 - 0.1),
        .maxx = @log(500.0 - 0.1),
        .miny = 0.001,
        .maxy = 0.999,
        .minz = -0.25 * pi,
        .maxz = 0.25 * pi,
    });
    var cache = try precompute.MetricCache.init(std.testing.allocator, grid, .{
        .coords = .mks2,
        .out_coords = .bl,
        .mp = puffy_mp,
    });
    defer cache.deinit();

    // centers (incl. a ghost cell)
    const cells = [_][2]i64{ .{ 0, 0 }, .{ 4, 3 }, .{ -3, -3 }, .{ 10, 8 } };
    for (cells) |c| {
        const geo = cache.fillGeometry(c[0], c[1], 0);
        const d = metric.compute(.mks2, puffy_mp, geo.xxvec);
        for (0..4) |i| for (0..4) |j| {
            try std.testing.expectEqual(d.gcov[i][j], geo.gg[i][j]);
            try std.testing.expectEqual(d.gcon[i][j], geo.GG[i][j]);
        };
        try std.testing.expectEqual(d.gdet, geo.gdet);
        try std.testing.expectEqual(d.gttpert, geo.gttpert);
        try expectClose(@sqrt(-1.0 / d.gcon[0][0]), geo.alpha, 1e-15, 0);
        // gcon column 4 must be deterministic and match the on-the-fly
        // geometryAt convention (rows 0..2 = 0, row 3 = gttpert). P1: storeBlocks
        // used to leave GG[0..2][4] as alloc-garbage, so the cached Geometry
        // differed bit-for-bit from the recomputed one and tripped msan.
        for (0..3) |i| try std.testing.expectEqual(@as(f64, 0), geo.GG[i][4]);
        try std.testing.expectEqual(d.gttpert, geo.GG[3][4]);
    }

    // faces: left x-face of cell 0 and the one-past-last x-face
    for ([_]i64{ -3, 0, 5, 11 }) |ix| {
        const geo = cache.fillGeometryFace(ix, 2, 0, 0);
        try std.testing.expectEqual(grid.xl(ix), geo.xxvec[1]);
        const d = metric.compute(.mks2, puffy_mp, geo.xxvec);
        try std.testing.expectEqual(d.gcov[1][1], geo.gg[1][1]);
        try std.testing.expectEqual(d.gdet, geo.gdet);
        // same col-4 determinism on the face path (fillGeometryFace)
        for (0..3) |i| try std.testing.expectEqual(@as(f64, 0), geo.GG[i][4]);
        try std.testing.expectEqual(d.gttpert, geo.GG[3][4]);
    }
    // y-face one past the last cell
    const geo_top = cache.fillGeometryFace(2, 6 + 3, 0, 1);
    try std.testing.expectEqual(grid.yl(9), geo_top.xxvec[2]);
}

test "precompute: Schwarzschild KS lapse α = 1/√(1+2/r) via fillGeometry" {
    const grid = Grid.init(.{
        .nx = 8,
        .ny = 6,
        .nz = 1,
        .ng = 2,
        .minx = 2.5,
        .maxx = 20.0,
        .miny = 0.3,
        .maxy = pi - 0.3,
        .minz = 0,
        .maxz = 0.5 * pi,
    });
    var cache = try precompute.MetricCache.init(std.testing.allocator, grid, .{
        .coords = .ks,
        .out_coords = .bl,
        .mp = .{ .a = 0 },
    });
    defer cache.deinit();
    var ix: i64 = 0;
    while (ix < 8) : (ix += 2) {
        const geo = cache.fillGeometry(ix, 3, 0);
        const r = geo.xxvec[1];
        try expectClose(1.0 / @sqrt(1.0 + 2.0 / r), geo.alpha, 1e-14, 0);
    }
}
