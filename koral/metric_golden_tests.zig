//! C-oracle golden comparisons (M1: metric layer).
//!
//! Reads binary KGLD files produced by oracle/harness_metric.c (built from
//! koral_lite with PROBLEM=147) and compares this implementation's output
//! at the recorded input points. Regenerate with tools/gen_golden.sh.
//!
//! Tolerances: all deviations are normalized by max(1, |C|, |zig|).
//! BL/KS/MINK quantities gate at 1e-13..1e-12 (pure algebraic/libm noise).
//! MKS2 *derived* quantities (inverse, dlgdet, Christoffels) gate at 1e-8:
//! C's Mathematica exports mix incompatible θ-map variants (truncated
//! `Pi` vs exact-π literals, full-θ vs (θ−π/2) forms), so C's own
//! g·G−I reaches 4.6e-9 near the axis — measured from the golden data
//! itself. Our AD implementation is internally consistent at 1e-15 and
//! matches C's *covariant* MKS2 metric (θ_C-form, see forms.zig) at
//! 1e-13; the looser MKS2 gates reflect C's internal spread, not ours.

const std = @import("std");
const metric = @import("metric/metric.zig");
const coco = @import("metric/coco.zig");
const precompute = @import("metric/precompute.zig");
const Grid = @import("grid.zig").Grid;
const gold = @import("testing/golden.zig");

const pi = std.math.pi;

/// PUFFY build parameters of the oracle (PROBLEMS/PUFFY/define.h).
const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };

fn expectGolden(c_val: f64, zig_val: f64, rtol: f64, what: []const u8, irec: usize) !void {
    const err = @abs(c_val - zig_val);
    const tol = rtol * @max(1.0, @max(@abs(c_val), @abs(zig_val)));
    if (!(err <= tol)) {
        std.debug.print(
            "golden mismatch [{s}] rec {d}: C {e:.17} vs zig {e:.17} (abs {e}, tol {e})\n",
            .{ what, irec, c_val, zig_val, err, tol },
        );
        return error.GoldenMismatch;
    }
}

test "golden: metric/inverse/gdet/dlgdet/Christoffels/gttpert vs C at 127 points" {
    var g = try gold.readGolden(std.testing.allocator, "metric/metric_points.kgld", 5, 101);
    defer g.deinit();

    // separate trackers: exact-form coords vs MKS2 (C-internal spread, see top)
    var t_gcov = gold.DevTracker{};
    var t_gcon = gold.DevTracker{};
    var t_gcon_mks2 = gold.DevTracker{};
    var t_gdet = gold.DevTracker{};
    var t_gdet_mks2 = gold.DevTracker{};
    var t_dlgdet = gold.DevTracker{};
    var t_dlgdet_mks2 = gold.DevTracker{};
    var t_kris = gold.DevTracker{};
    var t_kris_mks2 = gold.DevTracker{};
    var t_gttpert = gold.DevTracker{};

    for (0..g.nrec) |ir| {
        const r = g.rec(ir);
        const coords = try gold.coordsFromId(r.in[0]);
        const is_mks2 = coords == .mks2;
        const x = [4]f64{ r.in[1], r.in[2], r.in[3], r.in[4] };
        const d = metric.compute(coords, mp, x);

        var n: usize = 0;
        for (0..4) |i| {
            for (0..4) |j| {
                t_gcov.add(r.out[n], d.gcov[i][j], ir);
                n += 1;
            }
        }
        for (0..4) |i| {
            for (0..4) |j| {
                (if (is_mks2) &t_gcon_mks2 else &t_gcon).add(r.out[n], d.gcon[i][j], ir);
                n += 1;
            }
        }
        (if (is_mks2) &t_gdet_mks2 else &t_gdet).add(r.out[n], d.gdet, ir);
        n += 1;
        for (0..3) |i| {
            (if (is_mks2) &t_dlgdet_mks2 else &t_dlgdet).add(r.out[n], d.dlgdet[i], ir);
            n += 1;
        }
        for (0..4) |i| {
            for (0..4) |j| {
                for (0..4) |k| {
                    (if (is_mks2) &t_kris_mks2 else &t_kris).add(r.out[n], d.kris[i][j][k], ir);
                    n += 1;
                }
            }
        }
        t_gttpert.add(r.out[n], d.gttpert, ir);
    }

    try t_gcov.check(1e-13, "gcov");
    try t_gcon.check(1e-13, "gcon bl/ks/mink");
    try t_gcon_mks2.check(1e-8, "gcon mks2 (C-internal spread)");
    try t_gdet.check(1e-13, "gdet bl/ks/mink");
    try t_gdet_mks2.check(1e-8, "gdet mks2 (C-internal spread)");
    try t_dlgdet.check(1e-12, "dlgdet bl/ks/mink");
    try t_dlgdet_mks2.check(1e-8, "dlgdet mks2 (C-internal spread)");
    try t_kris.check(1e-12, "kris bl/ks/mink");
    try t_kris_mks2.check(1e-8, "kris mks2 (C-internal spread)");
    try t_gttpert.check(1e-13, "gttpert");
}

test "golden: coordinate transforms and Jacobians vs C" {
    var g = try gold.readGolden(std.testing.allocator, "metric/coco_dxdx.kgld", 6, 20);
    defer g.deinit();

    for (0..g.nrec) |ir| {
        const r = g.rec(ir);
        const from = try gold.coordsFromId(r.in[0]);
        const to = try gold.coordsFromId(r.in[1]);
        const x = [4]f64{ r.in[2], r.in[3], r.in[4], r.in[5] };

        const y = coco.cocoN(x, from, to, mp);
        for (0..4) |i| try expectGolden(r.out[i], y[i], 1e-12, "coco", ir);

        const jac = coco.dxdx(x, from, to, mp);
        var n: usize = 4;
        for (0..4) |i| {
            for (0..4) |j| {
                try expectGolden(r.out[n], jac[i][j], 1e-12, "dxdx", ir);
                n += 1;
            }
        }
    }
}

test "golden: gdet-corrected Christoffels on the PUFFY grid vs C" {
    var g = try gold.readGolden(std.testing.allocator, "metric/krzysie_grid.kgld", 2, 66);
    defer g.deinit();

    // The PUFFY grid exactly as the C build defines it (define.h → set_grid).
    const grid = Grid.init(.{
        .nx = 384,
        .ny = 360,
        .nz = 1,
        .ng = 3,
        .minx = @log(1.85 - 0.1),
        .maxx = @log(500.0 - 0.1),
        .miny = 0.001,
        .maxy = 1.0 - 0.001,
        .minz = -pi / 4.0,
        .maxz = pi / 4.0,
    });

    var tracker = gold.DevTracker{};
    for (0..g.nrec) |ir| {
        const r = g.rec(ir);
        const ix: i64 = @intFromFloat(r.in[0]);
        const iy: i64 = @intFromFloat(r.in[1]);

        // grid agreement first: cell centers must match C's get_x exactly
        try expectGolden(r.out[0], grid.xc(ix), 1e-14, "grid-x1", ir);
        try expectGolden(r.out[1], grid.yc(iy), 1e-14, "grid-x2", ir);

        const d = precompute.computeCorrected(.mks2, mp, &grid, ix, iy, 0);
        var n: usize = 2;
        for (0..4) |i| {
            for (0..4) |j| {
                for (0..4) |k| {
                    tracker.add(r.out[n], d.kris[i][j][k], ir);
                    n += 1;
                }
            }
        }
    }
    // MKS2 Christoffels inherit C's export spread (see header comment).
    try tracker.check(1e-8, "kris-corrected mks2");
}
