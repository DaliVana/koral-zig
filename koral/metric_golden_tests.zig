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
const build_options = @import("build_options");
const metric = @import("metric/metric.zig");
const coco = @import("metric/coco.zig");
const precompute = @import("metric/precompute.zig");
const Grid = @import("grid.zig").Grid;

const Coords = metric.Coords;
const pi = std.math.pi;

/// PUFFY build parameters of the oracle (PROBLEMS/PUFFY/define.h).
const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };

const Golden = struct {
    nrec: usize,
    nin: usize,
    nout: usize,
    data: []f64,
    a: std.mem.Allocator,

    fn deinit(self: *Golden) void {
        self.a.free(self.data);
    }

    fn rec(self: *const Golden, i: usize) struct { in: []const f64, out: []const f64 } {
        const w = self.nin + self.nout;
        const base = i * w;
        return .{
            .in = self.data[base .. base + self.nin],
            .out = self.data[base + self.nin .. base + w],
        };
    }
};

fn readGolden(a: std.mem.Allocator, comptime name: []const u8) !Golden {
    const path = build_options.golden_dir ++ "/metric/" ++ name;
    const raw = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 24)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("golden file missing: {s} (run tools/gen_golden.sh)\n", .{path});
            return error.SkipZigTest;
        },
        else => return err,
    };
    defer a.free(raw);

    if (raw.len < 24 or !std.mem.eql(u8, raw[0..4], "KGLD")) return error.BadGoldenFile;
    const version = std.mem.readInt(u32, raw[4..8], .little);
    if (version != 1) return error.BadGoldenVersion;
    const nrec: usize = @intCast(std.mem.readInt(u64, raw[8..16], .little));
    const nin = std.mem.readInt(u32, raw[16..20], .little);
    const nout = std.mem.readInt(u32, raw[20..24], .little);
    const nvals = nrec * (nin + nout);
    if (raw.len != 24 + nvals * 8) return error.BadGoldenFile;

    const data = try a.alloc(f64, nvals);
    for (0..nvals) |i| {
        data[i] = @bitCast(std.mem.readInt(u64, raw[24 + i * 8 ..][0..8], .little));
    }
    return .{ .nrec = nrec, .nin = nin, .nout = @intCast(nout), .data = data, .a = a };
}

fn coordsFromId(id: f64) Coords {
    return switch (@as(u32, @intFromFloat(id))) {
        1 => .bl, // KERRCOORDS/BLCOORDS/SCHWCOORDS
        2 => .ks,
        4 => .mink,
        10 => .mks2,
        else => unreachable,
    };
}

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

/// Accumulates the worst normalized deviation per quantity class, then
/// asserts a class-level bound — so a failure reports the observed maximum
/// instead of dying on the first element.
const DevTracker = struct {
    max_dev: f64 = 0,
    at_rec: usize = 0,
    c_val: f64 = 0,
    zig_val: f64 = 0,

    fn add(self: *DevTracker, c_val: f64, zig_val: f64, irec: usize) void {
        const dev = @abs(c_val - zig_val) / @max(1.0, @max(@abs(c_val), @abs(zig_val)));
        if (dev > self.max_dev) {
            self.* = .{ .max_dev = dev, .at_rec = irec, .c_val = c_val, .zig_val = zig_val };
        }
    }

    fn check(self: *const DevTracker, bound: f64, what: []const u8) !void {
        std.debug.print("golden [{s}]: max dev {e:.3} (bound {e:.1})\n", .{ what, self.max_dev, bound });
        if (self.max_dev > bound) {
            std.debug.print(
                "golden BOUND EXCEEDED [{s}] rec {d}: C {e:.17} vs zig {e:.17}\n",
                .{ what, self.at_rec, self.c_val, self.zig_val },
            );
            return error.GoldenMismatch;
        }
    }
};

test "golden: metric/inverse/gdet/dlgdet/Christoffels/gttpert vs C at 127 points" {
    var g = try readGolden(std.testing.allocator, "metric_points.kgld");
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 5), g.nin);
    try std.testing.expectEqual(@as(usize, 101), g.nout);

    // separate trackers: exact-form coords vs MKS2 (C-internal spread, see top)
    var t_gcov = DevTracker{};
    var t_gcon = DevTracker{};
    var t_gcon_mks2 = DevTracker{};
    var t_gdet = DevTracker{};
    var t_gdet_mks2 = DevTracker{};
    var t_dlgdet = DevTracker{};
    var t_dlgdet_mks2 = DevTracker{};
    var t_kris = DevTracker{};
    var t_kris_mks2 = DevTracker{};
    var t_gttpert = DevTracker{};

    for (0..g.nrec) |ir| {
        const r = g.rec(ir);
        const coords = coordsFromId(r.in[0]);
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
    var g = try readGolden(std.testing.allocator, "coco_dxdx.kgld");
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 6), g.nin);
    try std.testing.expectEqual(@as(usize, 20), g.nout);

    for (0..g.nrec) |ir| {
        const r = g.rec(ir);
        const from = coordsFromId(r.in[0]);
        const to = coordsFromId(r.in[1]);
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
    var g = try readGolden(std.testing.allocator, "krzysie_grid.kgld");
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 2), g.nin);
    try std.testing.expectEqual(@as(usize, 66), g.nout);

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

    var tracker = DevTracker{};
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
