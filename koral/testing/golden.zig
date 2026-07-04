//! Shared helpers for C-oracle golden tests: KGLD reader, per-class
//! deviation tracker, and record-embedded geometry reconstruction.

const std = @import("std");
const build_options = @import("build_options");
const config = @import("../config.zig");
const Geometry = @import("../geometry.zig").Geometry;

pub const Golden = struct {
    nrec: usize,
    nin: usize,
    nout: usize,
    data: []f64,
    a: std.mem.Allocator,

    pub fn deinit(self: *Golden) void {
        self.a.free(self.data);
    }

    pub fn rec(self: *const Golden, i: usize) struct { in: []const f64, out: []const f64 } {
        const w = self.nin + self.nout;
        const base = i * w;
        return .{
            .in = self.data[base .. base + self.nin],
            .out = self.data[base + self.nin .. base + w],
        };
    }
};

/// Read tests/golden/<relpath>; skips the test when the file is absent
/// (goldens not generated on this machine).
pub fn readGolden(a: std.mem.Allocator, comptime relpath: []const u8, nin: usize, nout: usize) !Golden {
    const path = build_options.golden_dir ++ "/" ++ relpath;
    const raw = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 24)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("golden file missing: {s} (run tools/gen_golden.sh)\n", .{path});
            return error.SkipZigTest;
        },
        else => return err,
    };
    defer a.free(raw);

    if (raw.len < 24 or !std.mem.eql(u8, raw[0..4], "KGLD")) return error.BadGoldenFile;
    if (std.mem.readInt(u32, raw[4..8], .little) != 1) return error.BadGoldenVersion;
    const nrec: usize = @intCast(std.mem.readInt(u64, raw[8..16], .little));
    try std.testing.expectEqual(nin, std.mem.readInt(u32, raw[16..20], .little));
    try std.testing.expectEqual(nout, std.mem.readInt(u32, raw[20..24], .little));
    const nvals = nrec * (nin + nout);
    if (raw.len != 24 + nvals * 8) return error.BadGoldenFile;

    const data = try a.alloc(f64, nvals);
    for (0..nvals) |i| {
        data[i] = @bitCast(std.mem.readInt(u64, raw[24 + i * 8 ..][0..8], .little));
    }
    return .{ .nrec = nrec, .nin = nin, .nout = nout, .data = data, .a = a };
}

/// Accumulates the worst normalized deviation per quantity class, then
/// asserts a class-level bound — a failure reports the observed maximum
/// instead of dying on the first element.
pub const DevTracker = struct {
    max_dev: f64 = 0,
    at_rec: usize = 0,
    c_val: f64 = 0,
    zig_val: f64 = 0,

    pub fn add(self: *DevTracker, c_val: f64, zig_val: f64, irec: usize) void {
        const dev = @abs(c_val - zig_val) / @max(1.0, @max(@abs(c_val), @abs(zig_val)));
        if (dev > self.max_dev) {
            self.* = .{ .max_dev = dev, .at_rec = irec, .c_val = c_val, .zig_val = zig_val };
        }
    }

    pub fn check(self: *const DevTracker, bound: f64, what: []const u8) !void {
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

/// Unpack a 10-value upper triangle into a symmetric 4×5 block (col 4 = 0).
pub fn sym10(vals: []const f64) [4][5]f64 {
    var m: [4][5]f64 = @splat(@splat(0));
    var n: usize = 0;
    for (0..4) |i| {
        for (i..4) |j| {
            m[i][j] = vals[n];
            m[j][i] = vals[n];
            n += 1;
        }
    }
    return m;
}

/// Geometry straight from a record: [gg10, GG10] (len 20) or
/// [gg10, GG10, gdet, alpha, gttpert] (len 23).
pub fn geomFromRecord(vals: []const f64, coords: config.Coords, x: [4]f64) Geometry {
    var geo = Geometry{
        .coords = coords,
        .ix = 0,
        .iy = 0,
        .iz = 0,
        .ifacedim = -1,
        .xxvec = x,
        .gg = sym10(vals[0..10]),
        .GG = sym10(vals[10..20]),
        .gdet = if (vals.len >= 23) vals[20] else 1.0,
        .alpha = undefined,
        .gttpert = if (vals.len >= 23) vals[22] else 0.0,
    };
    geo.alpha = if (vals.len >= 23) vals[21] else @sqrt(-1.0 / geo.GG[0][0]);
    return geo;
}

pub fn coordsFromId(id: f64) config.Coords {
    return switch (@as(u32, @intFromFloat(id))) {
        1 => .bl,
        2 => .ks,
        4 => .mink,
        10 => .mks2,
        else => unreachable,
    };
}
