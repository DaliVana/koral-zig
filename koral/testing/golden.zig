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

/// A forced-dt step file (oracle/harness_step.c):
/// "KSTP" u32 version, u32 nx,ny,nz,nv, u64 nrec, then per record
/// {t, dt, u[ncell·nv], p[ncell·nv], flags[nf·ncell]} — iv fastest, then
/// ix, iy, iz; record 0 is the post-init state. nf = 2 (ENTROPYFLAG,
/// HDFIXUPFLAG) or 4 (+ RADFIXUPFLAG, RADIMPFIXUPFLAG for RADIATION
/// builds), inferred from the file length.
pub const Kstp = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    nv: usize,
    nrec: usize,
    nflags: usize,
    data: []f64,
    a: std.mem.Allocator,

    pub fn deinit(self: *Kstp) void {
        self.a.free(self.data);
    }

    fn recLen(self: *const Kstp) usize {
        const ncell = self.nx * self.ny * self.nz;
        return 2 + 2 * ncell * self.nv + self.nflags * ncell;
    }

    pub const Record = struct {
        t: f64,
        dt: f64,
        u: []const f64,
        p: []const f64,
        flags: []const f64,
    };

    pub fn rec(self: *const Kstp, i: usize) Record {
        const ncell = self.nx * self.ny * self.nz;
        const base = i * self.recLen();
        const nu = ncell * self.nv;
        return .{
            .t = self.data[base],
            .dt = self.data[base + 1],
            .u = self.data[base + 2 .. base + 2 + nu],
            .p = self.data[base + 2 + nu .. base + 2 + 2 * nu],
            .flags = self.data[base + 2 + 2 * nu .. base + self.recLen()],
        };
    }

    /// flat index of (iv, ix, iy, iz) within a u/p slice
    pub fn idx(self: *const Kstp, iv: usize, ix: usize, iy: usize, iz: usize) usize {
        return ((iz * self.ny + iy) * self.nx + ix) * self.nv + iv;
    }
};

pub fn readKstp(a: std.mem.Allocator, comptime relpath: []const u8) !Kstp {
    const path = build_options.golden_dir ++ "/" ++ relpath;
    const raw = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 26)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("golden file missing: {s} (run tools/gen_golden.sh)\n", .{path});
            return error.SkipZigTest;
        },
        else => return err,
    };
    defer a.free(raw);

    if (raw.len < 32 or !std.mem.eql(u8, raw[0..4], "KSTP")) return error.BadGoldenFile;
    if (std.mem.readInt(u32, raw[4..8], .little) != 1) return error.BadGoldenVersion;
    var k = Kstp{
        .nx = std.mem.readInt(u32, raw[8..12], .little),
        .ny = std.mem.readInt(u32, raw[12..16], .little),
        .nz = std.mem.readInt(u32, raw[16..20], .little),
        .nv = std.mem.readInt(u32, raw[20..24], .little),
        .nrec = @intCast(std.mem.readInt(u64, raw[24..32], .little)),
        .nflags = 0,
        .data = undefined,
        .a = a,
    };
    // infer the flag count from the file length
    {
        const ncell = k.nx * k.ny * k.nz;
        const per_rec = (raw.len - 32) / 8 / k.nrec;
        if (per_rec < 2 + 2 * ncell * k.nv) return error.BadGoldenFile;
        k.nflags = (per_rec - 2 - 2 * ncell * k.nv) / ncell;
        if (k.nflags != 2 and k.nflags != 4) return error.BadGoldenFile;
    }
    const nvals = k.nrec * k.recLen();
    if (raw.len != 32 + nvals * 8) return error.BadGoldenFile;
    k.data = try a.alloc(f64, nvals);
    for (0..nvals) |i| {
        k.data[i] = @bitCast(std.mem.readInt(u64, raw[32 + i * 8 ..][0..8], .little));
    }
    return k;
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

// ---------------------------------------------------------------------------
// KINI reader — the PUFFY t=0 keystone snapshots (harness_init.c), gzipped.

/// A grid snapshot of selected primitive slots. Axes run [-gx..nx+gx) when
/// gx>0 (ghosts included), else [0..nx) by `stride`; same for y. iz is
/// always [0..nz). vars[] lists which primitive indices each record holds.
pub const Kini = struct {
    nx: i32,
    ny: i32,
    nz: i32,
    gx: i32,
    gy: i32,
    gz: i32,
    stride: i32,
    vars: []i32,
    /// values, laid out iz-slowest, then iy, then ix, vars fastest
    data: []f64,
    a: std.mem.Allocator,

    pub fn deinit(self: *Kini) void {
        self.a.free(self.vars);
        self.a.free(self.data);
    }

    /// number of sampled points along x / y (ghosts or strided domain)
    pub fn nxi(self: *const Kini) usize {
        return if (self.gx > 0) @intCast(self.nx + 2 * self.gx) else @intCast(@divTrunc(self.nx + self.stride - 1, self.stride));
    }
    pub fn nyi(self: *const Kini) usize {
        return if (self.gy > 0) @intCast(self.ny + 2 * self.gy) else @intCast(@divTrunc(self.ny + self.stride - 1, self.stride));
    }

    /// the C cell coordinate of sample column jx / row jy
    pub fn cellX(self: *const Kini, jx: usize) i64 {
        return if (self.gx > 0) @as(i64, @intCast(jx)) - self.gx else @as(i64, @intCast(jx)) * self.stride;
    }
    pub fn cellY(self: *const Kini, jy: usize) i64 {
        return if (self.gy > 0) @as(i64, @intCast(jy)) - self.gy else @as(i64, @intCast(jy)) * self.stride;
    }

    /// record base offset for sample (jx, jy, jz)
    pub fn base(self: *const Kini, jx: usize, jy: usize, jz: usize) usize {
        const nvv = self.vars.len;
        return ((jz * self.nyi() + jy) * self.nxi() + jx) * nvv;
    }
};

pub fn readKini(a: std.mem.Allocator, comptime relpath: []const u8) !Kini {
    const path = build_options.golden_dir ++ "/" ++ relpath;
    const gz = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 26)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("golden file missing: {s} (run tools/gen_golden.sh)\n", .{path});
            return error.SkipZigTest;
        },
        else => return err,
    };
    defer a.free(gz);

    // inflate (gzip container)
    const flate = std.compress.flate;
    var in: std.Io.Reader = .fixed(gz);
    var window: [flate.max_window_len]u8 = undefined;
    var dc: flate.Decompress = .init(&in, .gzip, &window);
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    _ = try dc.reader.streamRemaining(&aw.writer);
    const raw = aw.written();

    if (raw.len < 40 or !std.mem.eql(u8, raw[0..4], "KINI")) return error.BadGoldenFile;
    if (std.mem.readInt(u32, raw[4..8], .little) != 1) return error.BadGoldenVersion;
    const rd = struct {
        fn i32at(buf: []const u8, off: usize) i32 {
            return std.mem.readInt(i32, buf[off..][0..4], .little);
        }
    };
    var k = Kini{
        .nx = rd.i32at(raw, 8),
        .ny = rd.i32at(raw, 12),
        .nz = rd.i32at(raw, 16),
        .gx = rd.i32at(raw, 20),
        .gy = rd.i32at(raw, 24),
        .gz = rd.i32at(raw, 28),
        .stride = rd.i32at(raw, 32),
        .vars = undefined,
        .data = undefined,
        .a = a,
    };
    const nvars: usize = @intCast(rd.i32at(raw, 36));
    var off: usize = 40;
    k.vars = try a.alloc(i32, nvars);
    for (0..nvars) |i| {
        k.vars[i] = rd.i32at(raw, off);
        off += 4;
    }
    const npts = k.nxi() * k.nyi() * @as(usize, @intCast(k.nz));
    const nvals = npts * nvars;
    if (raw.len != off + nvals * 8) return error.BadGoldenFile;
    k.data = try a.alloc(f64, nvals);
    for (0..nvals) |i| {
        k.data[i] = @bitCast(std.mem.readInt(u64, raw[off + i * 8 ..][0..8], .little));
    }
    return k;
}
