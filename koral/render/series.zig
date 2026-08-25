//! KDMP time series for slow-light GRRT (see sweep.zig for the integrator).
//!
//! Fast light samples ONE snapshot everywhere; every pixel shows the flow
//! at a single instant, as if photons propagated infinitely fast. Slow light
//! samples the fluid at each photon's own coordinate time x⁰, which the
//! geodesic integrator already carries exactly (render.zig integrates the
//! full 4-vector; KS-based time is horizon-regular, so x⁰ is well behaved
//! all the way down to r_capture). This file supplies the time axis:
//!
//!  * `Series`; a directory of prims*.kdmp scanned by HEADER ONLY (44-byte
//!    v2 or 32-byte v1 reads, no bodies), sorted by the header time t and
//!    optionally strided. Dump cadence is NOT assumed uniform; the sgra_spin
//!    series changes cadence mid-run, so every consumer walks the actual
//!    sorted times.
//!  * `WindowSampler`; the time-interpolating sampler: trilinear-in-space
//!    (render.sampleData, both frames share the grid) and linear-in-time
//!    between a bracketing pair of frames. Outside [t_lo, t_hi] it clamps
//!    (holds the edge frame) and counts the clamp; the sweep only exposes
//!    out-of-range times at the series edges, so the counters measure how
//!    much of the render leaned on the ends of the series.
//!    The lerp is written `a + w·(b − a)`, so two frames with IDENTICAL
//!    bodies reproduce the fast-light sample BIT-EXACTLY for any w; the
//!    static-series identity gate in render_tests.zig pins this.
//!  * frame sources; `FileSource` (refcounted load/free of series frames;
//!    the sweep's sliding window keeps at most two frames + the reference
//!    frame resident) and `SliceSource` (frames already in memory; tests).

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const dump = @import("../io/dump.zig");
const render = @import("render.zig");

// ---- time-interpolating sampler --------------------------------------------

/// Sampler over a bracketing frame pair [t_lo, t_hi]: spatial trilinear from
/// both frames, then linear in the ray's own coordinate time x[0]. Mutable
/// (clamp counters); pass a pointer to render.advanceRay.
pub const WindowSampler = struct {
    lo: *const render.DumpData,
    hi: *const render.DumpData,
    t_lo: f64,
    t_hi: f64,
    /// samples clamped to the window edges (series-end hold). See file doc
    n_below: u64 = 0,
    n_above: u64 = 0,

    pub inline fn sample(self: *WindowSampler, comptime c: config.Config, s: *const render.Scene, x: [4]f64, pp: *[layout.VarLayout(c).count]f64) bool {
        const L = layout.VarLayout(c);
        // degenerate window (single frame): plain snapshot sampling
        if (self.lo == self.hi or !(self.t_hi > self.t_lo))
            return render.sampleData(c, &s.grid, self.lo, x, pp);
        var plo: [L.count]f64 = undefined;
        if (!render.sampleData(c, &s.grid, self.lo, x, &plo)) return false;
        var phi: [L.count]f64 = undefined;
        if (!render.sampleData(c, &s.grid, self.hi, x, &phi)) return false;
        var w = (x[0] - self.t_lo) / (self.t_hi - self.t_lo);
        if (!(w >= 0)) {
            w = 0;
            self.n_below += 1;
        } else if (w > 1) {
            w = 1;
            self.n_above += 1;
        }
        // one-multiply lerp: exact when both frames agree (identity gate)
        for (0..L.count) |iv| pp[iv] = plo[iv] + w * (phi[iv] - plo[iv]);
        return true;
    }
};

// ---- series scan -----------------------------------------------------------

pub const ScanError = error{ NoFrames, DimMismatch, Truncated, BadMagic, BadVersion };

/// A directory of KDMP snapshots: names + header times, ascending in t.
/// Bodies are NOT loaded here. See FileSource.
pub const Series = struct {
    dir: []u8,
    /// frame file names (inside `dir`), parallel to `ts`
    names: [][]u8,
    /// header times, strictly ascending after dedup
    ts: []f64,
    /// shape shared by every frame (t field = first frame's t)
    shape: dump.DumpHeader,

    /// Scan `dir_path` for *.kdmp, order by header t (ties: the lexically
    /// later name wins; a rewritten checkpoint replaces the original), then
    /// keep every `stride`-th frame COUNTING FROM THE LAST; the newest
    /// frame is always in the strided series, since the default t_obs
    /// anchors to it. Real dump directories accumulate foreign frames (a
    /// restart at different resolution, a resolution test): the shape of
    /// the NEWEST frame defines the series, and other shapes are skipped
    /// with a note rather than erroring.
    pub fn scan(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8, stride: usize) !Series {
        var d = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer d.close(io);

        const Entry = struct { name: []u8, h: dump.DumpHeader };
        var entries: std.ArrayList(Entry) = .empty;
        defer {
            for (entries.items) |e| allocator.free(e.name);
            entries.deinit(allocator);
        }

        var it = d.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".kdmp")) continue;
            var f = try d.openFile(io, entry.name, .{});
            defer f.close(io);
            var hdr: [dump.header_size]u8 = undefined;
            const got = try f.readPositionalAll(io, hdr[0..], 0);
            const lh = try render.parseHeaderLoose(hdr[0..got]);
            try entries.append(allocator, .{ .name = try allocator.dupe(u8, entry.name), .h = lh.h });
        }
        if (entries.items.len == 0) return error.NoFrames;

        std.mem.sort(Entry, entries.items, {}, struct {
            fn lt(_: void, a: Entry, b: Entry) bool {
                if (a.h.t != b.h.t) return a.h.t < b.h.t;
                return std.mem.order(u8, a.name, b.name) == .lt;
            }
        }.lt);

        // the newest frame's shape defines the series; skip foreign shapes,
        // dedup equal times (keep the later name), then stride from the end
        const shape = entries.items[entries.items.len - 1].h;
        var uniq: std.ArrayList(Entry) = .empty;
        defer uniq.deinit(allocator);
        var skipped: usize = 0;
        for (entries.items) |e| {
            if (e.h.nx != shape.nx or e.h.ny != shape.ny or e.h.nz != shape.nz or e.h.nv != shape.nv) {
                skipped += 1;
                continue;
            }
            if (uniq.items.len > 0 and uniq.items[uniq.items.len - 1].h.t == e.h.t) {
                uniq.items[uniq.items.len - 1] = e; // later name replaces
            } else {
                try uniq.append(allocator, e);
            }
        }
        if (skipped > 0) {
            std.debug.print("series: skipped {d} frame(s) not matching the newest frame's {d}x{d}x{d} shape\n", .{ skipped, shape.nx, shape.ny, shape.nz });
        }

        const st = @max(stride, 1);
        const nkeep = (uniq.items.len - 1) / st + 1;
        const names = try allocator.alloc([]u8, nkeep);
        errdefer allocator.free(names);
        const ts = try allocator.alloc(f64, nkeep);
        errdefer allocator.free(ts);
        for (0..nkeep) |i| {
            const src = uniq.items[uniq.items.len - 1 - (nkeep - 1 - i) * st];
            names[i] = try allocator.dupe(u8, src.name);
            ts[i] = src.h.t;
        }

        return .{
            .dir = try allocator.dupe(u8, dir_path),
            .names = names,
            .ts = ts,
            .shape = shape,
        };
    }

    pub fn deinit(self: *Series, allocator: std.mem.Allocator) void {
        for (self.names) |n| allocator.free(n);
        allocator.free(self.names);
        allocator.free(self.ts);
        allocator.free(self.dir);
        self.* = undefined;
    }

    /// Index of the frame nearest in time to t.
    pub fn nearest(self: *const Series, t: f64) usize {
        var best: usize = 0;
        for (self.ts, 0..) |tv, i| {
            if (@abs(tv - t) < @abs(self.ts[best] - t)) best = i;
        }
        return best;
    }

    /// Load frame `idx` in full (header + body).
    pub fn loadFrame(self: *const Series, allocator: std.mem.Allocator, io: std.Io, idx: usize) !render.DumpData {
        var pbuf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ self.dir, self.names[idx] });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 33));
        defer allocator.free(bytes);
        return render.DumpData.fromBytes(allocator, bytes);
    }
};

// ---- frame sources ---------------------------------------------------------

/// Disk-backed frame source with refcounted residency: the sweep's window
/// slide double-acquires the shared frame of consecutive pairs, and the CLI
/// holds the reference frame across the whole render; refcounts make both
/// safe while keeping at most window+reference frames in memory.
pub const FileSource = struct {
    ser: *const Series,
    allocator: std.mem.Allocator,
    io: std.Io,
    slots: []Slot,
    /// disk loads actually performed (cache misses)
    loads: usize = 0,

    const Slot = struct { data: ?render.DumpData = null, refs: usize = 0 };

    pub fn init(allocator: std.mem.Allocator, io: std.Io, ser: *const Series) !FileSource {
        const slots = try allocator.alloc(Slot, ser.ts.len);
        @memset(slots, .{});
        return .{ .ser = ser, .allocator = allocator, .io = io, .slots = slots };
    }

    pub fn deinit(self: *FileSource) void {
        for (self.slots) |*sl| {
            if (sl.data) |*d| d.deinit(self.allocator);
        }
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    pub fn times(self: *const FileSource) []const f64 {
        return self.ser.ts;
    }

    pub fn acquire(self: *FileSource, idx: usize) !*const render.DumpData {
        const sl = &self.slots[idx];
        if (sl.data == null) {
            sl.data = try self.ser.loadFrame(self.allocator, self.io, idx);
            self.loads += 1;
        }
        sl.refs += 1;
        return &sl.data.?;
    }

    pub fn release(self: *FileSource, idx: usize) void {
        const sl = &self.slots[idx];
        std.debug.assert(sl.refs > 0);
        sl.refs -= 1;
        if (sl.refs == 0) {
            sl.data.?.deinit(self.allocator);
            sl.data = null;
        }
    }
};

/// Frames already in memory (tests, or a fully preloaded short series).
pub const SliceSource = struct {
    ts: []const f64,
    frames: []const *const render.DumpData,

    pub fn times(self: *const SliceSource) []const f64 {
        return self.ts;
    }

    pub fn acquire(self: *const SliceSource, idx: usize) !*const render.DumpData {
        return self.frames[idx];
    }

    pub fn release(self: *const SliceSource, idx: usize) void {
        _ = self;
        _ = idx;
    }
};
