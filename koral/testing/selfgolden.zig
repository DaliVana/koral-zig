//! Self-goldens: a Zig-generated end-to-end regression baseline.
//!
//! **Provenance — read this before treating a mismatch as a bug.** These are
//! NOT the C-oracle goldens under `tests/golden/`. Those record what *KORAL C*
//! computes and are the authority on whether the transcription is faithful.
//! These record what *this repository* computed at the moment they were
//! generated, and are the authority on nothing at all except "the numbers moved
//! since then". A C-golden failure means the physics is wrong; a self-golden
//! failure means something changed, and it is the author's job to say whether
//! the change was intended.
//!
//! They exist because the C goldens are committed but not *regenerable* once
//! the Zig side stops matching koral_lite bit-for-bit: `tools/gen_golden.sh`
//! needs a sibling koral_lite checkout that, after a deliberate divergence, no
//! longer describes this code. Self-goldens are the replacement end-to-end net
//! for everything downstream of that point — they cover the assembled pipeline
//! (init → CFL → RK2IMEX → BCs → fixups), which the analytic theory tests do
//! not: those check identities and known solutions, not the composition.
//!
//! Format ("KSLF" v1, gzipped on disk). All values f64 little-endian, laid out
//! exactly as `Field.data` / `Sim.flags` are in memory — domain AND ghosts, so
//! a boundary-condition change is caught in the t=0 snapshot rather than after
//! it has propagated inward.
//!
//!   "KSLF" u32:version u32:nx u32:ny u32:nz u32:ng u32:nv
//!   u64:ncell(padded) u32:nflags u32:n_scalar_recs u32:n_field_recs
//!   u32:label_len label[label_len]   (zero-padded to an 8-byte boundary)
//!   scalars:     n_scalar_recs × Scalars.count
//!   field_steps: n_field_recs           (which step each snapshot came from)
//!   fields:      n_field_recs × (u[ncell·nv], p[ncell·nv], flags[ncell·nflags])
//!
//! Two granularities on purpose. A full field snapshot costs ~350 KiB gzipped,
//! so they are kept only at the endpoints — post-init and post-run — which is
//! enough to catch any change, since a perturbation at step k is still present
//! at step n. The per-step *scalars* (t, dt, the CFL denominators, the implicit
//! counters) are seven f64 each, so every step keeps them: they cost nothing
//! and are what tells you *when* a run started to diverge. Flags are widened to
//! f64 so the payload stays one homogeneous array.
//!
//! The dt sequence is the sim's own CFL choice, not a forced one — unlike the C
//! step goldens, which force C's dt to isolate the step arithmetic. Here there
//! is no other side to agree with, so the timestep machinery is pinned too.

const std = @import("std");
const build_options = @import("build_options");

pub const magic = "KSLF";
pub const version: u32 = 1;

/// Per-record scalars, in file order. Widening the two counters to f64 is
/// exact — both are far below 2^53.
pub const Scalars = struct {
    t: f64,
    dt: f64,
    own_dt: f64,
    tstepdenmax: f64,
    tstepdenmin: f64,
    n_radimp_failures: f64,
    n_radimp_iters: f64,

    pub const count = @typeInfo(Scalars).@"struct".fields.len;

    pub fn toArray(self: Scalars) [count]f64 {
        var out: [count]f64 = undefined;
        inline for (@typeInfo(Scalars).@"struct".fields, 0..) |f, i| {
            out[i] = @field(self, f.name);
        }
        return out;
    }

    pub fn fromSlice(vals: []const f64) Scalars {
        var out: Scalars = undefined;
        inline for (@typeInfo(Scalars).@"struct".fields, 0..) |f, i| {
            @field(out, f.name) = vals[i];
        }
        return out;
    }

    pub fn name(i: usize) []const u8 {
        inline for (@typeInfo(Scalars).@"struct".fields, 0..) |f, j| {
            if (i == j) return f.name;
        }
        unreachable;
    }
};

pub const Header = struct {
    nx: u32,
    ny: u32,
    nz: u32,
    ng: u32,
    nv: u32,
    ncell: u64,
    nflags: u32,
    n_scalar_recs: u32 = 0,
    n_field_recs: u32 = 0,
};

pub const File = struct {
    hdr: Header,
    label: []u8,
    /// n_scalar_recs × Scalars.count
    scalars: []f64,
    /// n_field_recs — the step index each snapshot came from
    field_steps: []f64,
    /// n_field_recs × fieldLen
    fields: []f64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *File) void {
        self.allocator.free(self.label);
        self.allocator.free(self.scalars);
        self.allocator.free(self.field_steps);
        self.allocator.free(self.fields);
    }

    pub fn fieldLen(hdr: Header) usize {
        const ncell: usize = @intCast(hdr.ncell);
        return 2 * ncell * hdr.nv + ncell * hdr.nflags;
    }

    pub fn scalarsAt(self: *const File, istep: usize) Scalars {
        return Scalars.fromSlice(self.scalars[istep * Scalars.count ..][0..Scalars.count]);
    }

    pub const Snapshot = struct {
        step: usize,
        u: []const f64,
        p: []const f64,
        flags: []const f64,
    };

    pub fn snapshot(self: *const File, i: usize) Snapshot {
        const ncell: usize = @intCast(self.hdr.ncell);
        const nu = ncell * self.hdr.nv;
        const base = i * fieldLen(self.hdr);
        return .{
            .step = @intFromFloat(self.field_steps[i]),
            .u = self.fields[base .. base + nu],
            .p = self.fields[base + nu .. base + 2 * nu],
            .flags = self.fields[base + 2 * nu .. base + fieldLen(self.hdr)],
        };
    }
};

// ---- writing --------------------------------------------------------------

/// Accumulates records in memory, then writes the gzipped file.
pub const Writer = struct {
    hdr: Header,
    label: []const u8,
    scalars: std.ArrayList(f64),
    field_steps: std.ArrayList(f64),
    fields: std.ArrayList(f64),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, hdr: Header, label: []const u8) Writer {
        return .{
            .hdr = hdr,
            .label = label,
            .scalars = .empty,
            .field_steps = .empty,
            .fields = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Writer) void {
        self.scalars.deinit(self.allocator);
        self.field_steps.deinit(self.allocator);
        self.fields.deinit(self.allocator);
    }

    /// Every step records its scalars.
    pub fn addScalars(self: *Writer, s: Scalars) !void {
        try self.scalars.appendSlice(self.allocator, &s.toArray());
        self.hdr.n_scalar_recs += 1;
    }

    /// Only the endpoints record full fields. `flags` is widened to f64 here.
    pub fn addSnapshot(self: *Writer, step: usize, u: []const f64, p: []const f64, flags: []const i32) !void {
        const ncell: usize = @intCast(self.hdr.ncell);
        std.debug.assert(u.len == ncell * self.hdr.nv);
        std.debug.assert(p.len == ncell * self.hdr.nv);
        std.debug.assert(flags.len == ncell * self.hdr.nflags);
        try self.field_steps.append(self.allocator, @floatFromInt(step));
        try self.fields.appendSlice(self.allocator, u);
        try self.fields.appendSlice(self.allocator, p);
        for (flags) |f| try self.fields.append(self.allocator, @floatFromInt(f));
        self.hdr.n_field_recs += 1;
    }

    /// Serialize to a gzipped byte buffer the caller owns.
    pub fn toGzip(self: *Writer, allocator: std.mem.Allocator) ![]u8 {
        var raw: std.ArrayList(u8) = .empty;
        defer raw.deinit(allocator);

        try raw.appendSlice(allocator, magic);
        try appendU32(allocator, &raw, version);
        try appendU32(allocator, &raw, self.hdr.nx);
        try appendU32(allocator, &raw, self.hdr.ny);
        try appendU32(allocator, &raw, self.hdr.nz);
        try appendU32(allocator, &raw, self.hdr.ng);
        try appendU32(allocator, &raw, self.hdr.nv);
        try appendU64(allocator, &raw, self.hdr.ncell);
        try appendU32(allocator, &raw, self.hdr.nflags);
        try appendU32(allocator, &raw, self.hdr.n_scalar_recs);
        try appendU32(allocator, &raw, self.hdr.n_field_recs);
        try appendU32(allocator, &raw, @intCast(self.label.len));
        try raw.appendSlice(allocator, self.label);
        while (raw.items.len % 8 != 0) try raw.append(allocator, 0);
        for (self.scalars.items) |v| try appendU64(allocator, &raw, @bitCast(v));
        for (self.field_steps.items) |v| try appendU64(allocator, &raw, @bitCast(v));
        for (self.fields.items) |v| try appendU64(allocator, &raw, @bitCast(v));

        const flate = std.compress.flate;
        var in: std.Io.Reader = .fixed(raw.items);
        var aw: std.Io.Writer.Allocating = try .initCapacity(allocator, 1 << 16);
        errdefer aw.deinit();
        const cbuf = try allocator.alloc(u8, flate.max_window_len);
        defer allocator.free(cbuf);
        // level_9: these are committed artifacts written rarely and read often
        var comp: flate.Compress = try .init(&aw.writer, cbuf, .gzip, .best);
        _ = try in.streamRemaining(&comp.writer);
        try comp.finish();
        return aw.toOwnedSlice();
    }
};

fn appendU32(a: std.mem.Allocator, list: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try list.appendSlice(a, &b);
}

fn appendU64(a: std.mem.Allocator, list: *std.ArrayList(u8), v: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    try list.appendSlice(a, &b);
}

// ---- reading --------------------------------------------------------------

pub const path_prefix = build_options.selfgolden_dir ++ "/";

/// Read `tests/selfgolden/<relpath>`. Unlike the C goldens a missing file is a
/// hard error, not a skip: self-goldens are always regenerable from this
/// repository alone, so absence means someone forgot to commit them.
pub fn read(allocator: std.mem.Allocator, comptime relpath: []const u8) !File {
    const path = path_prefix ++ relpath;
    const gz = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(1 << 28)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print(
                "self-golden missing: {s}\n  regenerate with: zig build update-self-goldens\n",
                .{path},
            );
            return error.SelfGoldenMissing;
        },
        else => return err,
    };
    defer allocator.free(gz);

    const flate = std.compress.flate;
    var in: std.Io.Reader = .fixed(gz);
    var window: [flate.max_window_len]u8 = undefined;
    var dc: flate.Decompress = .init(&in, .gzip, &window);
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    _ = try dc.reader.streamRemaining(&aw.writer);
    const raw = aw.written();

    if (raw.len < 40 or !std.mem.eql(u8, raw[0..4], magic)) return error.BadSelfGolden;
    if (std.mem.readInt(u32, raw[4..8], .little) != version) return error.BadSelfGoldenVersion;
    const rd = struct {
        fn u32at(b: []const u8, off: usize) u32 {
            return std.mem.readInt(u32, b[off..][0..4], .little);
        }
    };
    const hdr = Header{
        .nx = rd.u32at(raw, 8),
        .ny = rd.u32at(raw, 12),
        .nz = rd.u32at(raw, 16),
        .ng = rd.u32at(raw, 20),
        .nv = rd.u32at(raw, 24),
        .ncell = std.mem.readInt(u64, raw[28..36], .little),
        .nflags = rd.u32at(raw, 36),
        .n_scalar_recs = rd.u32at(raw, 40),
        .n_field_recs = rd.u32at(raw, 44),
    };
    const label_len = rd.u32at(raw, 48);
    var off: usize = 52;
    if (raw.len < off + label_len) return error.BadSelfGolden;
    const label = try allocator.alloc(u8, label_len);
    errdefer allocator.free(label);
    @memcpy(label, raw[off .. off + label_len]);
    off += label_len;
    off += (8 - off % 8) % 8;

    const n_sc = @as(usize, hdr.n_scalar_recs) * Scalars.count;
    const n_fs = @as(usize, hdr.n_field_recs);
    const n_fl = n_fs * File.fieldLen(hdr);
    if (raw.len != off + (n_sc + n_fs + n_fl) * 8) return error.BadSelfGolden;

    var f = File{
        .hdr = hdr,
        .label = label,
        .scalars = try allocator.alloc(f64, n_sc),
        .field_steps = undefined,
        .fields = undefined,
        .allocator = allocator,
    };
    errdefer allocator.free(f.scalars);
    f.field_steps = try allocator.alloc(f64, n_fs);
    errdefer allocator.free(f.field_steps);
    f.fields = try allocator.alloc(f64, n_fl);

    var i: usize = 0;
    for (f.scalars) |*v| {
        v.* = @bitCast(std.mem.readInt(u64, raw[off + i * 8 ..][0..8], .little));
        i += 1;
    }
    for (f.field_steps) |*v| {
        v.* = @bitCast(std.mem.readInt(u64, raw[off + i * 8 ..][0..8], .little));
        i += 1;
    }
    for (f.fields) |*v| {
        v.* = @bitCast(std.mem.readInt(u64, raw[off + i * 8 ..][0..8], .little));
        i += 1;
    }
    return f;
}

// ---- comparison -----------------------------------------------------------

/// Worst field-scale deviation over one array, normalized per variable by that
/// variable's own baseline magnitude — so a near-zero slot (FY on the axis)
/// cannot inflate the result the way a plain relative error would.
pub const Cmp = struct {
    max_dev: f64 = 0,
    iv: usize = 0,
    flat: usize = 0,
    base_val: f64 = 0,
    now_val: f64 = 0,
    n_bitdiff: usize = 0,
    n_total: usize = 0,

    /// `nv`-strided arrays (u/p) with a per-variable scale taken from the
    /// baseline record.
    pub fn addStrided(self: *Cmp, base: []const f64, now: []const f64, nv: usize) void {
        var scale: [64]f64 = @splat(0);
        std.debug.assert(nv <= scale.len);
        for (base, 0..) |b, i| {
            const iv = i % nv;
            scale[iv] = @max(scale[iv], @abs(b));
        }
        for (base, now, 0..) |b, n, i| {
            self.n_total += 1;
            if (@as(u64, @bitCast(b)) != @as(u64, @bitCast(n))) self.n_bitdiff += 1;
            const iv = i % nv;
            const s = if (scale[iv] > 0) scale[iv] else 1.0;
            const dev = @abs(b - n) / s;
            if (dev > self.max_dev) {
                self.* = .{
                    .max_dev = dev,
                    .iv = iv,
                    .flat = i,
                    .base_val = b,
                    .now_val = n,
                    .n_bitdiff = self.n_bitdiff,
                    .n_total = self.n_total,
                };
            }
        }
    }

    pub fn bitIdentical(self: *const Cmp) bool {
        return self.n_bitdiff == 0;
    }

    /// Gate on BIT identity, not on the tolerance. Both sides are this
    /// repository built the same way, so equality is achievable and is the same
    /// standard the threading determinism gate already holds to; a tolerance
    /// gate would let a real sub-1e-12 change through silently, which is
    /// exactly the class of accident a regression baseline exists to catch.
    ///
    /// `bound` is therefore not the pass/fail line but the *diagnosis* line: it
    /// separates "the numbers actually changed" from "only the last bits moved,
    /// which is what a toolchain or -Doptimize change looks like". Both fail —
    /// the author decides — but they get told which one they are looking at.
    pub fn check(self: *const Cmp, bound: f64, what: []const u8) !void {
        std.debug.print(
            "self-golden [{s}]: max dev {e:.3}, {d}/{d} slots differ in bits{s}\n",
            .{ what, self.max_dev, self.n_bitdiff, self.n_total, if (self.bitIdentical()) " — BIT-IDENTICAL" else "" },
        );
        if (self.bitIdentical()) return;

        if (self.max_dev > bound) {
            std.debug.print(
                "self-golden REGRESSION [{s}]: max dev {e:.3} exceeds {e:.1} — a real numerical change.\n",
                .{ what, self.max_dev, bound },
            );
        } else {
            std.debug.print(
                "self-golden DRIFT [{s}]: bits differ but max dev {e:.3} is below {e:.1}.\n" ++
                    "  Either a toolchain / -Doptimize change, or a real but very small\n" ++
                    "  code change — the magnitude alone cannot tell them apart, so if you\n" ++
                    "  did not change the toolchain, treat this as a code change.\n",
                .{ what, self.max_dev, bound },
            );
        }
        std.debug.print(
            "  worst: iv={d} flat={d}  baseline {e:.17} -> now {e:.17}\n" ++
                "  If the change was intended, regenerate: zig build update-self-goldens\n",
            .{ self.iv, self.flat, self.base_val, self.now_val },
        );
        return error.SelfGoldenMismatch;
    }
};
