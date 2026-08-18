//! Minimal binary + text output for a run, and the restart/continue path.
//! Serialization is pure (into a caller-owned buffer) so it is testable without
//! touching the filesystem. `writePrim` / `loadPrim` / `resolveRestartPath`
//! own the serial and MPI file protocol so drivers do not reimplement it.
//!
//!  * KDMP: a self-describing snapshot of the domain primitives (little-endian,
//!    iv-fastest AoS — the same ordering as the KSTP/KINI goldens and C's
//!    get_u). Because the header carries `t`, `nstep` and `out_idx`, one KDMP
//!    file is a complete checkpoint: a run can be *continued* from it with no
//!    side file (unlike C's split res####.head / res####.dat pair). The C
//!    restart dumps are bridged into this format by tools/res2kdmp.zig.
//!  * scalars.dat: one whitespace-separated text row per output, mirroring
//!    KORAL's scalars.dat (fileop.c:113) — the diagnostic time series.

const std = @import("std");
const scalars = @import("scalars.zig");
const invert = @import("../solve/invert.zig");

pub const kdmp_magic = "KDMP";
/// v2 adds nstep + out_idx to the header so a KDMP is a self-contained restart
/// point (v1 only had t). The reader rejects any other version.
pub const kdmp_version: u32 = 2;

/// Fixed-size KDMP header. `nx,ny,nz` are the *domain* (active) cell counts and
/// `nv` the primitive count — a restart onto a differently shaped grid is a
/// hard error, not a silent reinterpretation. `t`/`nstep`/`out_idx` restore the
/// clock and the output bookkeeping so a continued run neither rewinds nor
/// clobbers earlier frames.
pub const DumpHeader = struct {
    nx: u32,
    ny: u32,
    nz: u32,
    nv: u32,
    t: f64,
    nstep: u64,
    out_idx: u32,
};

/// Bytes on disk: magic(4) ver(4) nx,ny,nz,nv(4·4) t(8) nstep(8) out_idx(4) = 44.
pub const header_size: usize = 4 + 4 + 4 * 4 + 8 + 8 + 4;

pub const ReadError = error{ BadMagic, BadVersion, DimMismatch, Truncated };

/// Bytes a KDMP snapshot of the domain primitives occupies.
pub fn primDumpSize(comptime SimT: type, sim: *const SimT) usize {
    const ncell = sim.grid.nx * sim.grid.ny * sim.grid.nz;
    return header_size + ncell * SimT.nv * 8;
}

/// Bytes of one rank's body slab (the local interior, no header).
pub fn primBodySize(comptime SimT: type, sim: *const SimT) usize {
    return sim.grid.nx * sim.grid.ny * sim.grid.nz * SimT.nv * 8;
}

/// Byte offset of global z-plane `tok` in a KDMP file. The body is
/// `p[iz][iy][ix][iv]` global row-major, so a φ-slab (r and θ whole — the
/// φ-only decomposition) is one contiguous byte range: this offset is the
/// whole addressing scheme of the collective write/read (MPI plan §8.1 —
/// no file views, no subarray types). Serial ⇔ MPI checkpoints are
/// therefore the same bytes, and restart works at any rank count.
pub fn bodyOffset(nx: usize, ny: usize, nv: usize, tok: usize) u64 {
    return @as(u64, header_size) + @as(u64, tok) * @as(u64, nx) * ny * nv * 8;
}

/// Write the 44-byte KDMP header into `out` (single source of truth shared by
/// the live dump and the res2kdmp converter). Returns bytes written.
pub fn writeDumpHeader(out: []u8, h: DumpHeader) usize {
    var w: usize = 0;
    @memcpy(out[w..][0..4], kdmp_magic);
    w += 4;
    w += putU32(out[w..], kdmp_version);
    w += putU32(out[w..], h.nx);
    w += putU32(out[w..], h.ny);
    w += putU32(out[w..], h.nz);
    w += putU32(out[w..], h.nv);
    w += putF64(out[w..], h.t);
    w += putU64(out[w..], h.nstep);
    w += putU32(out[w..], h.out_idx);
    return w;
}

/// Parse and validate the magic/version, returning the header. Does not touch
/// the body — a caller with a `SimT` should use `loadPrimDump` (which also
/// checks the dimensions against the live grid).
pub fn parseDumpHeader(bytes: []const u8) ReadError!DumpHeader {
    if (bytes.len < header_size) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..4], kdmp_magic)) return error.BadMagic;
    var r: usize = 4;
    if (getU32(bytes, &r) != kdmp_version) return error.BadVersion;
    const nx = getU32(bytes, &r);
    const ny = getU32(bytes, &r);
    const nz = getU32(bytes, &r);
    const nv = getU32(bytes, &r);
    const t = getF64(bytes, &r);
    const nstep = getU64(bytes, &r);
    const out_idx = getU32(bytes, &r);
    return .{ .nx = nx, .ny = ny, .nz = nz, .nv = nv, .t = t, .nstep = nstep, .out_idx = out_idx };
}

/// Serialize the domain primitives into `out` (must be ≥ primDumpSize).
/// Header (see DumpHeader) then p[nz][ny][nx][nv] with iv fastest, iz slowest —
/// the same cell order the golden readers expect. Returns bytes written.
pub fn serializePrimDump(comptime SimT: type, sim: *const SimT, out_idx: u32, out: []u8) usize {
    var w = writeDumpHeader(out, .{
        .nx = @intCast(sim.grid.nx),
        .ny = @intCast(sim.grid.ny),
        .nz = @intCast(sim.grid.nz),
        .nv = @intCast(SimT.nv),
        .t = sim.t,
        .nstep = sim.nstep,
        .out_idx = out_idx,
    });

    w += serializePrimBody(SimT, sim, out[w..]);
    return w;
}

/// Serialize just the body — this sim's domain interior, iv fastest, iz
/// slowest — into `out` (≥ primBodySize). Under MPI this is one rank's
/// contiguous slab of the global file body (see bodyOffset); serially it
/// is the whole body. Returns bytes written.
pub fn serializePrimBody(comptime SimT: type, sim: *const SimT, out: []u8) usize {
    var w: usize = 0;
    var iz: i64 = 0;
    while (iz < sim.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < sim.nxi()) : (ix += 1) {
                var pp: [SimT.nv]f64 = undefined;
                sim.p.load(ix, iy, iz, &pp);
                for (0..SimT.nv) |iv| w += putF64(out[w..], pp[iv]);
            }
        }
    }
    return w;
}

/// Restart/continue: load a KDMP snapshot into `sim`, cell by cell, via
/// `initCell` (p stored verbatim, u = p2u without re-flooring — exactly C's
/// fread_restartfile_bin path, which trusts the saved primitives). The caller
/// must still run `finishInit` (halo + setBc + dt guess — ghosts and dt are
/// recomputed, never stored) and adopt the returned `t`/`nstep`/`out_idx`.
/// Errors if the grid or nv disagree with the file — no silent reshaping.
pub fn loadPrimDump(comptime SimT: type, sim: *SimT, bytes: []const u8) !DumpHeader {
    const h = try parseDumpHeader(bytes);
    if (h.nx != sim.grid.nx or h.ny != sim.grid.ny or h.nz != sim.grid.nz or h.nv != SimT.nv)
        return error.DimMismatch;

    const ncell = @as(usize, h.nx) * h.ny * h.nz;
    if (bytes.len < header_size + ncell * SimT.nv * 8) return error.Truncated;

    try loadPrimBody(SimT, sim, bytes[header_size..]);
    return h;
}

/// Load just a body slab — this sim's domain interior — from `bytes`
/// (≥ primBodySize; under MPI, the rank's slab read collectively at
/// bodyOffset). Same initCell semantics as loadPrimDump; the caller has
/// already validated the header against the GLOBAL grid.
pub fn loadPrimBody(comptime SimT: type, sim: *SimT, bytes: []const u8) !void {
    if (bytes.len < primBodySize(SimT, sim)) return error.Truncated;
    var r: usize = 0;
    var iz: i64 = 0;
    while (iz < sim.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < sim.nxi()) : (ix += 1) {
                var pp: [SimT.nv]f64 = undefined;
                for (0..SimT.nv) |iv| pp[iv] = getF64(bytes, &r);
                try sim.initCell(ix, iy, iz, pp);
            }
        }
    }
}

// ---- little-endian primitives ----------------------------------------------

fn putU32(out: []u8, v: u32) usize {
    std.mem.writeInt(u32, out[0..4], v, .little);
    return 4;
}
fn putU64(out: []u8, v: u64) usize {
    std.mem.writeInt(u64, out[0..8], v, .little);
    return 8;
}
fn putF64(out: []u8, v: f64) usize {
    std.mem.writeInt(u64, out[0..8], @bitCast(v), .little);
    return 8;
}
fn getU32(bytes: []const u8, r: *usize) u32 {
    const v = std.mem.readInt(u32, bytes[r.*..][0..4], .little);
    r.* += 4;
    return v;
}
fn getU64(bytes: []const u8, r: *usize) u64 {
    const v = std.mem.readInt(u64, bytes[r.*..][0..8], .little);
    r.* += 8;
    return v;
}
fn getF64(bytes: []const u8, r: *usize) f64 {
    const v: f64 = @bitCast(std.mem.readInt(u64, bytes[r.*..][0..8], .little));
    r.* += 8;
    return v;
}

/// Write a numbered checkpoint. Serial (`ntz ≤ 1`) uses std.Io; MPI writes
/// the body first and the 44-byte header last as a completion marker, so a
/// killed write fails `parseDumpHeader` instead of looking like a valid
/// full-length file of zero holes. Final bytes match the serial layout.
pub fn writePrim(
    comptime SimT: type,
    sim: *const SimT,
    comm: anytype,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    idx: u32,
) !void {
    if (sim.decomp.ntz <= 1) {
        const bytes = try allocator.alloc(u8, primDumpSize(SimT, sim));
        defer allocator.free(bytes);
        const n = serializePrimDump(SimT, sim, idx, bytes);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes[0..n] });
        return;
    }
    try writePrimMpi(SimT, sim, comm, allocator, path, idx);
}

fn writePrimMpi(
    comptime SimT: type,
    sim: *const SimT,
    comm: anytype,
    allocator: std.mem.Allocator,
    path: []const u8,
    idx: u32,
) !void {
    const g = sim.decomp.global;
    const body64 = try allocator.alloc(f64, primBodySize(SimT, sim) / 8);
    defer allocator.free(body64);
    const body = std.mem.sliceAsBytes(body64);
    _ = serializePrimBody(SimT, sim, body);

    var buf: [1024]u8 = undefined;
    const pathz = try std.fmt.bufPrintZ(&buf, "{s}", .{path});
    const total = bodyOffset(g.nx, g.ny, SimT.nv, g.nz);
    var f = try comm.fileCreate(pathz, total);
    comm.fileWriteAtAll(&f, bodyOffset(g.nx, g.ny, SimT.nv, @intCast(sim.decomp.tok)), body);
    comm.fileSync(&f);
    if (comm.rank() == 0) {
        var hdr: [header_size]u8 = undefined;
        _ = writeDumpHeader(&hdr, .{
            .nx = @intCast(g.nx),
            .ny = @intCast(g.ny),
            .nz = @intCast(g.nz),
            .nv = @intCast(SimT.nv),
            .t = sim.t,
            .nstep = sim.nstep,
            .out_idx = idx,
        });
        comm.fileWriteAt(&f, 0, hdr[0..]);
    }
    comm.fileClose(&f);
}

/// Load a checkpoint into `sim`. Serial reads the whole file; MPI reads the
/// header then this rank's φ-slab. Under MPI the header is folded so every
/// rank restarted from the same file (max == min) or the load fails together.
/// Caller adopts `t`/`nstep`/`out_idx` and runs `finishInit`.
pub fn loadPrim(
    comptime SimT: type,
    sim: *SimT,
    comm: anytype,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !DumpHeader {
    const h = if (sim.decomp.ntz <= 1) blk: {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 30));
        defer allocator.free(bytes);
        break :blk try loadPrimDump(SimT, sim, bytes);
    } else try loadPrimMpi(SimT, sim, comm, allocator, path);

    if (sim.decomp.ntz > 1) {
        const hv = [3]f64{ h.t, @floatFromInt(h.nstep), @floatFromInt(h.out_idx) };
        var hi = hv;
        var lo = hv;
        comm.allreduceMax(hi[0..]);
        comm.allreduceMin(lo[0..]);
        if (!std.mem.eql(f64, hi[0..], lo[0..])) return error.RestartRankDisagreement;
    }
    return h;
}

fn loadPrimMpi(
    comptime SimT: type,
    sim: *SimT,
    comm: anytype,
    allocator: std.mem.Allocator,
    path: []const u8,
) !DumpHeader {
    var pbuf: [1024]u8 = undefined;
    const pathz = try std.fmt.bufPrintZ(&pbuf, "{s}", .{path});
    const body64 = try allocator.alloc(f64, primBodySize(SimT, sim) / 8);
    defer allocator.free(body64);
    const body = std.mem.sliceAsBytes(body64);

    var f = try comm.fileOpenRead(pathz);
    defer comm.fileClose(&f);

    const fsize = comm.fileSize(&f);
    if (fsize < header_size) return error.Truncated;

    var hdr: [header_size]u8 = undefined;
    comm.fileReadAtAll(&f, 0, hdr[0..]);
    const h = try parseDumpHeader(hdr[0..]);
    const g = sim.decomp.global;
    if (h.nx != g.nx or h.ny != g.ny or h.nz != g.nz or h.nv != SimT.nv)
        return error.DimMismatch;
    if (fsize < bodyOffset(g.nx, g.ny, SimT.nv, g.nz)) return error.Truncated;

    comm.fileReadAtAll(&f, bodyOffset(g.nx, g.ny, SimT.nv, @intCast(sim.decomp.tok)), body);
    try loadPrimBody(SimT, sim, body);
    return h;
}

/// Resolve `--restart FILE|DIR` to a concrete KDMP path (caller frees).
/// A directory selects its lexical-max `*.kdmp` (zero-padded names sort by index).
pub fn resolveRestartPath(io: std.Io, allocator: std.mem.Allocator, arg: []const u8) ![]u8 {
    var d = std.Io.Dir.cwd().openDir(io, arg, .{ .iterate = true }) catch |err| switch (err) {
        error.NotDir => return allocator.dupe(u8, arg),
        else => return err,
    };
    defer d.close(io);

    var best: ?[]u8 = null;
    errdefer if (best) |b| allocator.free(b);
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".kdmp")) continue;
        if (best == null or std.mem.order(u8, entry.name, best.?) == .gt) {
            if (best) |b| allocator.free(b);
            best = try allocator.dupe(u8, entry.name);
        }
    }
    const name = best orelse return error.NoRestartFile;
    defer allocator.free(name);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ arg, name });
}

pub const Diag = struct {
    n_nan: u64 = 0,
    n_hd_fixup: u64 = 0,
};

pub fn collectDiag(comptime SimT: type, s: *const SimT) Diag {
    var d = Diag{};
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                var pp: [SimT.nv]f64 = undefined;
                s.p.load(ix, iy, iz, &pp);
                for (pp) |v| {
                    if (!std.math.isFinite(v)) {
                        d.n_nan += 1;
                        break;
                    }
                }
                if (s.getFlag(.hd_fixup, ix, iy, iz) != 0) d.n_hd_fixup += 1;
            }
        }
    }
    return d;
}

/// Globally-folded scalar row. Every rank must call this — it is collective.
/// `r_lum` / `r_scale` are the problem's diagnostic radii (PUFFY: 5000, 15).
pub fn scalarRow(comptime SimT: type, s: *SimT, dt: f64, r_lum: f64, r_scale: f64) !ScalarRow {
    const r_horizon = @import("../metric/metric.zig").rHorizonBL(s.opt.mp.a);
    const ix_h = scalars.radialShellIndex(SimT, s, r_horizon);
    const ix_l = scalars.radialShellIndex(SimT, s, r_lum);
    const diag = collectDiag(SimT, s);
    var counts = [4]f64{
        @floatFromInt(diag.n_hd_fixup),
        @floatFromInt(s.n_radimp_failures),
        @floatFromInt(diag.n_nan),
        @floatFromInt(invert.n_guard_recoveries.load(.monotonic)),
    };
    const gs = try scalars.globalScalars(SimT, s, ix_h, ix_l, r_scale, counts[0..]);
    return .{
        .t = s.t,
        .dt = dt,
        .nstep = s.nstep,
        .mass = gs.mass,
        .mdot = -gs.mdot,
        .radlum = gs.radlum,
        .totallum = gs.totallum,
        .scaleheight = gs.scaleheight,
        .max_pmag_ptot = gs.max_pmag_ptot,
        .n_hd_fixup = @intFromFloat(counts[0]),
        .n_radimp_fail = @intFromFloat(counts[1]),
        .n_nan = @intFromFloat(counts[2]),
        .n_floorguard = @intFromFloat(counts[3]),
    };
}

/// One row of scalars.dat — the diagnostic time series (all in code/GU units;
/// the executable may additionally print CGS/Eddington-scaled copies).
pub const ScalarRow = struct {
    t: f64,
    dt: f64,
    nstep: u64,
    mass: f64,
    mdot: f64, // already sign-flipped: >0 for accretion
    radlum: f64,
    totallum: f64,
    scaleheight: f64,
    max_pmag_ptot: f64, // max pmag/ptot over the domain
    n_hd_fixup: u64,
    n_radimp_fail: u64,
    n_nan: u64,
    /// Cumulative drift-floor guard recoveries (invert.n_guard_recoveries) —
    /// how often the koral_lite_puffy guard ladder diverted from the baseline
    /// drift-frame floor algebra. 0 on a healthy run.
    n_floorguard: u64,
};

pub const scalar_header =
    "# t dt nstep mass mdot radlum totallum H/R maxPmag/Ptot n_hdfix n_radimpfail n_nan n_floorguard\n";

/// Append one whitespace-separated row to a growing byte list (the executable
/// rewrites scalars.dat from this buffer each output cadence).
pub fn appendScalarLine(list: *std.ArrayList(u8), allocator: std.mem.Allocator, row: ScalarRow) !void {
    var buf: [512]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, "{e:.10} {e:.6} {d} {e:.10} {e:.10} {e:.10} {e:.10} {e:.6} {e:.6} {d} {d} {d} {d}\n", .{
        row.t,           row.dt,            row.nstep,   row.mass,
        row.mdot,          row.radlum,        row.totallum, row.scaleheight,
        row.max_pmag_ptot, row.n_hd_fixup,    row.n_radimp_fail, row.n_nan,
        row.n_floorguard,
    });
    try list.appendSlice(allocator, line);
}

test "bodyOffset: slabs tile the body contiguously and exactly" {
    // 4 ranks × 3 planes of a 384×360 r×θ grid: each slab starts where the
    // previous ended, plane 0 starts right after the header, and the last
    // slab ends at the total file size.
    const nx = 384;
    const ny = 360;
    const nv = 13;
    const plane: u64 = nx * ny * nv * 8;
    try std.testing.expectEqual(@as(u64, header_size), bodyOffset(nx, ny, nv, 0));
    var tok: usize = 0;
    while (tok < 12) : (tok += 3) {
        try std.testing.expectEqual(
            bodyOffset(nx, ny, nv, tok) + 3 * plane,
            bodyOffset(nx, ny, nv, tok + 3),
        );
    }
    try std.testing.expectEqual(
        @as(u64, header_size) + 12 * plane,
        bodyOffset(nx, ny, nv, 12),
    );
}

test "KDMP header round-trips through write/parse" {
    var buf: [header_size]u8 = undefined;
    const h = DumpHeader{ .nx = 384, .ny = 360, .nz = 4, .nv = 13, .t = 123.456, .nstep = 98765, .out_idx = 42 };
    const w = writeDumpHeader(&buf, h);
    try std.testing.expectEqual(header_size, w);
    const got = try parseDumpHeader(&buf);
    try std.testing.expectEqual(h, got);
}

test "parseDumpHeader rejects bad magic / version / truncation" {
    var buf: [header_size]u8 = undefined;
    _ = writeDumpHeader(&buf, .{ .nx = 1, .ny = 1, .nz = 1, .nv = 13, .t = 0, .nstep = 0, .out_idx = 0 });
    try std.testing.expectError(error.Truncated, parseDumpHeader(buf[0 .. header_size - 1]));
    var bad = buf;
    bad[0] = 'X';
    try std.testing.expectError(error.BadMagic, parseDumpHeader(&bad));
    var badver = buf;
    std.mem.writeInt(u32, badver[4..8], 999, .little);
    try std.testing.expectError(error.BadVersion, parseDumpHeader(&badver));
}
