//! Minimal binary + text output for a run (the plan defers HDF5/SILO; this is
//! the "binary KSTP-style dump + tiny reader" placeholder). Serialization is
//! pure (into a caller-owned buffer) so it is testable without touching the
//! filesystem — the executable does the actual std.Io writes.
//!
//!  * KDMP: a snapshot of the domain primitives (little-endian, iv-fastest
//!    AoS — the same ordering as the KSTP/KINI goldens and C's get_u), so a
//!    dump can be diffed against C or re-read as a restart prototype.
//!  * scalars.dat: one whitespace-separated text row per output, mirroring
//!    KORAL's scalars.dat (fileop.c:113) — the diagnostic time series.

const std = @import("std");

pub const kdmp_magic = "KDMP";
pub const kdmp_version: u32 = 1;

/// Bytes a KDMP snapshot of the domain primitives occupies.
pub fn primDumpSize(comptime SimT: type, sim: *const SimT) usize {
    const ncell = sim.grid.nx * sim.grid.ny * sim.grid.nz;
    return 4 + 4 + 4 * 4 + 8 + ncell * SimT.nv * 8;
}

/// Serialize the domain primitives into `out` (must be ≥ primDumpSize).
/// Header: "KDMP" u32 ver, u32 nx,ny,nz,nv, f64 t; then p[nz][ny][nx][nv]
/// (iv fastest). Returns the number of bytes written.
pub fn serializePrimDump(comptime SimT: type, sim: *const SimT, out: []u8) usize {
    var w: usize = 0;
    @memcpy(out[w..][0..4], kdmp_magic);
    w += 4;
    w += putU32(out[w..], kdmp_version);
    w += putU32(out[w..], @intCast(sim.grid.nx));
    w += putU32(out[w..], @intCast(sim.grid.ny));
    w += putU32(out[w..], @intCast(sim.grid.nz));
    w += putU32(out[w..], @intCast(SimT.nv));
    w += putF64(out[w..], sim.t);

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

fn putU32(out: []u8, v: u32) usize {
    std.mem.writeInt(u32, out[0..4], v, .little);
    return 4;
}
fn putF64(out: []u8, v: f64) usize {
    std.mem.writeInt(u64, out[0..8], @bitCast(v), .little);
    return 8;
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
};

pub const scalar_header =
    "# t dt nstep mass mdot radlum totallum H/R maxPmag/Ptot n_hdfix n_radimpfail n_nan\n";

/// Append one whitespace-separated row to a growing byte list (the executable
/// rewrites scalars.dat from this buffer each output cadence).
pub fn appendScalarLine(list: *std.ArrayList(u8), allocator: std.mem.Allocator, row: ScalarRow) !void {
    var buf: [512]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, "{e:.10} {e:.6} {d} {e:.10} {e:.10} {e:.10} {e:.10} {e:.6} {e:.6} {d} {d} {d}\n", .{
        row.t,           row.dt,            row.nstep,   row.mass,
        row.mdot,          row.radlum,        row.totallum, row.scaleheight,
        row.max_pmag_ptot, row.n_hd_fixup,    row.n_radimp_fail, row.n_nan,
    });
    try list.appendSlice(allocator, line);
}

test "KDMP round-trips the header" {
    const buf = try std.testing.allocator.alloc(u8, 64);
    defer std.testing.allocator.free(buf);
    var w: usize = 0;
    @memcpy(buf[w..][0..4], kdmp_magic);
    w += 4;
    _ = putU32(buf[w..], kdmp_version);
    try std.testing.expect(std.mem.eql(u8, buf[0..4], "KDMP"));
    try std.testing.expectEqual(kdmp_version, std.mem.readInt(u32, buf[4..8], .little));
}
