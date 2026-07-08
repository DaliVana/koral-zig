//! Round-trip gates for the KDMP checkpoint / restart path (io/dump.zig).
//!
//! A restart must reproduce the saved state exactly: the primitives are stored
//! verbatim and the conserveds are re-derived by p2u (no re-flooring), so a
//! serialize → fresh-sim → loadPrimDump cycle is required to be bit-identical
//! on both p and u, and the header must carry the clock/step/frame bookkeeping
//! that lets a run continue without rewinding or clobbering output.

const std = @import("std");
const config = @import("config.zig");
const grid_mod = @import("grid.zig");
const sim_mod = @import("sim.zig");
const dump = @import("io/dump.zig");

const Grid = grid_mod.Grid;

const cfg = config.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;
const NV = SimT.nv;

fn makeBox(a: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !SimT {
    const g = Grid.init(.{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .ng = 3,
        .minx = 0,
        .maxx = 2.0,
        .miny = 0,
        .maxy = 3.0,
        .minz = 0,
        .maxz = 4.0,
    });
    return SimT.init(a, g, .{ .coords = .mink, .gam = 5.0 / 3.0, .bc_x = .copy, .bc_y = .copy });
}

/// Fill every domain cell with a distinct, physical (p2u-valid) primitive
/// state so the round-trip actually exercises the cell ordering, not just a
/// constant.
fn fillPattern(s: *SimT) !void {
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                const idx: f64 = @floatFromInt(((iz * s.nyi() + iy) * s.nxi()) + ix);
                var pp: [NV]f64 = @splat(0);
                pp[L.index(.rho)] = 1.0 + 0.001 * idx;
                pp[L.index(.uu)] = 0.1 + 0.0001 * idx;
                pp[L.index(.vx)] = 0.01 * @sin(idx);
                pp[L.index(.vy)] = -0.02 + 0.0003 * idx;
                pp[L.index(.vz)] = 0.005;
                pp[L.index(.entr)] = 0.5 + 0.002 * idx;
                pp[L.index(.b1)] = 0.03 * @cos(idx);
                pp[L.index(.b2)] = 0.01 * idx;
                pp[L.index(.b3)] = -0.004;
                try s.initCell(ix, iy, iz, pp);
            }
        }
    }
}

test "KDMP restart round-trips p and u bit-for-bit and restores clock/step/frame" {
    const a = std.testing.allocator;
    var src = try makeBox(a, 5, 4, 3);
    defer src.deinit();
    try fillPattern(&src);
    src.t = 12.3456789;
    src.nstep = 4242;

    const out_idx: u32 = 7;
    const buf = try a.alloc(u8, dump.primDumpSize(SimT, &src));
    defer a.free(buf);
    const n = dump.serializePrimDump(SimT, &src, out_idx, buf);
    try std.testing.expectEqual(dump.primDumpSize(SimT, &src), n);

    // Header carries the full restart bookkeeping.
    const h = try dump.parseDumpHeader(buf);
    try std.testing.expectEqual(@as(u32, 5), h.nx);
    try std.testing.expectEqual(@as(u32, 4), h.ny);
    try std.testing.expectEqual(@as(u32, 3), h.nz);
    try std.testing.expectEqual(@as(u32, NV), h.nv);
    try std.testing.expectEqual(src.t, h.t);
    try std.testing.expectEqual(src.nstep, h.nstep);
    try std.testing.expectEqual(out_idx, h.out_idx);

    // A fresh sim of the same shape reloads to a bit-identical p AND u
    // (u is re-derived by p2u from the same primitives + geometry).
    var dst = try makeBox(a, 5, 4, 3);
    defer dst.deinit();
    const h2 = try dump.loadPrimDump(SimT, &dst, buf);
    try std.testing.expectEqual(h, h2);

    var iz: i64 = 0;
    while (iz < src.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < src.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < src.nxi()) : (ix += 1) {
                var ps: [NV]f64 = undefined;
                var pd: [NV]f64 = undefined;
                var us: [NV]f64 = undefined;
                var ud: [NV]f64 = undefined;
                src.p.load(ix, iy, iz, &ps);
                dst.p.load(ix, iy, iz, &pd);
                src.u.load(ix, iy, iz, &us);
                dst.u.load(ix, iy, iz, &ud);
                for (0..NV) |iv| {
                    try std.testing.expectEqual(ps[iv], pd[iv]);
                    try std.testing.expectEqual(us[iv], ud[iv]);
                }
            }
        }
    }
}

test "loadPrimDump rejects a grid-shape mismatch" {
    const a = std.testing.allocator;
    var src = try makeBox(a, 5, 4, 3);
    defer src.deinit();
    try fillPattern(&src);

    const buf = try a.alloc(u8, dump.primDumpSize(SimT, &src));
    defer a.free(buf);
    _ = dump.serializePrimDump(SimT, &src, 0, buf);

    var wrong = try makeBox(a, 6, 4, 3); // nx differs
    defer wrong.deinit();
    try std.testing.expectError(error.DimMismatch, dump.loadPrimDump(SimT, &wrong, buf));
}
