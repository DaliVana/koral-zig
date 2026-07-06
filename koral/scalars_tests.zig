//! M13 theory gates for the diagnostic scalar reductions (io/scalars.zig).
//!
//! The MYCOORDS reductions (mass, mdot, magnetization) have closed forms on a
//! uniform Minkowski state, so they are checked to machine precision here. The
//! BL-frame luminosity and the scale height run through the coordinate
//! transform and the density-weighted average and are pinned by the C golden
//! on the real PUFFY state (golden_puffystep_test.zig / harness_scalars.c) —
//! there is no clean flat-space analytic for them.

const std = @import("std");
const config = @import("config.zig");
const grid_mod = @import("grid.zig");
const sim_mod = @import("sim.zig");
const hydro = @import("physics/hydro.zig");
const scalars = @import("io/scalars.zig");

const Grid = grid_mod.Grid;

const cfg_mhd = config.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg_mhd);
const L = SimT.Layout;

const twopi = 2.0 * std.math.pi;

/// Build a uniform MINK box (Lx×Ly) with density ρ, internal energy u,
/// radial 3-velocity vx and magnetic field B, and return the initialized sim.
fn uniformBox(a: std.mem.Allocator, nx: usize, ny: usize, rho: f64, u: f64, vx: f64, b: [3]f64) !SimT {
    // z extent = 2π so the single axisymmetric slice spans a full revolution;
    // totalMass uses the actual cell dz (unlike mdot/lum, which force 2π).
    const g = Grid.init(.{ .nx = nx, .ny = ny, .ng = 3, .minx = 0, .maxx = 2.0, .miny = 0, .maxy = 3.0, .minz = 0, .maxz = twopi });
    var s = try SimT.init(a, g, .{ .coords = .mink, .gam = 5.0 / 3.0, .bc_x = .copy, .bc_y = .copy });
    errdefer s.deinit();
    var pp: [SimT.nv]f64 = @splat(0);
    pp[L.index(.rho)] = rho;
    pp[L.index(.uu)] = u;
    pp[L.index(.vx)] = vx;
    pp[L.index(.entr)] = hydro.sFromU(rho, u, 5.0 / 3.0);
    pp[L.index(.b1)] = b[0];
    pp[L.index(.b2)] = b[1];
    pp[L.index(.b3)] = b[2];
    var iy: i64 = 0;
    while (iy < @as(i64, @intCast(ny))) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < @as(i64, @intCast(nx))) : (ix += 1) {
            try s.initCell(ix, iy, 0, pp);
        }
    }
    return s;
}

test "scalars: totalMass = ρ·Lx·Ly·2π on a uniform MINK box (gdet=1)" {
    const a = std.testing.allocator;
    const rho = 3.5;
    var s = try uniformBox(a, 8, 6, rho, 0.1, 0.0, .{ 0, 0, 0 });
    defer s.deinit();
    const mass = scalars.totalMass(SimT, &s);
    // Lx=2, Ly=3, dz→2π
    try std.testing.expectApproxEqRel(rho * 2.0 * 3.0 * twopi, mass, 1e-14);
}

test "scalars: mdot through a shell = ρ·u^x·Ly·2π on uniform MINK flow" {
    const a = std.testing.allocator;
    const rho = 2.0;
    const vx = 0.3; // VELR: in MINK u^x = vx exactly
    var s = try uniformBox(a, 8, 6, rho, 0.1, vx, .{ 0, 0, 0 });
    defer s.deinit();
    // shell-independent (gdet=1, uniform); check a couple of shells
    for ([_]i64{ 0, 3, 7 }) |ix| {
        const md = try scalars.mdot(SimT, &s, ix);
        try std.testing.expectApproxEqRel(rho * vx * 3.0 * twopi, md, 1e-13);
    }
}

test "scalars: maxPmagPtot = |B|²/2 / ((γ−1)u) for B at rest (MINK)" {
    const a = std.testing.allocator;
    const u = 0.5;
    const b = [3]f64{ 0.2, 0.1, 0.05 };
    var s = try uniformBox(a, 4, 4, 1.0, u, 0.0, b);
    defer s.deinit();
    const bsq = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]; // MINK metric = identity spatial
    const want = (bsq / 2.0) / ((5.0 / 3.0 - 1.0) * u);
    const got = try scalars.maxPmagPtot(SimT, &s);
    try std.testing.expectApproxEqRel(want, got, 1e-13);
}

test "scalars: radialShellIndex picks the first cell past the radius" {
    const a = std.testing.allocator;
    var s = try uniformBox(a, 8, 4, 1.0, 0.1, 0.0, .{ 0, 0, 0 });
    defer s.deinit();
    // MINK: radiusBL == x-center; grid x∈[0,2], 8 cells → centers 0.125,0.375,…
    const ix = scalars.radialShellIndex(SimT, &s, 0.5);
    try std.testing.expect(scalars.radiusBL(SimT, &s, ix) > 0.5);
    try std.testing.expect(scalars.radiusBL(SimT, &s, ix - 1) <= 0.5);
}
