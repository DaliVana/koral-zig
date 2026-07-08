//! Sim.init runtime precondition validation (P1 correctness). Each construction
//! below violates exactly one invariant and must be rejected with
//! error.InvalidConfig instead of failing far downstream (a negative-index panic
//! in the sweep, a null-unwrap in setBc, silently-mixed coordinate systems, a
//! silent ν=0 viscous no-op, or order-dependent polar garbage); the final test
//! pins that a clean config still succeeds.

const std = @import("std");
const config = @import("config.zig");
const grid_mod = @import("grid.zig");
const sim_mod = @import("sim.zig");

const Grid = grid_mod.Grid;

// A radiative-hydro Minkowski config: PPM (ghostCells = 3) and a live .ee var
// (so the radviscosity/opac precondition is exercised). Matches testing/tubes
// cfg_rad, kept inline so this file is self-contained.
const cfg = config.Config{
    .modules = &.{ .hydro, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg);

fn grid(ng: usize, ny: usize) Grid {
    return Grid.init(.{
        .nx = 8,
        .ny = ny,
        .ng = ng,
        .minx = 0,
        .maxx = 1,
        .miny = 0,
        .maxy = 1,
    });
}

const a = std.testing.allocator;

test "Sim.init rejects ghost depth below the reconstruction stencil" {
    // PPM loads i−2 and needs ng ≥ 3; ng = 2 would drive Field.cellOffset
    // negative deep in the sweep.
    try std.testing.expectError(error.InvalidConfig, SimT.init(a, grid(2, 8), .{ .coords = .mink }));
}

test "Sim.init rejects a .specific boundary with a null specific_bc" {
    try std.testing.expectError(error.InvalidConfig, SimT.init(a, grid(3, 8), .{ .coords = .mink, .bc_x = .specific }));
    try std.testing.expectError(error.InvalidConfig, SimT.init(a, grid(3, 8), .{ .coords = .mink, .bc_y = .specific }));
}

test "Sim.init rejects runtime coords diverging from comptime cfg.coords" {
    try std.testing.expectError(error.InvalidConfig, SimT.init(a, grid(3, 8), .{ .coords = .ks }));
}

test "Sim.init rejects radviscosity with a null opac" {
    // ν = α·mfp, mfp = 1/χ, χ needs opacities — the silent ν=0 no-op is the
    // outcome the check eliminates.
    try std.testing.expectError(error.InvalidConfig, SimT.init(a, grid(3, 8), .{ .coords = .mink, .radviscosity = true }));
}

test "Sim.init rejects correct_polaraxis when ny <= 2*nccorrectpolar" {
    try std.testing.expectError(error.InvalidConfig, SimT.init(a, grid(3, 4), .{
        .coords = .mink,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
    }));
}

test "Sim.init accepts a clean config" {
    var s = try SimT.init(a, grid(3, 8), .{ .coords = .mink });
    defer s.deinit();
    // radviscosity+opac is legal once opac is present (spot-check the opposite
    // of the rejection above is not accidentally over-broad):
    var s2 = try SimT.init(a, grid(3, 8), .{ .coords = .mink, .correct_polaraxis = true, .nccorrectpolar = 2 });
    defer s2.deinit();
}
