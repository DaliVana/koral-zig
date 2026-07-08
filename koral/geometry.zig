//! Per-point geometry handed to kernels (C: struct geometry, ko.h:418).
//!
//! The 4×5 layout is C's: columns 0..3 are the metric, column 4 carries
//! extras — gg[0..2][4] = d(ln√−g)/dx^i, gg[3][4] = √−g, GG[3][4] = gttpert.

const config = @import("config.zig");

pub const Geometry = struct {
    coords: config.Coords,
    /// coordinates of the point; xxvec[0] = 0 (stationary metric).
    xxvec: [4]f64,
    gg: [4][5]f64,
    GG: [4][5]f64,
    gdet: f64,
    /// lapse α = 1/√(−g^tt)
    alpha: f64,
    gttpert: f64,
};
