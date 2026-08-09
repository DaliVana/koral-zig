//! Per-point geometry handed to kernels (C: struct geometry, ko.h:418).
//!
//! The 4×5 layout is C's: columns 0..3 are the metric, column 4 carries
//! extras — gg[0..2][4] = d(ln√−g)/dx^i, gg[3][4] = √−g, GG[3][4] = gttpert.

const config = @import("config.zig");

/// Distinct pointer view of the covariant block g_μν (`gg`). Same layout as
/// MetricConOf(T) but a nominally different type, so a *typed* value of one
/// cannot be passed where the other is expected. T is f64 or @Vector(W, f64)
/// (math/simd.zig).
///
/// The guarantee covers values obtained from `cov()`/`con()`; it does NOT
/// cover an anonymous literal, which coerces to whichever view the parameter
/// asks for — `wantsCon(.{ .m = &g.gg })` compiles. Build views with the
/// accessors, never with a bare `.{ .m = ... }` at an argument position.
pub fn MetricCovOf(comptime T: type) type {
    return struct { m: *const [4][5]T };
}

/// Distinct pointer view of the contravariant block g^μν (`GG`).
pub fn MetricConOf(comptime T: type) type {
    return struct { m: *const [4][5]T };
}

pub const MetricCov = MetricCovOf(f64);
pub const MetricCon = MetricConOf(f64);

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

    /// Typed view of g_μν for the lane-generic tensor helpers.
    pub fn cov(self: *const Geometry) MetricCov {
        return .{ .m = &self.gg };
    }

    /// Typed view of g^μν.
    pub fn con(self: *const Geometry) MetricCon {
        return .{ .m = &self.GG };
    }
};
