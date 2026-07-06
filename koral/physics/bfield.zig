//! Magnetic-field four-vector helpers (C: magn.c:11-144).
//!
//! `B` is the evolved primitive field B^i (C: pp[B1..B3], "curly B"),
//! `bcon`/`bcov` the fluid-frame magnetic four-vector with b·u = 0.

const simd = @import("../math/simd.zig");
const relele = @import("../relele.zig");

/// b^μ from B^i and the gas four-velocity (C: calc_bcon_4vel, magn.c:48).
pub fn bconFrom4vel(B: [3]f64, ucon: [4]f64, ucov: [4]f64) [4]f64 {
    return bconFrom4velG(f64, B, ucon, ucov);
}

/// bconFrom4vel over lane type T.
pub fn bconFrom4velG(comptime T: type, B: [3]T, ucon: [4]T, ucov: [4]T) [4]T {
    var bcon: [4]T = undefined;
    bcon[0] = B[0] * ucov[1] + B[1] * ucov[2] + B[2] * ucov[3];
    const inv_ut = simd.splat(T, 1.0) / ucon[0];
    for (1..4) |j| {
        bcon[j] = (B[j - 1] + bcon[0] * ucon[j]) * inv_ut;
    }
    return bcon;
}

pub fn Bfield4Of(comptime T: type) type {
    return struct { bcon: [4]T, bcov: [4]T, bsq: T };
}
pub const Bfield4 = Bfield4Of(f64);

/// b^μ, b_μ, b² in one go (C: calc_bcon_bcov_bsq_from_4vel, magn.c:11).
pub fn bconBcovBsqFrom4vel(B: [3]f64, ucon: [4]f64, ucov: [4]f64, gg: *const [4][5]f64) Bfield4 {
    return bconBcovBsqFrom4velG(f64, B, ucon, ucov, gg);
}

/// bconBcovBsqFrom4vel over lane type T.
pub fn bconBcovBsqFrom4velG(comptime T: type, B: [3]T, ucon: [4]T, ucov: [4]T, gg: *const [4][5]T) Bfield4Of(T) {
    const bcon = bconFrom4velG(T, B, ucon, ucov);
    const bcov = relele.indices21G(T, bcon, gg);
    return .{ .bcon = bcon, .bcov = bcov, .bsq = relele.dotG(T, bcon, bcov) };
}

/// Primitive B^i back from b^μ and the gas four-velocity
/// (C: calc_Bcon_4vel / the tail of calc_Bcon_prim, magn.c:75/:119).
pub fn bigBconFrom4vel(bcon: [4]f64, ucon: [4]f64) [4]f64 {
    var Bcon: [4]f64 = undefined;
    Bcon[0] = 0;
    for (1..4) |j| {
        Bcon[j] = bcon[j] * ucon[0] - bcon[0] * ucon[j];
    }
    return Bcon;
}
