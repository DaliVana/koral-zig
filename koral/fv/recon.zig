//! Cell-average → face reconstruction for finite-volume schemes
//! (C: avg2point, finite.c:97).
//!
//! Schemes:
//!   donor  — piecewise-constant (donor cell)
//!   linear — piecewise-linear with the minmod-θ limiter (θ = 1 is MinMod,
//!            θ = 2 is MC; PUFFY runs θ = 1.5). The MC/Superbee variants of
//!            C's FLUXLIMITER are not used by any current problem and are
//!            not implemented.
//!   ppm    — piecewise-parabolic (Colella & Woodward 1984, eqs 1.6–1.10)
//!            on a five-point stencil with per-cell widths.
//!
//! The kernel is `comptime T`-generic in the math/simd.zig style: `T` is
//! `f64` or `@Vector(W, f64)`. The scalar instantiation is the reference
//! (its chain is pinned by the C golden); the vector instantiation turns the
//! limiter branches into select-of-both-sides and reproduces the scalar
//! result bit-for-bit per lane (gated by the in-file lane test).

const std = @import("std");
const simd = @import("../math/simd.zig");

pub const Scheme = union(enum) {
    /// piecewise-constant (donor cell)
    donor,
    /// piecewise-linear, minmod-θ family
    linear: struct { theta: f64 },
    /// piecewise-parabolic (Colella & Woodward), nonuniform widths
    ppm,

    /// Stencil cells needed on each side of the center: 0, 1, 2.
    pub fn radius(s: Scheme) u8 {
        return switch (s) {
            .donor => 0,
            .linear => 1,
            .ppm => 2,
        };
    }
};

pub fn FacesOf(comptime T: type) type {
    return struct { left: T, right: T };
}
pub const Faces = FacesOf(f64);

/// Reconstruct one variable's left/right face values from the five-point
/// stencil `u` = {i−2 … i+2} with cell widths `dx`.
pub fn reconstruct(comptime T: type, scheme: Scheme, u: [5]T, dx: [5]T) FacesOf(T) {
    return switch (scheme) {
        .donor => .{ .left = u[2], .right = u[2] },
        .linear => |l| linearMinmod(T, u, l.theta),
        .ppm => ppm(T, u, dx),
    };
}

/// NV-variable convenience wrapper: `u[k]` is the k-th stencil cell's
/// variable vector (so the call site passes `.{ pm2, pm1, p0, pp1, pp2 }`);
/// `dx` is geometry, shared by all variables.
pub fn reconstructN(
    comptime n: usize,
    scheme: Scheme,
    u: [5][n]f64,
    dx: [5]f64,
) struct { left: [n]f64, right: [n]f64 } {
    var left: [n]f64 = undefined;
    var right: [n]f64 = undefined;
    for (0..n) |iv| {
        const f = reconstruct(f64, scheme, .{ u[0][iv], u[1][iv], u[2][iv], u[3][iv], u[4][iv] }, dx);
        left[iv] = f.left;
        right[iv] = f.right;
    }
    return .{ .left = left, .right = right };
}

/// C: finite.c:126-208 (Ramesh's rewrite: no divisions), FLUXLIMITER == 0.
/// At a local extremum the slope is selected to 0, which reproduces the
/// donor-cell fallback exactly (u ± 0.5·0 ≡ u bitwise).
fn linearMinmod(comptime T: type, u: [5]T, theta: f64) FacesOf(T) {
    const zero = simd.splat(T, 0.0);
    const half = simd.splat(T, 0.5);
    const th = simd.splat(T, theta);

    const deltam = u[2] - u[1];
    const deltap = u[3] - u[2];

    const smin = @min(@min(th * deltam, half * (deltam + deltap)), th * deltap);
    const smax = @max(@max(th * deltam, half * (deltam + deltap)), th * deltap);
    var slope = simd.select(T, deltam > zero, smin, smax);
    // local extremum → donor cell
    slope = simd.select(T, deltam * deltap <= zero, zero, slope);
    return .{ .left = u[2] - half * slope, .right = u[2] + half * slope };
}

/// C: my_sign — returns ±1 (0 → +1 is never hit: guarded by monotonicity).
fn sign(comptime T: type, x: T) T {
    return simd.select(T, x >= simd.splat(T, 0.0), simd.splat(T, 1.0), simd.splat(T, -1.0));
}

/// C: finite.c:210-332 — PPM with non-uniform cell widths.
fn ppm(comptime T: type, u: [5]T, dx: [5]T) FacesOf(T) {
    const zero = simd.splat(T, 0.0);
    const one = simd.splat(T, 1.0);
    const two = simd.splat(T, 2.0);
    const three = simd.splat(T, 3.0);

    const dxm2 = dx[0];
    const dxm1 = dx[1];
    const dx0 = dx[2];
    const dxp1 = dx[3];
    const dxp2 = dx[4];

    const dxp2_plus_dxp1 = dxp2 + dxp1;
    const dxp2_plus_dxp1_inv = one / dxp2_plus_dxp1;
    const dxp1_plus_dx0 = dxp1 + dx0;
    const dxp1_plus_dx0_inv = one / dxp1_plus_dx0;
    const dx0_plus_dxm1 = dx0 + dxm1;
    const dx0_plus_dxm1_inv = one / dx0_plus_dxm1;
    const dxm1_plus_dxm2 = dxm1 + dxm2;
    const dxm1_plus_dxm2_inv = one / dxm1_plus_dxm2;

    const dxm1_plus_dx0_plus_dxp1_inv = one / (dxm1 + dx0 + dxp1);
    const dx0_plus_dxp1_plus_dxp2_inv = one / (dx0 + dxp1 + dxp2);
    const dxm2_plus_dxm1_plus_dx0_inv = one / (dxm2 + dxm1 + dx0);

    const dx0_plus_twodxm1 = dx0 + two * dxm1;
    const dxp1_plus_twodx0 = dxp1 + two * dx0;
    const dxm1_plus_twodxm2 = dxm1 + two * dxm2;
    const twodxp1_plus_dx0 = two * dxp1 + dx0;
    const twodxp2_plus_dxp1 = two * dxp2 + dxp1;
    const twodx0_plus_dxm1 = two * dx0 + dxm1;

    // slopes Δa_{j-1}, Δa_j, Δa_{j+1} (C&W eq 1.7)
    var dri = dx0 * dxm1_plus_dx0_plus_dxp1_inv *
        (dx0_plus_twodxm1 * dxp1_plus_dx0_inv * (u[3] - u[2]) +
            twodxp1_plus_dx0 * dx0_plus_dxm1_inv * (u[2] - u[1]));
    var drip1 = dxp1 * dx0_plus_dxp1_plus_dxp2_inv *
        (dxp1_plus_twodx0 * dxp2_plus_dxp1_inv * (u[4] - u[3]) +
            twodxp2_plus_dxp1 * dxp1_plus_dx0_inv * (u[3] - u[2]));
    var drim1 = dxm1 * dxm2_plus_dxm1_plus_dx0_inv *
        (dxm1_plus_twodxm2 * dx0_plus_dxm1_inv * (u[2] - u[1]) +
            twodx0_plus_dxm1 * dxm1_plus_dxm2_inv * (u[1] - u[0]));

    // monotonize the slopes (C&W eq 1.8)
    dri = simd.select(
        T,
        (u[3] - u[2]) * (u[2] - u[1]) > zero,
        @min(@abs(dri), @min(two * @abs(u[2] - u[1]), two * @abs(u[2] - u[3]))) * sign(T, dri),
        zero,
    );
    drip1 = simd.select(
        T,
        (u[4] - u[3]) * (u[3] - u[2]) > zero,
        @min(@abs(drip1), @min(two * @abs(u[3] - u[2]), two * @abs(u[3] - u[4]))) * sign(T, drip1),
        zero,
    );
    drim1 = simd.select(
        T,
        (u[2] - u[1]) * (u[1] - u[0]) > zero,
        @min(@abs(drim1), @min(two * @abs(u[1] - u[0]), two * @abs(u[1] - u[2]))) * sign(T, drim1),
        zero,
    );

    // right face a_{j+1/2} (C&W eq 1.6)
    var z1 = dx0_plus_dxm1 / dxp1_plus_twodx0;
    var z2 = dxp2_plus_dxp1 / twodxp1_plus_dx0;
    var dx_inv = one / (dxm1 + dx0 + dxp1 + dxp2);
    var right = u[2] + dx0 * dxp1_plus_dx0_inv * (u[3] - u[2]) +
        dx_inv * ((two * dxp1 * dx0) * dxp1_plus_dx0_inv * (z1 - z2) * (u[3] - u[2]) -
            dx0 * z1 * drip1 + dxp1 * z2 * dri);

    // left face a_{j-1/2}
    z1 = dxm1_plus_dxm2 / dx0_plus_twodxm1;
    z2 = dxp1_plus_dx0 / twodx0_plus_dxm1;
    dx_inv = one / (dxm2 + dxm1 + dx0 + dxp1);
    var left = u[1] + dxm1 * dx0_plus_dxm1_inv * (u[2] - u[1]) +
        dx_inv * ((two * dx0 * dxm1) * dx0_plus_dxm1_inv * (z1 - z2) * (u[2] - u[1]) -
            dxm1 * z1 * dri + dx0 * z2 * drim1);

    // keep the parabola monotonic (equivalent of C&W eq 1.10)
    const flat = (right - u[2]) * (u[2] - left) <= zero;
    left = simd.select(T, flat, u[2], left);
    right = simd.select(T, flat, u[2], right);
    left = simd.select(
        T,
        (right - left) * (left - (three * u[2] - two * right)) < zero,
        three * u[2] - two * right,
        left,
    );
    right = simd.select(
        T,
        (right - left) * ((three * u[2] - two * left) - right) < zero,
        three * u[2] - two * left,
        right,
    );
    return .{ .left = left, .right = right };
}

//
// ---- tests ----------------------------------------------------------------
//
// Scheme-level theory gates (exactness, boundedness, mirror symmetry) live in
// flux_tests.zig; the C golden is flux_golden_tests.zig. Here we gate only
// the lane contract, mirroring simd.zig.
//

test "recon: vector lanes reproduce the scalar chain bit-for-bit" {
    const W = 4;
    const V = @Vector(W, f64);
    const schemes = [_]Scheme{
        .donor,
        .{ .linear = .{ .theta = 1.0 } },
        .{ .linear = .{ .theta = 1.5 } },
        .ppm,
    };

    var prng = std.Random.DefaultPrng.init(0x4ec0);
    const rng = prng.random();
    for (0..500) |round| {
        var u: [W][5]f64 = undefined;
        var dx: [W][5]f64 = undefined;
        for (0..W) |lane| {
            for (0..5) |i| {
                u[lane][i] = 2.0 * rng.float(f64) - 1.0;
                dx[lane][i] = 0.5 + 1.5 * rng.float(f64);
            }
        }
        // force the branchy cases onto fixed lanes: flat (Δ=0 ties),
        // monotone ramp, local extremum
        u[0] = @splat(0.25);
        u[1] = .{ -2.0, -1.0, 0.0, 1.0, 2.0 };
        u[2] = .{ 0.0, 1.0, 2.0, 1.0, 0.0 };
        if (round % 2 == 0) dx[0] = @splat(1.0);

        var uv: [5]V = undefined;
        var dxv: [5]V = undefined;
        for (0..5) |i| {
            var lu: [W]f64 = undefined;
            var ldx: [W]f64 = undefined;
            for (0..W) |lane| {
                lu[lane] = u[lane][i];
                ldx[lane] = dx[lane][i];
            }
            uv[i] = lu;
            dxv[i] = ldx;
        }

        for (schemes) |scheme| {
            const fv = reconstruct(V, scheme, uv, dxv);
            for (0..W) |lane| {
                const fs = reconstruct(f64, scheme, u[lane], dx[lane]);
                const lv: [W]f64 = fv.left;
                const rv: [W]f64 = fv.right;
                try std.testing.expectEqual(
                    @as(u64, @bitCast(fs.left)),
                    @as(u64, @bitCast(lv[lane])),
                );
                try std.testing.expectEqual(
                    @as(u64, @bitCast(fs.right)),
                    @as(u64, @bitCast(rv[lane])),
                );
            }
        }
    }
}

test "recon: scheme radius matches stencil use" {
    try std.testing.expectEqual(@as(u8, 0), (Scheme{ .donor = {} }).radius());
    try std.testing.expectEqual(@as(u8, 1), (Scheme{ .linear = .{ .theta = 1.5 } }).radius());
    try std.testing.expectEqual(@as(u8, 2), (Scheme{ .ppm = {} }).radius());
}
