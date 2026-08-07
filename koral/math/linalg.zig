//! Dense 4×4 determinant and inverse by cofactor expansion, generic over any
//! `Dual(N)` scalar. There is exactly one arithmetic implementation — the
//! dual methods — so every instantiation shares the same floating-point
//! chain by construction: `det4(Dual(0), m).v` is bit-identical to the value
//! slot of the `Dual(3)` metric path.
//!
//! For plain-f64 matrices, instantiate at `Dual(0)`: the derivative slots
//! are `[0]f64`, so all derivative work compiles away and only the value
//! chain remains.
//!
//! Cofactor expansion has no pivoting — fine for well-conditioned matrices
//! like metric tensors; use an LU route for anything near-singular.

const std = @import("std");

/// 3×3 minor of a 4×4: determinant of the submatrix picked out by row
/// indices `r` and column indices `c`.
fn minor3(comptime T: type, m: [4][4]T, r: [3]usize, c: [3]usize) T {
    const t0 = m[r[0]][c[0]].mul(m[r[1]][c[1]].mul(m[r[2]][c[2]]).sub(m[r[1]][c[2]].mul(m[r[2]][c[1]])));
    const t1 = m[r[0]][c[1]].mul(m[r[1]][c[0]].mul(m[r[2]][c[2]]).sub(m[r[1]][c[2]].mul(m[r[2]][c[0]])));
    const t2 = m[r[0]][c[2]].mul(m[r[1]][c[0]].mul(m[r[2]][c[1]]).sub(m[r[1]][c[1]].mul(m[r[2]][c[0]])));
    return t0.sub(t1).add(t2);
}

/// Complementary index triples: `complement[i]` are the three indices ≠ i.
const complement = [4][3]usize{
    .{ 1, 2, 3 },
    .{ 0, 2, 3 },
    .{ 0, 1, 3 },
    .{ 0, 1, 2 },
};

/// Determinant by cofactor expansion along row 0.
pub fn det4(comptime T: type, m: [4][4]T) T {
    var d = T.constant(0);
    var sign: f64 = 1.0;
    for (0..4) |j| {
        const minor = minor3(T, m, complement[0], complement[j]);
        d = d.add(m[0][j].mul(minor).scale(sign));
        sign = -sign;
    }
    return d;
}

/// Inverse via the adjugate: out[i][j] = cofactor C_ji / det. Valid for any
/// invertible 4×4 (the adjugate transpose is built in by the index swap).
pub fn inv4(comptime T: type, m: [4][4]T, det: T) [4][4]T {
    var out: [4][4]T = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            const minor = minor3(T, m, complement[j], complement[i]);
            const sign: f64 = if ((i + j) % 2 == 0) 1.0 else -1.0;
            out[i][j] = minor.scale(sign).div(det);
        }
    }
    return out;
}

//
// ---- tests ----------------------------------------------------------------
//

const dual = @import("dual.zig");
const Dual0 = dual.Dual(0);
const Dual3 = dual.Dual3;

fn wrap0(m: [4][4]f64) [4][4]Dual0 {
    var out: [4][4]Dual0 = undefined;
    for (0..4) |i| {
        for (0..4) |j| out[i][j] = Dual0.constant(m[i][j]);
    }
    return out;
}

test "linalg: det4/inv4 on Minkowski via Dual(0)" {
    var eta: [4][4]f64 = @splat(@splat(0));
    eta[0][0] = -1.0;
    for (1..4) |i| eta[i][i] = 1.0;
    const m = wrap0(eta);
    const det = det4(Dual0, m);
    try std.testing.expectEqual(@as(f64, -1.0), det.v);
    const inv = inv4(Dual0, m, det);
    for (0..4) |i| {
        for (0..4) |j| try std.testing.expectEqual(eta[i][j], inv[i][j].v);
    }
}

test "linalg: inv4 inverts a non-symmetric matrix" {
    const m = [4][4]f64{
        .{ 2.0, 1.0, 0.5, -1.0 },
        .{ 0.0, 3.0, 1.0, 2.0 },
        .{ -1.0, 0.5, 4.0, 0.0 },
        .{ 1.0, -2.0, 0.0, 5.0 },
    };
    const md = wrap0(m);
    const inv = inv4(Dual0, md, det4(Dual0, md));
    for (0..4) |i| {
        for (0..4) |j| {
            var s: f64 = 0;
            for (0..4) |k| s += m[i][k] * inv[k][j].v;
            const want: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(want, s, 1e-14);
        }
    }
}

test "linalg: Dual(0) and Dual(3) value slots agree bit-for-bit" {
    // a symmetric matrix with metric-like magnitudes
    const m = [4][4]f64{
        .{ -0.731, 0.269, 0.0, -0.42 },
        .{ 0.269, 1.269, 0.0, 0.83 },
        .{ 0.0, 0.0, 7.113, 0.0 },
        .{ -0.42, 0.83, 0.0, 10.041 },
    };
    const m0 = wrap0(m);
    var m3: [4][4]Dual3 = undefined;
    for (0..4) |i| {
        for (0..4) |j| m3[i][j] = Dual3.constant(m[i][j]);
    }
    const det0 = det4(Dual0, m0);
    const det3v = det4(Dual3, m3);
    try std.testing.expectEqual(@as(u64, @bitCast(det0.v)), @as(u64, @bitCast(det3v.v)));
    const inv0 = inv4(Dual0, m0, det0);
    const inv3 = inv4(Dual3, m3, det3v);
    for (0..4) |i| {
        for (0..4) |j| {
            try std.testing.expectEqual(
                @as(u64, @bitCast(inv0[i][j].v)),
                @as(u64, @bitCast(inv3[i][j].v)),
            );
        }
    }
}

test "linalg: det4 derivative through dual arithmetic is exact" {
    // det diag(-1, x, 1, 1) = -x ⇒ d(det)/dx = -1 exactly
    var m: [4][4]Dual3 = @splat(@splat(Dual3.constant(0)));
    m[0][0] = Dual3.constant(-1.0);
    m[1][1] = Dual3.variable(1.7, 0);
    m[2][2] = Dual3.constant(1.0);
    m[3][3] = Dual3.constant(1.0);
    const det = det4(Dual3, m);
    try std.testing.expectEqual(@as(f64, -1.7), det.v);
    try std.testing.expectEqual(@as(f64, -1.0), det.d[0]);
}
