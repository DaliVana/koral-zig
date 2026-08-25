//! Lane helpers for `comptime T`-generic numeric kernels (parallelization
//! plan §2.2): kernels are written once over `T` = `f64` or
//! `@Vector(W, f64)`; the scalar instantiation stays the golden-tested
//! reference, the vector instantiation is the production path.
//!
//! Bit-identity contract (what makes vector lanes reproduce the scalar
//! chain bit-for-bit, gated in simd_tests.zig):
//!  * +,−,×,÷, @sqrt, @abs vectorize losslessly; IEEE-exact per lane;
//!  * transcendentals (pow/cbrt/log1p/log/exp) loop the lanes through the
//!    SAME scalar routine the f64 instantiation calls;
//!  * data-dependent branches become `select` of both-sides-computed
//!    values: the selected value is what the scalar branch computes, the
//!    discarded side must be pure (NaN/inf lanes included);
//!  * horizontal reductions are forbidden here (they reassociate); the
//!    kernels keep per-lane accumulation in scalar order.

const std = @import("std");

/// Number of lanes (1 for the scalar instantiation).
pub inline fn lanes(comptime T: type) comptime_int {
    return if (T == f64) 1 else @typeInfo(T).vector.len;
}

/// The comparison-result type of T (`a < b` on vectors is a bool vector).
pub fn Mask(comptime T: type) type {
    return if (T == f64) bool else @Vector(lanes(T), bool);
}

/// Broadcast a scalar to all lanes (identity for T == f64).
pub inline fn splat(comptime T: type, x: anytype) T {
    return if (comptime T == f64) x else @splat(x);
}

/// `if (cond) a else b`, per lane. Both sides are evaluated at the call
/// site: use only where the discarded side is side-effect-free.
pub inline fn select(comptime T: type, cond: Mask(T), a: T, b: T) T {
    return if (comptime T == f64) (if (cond) a else b) else @select(f64, cond, a, b);
}

/// std.math.isFinite per lane (|x| < inf: false for NaN and ±inf).
pub inline fn isFinite(comptime T: type, x: T) Mask(T) {
    return @abs(x) < splat(T, std.math.inf(f64));
}

/// std.math.pow(f64, x, y) per lane (y uniform across lanes).
pub inline fn pow(comptime T: type, x: T, y: f64) T {
    if (comptime T == f64) return std.math.pow(f64, x, y);
    var r: T = undefined;
    inline for (0..lanes(T)) |k| r[k] = std.math.pow(f64, x[k], y);
    return r;
}

/// std.math.cbrt per lane.
pub inline fn cbrt(comptime T: type, x: T) T {
    if (comptime T == f64) return std.math.cbrt(x);
    var r: T = undefined;
    inline for (0..lanes(T)) |k| r[k] = std.math.cbrt(x[k]);
    return r;
}

/// std.math.log1p per lane.
pub inline fn log1p(comptime T: type, x: T) T {
    if (comptime T == f64) return std.math.log1p(x);
    var r: T = undefined;
    inline for (0..lanes(T)) |k| r[k] = std.math.log1p(x[k]);
    return r;
}

/// @log per lane (kept per-lane rather than the vector builtin so the
/// libm dispatch is literally the scalar one).
pub inline fn log(comptime T: type, x: T) T {
    if (comptime T == f64) return @log(x);
    var r: T = undefined;
    inline for (0..lanes(T)) |k| r[k] = @log(x[k]);
    return r;
}

/// @exp per lane (same rationale as `log`).
pub inline fn exp(comptime T: type, x: T) T {
    if (comptime T == f64) return @exp(x);
    var r: T = undefined;
    inline for (0..lanes(T)) |k| r[k] = @exp(x[k]);
    return r;
}

/// Broadcast one of C's 4×5 metric blocks to T lanes.
pub fn splatBlock(comptime T: type, m: *const [4][5]f64) [4][5]T {
    var out: [4][5]T = undefined;
    for (0..4) |i| {
        for (0..5) |j| {
            out[i][j] = splat(T, m[i][j]);
        }
    }
    return out;
}

/// Extract one lane of a vector-typed aggregate into its scalar (f64)
/// counterpart, recursively: extractLane(RadState, stv, 2) turns a
/// RadStateOf(@Vector(4, f64)) into the lane-2 RadStateOf(f64).
pub fn extractLane(comptime S: type, v: anytype, lane: usize) S {
    const V = @TypeOf(v);
    if (comptime S == f64) {
        if (comptime V == f64) return v;
        const a: [lanes(V)]f64 = v;
        return a[lane];
    }
    switch (@typeInfo(S)) {
        .array => |info| {
            var out: S = undefined;
            for (0..info.len) |i| {
                out[i] = extractLane(info.child, v[i], lane);
            }
            return out;
        },
        .@"struct" => |info| {
            var out: S = undefined;
            inline for (info.fields) |fld| {
                @field(out, fld.name) = extractLane(fld.type, @field(v, fld.name), lane);
            }
            return out;
        },
        // non-lane field types (bools, enums, …) pass through unchanged
        else => return v,
    }
}

//
// ---- tests ---------------------------------------------------------------
//

const V4 = @Vector(4, f64);

fn expectBits(want: f64, got: f64) !void {
    try std.testing.expectEqual(
        @as(u64, @bitCast(want)),
        @as(u64, @bitCast(got)),
    );
}

test "simd: transcendentals are per-lane bit-identical to the scalar routine" {
    // spans normals, a subnormal, zero, negatives (pow/log domain errors),
    // and inf — every lane must reproduce the scalar call's bits
    const samples = [_][4]f64{
        .{ 1.0, 2.5, 1.0e-310, 0.0 },
        .{ -3.0, 7.25e8, 1.4142135623730951, std.math.inf(f64) },
        .{ 4.5e8, 1.0e-20, 3.5e33, 0.3 },
    };
    const exps = [_]f64{ 0.2, -0.463, 0.69, 1.69, 0.86, 2.0 };
    for (samples) |s| {
        const v: V4 = s;
        for (exps) |y| {
            const pv: [4]f64 = pow(V4, v, y);
            for (0..4) |k| try expectBits(pow(f64, s[k], y), pv[k]);
        }
        const cv: [4]f64 = cbrt(V4, v);
        const lv: [4]f64 = log1p(V4, v);
        const gv: [4]f64 = log(V4, v);
        const ev: [4]f64 = exp(V4, v);
        for (0..4) |k| {
            try expectBits(cbrt(f64, s[k]), cv[k]);
            try expectBits(log1p(f64, s[k]), lv[k]);
            try expectBits(log(f64, s[k]), gv[k]);
            try expectBits(exp(f64, s[k]), ev[k]);
        }
    }
}

test "simd: select/isFinite mirror the scalar branch, NaN and inf included" {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    const s = [4]f64{ 1.0, nan, inf, -2.0 };
    const v: V4 = s;

    const finv = isFinite(V4, v);
    for (0..4) |k| {
        const want = std.math.isFinite(s[k]);
        const fina: [4]bool = finv;
        try std.testing.expectEqual(want, fina[k]);
        try std.testing.expectEqual(want, isFinite(f64, s[k]));
    }

    // scalar semantics of `if (x < 0) 1 else sqrt(x)` under select with the
    // discarded side evaluated (sqrt of negative = NaN, must not leak)
    const r: [4]f64 = select(V4, v < splat(V4, 0.0), splat(V4, 1.0), @sqrt(v));
    for (0..4) |k| {
        const want = if (s[k] < 0.0) 1.0 else @sqrt(s[k]);
        try expectBits(want, r[k]);
    }
}

test "simd: extractLane walks structs, arrays, and vectors" {
    const Inner = struct { a: f64, b: [2]f64 };
    const InnerV = struct { a: V4, b: [2]V4 };
    const vv = InnerV{
        .a = .{ 1, 2, 3, 4 },
        .b = .{ .{ 10, 20, 30, 40 }, .{ 100, 200, 300, 400 } },
    };
    const lane2 = extractLane(Inner, vv, 2);
    try expectBits(3, lane2.a);
    try expectBits(30, lane2.b[0]);
    try expectBits(300, lane2.b[1]);
}
