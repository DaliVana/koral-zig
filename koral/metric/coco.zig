//! Coordinate point transforms (C: coco_N, metric.c:2079) and analytic
//! Jacobians ∂x'^μ/∂x^ν (C: calc_dxdx_arb, metric.c:3253) between
//! BL ↔ KS ↔ MKS2. Routing goes through KS as the hub, exactly like C.
//!
//! Faithful C quirks preserved:
//! - coco BL↔KS shifts *t* only; the φ shift is omitted (C leaves it as a
//!   TODO), while dxdx_BL2KS *does* carry the a/Δ φ-column. Tensors are
//!   unaffected (axisymmetric stationary metric).
//! - The t-shift is undefined (NaN) between the horizons where Δ < 0 —
//!   same as C; spatial components are valid everywhere.

const std = @import("std");
const forms = @import("forms.zig");
const config = @import("../config.zig");

pub const Coords = config.Coords;
pub const MetricParams = forms.MetricParams;

const pi = std.math.pi;

fn blKsTimeShift(r: f64, a: f64) f64 {
    // 2/√(1−a²) · atanh(√(1−a²)/(1−r)) + ln Δ   (C: coco_BL2KS)
    const delta = r * r - 2.0 * r + a * a;
    const sqrta = @sqrt(1.0 - a * a);
    return 2.0 / sqrta * std.math.atanh(sqrta / (1.0 - r)) + @log(delta);
}

pub fn blToKs(x: [4]f64, mp: MetricParams) [4]f64 {
    return .{ x[0] + blKsTimeShift(x[1], mp.a), x[1], x[2], x[3] };
}

pub fn ksToBl(x: [4]f64, mp: MetricParams) [4]f64 {
    return .{ x[0] - blKsTimeShift(x[1], mp.a), x[1], x[2], x[3] };
}

pub fn mks2ToKs(x: [4]f64, mp: MetricParams) [4]f64 {
    const Dual3 = @import("../math/dual.zig").Dual3;
    return .{
        x[0],
        @exp(x[1]) + mp.mksr0,
        forms.mks2Theta(Dual3.constant(x[2]), mp.mksh0).v,
        x[3],
    };
}

pub fn ksToMks2(x: [4]f64, mp: MetricParams) [4]f64 {
    return .{
        x[0],
        @log(x[1] - mp.mksr0),
        forms.mks2X2FromTheta(x[2], mp.mksh0),
        x[3],
    };
}

/// General point transform, KS-hub routing as in C's coco_N.
pub fn cocoN(x: [4]f64, from: Coords, to: Coords, mp: MetricParams) [4]f64 {
    if (from == to) return x;
    return switch (from) {
        .bl => switch (to) {
            .ks => blToKs(x, mp),
            .mks2 => ksToMks2(blToKs(x, mp), mp),
            else => unsupported(from, to),
        },
        .ks => switch (to) {
            .bl => ksToBl(x, mp),
            .mks2 => ksToMks2(x, mp),
            else => unsupported(from, to),
        },
        .mks2 => switch (to) {
            .ks => mks2ToKs(x, mp),
            .bl => ksToBl(mks2ToKs(x, mp), mp),
            else => unsupported(from, to),
        },
        .mink => unsupported(from, to),
    };
}

fn unsupported(from: Coords, to: Coords) [4]f64 {
    std.debug.panic("coco: unsupported transform {s} -> {s}", .{ @tagName(from), @tagName(to) });
}

fn eye() [4][4]f64 {
    var m: [4][4]f64 = @splat(@splat(0));
    for (0..4) |i| m[i][i] = 1;
    return m;
}

/// C: dxdx_BL2KS (metric.c:3435) — includes the a/Δ φ-column.
pub fn dxdxBlToKs(x: [4]f64, mp: MetricParams) [4][4]f64 {
    const r = x[1];
    const delta = r * r - 2.0 * r + mp.a * mp.a;
    var m = eye();
    m[0][1] = 2.0 * r / delta;
    m[3][1] = mp.a / delta;
    return m;
}

/// C: dxdx_KS2BL (metric.c:3458).
pub fn dxdxKsToBl(x: [4]f64, mp: MetricParams) [4][4]f64 {
    const r = x[1];
    const delta = r * r - 2.0 * r + mp.a * mp.a;
    var m = eye();
    m[0][1] = -2.0 * r / delta;
    m[3][1] = -mp.a / delta;
    return m;
}

/// C: dxdx_MKS22KS (metric.c:3689) — metric flavor using the same exact π
/// as the coco transform. Input x in MKS2.
pub fn dxdxMks2ToKs(x: [4]f64, mp: MetricParams) [4][4]f64 {
    const Dual3 = @import("../math/dual.zig").Dual3;
    var m = eye();
    m[1][1] = @exp(x[1]);
    m[2][2] = forms.mks2DThetaDx2Metric(Dual3.constant(x[2]), mp.mksh0).v;
    return m;
}

/// C: dxdx_KS2MKS2 (metric.c:3638) — unified exact π in both the transform
/// and its Jacobian. Input x in KS.
pub fn dxdxKsToMks2(x: [4]f64, mp: MetricParams) [4][4]f64 {
    const h0 = mp.mksh0;
    var m = eye();
    m[1][1] = 1.0 / (x[1] - mp.mksr0);
    // dx2/dθ = 2 tan(πH0/2) / (H0 Pi² (1 + v²)),  v = (2/π)(θ−π/2)·tan(πH0/2)
    const v = (2.0 / pi) * (x[2] - 0.5 * pi) * @tan(0.5 * pi * h0);
    m[2][2] = 2.0 * @tan(0.5 * pi * h0) / (h0 * pi * pi * (1.0 + v * v));
    return m;
}

fn matmul(a: [4][4]f64, b: [4][4]f64) [4][4]f64 {
    var c: [4][4]f64 = @splat(@splat(0));
    for (0..4) |i| {
        for (0..4) |j| {
            var s: f64 = 0;
            for (0..4) |k| s += a[i][k] * b[k][j];
            c[i][j] = s;
        }
    }
    return c;
}

/// Jacobian dx_to/dx_from at point x (given in `from` coordinates).
/// Composite pairs chain through KS: J = J2(coco(x)) · J1(x).
pub fn dxdx(x: [4]f64, from: Coords, to: Coords, mp: MetricParams) [4][4]f64 {
    if (from == to) return eye();
    return switch (from) {
        .bl => switch (to) {
            .ks => dxdxBlToKs(x, mp),
            .mks2 => matmul(dxdxKsToMks2(blToKs(x, mp), mp), dxdxBlToKs(x, mp)),
            else => unsupportedJ(from, to),
        },
        .ks => switch (to) {
            .bl => dxdxKsToBl(x, mp),
            .mks2 => dxdxKsToMks2(x, mp),
            else => unsupportedJ(from, to),
        },
        .mks2 => switch (to) {
            .ks => dxdxMks2ToKs(x, mp),
            .bl => matmul(dxdxKsToBl(mks2ToKs(x, mp), mp), dxdxMks2ToKs(x, mp)),
            else => unsupportedJ(from, to),
        },
        .mink => unsupportedJ(from, to),
    };
}

fn unsupportedJ(from: Coords, to: Coords) [4][4]f64 {
    std.debug.panic("dxdx: unsupported transform {s} -> {s}", .{ @tagName(from), @tagName(to) });
}
