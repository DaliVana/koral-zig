//! Adaptive Gauss–Kronrod 21-point quadrature.
//!
//! Replaces the C side's `gsl_integration_qags` (PROBLEMS/PUFFY/tools.c:276,
//! epsrel 1e-8) for the limotorus ln f integral. GSL's qags is GK21 +
//! worst-first interval subdivision + Wynn ε-extrapolation; the integrands we
//! feed it (smooth up to two kinks at the angular-momentum breaks) need no
//! extrapolation, so this is plain GK21 with worst-first bisection run to a
//! tighter tolerance (callers use rtol 1e-12 where C used 1e-8). Agreement
//! with C is then bounded by C's own epsrel — pinned by the epsrel-1e-12
//! oracle variant of the PUFFY t=0 golden.
//!
//! Nodes/weights are QUADPACK's dqk21 constants. The error estimate is the
//! plain |GK21 − G10| difference — strictly more conservative than
//! QUADPACK's rescaled estimate, which only ever shrinks it.

const std = @import("std");

/// Positive-half Kronrod abscissae; even indices are Kronrod-only points,
/// odd indices are the embedded 10-point Gauss nodes, index 10 is x = 0.
const xgk = [11]f64{
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.000000000000000000000000000000000,
};

const wgk = [11]f64{
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
};

/// Weights of the embedded 10-point Gauss rule (nodes xgk[1,3,5,7,9]).
const wg = [5]f64{
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
};

pub const Panel = struct {
    value: f64,
    /// |GK21 − G10| · |half-width| — conservative error estimate.
    abserr: f64,
};

/// One GK21 panel over [a, b]. `f` is anything with `eval(self, x: f64) f64`.
pub fn gk21(f: anytype, a: f64, b: f64) Panel {
    const c = 0.5 * (a + b);
    const h = 0.5 * (b - a);
    const fc = f.eval(c);
    var resg: f64 = 0.0;
    var resk: f64 = wgk[10] * fc;
    var j: usize = 0;
    while (j < 5) : (j += 1) {
        const jtw = 2 * j + 1; // Gauss nodes
        const x = h * xgk[jtw];
        const fsum = f.eval(c - x) + f.eval(c + x);
        resg += wg[j] * fsum;
        resk += wgk[jtw] * fsum;
    }
    j = 0;
    while (j < 5) : (j += 1) {
        const jtw = 2 * j; // Kronrod-only nodes
        const x = h * xgk[jtw];
        resk += wgk[jtw] * (f.eval(c - x) + f.eval(c + x));
    }
    return .{
        .value = resk * h,
        .abserr = @abs((resk - resg) * h),
    };
}

pub const QuadResult = struct {
    value: f64,
    abserr: f64,
    /// false when max_segments was exhausted before the tolerance was met —
    /// callers mirror C's GSL-failure branch in that case.
    converged: bool,
};

const max_segments = 512;

const Segment = struct { a: f64, b: f64, value: f64, abserr: f64 };

/// Adaptive GK21 over [a, b] (either orientation) to
/// max(epsabs, epsrel·|I|), splitting the worst segment first — the same
/// strategy qags uses, minus the extrapolation stage.
pub fn integrate(f: anytype, a: f64, b: f64, epsabs: f64, epsrel: f64) QuadResult {
    if (a == b) return .{ .value = 0.0, .abserr = 0.0, .converged = true };

    var segs: [max_segments]Segment = undefined;
    var nseg: usize = 1;
    const p0 = gk21(f, a, b);
    segs[0] = .{ .a = a, .b = b, .value = p0.value, .abserr = p0.abserr };

    while (true) {
        var total: f64 = 0.0;
        var toterr: f64 = 0.0;
        var worst: usize = 0;
        for (segs[0..nseg], 0..) |s, i| {
            total += s.value;
            toterr += s.abserr;
            if (s.abserr > segs[worst].abserr) worst = i;
        }
        if (toterr <= @max(epsabs, epsrel * @abs(total)))
            return .{ .value = total, .abserr = toterr, .converged = true };
        if (nseg == max_segments)
            return .{ .value = total, .abserr = toterr, .converged = false };

        const s = segs[worst];
        const mid = 0.5 * (s.a + s.b);
        const left = gk21(f, s.a, mid);
        const right = gk21(f, mid, s.b);
        segs[worst] = .{ .a = s.a, .b = mid, .value = left.value, .abserr = left.abserr };
        segs[nseg] = .{ .a = mid, .b = s.b, .value = right.value, .abserr = right.abserr };
        nseg += 1;
    }
}

// ---------------------------------------------------------------------------
// tests

const Poly = struct {
    // f(x) = x^n
    n: u32,
    pub fn eval(self: Poly, x: f64) f64 {
        return std.math.pow(f64, x, @floatFromInt(self.n));
    }
};

test "single GK21 panel integrates degree-31 polynomials exactly (pins nodes+weights)" {
    // Kronrod-21 is exact through degree 31; x^31 on [0,1] = 1/32.
    var n: u32 = 0;
    while (n <= 31) : (n += 1) {
        const p = gk21(Poly{ .n = n }, 0.0, 1.0);
        const exact = 1.0 / @as(f64, @floatFromInt(n + 1));
        try std.testing.expect(@abs(p.value - exact) <= 1e-15 * exact + 1e-16);
    }
}

test "embedded G10 exact through degree 19: panel error estimate vanishes" {
    // For deg ≤ 19 both rules are exact, so |GK−G| must be at rounding level.
    var n: u32 = 0;
    while (n <= 19) : (n += 1) {
        const p = gk21(Poly{ .n = n }, 0.0, 1.0);
        try std.testing.expect(p.abserr <= 1e-14);
    }
}

const Runge = struct {
    pub fn eval(_: Runge, x: f64) f64 {
        return 1.0 / (1.0 + 25.0 * x * x);
    }
};

const ExpF = struct {
    pub fn eval(_: ExpF, x: f64) f64 {
        return @exp(x);
    }
};

const Kinked = struct {
    // |x - 1/3|: C1-discontinuous inside the interval, like l3d at a break.
    pub fn eval(_: Kinked, x: f64) f64 {
        return @abs(x - 1.0 / 3.0);
    }
};

test "adaptive GK21 closed forms at 1e-12" {
    // ∫_{-1}^{1} dx/(1+25x²) = (2/5)·atan(5)
    const r1 = integrate(Runge{}, -1.0, 1.0, 0.0, 1e-12);
    try std.testing.expect(r1.converged);
    const exact1 = 0.4 * std.math.atan(@as(f64, 5.0));
    try std.testing.expect(@abs(r1.value - exact1) <= 1e-12 * exact1);

    // ∫_0^{10} e^x dx = e^10 − 1
    const r2 = integrate(ExpF{}, 0.0, 10.0, 0.0, 1e-12);
    try std.testing.expect(r2.converged);
    const exact2 = @exp(@as(f64, 10.0)) - 1.0;
    try std.testing.expect(@abs(r2.value - exact2) <= 1e-12 * exact2);

    // kink inside the interval: ∫_0^1 |x−1/3| dx = 5/18
    const r3 = integrate(Kinked{}, 0.0, 1.0, 0.0, 1e-12);
    try std.testing.expect(r3.converged);
    try std.testing.expect(@abs(r3.value - 5.0 / 18.0) <= 1e-12);

    // reversed orientation flips the sign
    const r4 = integrate(ExpF{}, 10.0, 0.0, 0.0, 1e-12);
    try std.testing.expect(@abs(r4.value + exact2) <= 1e-12 * exact2);
}
