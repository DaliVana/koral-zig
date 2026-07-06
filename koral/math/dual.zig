//! Forward-mode automatic differentiation over (x1,x2,x3).
//!
//! The metric is stationary (∂_t g = 0), so a scalar carries its value and
//! three spatial partials. Evaluating the analytic metric in `Dual3`
//! arithmetic yields ∂g/∂x^i exact to roundoff — the same mathematics as
//! KORAL's Mathematica-exported closed forms (metric.c `*_ana`), without
//! transcribing them. Christoffels, d(ln√−g)/dx^i, and Jacobian pushforwards
//! all fall out of the chain rule.

const std = @import("std");

pub const Dual3 = struct {
    v: f64,
    d: [3]f64 = .{ 0, 0, 0 },

    /// Constant (zero derivative).
    pub fn constant(x: f64) Dual3 {
        return .{ .v = x };
    }

    /// Independent variable seeded in derivative slot `slot` (0 ↔ x1 …).
    pub fn variable(x: f64, slot: usize) Dual3 {
        var r = Dual3{ .v = x };
        r.d[slot] = 1;
        return r;
    }

    pub fn add(a: Dual3, b: Dual3) Dual3 {
        return .{ .v = a.v + b.v, .d = .{ a.d[0] + b.d[0], a.d[1] + b.d[1], a.d[2] + b.d[2] } };
    }

    pub fn sub(a: Dual3, b: Dual3) Dual3 {
        return .{ .v = a.v - b.v, .d = .{ a.d[0] - b.d[0], a.d[1] - b.d[1], a.d[2] - b.d[2] } };
    }

    pub fn mul(a: Dual3, b: Dual3) Dual3 {
        return .{ .v = a.v * b.v, .d = .{
            a.d[0] * b.v + a.v * b.d[0],
            a.d[1] * b.v + a.v * b.d[1],
            a.d[2] * b.v + a.v * b.d[2],
        } };
    }

    pub fn div(a: Dual3, b: Dual3) Dual3 {
        const inv = 1.0 / b.v;
        const q = a.v * inv;
        return .{ .v = q, .d = .{
            (a.d[0] - q * b.d[0]) * inv,
            (a.d[1] - q * b.d[1]) * inv,
            (a.d[2] - q * b.d[2]) * inv,
        } };
    }

    pub fn neg(a: Dual3) Dual3 {
        return .{ .v = -a.v, .d = .{ -a.d[0], -a.d[1], -a.d[2] } };
    }

    /// Multiply by an f64 constant.
    pub fn scale(a: Dual3, c: f64) Dual3 {
        return .{ .v = a.v * c, .d = .{ a.d[0] * c, a.d[1] * c, a.d[2] * c } };
    }

    /// Add an f64 constant.
    pub fn addc(a: Dual3, c: f64) Dual3 {
        return .{ .v = a.v + c, .d = a.d };
    }

    pub fn sq(a: Dual3) Dual3 {
        return a.mul(a);
    }

    fn chain(a: Dual3, v: f64, dv: f64) Dual3 {
        return .{ .v = v, .d = .{ a.d[0] * dv, a.d[1] * dv, a.d[2] * dv } };
    }

    pub fn sin(a: Dual3) Dual3 {
        return chain(a, @sin(a.v), @cos(a.v));
    }

    pub fn cos(a: Dual3) Dual3 {
        return chain(a, @cos(a.v), -@sin(a.v));
    }

    pub fn tan(a: Dual3) Dual3 {
        const t = @tan(a.v);
        return chain(a, t, 1.0 + t * t);
    }

    pub fn exp(a: Dual3) Dual3 {
        const e = @exp(a.v);
        return chain(a, e, e);
    }

    pub fn log(a: Dual3) Dual3 {
        return chain(a, @log(a.v), 1.0 / a.v);
    }

    pub fn sqrt(a: Dual3) Dual3 {
        const s = @sqrt(a.v);
        return chain(a, s, 0.5 / s);
    }

    pub fn atan(a: Dual3) Dual3 {
        return chain(a, std.math.atan(a.v), 1.0 / (1.0 + a.v * a.v));
    }
};

//
// ---- tests ----------------------------------------------------------------
//

fn fdCheck(comptime f: fn (Dual3) Dual3, x: f64, tol: f64) !void {
    // Richardson-extrapolated central difference vs the dual derivative.
    const g = f(Dual3.variable(x, 1));
    const h = 1e-5 * @max(1.0, @abs(x));
    const d1 = (f(Dual3.constant(x + h)).v - f(Dual3.constant(x - h)).v) / (2 * h);
    const d2 = (f(Dual3.constant(x + h / 2)).v - f(Dual3.constant(x - h / 2)).v) / h;
    const fd = (4.0 * d2 - d1) / 3.0;
    try std.testing.expectApproxEqRel(fd, g.d[1], tol);
}

test "dual: derivatives of a composite expression match Richardson FD" {
    const F = struct {
        fn f(x: Dual3) Dual3 {
            // f(x) = sin(x)·exp(x/3) / (1 + x²) + atan(tan(x/4))
            const t1 = x.sin().mul(x.scale(1.0 / 3.0).exp());
            const t2 = Dual3.constant(1).add(x.sq());
            return t1.div(t2).add(x.scale(0.25).tan().atan());
        }
    };
    for ([_]f64{ 0.3, 1.1, 2.7, -0.8 }) |x| {
        try fdCheck(F.f, x, 1e-9);
    }
}

test "dual: exact rules on simple functions" {
    const x = Dual3.variable(2.0, 0);
    const y = x.sq().mul(x); // x^3, d = 3x^2 = 12
    try std.testing.expectApproxEqRel(@as(f64, 8.0), y.v, 1e-15);
    try std.testing.expectApproxEqRel(@as(f64, 12.0), y.d[0], 1e-15);
    const z = x.sqrt(); // d = 1/(2√2)
    try std.testing.expectApproxEqRel(1.0 / (2.0 * @sqrt(2.0)), z.d[0], 1e-15);
    const w = Dual3.constant(5.0).div(x); // d = -5/4
    try std.testing.expectApproxEqRel(@as(f64, -1.25), w.d[0], 1e-15);
}
