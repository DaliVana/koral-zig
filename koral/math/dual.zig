//! Forward-mode automatic differentiation over N independent variables.
//!
//! `Dual(N)` carries a value and N partial derivatives; evaluating a closed
//! form in dual arithmetic yields its gradient exact to roundoff via the
//! chain rule. Each derivative slot is computed by the same expression the
//! scalar (unrolled) version would use, so widening N never changes the
//! floating-point chain of existing slots.
//!
//! `Dual3` is the (x1,x2,x3) instantiation used by the metric: the metric is
//! stationary (∂_t g = 0), so a scalar carries its value and three spatial
//! partials. Evaluating the analytic metric in `Dual3` arithmetic yields
//! ∂g/∂x^i exact to roundoff; the same mathematics as KORAL's
//! Mathematica-exported closed forms (metric.c `*_ana`), without
//! transcribing them. Christoffels, d(ln√−g)/dx^i, and Jacobian pushforwards
//! all fall out of the chain rule.

const std = @import("std");

pub fn Dual(comptime N: usize) type {
    return struct {
        const Self = @This();

        v: f64,
        d: [N]f64 = @splat(0),

        /// Constant (zero derivative).
        pub fn constant(x: f64) Self {
            return .{ .v = x };
        }

        /// Independent variable seeded in derivative slot `slot` (0 ↔ x1 …).
        pub fn variable(x: f64, slot: usize) Self {
            var r = Self{ .v = x };
            r.d[slot] = 1;
            return r;
        }

        pub fn add(a: Self, b: Self) Self {
            var r = Self{ .v = a.v + b.v };
            inline for (0..N) |i| r.d[i] = a.d[i] + b.d[i];
            return r;
        }

        pub fn sub(a: Self, b: Self) Self {
            var r = Self{ .v = a.v - b.v };
            inline for (0..N) |i| r.d[i] = a.d[i] - b.d[i];
            return r;
        }

        pub fn mul(a: Self, b: Self) Self {
            var r = Self{ .v = a.v * b.v };
            inline for (0..N) |i| r.d[i] = a.d[i] * b.v + a.v * b.d[i];
            return r;
        }

        pub fn div(a: Self, b: Self) Self {
            const inv = 1.0 / b.v;
            const q = a.v * inv;
            var r = Self{ .v = q };
            inline for (0..N) |i| r.d[i] = (a.d[i] - q * b.d[i]) * inv;
            return r;
        }

        pub fn neg(a: Self) Self {
            var r = Self{ .v = -a.v };
            inline for (0..N) |i| r.d[i] = -a.d[i];
            return r;
        }

        /// Multiply by an f64 constant.
        pub fn scale(a: Self, c: f64) Self {
            var r = Self{ .v = a.v * c };
            inline for (0..N) |i| r.d[i] = a.d[i] * c;
            return r;
        }

        /// Add an f64 constant.
        pub fn addc(a: Self, c: f64) Self {
            return .{ .v = a.v + c, .d = a.d };
        }

        pub fn sq(a: Self) Self {
            return a.mul(a);
        }

        fn chain(a: Self, v: f64, dv: f64) Self {
            var r = Self{ .v = v };
            inline for (0..N) |i| r.d[i] = a.d[i] * dv;
            return r;
        }

        pub fn sin(a: Self) Self {
            return chain(a, @sin(a.v), @cos(a.v));
        }

        pub fn cos(a: Self) Self {
            return chain(a, @cos(a.v), -@sin(a.v));
        }

        pub fn tan(a: Self) Self {
            const t = @tan(a.v);
            return chain(a, t, 1.0 + t * t);
        }

        pub fn exp(a: Self) Self {
            const e = @exp(a.v);
            return chain(a, e, e);
        }

        pub fn log(a: Self) Self {
            return chain(a, @log(a.v), 1.0 / a.v);
        }

        pub fn sqrt(a: Self) Self {
            const s = @sqrt(a.v);
            return chain(a, s, 0.5 / s);
        }

        pub fn atan(a: Self) Self {
            return chain(a, std.math.atan(a.v), 1.0 / (1.0 + a.v * a.v));
        }
    };
}

pub const Dual3 = Dual(3);

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

test "dual: Dual(N) slots are independent of N and of each other" {
    // The same expression through Dual(1), Dual(3), and Dual(6) must produce
    // bit-identical value and per-slot derivatives — widening N only adds
    // slots, it never perturbs existing ones.
    const expr = struct {
        fn f(comptime D: type, x: D, y: D) D {
            return x.sq().mul(y.sin()).div(y.addc(2.0)).add(x.log());
        }
    };
    const xv = 1.7;
    const yv = 0.6;

    const r1x = expr.f(Dual(1), Dual(1).variable(xv, 0), Dual(1).constant(yv));
    const r1y = expr.f(Dual(1), Dual(1).constant(xv), Dual(1).variable(yv, 0));
    const r3 = expr.f(Dual3, Dual3.variable(xv, 0), Dual3.variable(yv, 2));
    const r6 = expr.f(Dual(6), Dual(6).variable(xv, 1), Dual(6).variable(yv, 5));

    const vbits: u64 = @bitCast(r1x.v);
    try std.testing.expectEqual(vbits, @as(u64, @bitCast(r3.v)));
    try std.testing.expectEqual(vbits, @as(u64, @bitCast(r6.v)));

    const dxbits: u64 = @bitCast(r1x.d[0]);
    try std.testing.expectEqual(dxbits, @as(u64, @bitCast(r3.d[0])));
    try std.testing.expectEqual(dxbits, @as(u64, @bitCast(r6.d[1])));

    const dybits: u64 = @bitCast(r1y.d[0]);
    try std.testing.expectEqual(dybits, @as(u64, @bitCast(r3.d[2])));
    try std.testing.expectEqual(dybits, @as(u64, @bitCast(r6.d[5])));

    // untouched slots stay exactly zero
    try std.testing.expectEqual(@as(f64, 0.0), r3.d[1]);
    try std.testing.expectEqual(@as(f64, 0.0), r6.d[0]);
}
