//! Metric assembly: from the Dual3 covariant metric of `forms.zig`, produce
//! everything the solver caches per point — g_μν, g^μν, √−g, d(ln√−g)/dx^i,
//! Christoffels Γ^λ_μν, and the g_tt perturbation (C: calc_g/calc_G/
//! calc_gdet/calc_dlgdet/calc_Krzysie/calc_gttpert in metric.c).
//!
//! Derivatives come from the dual numbers (exact to roundoff); the inverse
//! and determinant are computed by cofactor expansion of the 4×4.

const std = @import("std");
const Dual3 = @import("../math/dual.zig").Dual3;
const forms = @import("forms.zig");
const config = @import("../config.zig");

pub const Coords = config.Coords;
pub const MetricParams = forms.MetricParams;

/// C: r_horizon_BL (metric.c:245) — the outer Kerr event horizon in
/// Boyer–Lindquist radius, 1 + √(1−a²) (→ 2 for a = 0).
pub fn rHorizonBL(a: f64) f64 {
    return 1.0 + @sqrt(1.0 - a * a);
}

/// C: r_ISCO_BL (metric.c:252) — the prograde ISCO in Boyer–Lindquist
/// radius, in the literal pow(·,1/3) shape (→ 6 for a = 0).
pub fn rIscoBL(ac: f64) f64 {
    const c3 = 1.0 / 3.0;
    const z1 = 1.0 + std.math.pow(f64, 1.0 - ac * ac, c3) *
        (std.math.pow(f64, 1.0 + ac, c3) + std.math.pow(f64, 1.0 - ac, c3));
    const z2 = std.math.pow(f64, 3.0 * ac * ac + z1 * z1, 1.0 / 2.0);
    return 3.0 + z2 - std.math.pow(f64, (3.0 - z1) * (3.0 + z1 + 2.0 * z2), 1.0 / 2.0);
}

/// Everything known about the metric at one point.
pub const CoordData = struct {
    gcov: [4][4]f64,
    gcon: [4][4]f64,
    /// √−g
    gdet: f64,
    /// ∂ ln √−g / ∂x^i, i = 1..3
    dlgdet: [3]f64,
    /// Γ^i_jk (C: Krzysie)
    kris: [4][4][4]f64,
    /// perturbed part of g_tt: 2r/Σ for Kerr-based coords (C: calc_gttpert)
    gttpert: f64,
};

pub fn gcovDual(coords: Coords, mp: MetricParams, x: [4]f64) [4][4]Dual3 {
    const xd = [4]Dual3{
        Dual3.constant(x[0]),
        Dual3.variable(x[1], 0),
        Dual3.variable(x[2], 1),
        Dual3.variable(x[3], 2),
    };
    return switch (coords) {
        .mink => forms.gcovMink(xd, mp),
        .bl => forms.gcovBl(xd, mp),
        .ks => forms.gcovKs(xd, mp),
        .mks2 => forms.gcovMks2(xd, mp),
    };
}

fn det3(m: [4][4]Dual3, r: [3]usize, c: [3]usize) Dual3 {
    const t0 = m[r[0]][c[0]].mul(m[r[1]][c[1]].mul(m[r[2]][c[2]]).sub(m[r[1]][c[2]].mul(m[r[2]][c[1]])));
    const t1 = m[r[0]][c[1]].mul(m[r[1]][c[0]].mul(m[r[2]][c[2]]).sub(m[r[1]][c[2]].mul(m[r[2]][c[0]])));
    const t2 = m[r[0]][c[2]].mul(m[r[1]][c[0]].mul(m[r[2]][c[1]]).sub(m[r[1]][c[1]].mul(m[r[2]][c[0]])));
    return t0.sub(t1).add(t2);
}

const others = [4][3]usize{
    .{ 1, 2, 3 },
    .{ 0, 2, 3 },
    .{ 0, 1, 3 },
    .{ 0, 1, 2 },
};

fn det4(m: [4][4]Dual3) Dual3 {
    var d = Dual3.constant(0);
    var sign: f64 = 1.0;
    for (0..4) |j| {
        const minor = det3(m, others[0], others[j]);
        d = d.add(m[0][j].mul(minor).scale(sign));
        sign = -sign;
    }
    return d;
}

/// Inverse of a symmetric 4×4 in dual arithmetic (adjugate / det).
fn inv4(m: [4][4]Dual3, det: Dual3) [4][4]Dual3 {
    var out: [4][4]Dual3 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            // cofactor C_ji / det (matrix symmetric ⇒ inverse symmetric)
            const minor = det3(m, others[j], others[i]);
            const sign: f64 = if ((i + j) % 2 == 0) 1.0 else -1.0;
            out[i][j] = minor.scale(sign).div(det);
        }
    }
    return out;
}

/// Full metric data at one point (C: the per-cell part of calc_metric).
pub fn compute(coords: Coords, mp: MetricParams, x: [4]f64) CoordData {
    const g = gcovDual(coords, mp, x);
    const det = det4(g);
    const ginv = inv4(g, det);

    var out: CoordData = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            out.gcov[i][j] = g[i][j].v;
            out.gcon[i][j] = ginv[i][j].v;
        }
    }

    out.gdet = @sqrt(-det.v);
    // ln √−g = ½ ln(−det) ⇒ d = ½ det'/det
    for (0..3) |i| out.dlgdet[i] = 0.5 * det.d[i] / det.v;

    // Γ^i_jk = ½ g^il (∂_j g_lk + ∂_k g_lj − ∂_l g_jk); ∂_0 ≡ 0 (stationary).
    for (0..4) |i| {
        for (0..4) |j| {
            for (j..4) |k| {
                var s: f64 = 0;
                for (0..4) |l| {
                    const t = dpart(g, l, k, j) + dpart(g, l, j, k) - dpart(g, j, k, l);
                    s += out.gcon[i][l] * t;
                }
                out.kris[i][j][k] = 0.5 * s;
                out.kris[i][k][j] = 0.5 * s;
            }
        }
    }

    out.gttpert = gttpert(coords, mp, x);
    return out;
}

/// ∂_m g_ab from the dual derivatives (∂_t = 0).
inline fn dpart(g: [4][4]Dual3, a: usize, b: usize, m: usize) f64 {
    return if (m == 0) 0.0 else g[a][b].d[m - 1];
}

/// C: calc_gttpert_arb (metric.c:4391). For all Kerr-based coordinates with
/// a diagonal map to KS/BL (our ks, bl, mks2) the transformation collapses
/// to the base perturbation 2r/Σ; Minkowski is 0.
pub fn gttpert(coords: Coords, mp: MetricParams, x: [4]f64) f64 {
    switch (coords) {
        .mink => return 0.0,
        .bl, .ks => {
            const r = x[1];
            const c = @cos(x[2]);
            return 2.0 * r / (r * r + mp.a * mp.a * c * c);
        },
        .mks2 => {
            const r = @exp(x[1]) + mp.mksr0;
            const th = forms.mks2Theta(Dual3.constant(x[2]), mp.mksh0).v;
            const c = @cos(th);
            return 2.0 * r / (r * r + mp.a * mp.a * c * c);
        },
    }
}
