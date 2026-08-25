//! Covariant metric forms g_μν per coordinate system, in Dual3 arithmetic.
//!
//! Base forms transcribed from KORAL's calc_g_arb_ana (metric.c:534):
//! MINKCOORDS, KERRCOORDS (Boyer-Lindquist), KSCOORDS (Kerr-Schild), in
//! G = c = M = 1 units. MKS2 is *not* transcribed (the C version is a huge
//! Mathematica export): it is the exact pushforward of KS through the
//! diagonal map r = R0 + e^{x1}, θ = θ(x2); mathematically identical,
//! with all derivatives supplied by the chain rule through Dual3.

const std = @import("std");
const Dual3 = @import("../math/dual.zig").Dual3;

pub const pi: f64 = std.math.pi;

pub const MetricParams = struct {
    /// BH spin (C: BHSPIN).
    a: f64 = 0,
    /// MKS2 radial offset (C: MKSR0).
    mksr0: f64 = 0,
    /// MKS2 polar squeeze (C: MKSH0).
    mksh0: f64 = 1,
};

/// Minkowski, Cartesian (C: MINKCOORDS).
pub fn gcovMink(x: [4]Dual3, mp: MetricParams) [4][4]Dual3 {
    _ = x;
    _ = mp;
    var g: [4][4]Dual3 = @splat(@splat(Dual3.constant(0)));
    g[0][0] = Dual3.constant(-1);
    g[1][1] = Dual3.constant(1);
    g[2][2] = Dual3.constant(1);
    g[3][3] = Dual3.constant(1);
    return g;
}

/// Kerr in Boyer-Lindquist coordinates (C: KERRCOORDS), x = (t, r, θ, φ).
pub fn gcovBl(x: [4]Dual3, mp: MetricParams) [4][4]Dual3 {
    const a = mp.a;
    const r = x[1];
    const sth = x[2].sin();
    const cth = x[2].cos();
    const s2 = sth.sq();
    // Σ = r² + a² cos²θ ; Δ = a² + (r−2) r
    const sigma = r.sq().add(cth.sq().scale(a * a));
    const delta = r.addc(-2.0).mul(r).addc(a * a);
    const two_r_sigma = r.scale(2.0).div(sigma); // 2r/Σ

    var g: [4][4]Dual3 = @splat(@splat(Dual3.constant(0)));
    g[0][0] = two_r_sigma.addc(-1.0);
    g[0][3] = r.mul(s2).scale(-2.0 * a).div(sigma);
    g[1][1] = sigma.div(delta);
    g[2][2] = sigma;
    // g33 = sin²θ (a² + r² + 2 a² r sin²θ / Σ)
    g[3][3] = s2.mul(r.sq().addc(a * a).add(r.mul(s2).scale(2.0 * a * a).div(sigma)));
    g[3][0] = g[0][3];
    return g;
}

/// Kerr in horizon-penetrating Kerr-Schild coordinates (C: KSCOORDS).
pub fn gcovKs(x: [4]Dual3, mp: MetricParams) [4][4]Dual3 {
    const a = mp.a;
    const r = x[1];
    const sth = x[2].sin();
    const cth = x[2].cos();
    const s2 = sth.sq();
    const sigma = r.sq().add(cth.sq().scale(a * a));
    const two_r_sigma = r.scale(2.0).div(sigma);
    const one_p = two_r_sigma.addc(1.0); // 1 + 2r/Σ

    var g: [4][4]Dual3 = @splat(@splat(Dual3.constant(0)));
    g[0][0] = two_r_sigma.addc(-1.0);
    g[0][1] = two_r_sigma;
    g[0][3] = r.mul(s2).scale(-2.0 * a).div(sigma);
    g[1][1] = one_p;
    g[1][3] = one_p.mul(s2).scale(-a);
    g[2][2] = sigma;
    // g33 = sin²θ (Σ + a² (1 + 2r/Σ) sin²θ)
    g[3][3] = s2.mul(sigma.add(one_p.mul(s2).scale(a * a)));
    g[1][0] = g[0][1];
    g[3][0] = g[0][3];
    g[3][1] = g[1][3];
    return g;
}

// --- the MKS2 map (C: coco_MKS22KS metric.c:2626, dxdx_MKS22KS :3689) ---
//
// The Zig port now uses one exact value of π for both the MKS2 metric flavor
// and the coco point transforms. That keeps metric-flavor θ and coco-flavor θ
// consistent instead of preserving the old C split.

/// cot(π H0 / 2) with the exact-π/2 literal (C: Cot(1.5707963267948966*H0)
/// in both flavors).
pub fn mks2Cot(h0: f64) f64 {
    return 1.0 / @tan(0.5 * pi * h0);
}

/// Point-transform flavor (exact π): θ(x2) = π/2·(1 + cot(πH0/2)·tan(H0π(x2−1/2))).
pub fn mks2Theta(x2: Dual3, h0: f64) Dual3 {
    const u = x2.addc(-0.5).scale(h0 * pi);
    return u.tan().scale(0.5 * pi * mks2Cot(h0)).addc(0.5 * pi);
}

/// Metric flavor: θ = (π/2)·(1 + cot(πH0/2)·tan(H0·π·(x2−1/2))) using the
/// same exact π as the coco transform.
pub fn mks2ThetaMetric(x2: Dual3, h0: f64) Dual3 {
    const u = x2.addc(-0.5).scale(h0 * pi);
    return u.tan().scale(mks2Cot(h0)).addc(1.0).scale(0.5 * pi);
}

/// Metric flavor dθ/dx2 = (H0 π²/2) cot(πH0/2) sec²(H0 π (x2 − 1/2)) using
/// the same exact π as the coco transform.
pub fn mks2DThetaDx2Metric(x2: Dual3, h0: f64) Dual3 {
    const u = x2.addc(-0.5).scale(h0 * pi);
    const t = u.tan();
    return t.sq().addc(1.0).scale(0.5 * h0 * pi * pi * mks2Cot(h0));
}

/// x2(θ): inverse point transform (C: coco_KS2MKS2, exact-π literals).
pub fn mks2X2FromTheta(theta: f64, h0: f64) f64 {
    return (0.5 * h0 + (1.0 / pi) * std.math.atan((1.0 / pi) * (2.0 * theta - pi) * @tan(0.5 * pi * h0))) / h0;
}

/// Modified Kerr-Schild type 2 (C: MKS2COORDS): x = (t, x1, x2, φ),
/// r = R0 + e^{x1}, θ = θ(x2). Exact pushforward of KS in the metric flavor.
pub fn gcovMks2(x: [4]Dual3, mp: MetricParams) [4][4]Dual3 {
    const j1 = x[1].exp(); // dr/dx1 (also r − R0)
    const r = j1.addc(mp.mksr0);
    const th = mks2ThetaMetric(x[2], mp.mksh0);
    const j2 = mks2DThetaDx2Metric(x[2], mp.mksh0);

    const gks = gcovKs(.{ x[0], r, th, x[3] }, mp);
    const jf = [4]Dual3{ Dual3.constant(1), j1, j2, Dual3.constant(1) };

    var g: [4][4]Dual3 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            g[i][j] = gks[i][j].mul(jf[i]).mul(jf[j]);
        }
    }
    return g;
}
