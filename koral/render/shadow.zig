//! The analytic Kerr shadow (Bardeen 1973): the critical curve of photon
//! capture, parametrized by the radius r_p of unstable spherical photon
//! orbits. With G = c = M = 1 and the radial potential
//!     R(r) = (r² + a² − aξ)² − Δ·[η + (ξ − a)²],   Δ = r² − 2r + a²,
//! solving R = R' = 0 gives the critical impact parameters
//!     ξ(r_p) = [r_p²(3 − r_p) − a²(r_p + 1)] / [a(r_p − 1)]
//!     η(r_p) = r_p³[4a² − r_p(r_p − 3)²] / [a²(r_p − 1)²]
//! (ξ = L_z/E, η = Q/E²; both are exact constants of motion, so a capture
//! test phrased in (ξ, η) is exact even for a camera at finite radius).
//! A ray from far away is captured iff η < η_crit(ξ); on the image plane
//! of an observer at inclination i the curve appears at
//!     α = −ξ/sin i,   β = ±√(η + a²cos²i − ξ²cot²i).
//!
//! Everything here is fluid-free; it validates the optics stack
//! (camera + tetrad + geodesics + capture logic) end to end.

const std = @import("std");

/// Prograde/retrograde equatorial photon-orbit radii,
/// r_ph = 2(1 + cos((2/3)·arccos(∓a))); the r_p range of the shell.
pub fn photonShell(a: f64) struct { r_pro: f64, r_retro: f64 } {
    return .{
        .r_pro = 2.0 * (1.0 + @cos((2.0 / 3.0) * std.math.acos(-a))),
        .r_retro = 2.0 * (1.0 + @cos((2.0 / 3.0) * std.math.acos(a))),
    };
}

pub const XiEta = struct { xi: f64, eta: f64 };

/// Critical (ξ, η) of the spherical photon orbit at r_p. Requires a ≠ 0
/// (the a → 0 limit is the ξ² + η = 27 circle).
pub fn xiEta(rp: f64, a: f64) XiEta {
    const denom = a * (rp - 1.0);
    return .{
        .xi = (rp * rp * (3.0 - rp) - a * a * (rp + 1.0)) / denom,
        .eta = rp * rp * rp * (4.0 * a * a - rp * (rp - 3.0) * (rp - 3.0)) /
            (a * a * (rp - 1.0) * (rp - 1.0)),
    };
}

/// η on the critical curve at the given ξ, by bisection over the photon
/// shell (ξ(r_p) is monotone decreasing from prograde to retrograde).
/// null when ξ lies outside the shell's ξ-range; no critical orbit
/// exists there and such rays always escape.
pub fn etaCritForXi(xi: f64, a: f64) ?f64 {
    const shell = photonShell(a);
    var lo = shell.r_pro + 1e-9;
    var hi = shell.r_retro - 1e-9;
    const xi_lo = xiEta(lo, a).xi; // max ξ (prograde)
    const xi_hi = xiEta(hi, a).xi; // min ξ (retrograde)
    if (xi > xi_lo or xi < xi_hi) return null;
    var iter: usize = 0;
    while (iter < 200) : (iter += 1) {
        const mid = 0.5 * (lo + hi);
        if (xiEta(mid, a).xi > xi) lo = mid else hi = mid;
        if (hi - lo < 1e-13 * hi) break;
    }
    return xiEta(0.5 * (lo + hi), a).eta;
}

/// Would a ray with constants (ξ, η), sent inward from well outside the
/// photon shell, be captured?
pub fn predictCaptured(xi: f64, eta: f64, a: f64) bool {
    const crit = etaCritForXi(xi, a) orelse return false;
    return eta < crit;
}

pub const CurvePoint = struct { alpha: f64, beta: f64 };

/// Image-plane point of the critical curve for the orbit at r_p, seen from
/// inclination incl (radians from the spin axis). null where the orbit is
/// invisible from this inclination (β² < 0). β ≥ 0; mirror for the lower
/// half.
pub fn curvePoint(rp: f64, a: f64, incl: f64) ?CurvePoint {
    const ce = xiEta(rp, a);
    const si = @sin(incl);
    const ci = @cos(incl);
    const b2 = ce.eta + a * a * ci * ci - ce.xi * ce.xi * (ci / si) * (ci / si);
    if (b2 < 0) return null;
    return .{ .alpha = -ce.xi / si, .beta = @sqrt(b2) };
}
