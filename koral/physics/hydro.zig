//! Gas thermodynamics and stress-energy tensor (C: physics.c).
//!
//! `gamma_adiab` is the adiabatic index Γ (C: GAMMA / pgamma; PUFFY: 5/3).
//! CONSISTENTGAMMA is off for PUFFY, so Γ is a plain parameter here.

const std = @import("std");
const relele = @import("../relele.zig");
const mhd = @import("bfield.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const Geometry = @import("../geometry.zig").Geometry;

/// Specific-entropy-like conserved S(ρ, u) (C: calc_Sfromu, physics.c:1457;
/// log form — NOLOGINS is not defined for PUFFY).
pub fn sFromU(rho: f64, u: f64, gamma_adiab: f64) f64 {
    const gammam1 = gamma_adiab - 1.0;
    const indexn = 1.0 / gammam1;
    const pre = gammam1 * u;
    return rho * @log(std.math.pow(f64, pre, indexn) / std.math.pow(f64, rho, indexn + 1.0));
}

/// Inverse of sFromU (C: calc_ufromS, physics.c:1497).
pub fn uFromS(s: f64, rho: f64, gamma_adiab: f64) f64 {
    const gammam1 = gamma_adiab - 1.0;
    const indexn = 1.0 / gammam1;
    return std.math.pow(
        f64,
        std.math.pow(f64, rho, indexn + 1.0) * @exp(s / rho),
        gammam1,
    ) / gammam1;
}

/// MHD stress-energy tensor T^μν (C: calc_Tij, physics.c:1349):
/// T^μν = (ρ + u + p + b²) u^μ u^ν + (p + b²/2) g^μν − b^μ b^ν.
pub fn calcTij(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma_adiab: f64,
) relele.Error![4][4]f64 {
    const L = layout.VarLayout(cfg);
    const rho = pp[L.index(.rho)];
    const uu = pp[L.index(.uu)];

    const u = try relele.uconUcovFromPrims(
        .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
        geom,
    );

    var bcon: [4]f64 = @splat(0);
    var bsq: f64 = 0;
    if (comptime L.hasVar(.b1)) {
        const b = mhd.bconBcovBsqFrom4vel(
            .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
            u.con,
            u.cov,
            &geom.gg,
        );
        bcon = b.bcon;
        bsq = b.bsq;
    }

    const p = (gamma_adiab - 1.0) * uu;
    const w = rho + uu + p;
    const eta = w + bsq;
    const ptot = p + 0.5 * bsq;

    var t: [4][4]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            t[i][j] = eta * u.con[i] * u.con[j] + ptot * geom.GG[i][j] - bcon[i] * bcon[j];
        }
    }
    return t;
}
