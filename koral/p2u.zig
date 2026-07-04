//! Primitives → conserved (C: p2u.c). GDETIN == 1 for PUFFY, so the
//! conserved variables carry √−g (gdetu = gdet) — asserted, not optional.
//!
//! The energy conserved uu[UU] is C's T^t_t + ρu^t, assembled through
//! utp1 = 1 + u_t (p2u.c:95) to avoid catastrophic cancellation in the
//! non-relativistic limit; gttpert = g_tt + 1 feeds that path.

const std = @import("std");
const relele = @import("relele.zig");
const mhd = @import("physics/mhd.zig");
const config = @import("config.zig");
const layout = @import("layout.zig");
const Geometry = @import("geometry.zig").Geometry;

const four_third: f64 = 4.0 / 3.0;
const one_third: f64 = 1.0 / 3.0;

/// 1 + u_t computed cancellation-free (C: calc_utp1, p2u.c:163; the
/// VELPRIM == VELR branch — the only one KORAL's defaults ever take).
pub fn calcUtp1(vcon: [4]f64, geom: *const Geometry) f64 {
    const gg = &geom.gg;
    var qsq: f64 = 0;
    for (1..4) |i| {
        for (1..4) |j| {
            qsq += vcon[i] * vcon[j] * gg[i][j];
        }
    }
    const alpha = geom.alpha;
    const alphasq = alpha * alpha;
    const alpgam = @sqrt(alphasq * (1.0 + qsq));

    // β^i β_i / α² = g^{ti} g_{ti}
    const betasqoalphasq = gg[0][1] * geom.GG[0][1] + gg[0][2] * geom.GG[0][2] + gg[0][3] * geom.GG[0][3];

    var ud0tilde: f64 = 0;
    for (1..4) |j| ud0tilde += vcon[j] * gg[0][j];

    return ud0tilde + (geom.gttpert - alphasq * (betasqoalphasq + qsq)) / (1.0 + alpgam);
}

/// C: p2u_mhd (p2u.c:32) — writes the hydro+magnetic slots of `uu`.
pub fn p2uMhd(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    uu: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma: f64,
) relele.Error!void {
    const L = layout.VarLayout(cfg);
    const gdetu = geom.gdet; // GDETIN == 1

    const rho = pp[L.index(.rho)];
    const ugas = pp[L.index(.uu)];
    const s = pp[L.index(.entr)];

    const vcon = [4]f64{ 0, pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] };
    const u = try relele.convVelsBoth(vcon, .velr, &geom.gg, &geom.GG);

    var bcon: [4]f64 = @splat(0);
    var bcov: [4]f64 = @splat(0);
    var bsq: f64 = 0;
    if (comptime L.hasVar(.b1)) {
        const b = mhd.bconBcovBsqFrom4vel(
            .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
            u.con,
            u.cov,
            &geom.gg,
        );
        bcon = b.bcon;
        bcov = b.bcov;
        bsq = b.bsq;
    }

    const ut = u.con[0];
    const rhout = rho * ut;
    const sut = s * ut;

    const pre = (gamma - 1.0) * ugas;
    const w = rho + ugas + pre;
    const eta = w + bsq;
    const etap = ugas + pre + bsq; // eta - rho
    const ptot = pre + 0.5 * bsq;

    const utp1 = calcUtp1(vcon, geom);

    const tttt = etap * u.con[0] * u.cov[0] + rho * u.con[0] * utp1 + ptot - bcon[0] * bcov[0];
    const ttr = eta * u.con[0] * u.cov[1] - bcon[0] * bcov[1];
    const ttth = eta * u.con[0] * u.cov[2] - bcon[0] * bcov[2];
    const ttph = eta * u.con[0] * u.cov[3] - bcon[0] * bcov[3];

    uu[L.index(.rho)] = gdetu * rhout;
    uu[L.index(.uu)] = gdetu * tttt;
    uu[L.index(.vx)] = gdetu * ttr;
    uu[L.index(.vy)] = gdetu * ttth;
    uu[L.index(.vz)] = gdetu * ttph;
    uu[L.index(.entr)] = gdetu * sut;

    if (comptime L.hasVar(.b1)) {
        uu[L.index(.b1)] = gdetu * pp[L.index(.b1)];
        uu[L.index(.b2)] = gdetu * pp[L.index(.b2)];
        uu[L.index(.b3)] = gdetu * pp[L.index(.b3)];
    }
}

/// C: p2u_rad (p2u.c:300) — M1 closure: R^t_μ from (Êrf, ũ_rad).
pub fn p2uRad(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    uu: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
) relele.Error!void {
    const L = layout.VarLayout(cfg);
    const gdetu = geom.gdet; // GDETIN == 1

    const erf = pp[L.index(.ee)];
    const urf = try relele.convVels(
        .{ 0, pp[L.index(.fx)], pp[L.index(.fy)], pp[L.index(.fz)] },
        .velr,
        .vel4,
        &geom.gg,
        &geom.GG,
    );

    var rtopp: [4]f64 = undefined;
    for (0..4) |i| {
        rtopp[i] = four_third * erf * urf[0] * urf[i] + one_third * erf * geom.GG[0][i];
    }
    rtopp = relele.indices21(rtopp, &geom.gg); // R^t_μ

    for (0..4) |i| {
        uu[L.index(.ee) + i] = gdetu * rtopp[i];
    }
}

/// C: p2u (p2u.c:14) — full conserved vector.
pub fn p2u(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma: f64,
) relele.Error![layout.VarLayout(cfg).count]f64 {
    const L = layout.VarLayout(cfg);
    var uu: [L.count]f64 = @splat(0);
    try p2uMhd(cfg, pp, &uu, geom, gamma);
    if (comptime L.hasVar(.ee)) {
        try p2uRad(cfg, pp, &uu, geom);
    }
    return uu;
}
