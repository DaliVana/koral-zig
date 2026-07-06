//! M1 radiation basics (M7): the lab-frame radiation stress tensor, the
//! fluid-frame energy density, and the radiative characteristic speeds with
//! the optical-depth limiter.
//!
//!   rad.c:3098  calc_Rij_M1
//!   rad.c:3164  calc_ff_Rtt   (fluid-frame Ehat = −R^t_t(ff) + gas ucon)
//!   rad.c:3598  calc_rad_wavespeeds (Sądowski+13a τ-limiter)
//!
//! C-fidelity notes:
//!  * PUFFY / choices.h default damping branch: rv2τ = (4/3)²/τ² (none of
//!    SKIPRADWAVESPEEDLIMITER / FULLRADWAVESPEEDS / DAMPRADWAVESPEEDSQRTTAU /
//!    MULTIPLYRADWAVESPEEDCONSTANT / DAMPRADWAVESPEEDLIMITERINSIDE defined).
//!  * rad.c:3702 assigns rv2z = rv2dim[1] — the z-direction limiter uses the
//!    *y* optical depth. Transcribed as-is (pinned by a theory test).
//!  * The opacity χ that feeds τ = χ·dx arrives with M8; callers pass tautot.

const relele = @import("../relele.zig");
const wavespeeds = @import("wavespeeds.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const frames = @import("../frames.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const Geometry = @import("../geometry.zig").Geometry;

const four_third: f64 = 4.0 / 3.0;
const one_third: f64 = 1.0 / 3.0;

/// Rad-rest-frame 4-velocity from the (VELR) radiative primitives.
pub fn urfCon(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
) relele.Error![4]f64 {
    const L = layout.VarLayout(cfg);
    return relele.convVels(
        .{ 0, pp[L.index(.fx)], pp[L.index(.fy)], pp[L.index(.fz)] },
        .velr,
        .vel4,
        &geom.gg,
        &geom.GG,
    );
}

/// Lab-frame R^μν of the M1 closure (C: calc_Rij_M1, rad.c:3098):
/// R^μν = (4/3) Ê u_r^μ u_r^ν + (1/3) Ê g^μν.
pub fn calcRij(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
) relele.Error![4][4]f64 {
    const L = layout.VarLayout(cfg);
    const erf = pp[L.index(.ee)];
    const urf = try urfCon(cfg, pp, geom);

    var rij: [4][4]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            rij[i][j] = four_third * erf * urf[i] * urf[j] + one_third * erf * geom.GG[i][j];
        }
    }
    return rij;
}

pub const FfRtt = struct { rtt: f64, ucon: [4]f64 };

/// C: calc_ff_Rtt (rad.c:3164) — fluid-frame R^t_t (Ehat = −rtt) and the
/// gas lab-frame 4-velocity: rtt = −R^μ_ν u^ν u_μ.
pub fn calcFfRtt(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
) relele.Error!FfRtt {
    const L = layout.VarLayout(cfg);
    const u = try relele.uconUcovFromPrims(
        .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
        geom,
    );

    const rij_up = try calcRij(cfg, pp, geom);
    const rij = relele.indices2221(rij_up, &geom.gg);
    var rtt: f64 = 0;
    for (0..4) |ia| {
        for (0..4) |ib| {
            rtt += -rij[ia][ib] * u.con[ib] * u.cov[ia];
        }
    }
    return .{ .rtt = rtt, .ucon = u.con };
}

/// C: calc_ff_Ehat (rad.c:3194).
pub fn calcFfEhat(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
) relele.Error!f64 {
    const r = try calcFfRtt(cfg, pp, geom);
    return -r.rtt;
}

/// C: prad_ff2lab (frames.c:1890) — fluid-frame radiative primitives
/// (Ê, û_r = 0 relative to gas) → lab-frame primitives in the same coords.
/// Rij from the M1 closure on pp, ff→lab boost (the `_with_alpha` variant's
/// α-correction is dead code: the copy the boost reads is taken before the
/// scaling, so it reduces to the plain boost), then u2p_rad on gdetu·R^0_μ.
/// The C caller ignores u2p_rad's return code; so do we. Modifies pp's
/// EE..FZ slots in place; MHD slots pass through untouched.
pub fn pradFf2Lab(
    comptime cfg: config.Config,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    rp: invert_rad.RadParams,
) relele.Error!void {
    const L = layout.VarLayout(cfg);
    var rij = try calcRij(cfg, pp.*, geom);
    rij = try frames.boost22Ff2Lab(
        rij,
        .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
        &geom.gg,
        &geom.GG,
    );
    rij = relele.indices2221(rij, &geom.gg);

    const gdetu = geom.gdet; // GDETIN == 1
    var uu = [_]f64{0} ** L.count;
    uu[L.index(.ee)] = gdetu * rij[0][0];
    uu[L.index(.fx)] = gdetu * rij[0][1];
    uu[L.index(.fy)] = gdetu * rij[0][2];
    uu[L.index(.fz)] = gdetu * rij[0][3];

    _ = invert_rad.u2pRad(cfg, uu, pp, geom, rp);
}

/// C: calc_rad_wavespeeds (rad.c:3598) — aval[0..6] are the unlimited
/// speeds (rv² = 1/3 in the rad rest frame, used for the timestep),
/// aval[6..12] the τ-damped ones (used for the fluxes). No co-going
/// clamps here — the caller applies them (physics.c:609-621).
pub fn calcRadWavespeeds(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    tautot: [3]f64,
    dims: [3]bool,
) relele.Error![12]f64 {
    const urf = try urfCon(cfg, pp, geom);

    const rv2rad = one_third;
    var aval: [12]f64 = undefined;

    // unlimited rad wavespeeds, used for the time step
    const a0 = wavespeeds.lrCore(urf, &geom.GG, .{ rv2rad, rv2rad, rv2rad }, dims);
    for (0..6) |i| aval[i] = a0[i];

    // damped by the optical depth (Sądowski+13a); default branch only
    var rv2dim: [3]f64 = undefined;
    for (0..3) |dim| {
        if (tautot[dim] <= 0.0) {
            rv2dim[dim] = rv2rad;
        } else {
            const rv2tau = (four_third * four_third) / (tautot[dim] * tautot[dim]);
            rv2dim[dim] = @min(rv2rad, rv2tau);
        }
    }
    const rv2x = rv2dim[0];
    const rv2y = rv2dim[1];
    const rv2z = rv2dim[1]; // C quirk (rad.c:3702): z uses the y depth

    const a1 = wavespeeds.lrCore(urf, &geom.GG, .{ rv2x, rv2y, rv2z }, dims);
    for (0..6) |i| aval[6 + i] = a1[i];

    return aval;
}
