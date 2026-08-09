//! Radiative conserved → primitives inversion (C: u2prad.c) and the
//! radiative floors (check_floors_rad).
//!
//! The M1 inversion is closed-form: from R^t_μ, get the rad-frame Lorentz
//! factor γ² from JCM's Mathematica solution (u2p_inversion.nb), then
//! Ê = 3 R^tt α²/(4γ²−1) and the relative 4-velocity. Two cancellation
//! branches exist for γ² (gamma2a for the generic case, gamma2b when Rtt
//! is small); failures fall back to the "cold" solver that discards R^t_t,
//! keeps R^t_i, and pins γ to either 1 or GAMMAMAXRAD — whichever
//! reproduces the original R^t_t more closely (u2prad.c:369).
//!
//! C-fidelity notes:
//!  * delta/numerator/divisor are vestigial (always 0) — the failure1/2/3
//!    flags they feed only guard a never-taken diagnostic print.
//!  * On a cold-branch non-finite result u2p_rad returns −1 *without*
//!    touching pp (primitives keep their previous values), but `corrected`
//!    was already set — the caller still flags the cell for rad fixups.
//!  * Power(x,2) in the C source is pow(x,2.), which clang -O2 folds to
//!    x*x — transcribed as plain multiplication.
//!  * GDETIN == 1 (gdetu = gdet), VELPRIMRAD == VELR (the trailing
//!    conv_vels is an identity copy and is skipped).

const std = @import("std");
const relele = @import("../relele.zig");
const radiation = @import("../physics/radiation.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const Geometry = @import("../geometry.zig").Geometry;

/// C: GAMMASMALLLIMIT (choices.h:417).
const gammasmalllimit: f64 = 1.0 - 1e-10;
/// C: NUMEPSILON = DBL_EPSILON (ko.h:9).
const numepsilon: f64 = std.math.floatEps(f64);

/// Radiation caps/floors (C: choices.h defaults; PROBLEMS/*/define.h
/// overrides). ERADFLOOR = 10·SMALL is not overridden by any target.
pub const RadParams = struct {
    gammamaxrad: f64 = 100.0, // choices.h:412
    eradfloor: f64 = 10.0 * 1.0e-80, // choices.h:358, SMALL = 1e-80
    eerhoratiomin: f64 = 1.0e-30,
    eerhoratiomax: f64 = 1.0e30,
    eeuuratiomin: f64 = 1.0e-30,
    eeuuratiomax: f64 = 1.0e30,

    pub const cdefault = RadParams{};

    /// PROBLEMS/PUFFY/define.h:115-132.
    pub const puffy = RadParams{
        .gammamaxrad = 10.0,
        .eerhoratiomin = 1.0e-20,
        .eerhoratiomax = 1.0e4,
        .eeuuratiomin = 1.0e-20,
        .eeuuratiomax = 1.0e20,
    };
};

pub const U2pRadResult = struct {
    /// C return code: 0 ok, −1 non-finite cold result (pp untouched).
    ret: i32,
    /// C `corrected` — cell needs rad fixup consideration (RADFIXUPFLAG).
    corrected: bool,
};

/// C: get_m1closure_gammarel2 (u2prad.c:236). Always "succeeds"; may
/// produce NaN/inf which the caller's failure logic handles.
fn m1GammaRel2(geom: *const Geometry, avcov: [4]f64) f64 {
    const GG = &geom.GG;
    const gn11 = GG[0][0];
    const gn12 = GG[0][1];
    const gn13 = GG[0][2];
    const gn14 = GG[0][3];
    const gn22 = GG[1][1];
    const gn23 = GG[1][2];
    const gn24 = GG[1][3];
    const gn33 = GG[2][2];
    const gn34 = GG[2][3];
    const gn44 = GG[3][3];

    const rdtt = avcov[0];
    const rdtx = avcov[1];
    const rdty = avcov[2];
    const rdtz = avcov[3];

    // the discriminant Sqrt(...) shared by both branches (C repeats the
    // expression textually; identical operands give identical bits)
    const mix = gn12 * rdtx + gn13 * rdty + gn14 * rdtz;
    const sq = @sqrt(4.0 * (gn11 * gn11) * (rdtt * rdtt) + mix * mix +
        gn11 * (8.0 * gn12 * rdtt * rdtx + 3.0 * gn22 * (rdtx * rdtx) + 8.0 * gn13 * rdtt * rdty +
            6.0 * gn23 * rdtx * rdty + 3.0 * gn33 * (rdty * rdty) +
            8.0 * gn14 * rdtt * rdtz + 6.0 * gn24 * rdtx * rdtz + 6.0 * gn34 * rdty * rdtz +
            3.0 * gn44 * (rdtz * rdtz)));

    const den = gn11 * (rdtt * rdtt) + 2.0 * gn12 * rdtt * rdtx + gn22 * (rdtx * rdtx) +
        2.0 * gn13 * rdtt * rdty + 2.0 * gn23 * rdtx * rdty + gn33 * (rdty * rdty) +
        2.0 * (gn14 * rdtt + gn24 * rdtx + gn34 * rdty) * rdtz + gn44 * (rdtz * rdtz);

    const gamma2a = (-0.25 * (2.0 * (gn11 * gn11) * (rdtt * rdtt) + mix * (mix + sq) +
        gn11 * (4.0 * gn12 * rdtt * rdtx + gn22 * (rdtx * rdtx) + 2.0 * gn23 * rdtx * rdty +
            gn33 * (rdty * rdty) + 2.0 * gn24 * rdtx * rdtz + 2.0 * gn34 * rdty * rdtz +
            gn44 * (rdtz * rdtz) +
            rdtt * (4.0 * gn13 * rdty + 4.0 * gn14 * rdtz + sq)))) / den;

    var gamma2: f64 = undefined;
    if (gamma2a < gammasmalllimit or !std.math.isFinite(gamma2a)) {
        // mathematica solution avoiding catastrophic cancellation when Rtt
        // is very small (u2p_inversion.nb)
        const gamma2b = (0.25 * (-2.0 * (gn11 * gn11) * (rdtt * rdtt) -
            1.0 * gn11 * (4.0 * gn12 * rdtt * rdtx + gn22 * (rdtx * rdtx) +
                rdty * (4.0 * gn13 * rdtt + 2.0 * gn23 * rdtx + gn33 * rdty) +
                2.0 * (2.0 * gn14 * rdtt + gn24 * rdtx + gn34 * rdty) * rdtz +
                gn44 * (rdtz * rdtz)) +
            gn11 * rdtt * sq +
            mix * (-1.0 * gn12 * rdtx - 1.0 * gn13 * rdty - 1.0 * gn14 * rdtz + sq))) / den;
        gamma2 = gamma2b;
    } else {
        gamma2 = gamma2a;
    }

    // relative-velocity γ², always ≥ 1 even in GR; machine-error snap to 1
    const alpha = geom.alpha;
    var gammarel2 = gamma2 * alpha * alpha;
    if (gammarel2 > gammasmalllimit and gammarel2 < 1.0) gammarel2 = 1.0;
    return gammarel2;
}

/// C: get_m1closure_Erf (u2prad.c:351).
fn m1Erf(geom: *const Geometry, avcon0: f64, gammarel2: f64) f64 {
    const alpha = geom.alpha;
    return 3.0 * avcon0 * alpha * alpha / (4.0 * gammarel2 - 1.0);
}

const ColdResult = struct {
    avcov: [4]f64,
    erf: f64,
    urfconrel: [4]f64,
};

/// C: get_m1closure_gammarel2_cold (u2prad.c:136) — fixed γ², discard
/// R^t_t, keep R^t_i.
fn m1Cold(geom: *const Geometry, avcov_in: [4]f64, gammarel2: f64) ColdResult {
    const GG = &geom.GG;
    const gn11 = GG[0][0];
    const gn12 = GG[0][1];
    const gn13 = GG[0][2];
    const gn14 = GG[0][3];
    const gn22 = GG[1][1];
    const gn23 = GG[1][2];
    const gn24 = GG[1][3];
    const gn33 = GG[2][2];
    const gn34 = GG[2][3];
    const gn44 = GG[3][3];
    const alpha = geom.alpha;

    const rdtx = avcov_in[1];
    const rdty = avcov_in[2];
    const rdtz = avcov_in[3];

    const utsq = gammarel2 / (alpha * alpha);

    var avcov = avcov_in;

    const mix = gn13 * rdty + gn14 * rdtz;
    const disc = @sqrt(((gn12 * gn12) * (rdtx * rdtx) + 2.0 * gn12 * rdtx * mix + mix * mix -
        1.0 * gn11 * (gn22 * (rdtx * rdtx) + 2.0 * gn23 * rdtx * rdty + gn33 * (rdty * rdty) +
            2.0 * gn24 * rdtx * rdtz + 2.0 * gn34 * rdty * rdtz + gn44 * (rdtz * rdtz))) *
        utsq * (gn11 + utsq) * ((gn11 + 4.0 * utsq) * (gn11 + 4.0 * utsq)));

    avcov[0] = (0.25 * (-4.0 * (gn12 * rdtx + gn13 * rdty + gn14 * rdtz) * utsq * (gn11 + utsq) +
        disc)) / (gn11 * utsq * (gn11 + utsq));

    const erf = (0.75 * disc) / (utsq * (gn11 + utsq) * (gn11 + 4.0 * utsq));

    const avcon = relele.raiseVec(avcov, geom);

    const gammarel = @sqrt(gammarel2);
    var urfconrel: [4]f64 = @splat(0);
    if (erf > 0.0) {
        for (1..4) |jj| {
            urfconrel[jj] = alpha * (avcon[jj] + 1.0 / 3.0 * erf * GG[0][jj] * (4.0 * gammarel2 - 1.0)) /
                (4.0 / 3.0 * erf * gammarel);
        }
    }

    return .{ .avcov = avcov, .erf = erf, .urfconrel = urfconrel };
}

/// C: u2p_rad / u2p_rad_urf (u2prad.c:31/:67) — no EVOLVEPHOTONNUMBER.
/// Writes pp[EE..FZ] on success; leaves pp untouched when ret == −1.
pub fn u2pRad(
    comptime cfg: config.Config,
    uu: [layout.VarLayout(cfg).count]f64,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    rp: RadParams,
) U2pRadResult {
    const L = layout.VarLayout(cfg);
    const gdetu = geom.gdet; // GDETIN == 1
    const alpha = geom.alpha;
    const GG = &geom.GG;

    // conserved R^t_μ and R^tμ
    const avcov = [4]f64{
        uu[L.index(.ee)] / gdetu,
        uu[L.index(.fx)] / gdetu,
        uu[L.index(.fy)] / gdetu,
        uu[L.index(.fz)] / gdetu,
    };
    var avcon = relele.raiseVec(avcov, geom);

    var gammarel2 = m1GammaRel2(geom, avcov);
    var erf = m1Erf(geom, avcon[0], gammarel2);

    // C: get_m1closure_urfconrel (u2prad.c:369)
    const gammamax = rp.gammamaxrad;
    const nonfailure = gammarel2 >= 1.0 and erf > rp.eradfloor and
        gammarel2 <= gammamax * gammamax / gammasmalllimit / gammasmalllimit;

    var urfcon: [4]f64 = undefined;
    var corrected = false;

    if (nonfailure) {
        const gammarel = @sqrt(gammarel2);
        for (0..4) |ii| {
            urfcon[ii] = alpha * (avcon[ii] + 1.0 / 3.0 * erf * GG[0][ii] * (4.0 * gammarel2 - 1.0)) /
                (4.0 / 3.0 * erf * gammarel);
        }
    } else {
        // cold fallbacks: γ ≈ 1 and γ = γmax; keep the one whose recomputed
        // R^t_t is closer to the original
        const one_eps = 1.0 + 10.0 * numepsilon;
        const slow = m1Cold(geom, avcov, one_eps * one_eps);
        const fast = m1Cold(geom, avcov, gammamax * gammamax);

        if (@abs(slow.avcov[0] - avcov[0]) > @abs(fast.avcov[0] - avcov[0])) {
            avcon = relele.raiseVec(fast.avcov, geom);
            urfcon = fast.urfconrel;
            gammarel2 = gammamax * gammamax;
            erf = fast.erf;
        } else {
            avcon = relele.raiseVec(slow.avcov, geom);
            urfcon = slow.urfconrel;
            gammarel2 = one_eps * one_eps;
            erf = slow.erf;
        }
        corrected = true;

        if (!std.math.isFinite(erf) or !std.math.isFinite(gammarel2) or
            !std.math.isFinite(urfcon[1]) or !std.math.isFinite(urfcon[2]) or
            !std.math.isFinite(urfcon[3]))
        {
            return .{ .ret = -1, .corrected = corrected };
        }
    }

    // VELPRIMRAD == VELR: conv_vels(VELR, VELR) is a copy
    pp[L.index(.ee)] = erf;
    pp[L.index(.fx)] = urfcon[1];
    pp[L.index(.fy)] = urfcon[2];
    pp[L.index(.fz)] = urfcon[3];
    return .{ .ret = 0, .corrected = corrected };
}

/// C: check_floors_rad (u2prad.c:526) — rad-frame E floor plus Ehat/ρ and
/// Ehat/u ratio floors (no SKIPRADSOURCE, no EVOLVEPHOTONNUMBER; the
/// magnetic block is behind the never-defined SKIP_MAGNFIELD).
pub fn checkFloorsRad(
    comptime cfg: config.Config,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    rp: RadParams,
) relele.Error!i32 {
    const L = layout.VarLayout(cfg);
    var ret: i32 = 0;

    const ff = try radiation.calcFfRtt(cfg, pp.*, geom);
    const ehat = -ff.rtt;
    const eratio = pp[L.index(.ee)] / ehat; // ratio of the two frames' energies

    if (pp[L.index(.ee)] < rp.eradfloor) {
        pp[L.index(.ee)] = rp.eradfloor;
        ret = -1;
    }

    if (ehat < rp.eerhoratiomin * pp[L.index(.rho)]) {
        pp[L.index(.ee)] = eratio * rp.eerhoratiomin * pp[L.index(.rho)];
        ret = -1;
    }
    if (ehat > rp.eerhoratiomax * pp[L.index(.rho)]) {
        pp[L.index(.rho)] = 1.0 / rp.eerhoratiomax * ehat;
        ret = -1;
    }

    if (ehat < rp.eeuuratiomin * pp[L.index(.uu)]) {
        pp[L.index(.ee)] = eratio * rp.eeuuratiomin * pp[L.index(.uu)];
        ret = -1;
    }
    if (ehat > rp.eeuuratiomax * pp[L.index(.uu)]) {
        pp[L.index(.uu)] = 1.0 / rp.eeuuratiomax * ehat;
        ret = -1;
    }

    return ret;
}
