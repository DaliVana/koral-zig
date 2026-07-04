//! Conserved → primitives inversion (C: u2p.c).
//!
//! PUFFY / KORAL-default build choices, hardwired here:
//!   U2P_SOLVER = U2P_SOLVER_W (1-D Newton in W = w γ², u2p.c:1024)
//!   U2P_EQS    = U2P_EQS_NOBLE energy residual
//!   VELPRIM    = VELR, GDETIN = 1, ALLOWENTROPYU2P = 1, no CONSISTENTGAMMA
//!
//! Solver return codes are C's exactly (they are golden-compared):
//!   0 ok · -101 damping failed · -102 iteration limit · -103 W out of
//!   bounds / not finite · -104 negative uint · -105 negative rho ·
//!   -120 utcon not finite · -150 no valid initial W.
//!
//! The Newton iteration is transcribed literally — including C's entropy
//! residual using dwmrho0/dW = 1/γ where 1/γ² would be exact (u2p.c:2185).
//! The derivative is only a Newton direction; the converged root is the
//! same, but the iteration path must match C's for golden agreement.

const std = @import("std");
const relele = @import("../relele.zig");
const mhd = @import("../physics/mhd.zig");
const hydro = @import("../physics/hydro.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const Geometry = @import("../geometry.zig").Geometry;

/// C: U2PCONV (PUFFY define.h:27).
pub const u2pconv: f64 = 1.0e-10;
/// C: BIG = 1/SMALL (ko.h:7-8).
const big: f64 = 1.0 / 1.0e-80;

pub const Etype = enum { hot, entropy };

/// Floor thresholds (C: choices.h defaults, overridden by PROBLEMS/*/define.h).
pub const FloorParams = struct {
    rhofloor: f64 = 1.0e-50,
    uurhoratiomin: f64 = 1.0e-10,
    uurhoratiomax: f64 = 1.0e2,
    b2rhoratiomax: f64 = 100.0,
    b2uuratiomax: f64 = 100.0,
    gammamaxhd: f64 = 100.0,

    /// PROBLEMS/PUFFY/define.h:112-135 (B2RHOFLOORFRAME = DRIFTFRAME).
    pub const puffy = FloorParams{
        .rhofloor = 1.0e-30,
        .uurhoratiomin = 1.0e-8,
        .uurhoratiomax = 1.0e0,
        .b2rhoratiomax = 50.0,
        .b2uuratiomax = 50.0,
        .gammamaxhd = 10.0,
    };
};

const Residual = struct { f: f64, df: f64, err: f64 };

/// Common kinematics at a trial W (shared by both residuals).
fn wKinematics(W: f64, Wp: f64, D: f64, Qtsq: f64, QdotBsq: f64, Bsq: f64) struct {
    v2: f64,
    gamma: f64,
    gamma2: f64,
    rho0: f64,
    wmrho0: f64,
} {
    const x = Bsq + W;
    const wsq = W * W;
    const v2 = (wsq * Qtsq + QdotBsq * (Bsq + 2.0 * W)) / (wsq * (x * x));
    const gamma2 = 1.0 / (1.0 - v2);
    const gamma = @sqrt(gamma2);
    return .{
        .v2 = v2,
        .gamma = gamma,
        .gamma2 = gamma2,
        .rho0 = D / gamma,
        .wmrho0 = Wp / gamma2 - D * v2 / (1.0 + gamma),
    };
}

/// C: f_u2p_hot (u2p.c:2009), U2P_EQS_NOBLE branch.
fn fU2pHot(Wp: f64, cons: [7]f64, pgamma: f64) Residual {
    const Qn = cons[0];
    const Qtsq = cons[1];
    const D = cons[2];
    const QdotBsq = cons[3];
    const Bsq = cons[4];

    const W = Wp + D;
    const x = Bsq + W;
    const wsq = W * W;
    const w3 = wsq * W;
    const x3 = x * x * x;

    const k = wKinematics(W, Wp, D, Qtsq, QdotBsq, Bsq);
    const u = k.wmrho0 / pgamma;
    const p = (pgamma - 1.0) * u;

    const f = Qn + W - p + 0.5 * Bsq * (1.0 + k.v2) - QdotBsq / 2.0 / wsq;
    const err = @abs(f) /
        (@abs(Qn) + @abs(W) + @abs(p) + @abs(0.5 * Bsq * (1.0 + k.v2)) + @abs(QdotBsq / 2.0 / wsq));

    const dvsq = -2.0 / x3 * (Qtsq + QdotBsq * (3.0 * W * x + Bsq * Bsq) / w3);
    const dp1 = (pgamma - 1.0) * (1.0 - k.v2) / pgamma; // C: dpdWp_calc_vsq
    const idwmrho0dp = (pgamma - 1.0) / pgamma;
    const dwmrho0dvsq = D * (k.gamma * 0.5 - 1.0) - Wp;
    // drho0dvsq · idrho0dp == 0 in C (compute_idrho0dp returns 0)
    const dp2 = dwmrho0dvsq * idwmrho0dp;
    const dpdW = dp1 + dp2 * dvsq;

    const df = 1.0 - dpdW + QdotBsq / (wsq * W) + 0.5 * Bsq * dvsq;
    return .{ .f = f, .df = df, .err = err };
}

/// C: f_u2p_entropy (u2p.c:2143).
fn fU2pEntropy(Wp: f64, cons: [7]f64, pgamma: f64) Residual {
    const Qtsq = cons[1];
    const D = cons[2];
    const QdotBsq = cons[3];
    const Bsq = cons[4];
    const Sc = cons[5];

    const W = Wp + D;
    const x = Bsq + W;
    const wsq = W * W;
    const w3 = wsq * W;
    const x3 = x * x * x;

    const k = wKinematics(W, Wp, D, Qtsq, QdotBsq, Bsq);

    // specific entropy ln(p^n/ρ^{n+1}) at (ρ0, χ = u+p)
    const pressure = (pgamma - 1.0) / pgamma * k.wmrho0;
    const indexn = 1.0 / (pgamma - 1.0);
    const ssofchi = @log(std.math.pow(f64, pressure, indexn) / std.math.pow(f64, k.rho0, indexn + 1.0));

    const f = -Sc + D * ssofchi;
    const err = @abs(f) / (@abs(Sc) + @abs(D * ssofchi));

    const dssdrho = pgamma / ((1.0 - pgamma) * k.rho0);
    const dssdchi = 1.0 / ((pgamma - 1.0) * k.wmrho0);

    const dwmrho0dw = 1.0 / k.gamma; // C's 1/γ (sic), u2p.c:2185
    const dwmrho0dvsq = D * (k.gamma * 0.5 - 1.0) - Wp;
    const drho0dvsq = -D * k.gamma * 0.5;
    const dvsq = -2.0 / x3 * (Qtsq + QdotBsq * (3.0 * W * x + Bsq * Bsq) / w3);

    const dssdw = dwmrho0dw * dssdchi; // drho0dW = 0
    const dssdvsq = drho0dvsq * dssdrho + dwmrho0dvsq * dssdchi;
    const dssdwp = dssdw + dssdvsq * dvsq;

    return .{ .f = f, .df = D * dssdwp, .err = err };
}

fn residual(etype: Etype, Wp: f64, cons: [7]f64, pgamma: f64) Residual {
    return switch (etype) {
        .hot => fU2pHot(Wp, cons, pgamma),
        .entropy => fU2pEntropy(Wp, cons, pgamma),
    };
}

/// C's "unphysical W" guard used in the init and damping loops (u2p.c:1195):
/// true when W³(W+2B²) − (Q·B)²(2W+B²) ≤ W²(Q̃²−B⁴), i.e. v² would be ≥ 1.
fn wOutOfBounds(W: f64, Qtsq: f64, QdotBsq: f64, Bsq: f64, res: Residual) bool {
    return (W * W * W * (W + 2.0 * Bsq) - QdotBsq * (2.0 * W + Bsq)) <= W * W * (Qtsq - Bsq * Bsq) or
        !std.math.isFinite(res.f) or !std.math.isFinite(res.df);
}

/// C: u2p_solver_W (u2p.c:1024). Returns C's status code; on 0 the
/// MHD primitive slots of `pp` are overwritten.
pub fn u2pSolverW(
    comptime cfg: config.Config,
    uu: [layout.VarLayout(cfg).count]f64,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    etype: Etype,
    pgamma: f64,
) i32 {
    const L = layout.VarLayout(cfg);
    const gg = &geom.gg;
    const GG = &geom.GG;
    const gdetu_inv = 1.0 / geom.gdet; // GDETIN == 1
    const alpha = geom.alpha;

    const D = uu[L.index(.rho)] * gdetu_inv * alpha;
    const Sc = uu[L.index(.entr)] * gdetu_inv * alpha;

    const Qcov = [4]f64{
        (uu[L.index(.uu)] * gdetu_inv - uu[L.index(.rho)] * gdetu_inv) * alpha,
        uu[L.index(.vx)] * gdetu_inv * alpha,
        uu[L.index(.vy)] * gdetu_inv * alpha,
        uu[L.index(.vz)] * gdetu_inv * alpha,
    };
    const Qcovp = [4]f64{
        uu[L.index(.uu)] * gdetu_inv * alpha,
        uu[L.index(.vx)] * gdetu_inv * alpha,
        uu[L.index(.vy)] * gdetu_inv * alpha,
        uu[L.index(.vz)] * gdetu_inv * alpha,
    };
    const Qconp = relele.indices12(Qcovp, GG);
    const Qcon = relele.indices12(Qcov, GG);

    var Bcon: [4]f64 = @splat(0);
    var Bsq: f64 = 0;
    var QdotB: f64 = 0;
    if (comptime L.hasVar(.b1)) {
        Bcon = .{
            0,
            uu[L.index(.b1)] * gdetu_inv * alpha,
            uu[L.index(.b2)] * gdetu_inv * alpha,
            uu[L.index(.b3)] * gdetu_inv * alpha,
        };
        const Bcov = relele.indices21(Bcon, gg);
        Bsq = relele.dot(Bcon, Bcov);
        QdotB = relele.dot(Qcov, Bcon);
    }
    const QdotBsq = QdotB * QdotB;

    const ncov = [4]f64{ -alpha, 0, 0, 0 };
    const ncon = relele.indices12(ncov, GG);
    const Qn = Qcon[0] * ncov[0];

    // Q̃^ν = (δ^ν_μ + n^ν n_μ) Q^μ
    var Qtcon: [4]f64 = undefined;
    for (0..4) |i| {
        Qtcon[i] = 0;
        for (0..4) |j| {
            Qtcon[i] += (relele.kron(i, j) + ncon[i] * ncov[j]) * Qcon[j];
        }
    }
    const Qtcov = relele.indices21(Qtcon, gg);
    const Qt2 = relele.dot(Qtcon, Qtcov);
    const Qtsq = Qt2;

    const betasqoalphasq = gg[0][1] * GG[0][1] + gg[0][2] * GG[0][2] + gg[0][3] * GG[0][3];
    const alphasq = alpha * alpha;
    const Dfactor = (-geom.gttpert + alphasq * betasqoalphasq) / (alphasq + alpha);
    const Qdotnp = Qconp[0] * ncov[0] + D * Dfactor;

    // initial guess from current primitives (VELPRIM == VELR: no conversion)
    var rho = pp[L.index(.rho)];
    var uint = pp[L.index(.uu)];
    const utcon0 = [4]f64{ 0, pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] };
    var qsq: f64 = 0;
    for (1..4) |i| {
        for (1..4) |j| {
            qsq += utcon0[i] * utcon0[j] * gg[i][j];
        }
    }
    var gamma2 = 1.0 + qsq;
    var W = (rho + pgamma * uint) * gamma2;

    const cons = [7]f64{ Qn, Qt2, D, QdotBsq, Bsq, Sc, Qdotnp };
    const conv = u2pconv;

    // make sure the initial W gives v² < 1
    var i_increase: usize = 0;
    while (true) {
        const res = residual(etype, W - D, cons, pgamma);
        if (wOutOfBounds(W, Qtsq, QdotBsq, Bsq, res) and i_increase < 50) {
            W *= 10.0;
            i_increase += 1;
            continue;
        }
        break;
    }
    if (i_increase >= 50) return -150;

    // 1-D Newton
    var iter: usize = 0;
    var f0: f64 = undefined;
    var dfdW: f64 = undefined;
    var errr: f64 = undefined;
    while (iter < 50) {
        const Wprev = W;
        iter += 1;

        const res = residual(etype, W - D, cons, pgamma);
        f0 = res.f;
        dfdW = res.df;
        errr = res.err;

        if (errr < conv) break;
        if (dfdW == 0.0) {
            W *= 1.1;
            continue;
        }

        var Wnew = W - f0 / dfdW;
        var idump: usize = 0;
        var dumpfac: f64 = 1.0;
        while (true) {
            const rtmp = residual(etype, Wnew - D, cons, pgamma);
            if (wOutOfBounds(Wnew, Qtsq, QdotBsq, Bsq, rtmp) and idump < 100) {
                idump += 1;
                dumpfac /= 2.0;
                Wnew = W - dumpfac * f0 / dfdW;
                continue;
            }
            break;
        }
        if (idump >= 100) return -101;

        W = Wnew;
        if (@abs(W) > big) return -103;
        if (@abs((W - Wprev) / Wprev) < conv and errr < 1.0e-1) break;
    }
    if (iter >= 50) return -102;
    if (!std.math.isFinite(W)) return -103;

    // unpack the root
    const wsq = W * W;
    const xsq = (Bsq + W) * (Bsq + W);
    const v2 = (wsq * Qtsq + QdotBsq * (Bsq + 2.0 * W)) / (wsq * xsq);

    gamma2 = 1.0 / (1.0 - v2);
    const gam = @sqrt(gamma2);
    rho = D / gam;
    const entr = Sc / gam;
    uint = 1.0 / pgamma * (W / gamma2 - rho);
    var utcon: [4]f64 = undefined;
    utcon[0] = 0;
    for (1..4) |i| {
        utcon[i] = gam / (W + Bsq) * (Qtcon[i] + QdotB * Bcon[i] / W);
    }

    if (!std.math.isFinite(utcon[1])) return -120;
    if (uint < 0.0 or gamma2 < 0.0 or std.math.isNan(W) or !std.math.isFinite(W)) return -104;

    // VELR == VELPRIM: no conversion
    pp[L.index(.rho)] = rho;
    pp[L.index(.uu)] = uint;
    pp[L.index(.vx)] = utcon[1];
    pp[L.index(.vy)] = utcon[2];
    pp[L.index(.vz)] = utcon[3];

    if (rho < 0.0) return -105;

    pp[L.index(.entr)] = entr;

    if (comptime L.hasVar(.b1)) {
        pp[L.index(.b1)] = uu[L.index(.b1)] * gdetu_inv;
        pp[L.index(.b2)] = uu[L.index(.b2)] * gdetu_inv;
        pp[L.index(.b3)] = uu[L.index(.b3)] * gdetu_inv;
    }
    return 0;
}

pub const U2pResult = struct {
    /// C's u2p() return: 0 clean, -1 entropy used, -2 fixup requested, -3 both solvers failed.
    ret: i32,
    /// corrected[0]: the entropy equation was used.
    entropy_used: bool,
    /// fixups[0]: cell needs averaging from neighbors.
    fixup: bool,
};

/// The MHD part of C's u2p() cascade (u2p.c:131): hot energy inversion,
/// entropy fallback, primitives restored on total failure. The radiation
/// inversion (u2p_rad) arrives with M7.
pub fn u2pMhd(
    comptime cfg: config.Config,
    uu_in: [layout.VarLayout(cfg).count]f64,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    pgamma: f64,
    floors: FloorParams,
) U2pResult {
    const L = layout.VarLayout(cfg);
    var uu = uu_in;
    const ppbak = pp.*;
    var ppmhd = pp.*;

    const gdetu = geom.gdet; // GDETIN == 1
    var ret: i32 = 0;
    var entropy_used = false;

    // negative conserved density → floor it and demand a fixup (u2p.c:217)
    if (uu[L.index(.rho)] * geom.alpha / gdetu < 0.0) {
        ppmhd[L.index(.rho)] = floors.rhofloor;
        uu[L.index(.rho)] = floors.rhofloor * gdetu / geom.alpha;
        ret = -2;
    }

    var u2pret = u2pSolverW(cfg, uu, &ppmhd, geom, .hot, pgamma);

    if (u2pret < 0) { // ALLOWENTROPYU2P == 1
        ret = -1;
        entropy_used = true;
        u2pret = u2pSolverW(cfg, uu, &ppmhd, geom, .entropy, pgamma);
        if (u2pret < 0) ret = -2;
    }

    if (u2pret >= 0) {
        pp.* = ppmhd;
    } else {
        pp.* = ppbak; // leave primitives unchanged
        ret = -3;
    }

    return .{
        .ret = ret,
        .entropy_used = entropy_used,
        .fixup = ret < -1,
    };
}

/// C: check_floors_mhd (u2p.c:416) with PUFFY's build: no RHOFLOOR_BH/INIT,
/// B2RHOFLOORFRAME == DRIFTFRAME (Ressler+2017), VELPRIM == VELR,
/// no electron/relel species. Returns -1 if anything was floored, else 0;
/// pp is modified in place and ENTR re-synced after a floor hits.
pub fn checkFloorsMhd(
    comptime cfg: config.Config,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    pgamma: f64,
    floors: FloorParams,
) relele.Error!i32 {
    const L = layout.VarLayout(cfg);
    const gg = &geom.gg;
    const GG = &geom.GG;
    var ret: i32 = 0;

    // rho too small
    if (pp[L.index(.rho)] < floors.rhofloor) {
        pp[L.index(.rho)] = floors.rhofloor;
        ret = -1;
    }

    // uint too small (too cold)
    if (pp[L.index(.uu)] < floors.uurhoratiomin * pp[L.index(.rho)]) {
        pp[L.index(.uu)] = floors.uurhoratiomin * pp[L.index(.rho)];
        ret = -1;
    }

    // uint too large (too hot)
    if (pp[L.index(.uu)] > floors.uurhoratiomax * pp[L.index(.rho)]) {
        pp[L.index(.uu)] = floors.uurhoratiomax * pp[L.index(.rho)];
        ret = -1;
    }

    // too magnetized → add mass/energy in the drift frame (Ressler+2017)
    if (comptime L.hasVar(.b1)) {
        const u = try relele.uconUcovFromPrims(
            .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
            geom,
        );
        const b = mhd.bconBcovBsqFrom4vel(
            .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
            u.con,
            u.cov,
            gg,
        );

        var f: f64 = 1.0;
        var fuu: f64 = 1.0;
        var floored = false;
        if (b.bsq > floors.b2rhoratiomax * pp[L.index(.rho)]) {
            f = b.bsq / (floors.b2rhoratiomax * pp[L.index(.rho)]);
            floored = true;
        }
        if (b.bsq > floors.b2uuratiomax * pp[L.index(.uu)]) {
            fuu = b.bsq / (floors.b2uuratiomax * pp[L.index(.uu)]);
            floored = true;
        }

        if (floored) {
            const wold = pp[L.index(.rho)] + pp[L.index(.uu)] * pgamma;
            pp[L.index(.rho)] *= f;
            pp[L.index(.uu)] *= fuu;
            const wnew = pp[L.index(.rho)] + pp[L.index(.uu)] * pgamma;

            const betapar = -b.bcon[0] / (b.bsq * u.con[0]);
            var betasq = betapar * betapar * b.bsq;
            const betasqmax = 1.0 - 1.0 / (floors.gammamaxhd * floors.gammamaxhd);
            if (betasq > betasqmax) betasq = betasqmax;
            const gammapar = 1.0 / @sqrt(1.0 - betasq);

            var ucondr: [4]f64 = undefined;
            for (0..4) |iv| ucondr[iv] = gammapar * (u.con[iv] + betapar * b.bcon[iv]);

            const Bcon = [4]f64{ 0, pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] };
            const Bcov = relele.indices21(Bcon, gg);
            const Bmag = @sqrt(relele.dot(Bcon, Bcov));

            const udotB = relele.dot(u.con, Bcov);
            const QdotB = udotB * wold * u.con[0];

            const xx = 2.0 * QdotB / (Bmag * wnew * ucondr[0]);
            const vpar = xx / (ucondr[0] * (1.0 + @sqrt(1.0 + xx * xx)));

            var vcon: [4]f64 = undefined;
            vcon[0] = 1.0;
            for (1..4) |iv| {
                vcon[iv] = vpar * Bcon[iv] / Bmag + ucondr[iv] / ucondr[0];
            }

            const ucont = try relele.convVels(vcon, .vel3, .velr, gg, GG);
            pp[L.index(.vx)] = ucont[1];
            pp[L.index(.vy)] = ucont[2];
            pp[L.index(.vz)] = ucont[3];
            ret = -1;
        }
    }

    // too fast (VELPRIM == VELR)
    {
        var qsq: f64 = 0;
        const vx = L.index(.vx);
        for (1..4) |i| {
            for (1..4) |j| {
                qsq += pp[vx - 1 + i] * pp[vx - 1 + j] * gg[i][j];
            }
        }
        const gamma2 = 1.0 + qsq;
        if (gamma2 > floors.gammamaxhd * floors.gammamaxhd) {
            const qsqmax = floors.gammamaxhd * floors.gammamaxhd - 1.0;
            const a = @sqrt(qsqmax / qsq);
            for (1..4) |j| pp[vx - 1 + j] *= a;
            ret = -1;
        }
    }

    // re-sync entropy after any floor
    if (ret < 0) {
        pp[L.index(.entr)] = hydro.sFromU(pp[L.index(.rho)], pp[L.index(.uu)], pgamma);
    }
    return ret;
}
