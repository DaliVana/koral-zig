//! Frame changes: Lorentz boosts between the lab (normal-observer) frame
//! and the fluid / radiation rest frames, and coordinate transforms of
//! vectors, tensors and whole primitive states (C: frames.c).
//!
//! The boost matrix is L = δ + Ω + Ω²/(1+γ) with Ω^μ_ν = u^μ w_ν − w^μ u_ν,
//! where u is the target frame's four-velocity and w the source frame's
//! (γ = −w·u). L is an exact Lorentz transform of the coordinate-basis
//! metric: g(Lx, Ly) = g(x, y).
//!
//! Faithful C quirks (verified against frames.c and mirrored exactly):
//! - lab2ff boosts multiply the 0-row/column of the *output* by α
//!   ("orthonormality correction", frames.c:396); T[0][0] gets α twice.
//! - ff2lab's corresponding α *division* is dead code — C divides the input
//!   array only after the working copy was taken (frames.c:435-445), so the
//!   output carries no α. The two directions are therefore NOT mutual
//!   inverses of each other (the bare L matrices are).
//! - boost22_lab2rf's α multiplication is likewise dead (recopied at :600).
//!   rf2lab (:534) does divide the whole output tensor by α.
//! - C's trans_pall_coco only works when ppin and ppout alias: the init
//!   loop of trans_prad_coco (frames.c:154) copies ppin over ALL of ppout,
//!   clobbering trans_pmhd_coco's result if the arrays are distinct. Our
//!   functional chaining in transPallCoco matches the aliased behavior.

const std = @import("std");
const relele = @import("relele.zig");
const mhd = @import("physics/bfield.zig");
const coco = @import("metric/coco.zig");
const config = @import("config.zig");
const layout = @import("layout.zig");
const Geometry = @import("geometry.zig").Geometry;
const MetricParams = @import("metric/forms.zig").MetricParams;

const VelType = relele.VelType;

//
// ---- Lorentz matrices (C: frames.c:202, :301) -----------------------------
//

fn lorentzFromPair(ucon: [4]f64, ucov: [4]f64, wcon: [4]f64, wcov: [4]f64) [4][4]f64 {
    var om: [4][4]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            om[i][j] = ucon[i] * wcov[j] - wcon[i] * ucov[j];
        }
    }
    const gam = -relele.dot(wcon, ucov);

    var l: [4][4]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            var omsum: f64 = 0;
            for (0..4) |k| {
                omsum += om[i][k] * om[k][j];
            }
            l[i][j] = relele.kron(i, j) + 1.0 / (1.0 + gam) * omsum + om[i][j];
        }
    }
    return l;
}

fn normalObsConCov(GG: *const [4][5]f64) relele.ConCov {
    const alpha = @sqrt(-1.0 / GG[0][0]);
    const wcov = [4]f64{ -alpha, 0, 0, 0 };
    return .{ .con = relele.indices12(wcov, GG), .cov = wcov };
}

/// C: calc_Lorentz_lab2ff — u = fluid (from VELPRIM prims), w = normal observer.
pub fn lorentzLab2Ff(vprim: [3]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) relele.Error![4][4]f64 {
    const u = try relele.convVelsBoth(.{ 0, vprim[0], vprim[1], vprim[2] }, .velr, gg, GG);
    const w = normalObsConCov(GG);
    return lorentzFromPair(u.con, u.cov, w.con, w.cov);
}

/// C: calc_Lorentz_ff2lab — u = normal observer, w = fluid.
pub fn lorentzFf2Lab(vprim: [3]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) relele.Error![4][4]f64 {
    const w = try relele.convVelsBoth(.{ 0, vprim[0], vprim[1], vprim[2] }, .velr, gg, GG);
    const u = normalObsConCov(GG);
    return lorentzFromPair(u.con, u.cov, w.con, w.cov);
}

//
// ---- vector boosts (C: frames.c:634, :790, :732) --------------------------
//

/// A2 = α · (L_lab2ff A1).
pub fn boost2Lab2Ff(a1: [4]f64, vprim: [3]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) relele.Error![4]f64 {
    const l = try lorentzLab2Ff(vprim, gg, GG);
    var a2 = relele.multiply2(a1, l);
    const alpha = @sqrt(-1.0 / GG[0][0]);
    for (0..4) |i| a2[i] *= alpha;
    return a2;
}

/// A2 = L_ff2lab (A1 / α).
pub fn boost2Ff2Lab(a1: [4]f64, vprim: [3]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) relele.Error![4]f64 {
    const l = try lorentzFf2Lab(vprim, gg, GG);
    const alpha = @sqrt(-1.0 / GG[0][0]);
    var at: [4]f64 = undefined;
    for (0..4) |i| at[i] = a1[i] / alpha;
    return relele.multiply2(at, l);
}

/// A2 = L_lab2ff (α A1), with L built from the radiation rest-frame velocity
/// (C: boost2_lab2rf; VELPRIMRAD == VELPRIM == VELR so the substitution is a copy).
pub fn boost2Lab2Rf(a1: [4]f64, vradprim: [3]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) relele.Error![4]f64 {
    const l = try lorentzLab2Ff(vradprim, gg, GG);
    const alpha = @sqrt(-1.0 / GG[0][0]);
    var at: [4]f64 = undefined;
    for (0..4) |i| at[i] = a1[i] * alpha;
    return relele.multiply2(at, l);
}

//
// ---- tensor boosts (C: frames.c:352, :416, :478, :556) --------------------
//

fn boostTensor(t1: [4][4]f64, l: [4][4]f64) [4][4]f64 {
    return relele.multiply22(t1, l);
}

/// T2 = L L T1, then 0-row/col × α (C: boost22_lab2ff).
pub fn boost22Lab2Ff(t1: [4][4]f64, vprim: [3]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) relele.Error![4][4]f64 {
    const l = try lorentzLab2Ff(vprim, gg, GG);
    var t2 = boostTensor(t1, l);
    const alpha = @sqrt(-1.0 / GG[0][0]);
    for (0..4) |i| {
        t2[i][0] *= alpha;
        t2[0][i] *= alpha;
    }
    return t2;
}

/// T2 = L L T1 — no α correction reaches the output in C (dead code, see top).
pub fn boost22Ff2Lab(t1: [4][4]f64, vprim: [3]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) relele.Error![4][4]f64 {
    const l = try lorentzFf2Lab(vprim, gg, GG);
    return boostTensor(t1, l);
}

/// Radiation-rest-frame → lab: T2 = (L L T1) / α (C: boost22_rf2lab).
pub fn boost22Rf2Lab(t1: [4][4]f64, vradprim: [3]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) relele.Error![4][4]f64 {
    const l = try lorentzFf2Lab(vradprim, gg, GG);
    var t2 = boostTensor(t1, l);
    const alpha = @sqrt(-1.0 / GG[0][0]);
    for (0..4) |i| {
        for (0..4) |j| {
            t2[i][j] /= alpha;
        }
    }
    return t2;
}

/// Lab → radiation rest frame: T2 = L L T1 (C's α multiply is dead, see top).
pub fn boost22Lab2Rf(t1: [4][4]f64, vradprim: [3]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) relele.Error![4][4]f64 {
    const l = try lorentzLab2Ff(vradprim, gg, GG);
    return boostTensor(t1, l);
}

//
// ---- coordinate transforms (C: frames.c:992 trans2_coco, :1207 trans22) ---
//

/// u^μ between coordinate systems (C: trans2_coco). Chains through KS for
/// composite pairs exactly like C (Jacobian product, intermediate point).
pub fn trans2Coco(x: [4]f64, u: [4]f64, from: config.Coords, to: config.Coords, mp: MetricParams) [4]f64 {
    if (from == to) return u;
    return relele.multiply2(u, coco.dxdx(x, from, to, mp));
}

/// T^μν between coordinate systems (C: trans22_coco).
pub fn trans22Coco(x: [4]f64, t: [4][4]f64, from: config.Coords, to: config.Coords, mp: MetricParams) [4][4]f64 {
    if (from == to) return t;
    return relele.multiply22(t, coco.dxdx(x, from, to, mp));
}

//
// ---- whole-primitive-state transforms (C: frames.c:29, :141, :14) ---------
//

/// MHD primitives between coordinates (C: trans_pmhd_coco). Velocity slots
/// go VELPRIM → VEL4 → (Jacobian) → VELPRIM; B^i via b^μ. The point is
/// geom1's position; geom1/geom2 supply the metric in each system.
pub fn transPmhdCoco(
    comptime cfg: config.Config,
    ppin: [layout.VarLayout(cfg).count]f64,
    geom1: *const Geometry,
    geom2: *const Geometry,
    mp: MetricParams,
) relele.Error![layout.VarLayout(cfg).count]f64 {
    const L = layout.VarLayout(cfg);
    var pp2 = ppin;
    if (geom1.coords == geom2.coords) return pp2;

    const ug1 = try relele.uconUcovFromPrims(
        .{ ppin[L.index(.vx)], ppin[L.index(.vy)], ppin[L.index(.vz)] },
        geom1,
    );

    var bcon: [4]f64 = @splat(0);
    if (comptime L.hasVar(.b1)) {
        bcon = mhd.bconFrom4vel(
            .{ ppin[L.index(.b1)], ppin[L.index(.b2)], ppin[L.index(.b3)] },
            ug1.con,
            ug1.cov,
        );
        bcon = trans2Coco(geom1.xxvec, bcon, geom1.coords, geom2.coords, mp);
    }

    var ucon = trans2Coco(geom1.xxvec, ug1.con, geom1.coords, geom2.coords, mp);
    ucon = try relele.convVelsUt(ucon, .vel4, .velr, &geom2.gg, &geom2.GG);

    pp2[L.index(.vx)] = ucon[1];
    pp2[L.index(.vy)] = ucon[2];
    pp2[L.index(.vz)] = ucon[3];

    if (comptime L.hasVar(.b1)) {
        // C: calc_Bcon_prim recomputes the four-velocity from the *new* prims.
        const ug2 = try relele.uconUcovFromPrims(
            .{ pp2[L.index(.vx)], pp2[L.index(.vy)], pp2[L.index(.vz)] },
            geom2,
        );
        const Bcon = mhd.bigBconFrom4vel(bcon, ug2.con);
        pp2[L.index(.b1)] = Bcon[1];
        pp2[L.index(.b2)] = Bcon[2];
        pp2[L.index(.b3)] = Bcon[3];
    }
    return pp2;
}

/// Radiative primitives between coordinates (C: trans_prad_coco); Erf is a
/// scalar, the rad-frame velocity transforms like the gas velocity.
pub fn transPradCoco(
    comptime cfg: config.Config,
    ppin: [layout.VarLayout(cfg).count]f64,
    geom1: *const Geometry,
    geom2: *const Geometry,
    mp: MetricParams,
) relele.Error![layout.VarLayout(cfg).count]f64 {
    const L = layout.VarLayout(cfg);
    var pp2 = ppin;
    if (comptime !L.hasVar(.ee)) return pp2;
    if (geom1.coords == geom2.coords) return pp2;

    var ucon = try relele.convVels(
        .{ 0, ppin[L.index(.fx)], ppin[L.index(.fy)], ppin[L.index(.fz)] },
        .velr,
        .vel4,
        &geom1.gg,
        &geom1.GG,
    );
    ucon = trans2Coco(geom1.xxvec, ucon, geom1.coords, geom2.coords, mp);
    ucon = try relele.convVelsUt(ucon, .vel4, .velr, &geom2.gg, &geom2.GG);

    pp2[L.index(.fx)] = ucon[1];
    pp2[L.index(.fy)] = ucon[2];
    pp2[L.index(.fz)] = ucon[3];
    return pp2;
}

/// All primitives (C: trans_pall_coco).
pub fn transPallCoco(
    comptime cfg: config.Config,
    ppin: [layout.VarLayout(cfg).count]f64,
    geom1: *const Geometry,
    geom2: *const Geometry,
    mp: MetricParams,
) relele.Error![layout.VarLayout(cfg).count]f64 {
    var pp2 = try transPmhdCoco(cfg, ppin, geom1, geom2, mp);
    pp2 = try transPradCoco(cfg, pp2, geom1, geom2, mp);
    return pp2;
}
