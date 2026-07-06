//! Advective fluxes of the conserved variables through a face in direction
//! `idim` (C: f_flux_prime, physics.c:1167 + f_flux_prime_rad, rad.c:3858).
//!
//! The energy flux uses the same cancellation-free utp1 assembly as p2u.
//! The radiative rows here are the pure M1 tensor; PUFFY's shear-viscosity
//! correction (Rijvisc) is added on top at the faces by the sweep
//! (sim.rijviscFace + radvisc.addRadViscFlux, M12), so it is absent from
//! this function's golden records (harness_flux zeroes Rijviscglobal).

const std = @import("std");
const relele = @import("../relele.zig");
const mhd = @import("bfield.zig");
const hydro = @import("hydro.zig");
const radiation = @import("radiation.zig");
const p2u_mod = @import("../p2u.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const Geometry = @import("../geometry.zig").Geometry;

/// C: f_flux_prime — flux vector at the face whose geometry is `geom`
/// (idim: 0/1/2 for x/y/z). GDETIN == 1: gdetu = geom.gdet.
pub fn fFluxPrime(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    idim: usize,
    geom: *const Geometry,
    gamma_adiab: f64,
) relele.Error![layout.VarLayout(cfg).count]f64 {
    const L = layout.VarLayout(cfg);
    var ff: [L.count]f64 = @splat(0);
    const gdetu = geom.gdet;

    // T^μν → T^μ_ν
    const tij_uu = try hydro.calcTij(cfg, pp, geom, gamma_adiab);
    const tij_ud = relele.indices2221(tij_uu, &geom.gg);

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

    const pre = (gamma_adiab - 1.0) * ugas;
    const etap = ugas + pre + bsq; // eta - rho

    const utp1 = p2u_mod.calcUtp1(vcon, geom);

    // hydro rows
    ff[L.index(.rho)] = gdetu * rho * u.con[idim + 1];
    ff[L.index(.entr)] = gdetu * s * u.con[idim + 1];

    // energy row without the T^i_t + rho u^i cancellation (physics.c:1261)
    ff[L.index(.uu)] = gdetu * (etap * u.con[idim + 1] * u.cov[0] + rho * u.con[idim + 1] * utp1);
    if (comptime L.hasVar(.b1)) {
        ff[L.index(.uu)] += -gdetu * bcon[idim + 1] * bcov[0];
    }

    ff[L.index(.vx)] = gdetu * tij_ud[idim + 1][1];
    ff[L.index(.vy)] = gdetu * tij_ud[idim + 1][2];
    ff[L.index(.vz)] = gdetu * tij_ud[idim + 1][3];

    // magnetic rows: induction-equation fluxes b^i u^d − b^d u^i
    if (comptime L.hasVar(.b1)) {
        ff[L.index(.b1)] = gdetu * (bcon[1] * u.con[idim + 1] - bcon[idim + 1] * u.con[1]);
        ff[L.index(.b2)] = gdetu * (bcon[2] * u.con[idim + 1] - bcon[idim + 1] * u.con[2]);
        ff[L.index(.b3)] = gdetu * (bcon[3] * u.con[idim + 1] - bcon[idim + 1] * u.con[3]);
    }

    // radiative rows: R^d_ν (pure M1; + Rijvisc in M12)
    if (comptime L.hasVar(.ee)) {
        const rij_uu = try radiation.calcRij(cfg, pp, geom);
        const rij_ud = relele.indices2221(rij_uu, &geom.gg);
        for (0..4) |nu| {
            ff[L.index(.ee) + nu] = gdetu * rij_ud[idim + 1][nu];
        }
    }

    return ff;
}
