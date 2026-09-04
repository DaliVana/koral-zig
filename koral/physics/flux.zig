//! Advective fluxes of the conserved variables through a face in direction
//! `idim` (C: f_flux_prime, physics.c:1167 + f_flux_prime_rad, rad.c:3858).
//!
//! Physics: this is F in the finite-volume law ∂_t(√−g U) + ∂_i(√−g F^i) = S;
//! the √−g factor is what turns the curvilinear divergence into a plain
//! difference of face fluxes. Mass and entropy advect as ρu^i and S·u^i;
//! the momentum rows are the mixed stress tensor T^i_j; the energy row is
//! T^i_t, assembled cancellation-free; in the Newtonian limit
//! T^t_t ≈ −ρ(1 + small), so the evolved energy is T^t_t + ρu^t and its
//! flux uses u_t + 1 (calcUtp1) to dodge the rest-mass cancellation. The
//! induction rows b^i u^j − b^j u^i are the dual Maxwell tensor: field
//! lines advect with the flow, and the normal component's self-flux
//! vanishes by antisymmetry. The radiative rows are the M1 stress tensor
//! R^i_ν (physics/radiation.zig).
//!
//! The energy flux uses the same cancellation-free utp1 assembly as p2u.
//! The radiative rows here are the pure M1 tensor; PUFFY's shear-viscosity
//! correction (Rijvisc) is added on top at the faces by the sweep
//! (sim/rijvisc.zig faceAvg + addRadViscFlux, M12), so it is absent
//! from this function's golden records (harness_flux zeroes Rijviscglobal).
//!
//! Divergence from C: C's f_flux_prime calls my_err + exit(-1) on a NaN stress
//! tensor (physics.c:1230-1247); we return error.NanInFlux instead so the
//! sweep's `try` carries it to the driver's step-failure path (see the check
//! at the end of fFluxPrime).

const std = @import("std");
const relele = @import("../relele.zig");
const mhd = @import("bfield.zig");
const hydro = @import("hydro.zig");
const radiation = @import("radiation.zig");
const p2u_mod = @import("../p2u.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const Geometry = @import("../geometry.zig").Geometry;

/// C: f_flux_prime. Flux vector at the face whose geometry is `geom`
/// (idim: 0/1/2 for x/y/z). GDETIN == 1: gdetu = geom.gdet. Returns
/// relele.Error.NanInFlux if the assembled flux is non-finite (see below).
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

    const rho = pp[L.index(.rho)];
    const ugas = pp[L.index(.uu)];
    const s = pp[L.index(.entr)];

    // Gas 4-velocity and magnetic four-vector, solved once and shared with the
    // stress tensor below (finding #3: calcTij used to re-derive both from the
    // same pp/geom). convertBoth + bconBcovBsqFrom4vel are pure, so calcTij's
    // internal result and this one are the same bits.
    const vcon = [4]f64{ 0, pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] };
    const u = try relele.convertBoth(vcon, .velr, geom);

    var bcon: [4]f64 = @splat(0);
    var bcov: [4]f64 = @splat(0);
    var bsq: f64 = 0;
    if (comptime L.hasVar(.b1)) {
        const b = mhd.bconBcovBsqFrom4vel(
            .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
            u.con,
            u.cov,
            geom,
        );
        bcon = b.bcon;
        bcov = b.bcov;
        bsq = b.bsq;
    }

    // T^μν → T^μ_ν, but only the row this face flux consumes (idim+1):
    // ff[vx/vy/vz] read tij_ud[idim+1][1..3]. Row-restricted lowering is
    // bit-identical to lowerSecond(...)[idim+1] at ¼ the madds (P2 #7).
    const tij_uu = hydro.calcTijFromState(cfg, pp, geom, gamma_adiab, u, bcon, bsq);
    const tij_ud_row = relele.lowerSecondRow(tij_uu, geom, idim + 1);

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

    ff[L.index(.vx)] = gdetu * tij_ud_row[1];
    ff[L.index(.vy)] = gdetu * tij_ud_row[2];
    ff[L.index(.vz)] = gdetu * tij_ud_row[3];

    // magnetic rows: induction-equation fluxes b^i u^d − b^d u^i
    if (comptime L.hasVar(.b1)) {
        ff[L.index(.b1)] = gdetu * (bcon[1] * u.con[idim + 1] - bcon[idim + 1] * u.con[1]);
        ff[L.index(.b2)] = gdetu * (bcon[2] * u.con[idim + 1] - bcon[idim + 1] * u.con[2]);
        ff[L.index(.b3)] = gdetu * (bcon[3] * u.con[idim + 1] - bcon[idim + 1] * u.con[3]);
    }

    // radiative rows: R^d_ν (pure M1; + Rijvisc in M12) — only row idim+1 is
    // read, so lower just that row (bit-identical to lowerSecond(...)[idim+1]).
    if (comptime L.hasVar(.ee)) {
        const rij_uu = try radiation.calcRij(cfg, pp, geom);
        const rij_ud_row = relele.lowerSecondRow(rij_uu, geom, idim + 1);
        for (0..4) |nu| {
            ff[L.index(.ee) + nu] = gdetu * rij_ud_row[nu];
        }
    }

    // C: f_flux_prime hard-aborts on any NaN in the stress tensor right after
    // calc_Tij ('nan in flux_prime', physics.c:1230-1247 → my_err + exit(-1),
    // compiled in for PUFFY). We check the assembled flux instead (so the
    // radiative rows are covered too) and surface it as error.NanInFlux, which
    // the sweep's `try` (sim/explicit.zig) propagates to the driver's step-failure path.
    // The point is to pin the origin cell and close the between-outputs window
    // where cell_fixup could otherwise neighbour-average a transient NaN away —
    // making the run finish with quietly wrong physics and no trace. We do NOT
    // exit(-1) (Zig-native error propagation); healthy states never trip this,
    // so golden records are unaffected.
    for (ff) |v| {
        if (!std.math.isFinite(v)) return error.NanInFlux;
    }

    return ff;
}
