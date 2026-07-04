//! Radiation–gas coupling: the state fill (the radiation-relevant subset
//! of C's fill_struct_of_state, physics.c:12), the thermal four-force
//! calc_Gi (rad.c:2653 calc_all_Gi_with_state) with its Comptonization
//! term (rad.c:2907), and the pp-level opacity wrappers calc_kappa /
//! calc_kappaes / calc_chi (opacities.c).
//!
//! C-fidelity notes:
//!  * Two kappaes flavors coexist in C: fill_struct_of_state calls
//!    calc_kappaes_with_temperatures with Trad = TradBB (used by calc_Gi),
//!    while the standalone calc_kappaes(pp,ggg) — the one calc_chi and the
//!    wavespeed limiter use — sets Trad = Te. Both preserved here.
//!  * calc_all_Gi_with_state's return value is 1/(Tgas²·kappa) with a
//!    NEVER-ASSIGNED local kappa — uninitialized stack garbage. Every call
//!    site discards it; we return void.
//!  * Gith_ff is the α-scaled lab→ff boost of Gith_lab with the time
//!    component then overwritten by the exact fluid-frame expression
//!    −κ_gasAbs·4πB + κ_radAbs·Ê (rad.c:2724-2727).
//!  * COMPTONIZATION is on in PUFFY; DAMPCOMPTONIZATIONATBH is not.

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const relele = @import("../relele.zig");
const frames = @import("../frames.zig");
const hydro = @import("hydro.zig");
const mhd = @import("mhd.zig");
const radiation = @import("radiation.zig");
const thermo = @import("thermo.zig");
const opacities = @import("opacities.zig");
const units_mod = @import("../units.zig");
const Geometry = @import("../geometry.zig").Geometry;

/// Everything opacity/four-force evaluation needs besides the state.
pub const Params = struct {
    c: thermo.Consts,
    ch: opacities.Channels = .{},

    pub fn init(u: units_mod.Units, comp: thermo.Composition) Params {
        return .{ .c = thermo.Consts.init(u, comp) };
    }

    /// The PUFFY build: MASS = 10 M☉, HFRAC = 1 with the direct
    /// MU_GAS/MU_I/MU_E = 1/2/2 overrides, all channels on.
    pub fn puffy() Params {
        return init(units_mod.Units.init(10.0), thermo.Composition.puffy);
    }
};

/// The radiation-relevant subset of C's struct_of_state.
pub const RadState = struct {
    rho: f64,
    uint: f64,
    tgas: f64,
    te: f64,
    ti: f64,
    /// Sgas = kB/(μ_gas m_p) · calc_Sfromu (physics.c:41) — note the
    /// different normalization from the ENTR primitive
    sgas: f64,
    ne: f64,
    ucon: [4]f64,
    ucov: [4]f64,
    bsq: f64,
    rij: [4][4]f64,
    ehat: f64,
    tradbb: f64,
    trad: f64,
    kappaes: f64,
    kappa: f64,
    opac: opacities.Opac,
};

/// C: fill_struct_of_state (physics.c:12), the RADIATION branch without
/// photon number (Trad = TradBB) or electrons (Te = Ti = Tgas).
pub fn fillRadState(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gam: f64,
    p: *const Params,
) relele.Error!RadState {
    const L = layout.VarLayout(cfg);
    const c = &p.c;

    const rho = pp[L.index(.rho)];
    const uint = pp[L.index(.uu)];
    const temps = thermo.tempsFromUrho(c, uint, rho, gam);
    const sgas = c.kb_over_mugas_mp * hydro.sFromU(rho, uint, gam);
    const ne = thermo.thermalNe(c, rho);

    const u = try relele.uconUcovFromPrims(
        .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
        geom,
    );

    var bsq: f64 = 0;
    if (comptime L.hasVar(.b1)) {
        const b = mhd.bconBcovBsqFrom4vel(
            .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
            u.con,
            u.cov,
            &geom.gg,
        );
        bsq = b.bsq;
    }

    const rij = try radiation.calcRij(cfg, pp, geom);

    // C: calc_Ehat_from_Rij_ucov (rad.c:3054) — Ê = R^ab u_a u_b
    var ehat: f64 = 0;
    for (0..4) |i| {
        for (0..4) |j| {
            ehat += rij[i][j] * u.cov[i] * u.cov[j];
        }
    }

    const tradbb = c.lteTfromE(ehat);
    const trad = tradbb; // no EVOLVEPHOTONNUMBER

    const kappaes = opacities.kappaEsPuffy(c, p.ch, rho, trad);

    const kr = opacities.kappaFromState(c, p.ch, .{
        .rho = rho,
        .tgas = temps.tgas,
        .te = temps.te,
        .trad = trad,
        .tradbb = tradbb,
        .ne = ne,
        .bsq = bsq,
    });

    return .{
        .rho = rho,
        .uint = uint,
        .tgas = temps.tgas,
        .te = temps.te,
        .ti = temps.ti,
        .sgas = sgas,
        .ne = ne,
        .ucon = u.con,
        .ucov = u.cov,
        .bsq = bsq,
        .rij = rij,
        .ehat = ehat,
        .tradbb = tradbb,
        .trad = trad,
        .kappaes = kappaes,
        .kappa = kr.kappa,
        .opac = kr.opac,
    };
}

pub const Gi = struct {
    /// fluid-frame thermal four-force (C: type 0/2)
    ff: [4]f64,
    /// lab-frame thermal four-force (C: type 1/3)
    lab: [4]f64,
};

/// C: calc_Compt_Gi_with_state (rad.c:2907) — thermal Comptonization,
/// no RELELECTRONS correction. Returns coeff so both frames reuse it.
pub fn comptComptonCoeff(c: *const thermo.Consts, st: *const RadState) f64 {
    const thetae = c.kb_over_me * st.te;
    return st.kappaes * st.ehat * (4.0 * c.kb_over_me * (st.trad - st.te)) *
        (1.0 + 3.683 * thetae + 4.0 * thetae * thetae) / (1.0 + 4.0 * thetae);
}

/// C: calc_all_Gi_with_state (rad.c:2653), thermal part (no RELELECTRONS,
/// so total == thermal). `vprim` are the gas velocity primitives used for
/// the lab→ff boost.
pub fn calcGiFromState(
    st: *const RadState,
    vprim: [3]f64,
    geom: *const Geometry,
    p: *const Params,
) relele.Error!Gi {
    const c = &p.c;
    const b = c.sigma_rad_over_pi * st.te * st.te * st.te * st.te;

    const k_rad_abs = st.opac.rad_abs;
    const k_gas_abs = st.opac.gas_abs;
    const k_rad_ross = st.opac.rad_ross;
    const kappaes = st.kappaes;
    const ruu = st.ehat;

    var gi_lab: [4]f64 = undefined;
    for (0..4) |i| {
        var ru: f64 = 0;
        for (0..4) |j| {
            ru += st.rij[i][j] * st.ucov[j];
        }
        gi_lab[i] = -(k_rad_ross + kappaes) * ru -
            ((k_rad_ross + kappaes - k_rad_abs) * ruu + k_gas_abs * c.fourpi * b) * st.ucon[i];
    }

    // boost to the fluid frame, then rewrite the time component directly
    var gi_ff = try frames.boost2Lab2Ff(gi_lab, vprim, &geom.gg, &geom.GG);
    gi_ff[0] = -k_gas_abs * c.fourpi * b + k_rad_abs * st.ehat;

    if (p.ch.comptonization) {
        const coeff = comptComptonCoeff(c, st);
        // lab: coeff·u^μ; ff: coeff·(1,0,0,0)
        for (0..4) |i| gi_lab[i] += coeff * st.ucon[i];
        gi_ff[0] += coeff;
    }

    return .{ .ff = gi_ff, .lab = gi_lab };
}

/// C: calc_Gi (rad.c:2588). The C return value is uninitialized garbage
/// discarded by all callers — not reproduced.
pub fn calcGi(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gam: f64,
    p: *const Params,
) relele.Error!Gi {
    const L = layout.VarLayout(cfg);
    const st = try fillRadState(cfg, pp, geom, gam, p);
    return calcGiFromState(&st, .{
        pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)],
    }, geom, p);
}

/// C: calc_kappa (opacities.c:22) — state fill + default opacity code.
pub fn calcKappa(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gam: f64,
    p: *const Params,
) relele.Error!f64 {
    const st = try fillRadState(cfg, pp, geom, gam, p);
    return st.kappa;
}

/// C: calc_kappaes (opacities.c:126) — the standalone flavor with
/// Trad = Te (NOT TradBB).
pub fn calcKappaes(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    gam: f64,
    p: *const Params,
) f64 {
    const L = layout.VarLayout(cfg);
    const temps = thermo.tempsFromUrho(&p.c, pp[L.index(.uu)], pp[L.index(.rho)], gam);
    const trad = temps.te;
    return opacities.kappaEsPuffy(&p.c, p.ch, pp[L.index(.rho)], trad);
}

/// C: calc_chi (opacities.c:148) — κ + κ_es for the wavespeed τ-limiter.
pub fn calcChi(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gam: f64,
    p: *const Params,
) relele.Error!f64 {
    const kappa = try calcKappa(cfg, pp, geom, gam, p);
    return kappa + calcKappaes(cfg, pp, gam, p);
}
