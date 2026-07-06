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
const simd = @import("../math/simd.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const relele = @import("../relele.zig");
const frames = @import("../frames.zig");
const hydro = @import("hydro.zig");
const mhd = @import("bfield.zig");
const radiation = @import("radiation.zig");
const thermo = @import("thermo.zig");
const opacities = @import("opacities.zig");
const units_mod = @import("../units.zig");
const Geometry = @import("../geometry.zig").Geometry;

/// C: PR_KAPPA — the per-problem absorption-opacity hook textually included
/// into calc_kappa_from_state (opacities.c:72). `.default` is the no-PR_KAPPA
/// branch (calc_opacities_from_state). `.grey` mirrors a problem kappa.c of
/// the shape `kappa = COEFF*rho;` that also assigns all six opac slots (the
/// C wrapper otherwise reads UNINITIALIZED stack in its `kappaGasAbs >= 0.`
/// broadcast test — LRTORUS/kappa.c really does that; our oracle problems
/// set the slots explicitly to stay deterministic).
pub const KappaMode = union(enum) {
    default,
    /// κ = coeff·ρ (both in geometrical units), all slots equal,
    /// totEmissivity = κ·4πB.
    grey: f64,
};

/// C: PR_KAPPAES — the scattering hook (opacities.c:114/138). Without it C
/// returns 0; PUFFY's kappaes.c is the Klein–Nishina-corrected Thomson value.
pub const KappaesMode = union(enum) {
    none,
    puffy,
    /// κ_es = coeff·ρ (both in geometrical units).
    grey: f64,
};

/// Everything opacity/four-force evaluation needs besides the state.
pub const Params = struct {
    consts: thermo.Consts,
    channels: opacities.Channels = .{},
    kappa: KappaMode = .default,
    /// C-faithful neutral default: without a PR_KAPPAES hook C returns 0.
    /// PUFFY's Klein–Nishina scattering is opt-in via `Params.puffy()`, so a
    /// non-PUFFY problem never silently inherits PUFFY scattering physics.
    kappaes: KappaesMode = .none,

    pub fn init(units: units_mod.Units, comp: thermo.Composition) Params {
        return .{ .consts = thermo.Consts.init(units, comp) };
    }

    /// The PUFFY build: MASS = 10 M☉, HFRAC = 1 with the direct
    /// MU_GAS/MU_I/MU_E = 1/2/2 overrides, all channels on, and the
    /// Klein–Nishina scattering hook (PR_KAPPAES) enabled explicitly.
    pub fn puffy() Params {
        var p = init(units_mod.Units.init(10.0), thermo.Composition.puffy);
        p.kappaes = .puffy;
        return p;
    }

    /// A grey test problem (radtube/radpulse): PR_KAPPA `kappa = kabs*rho`
    /// (+ explicit opac-slot assignment) and PR_KAPPAES `return kes*rho`
    /// (omitted entirely when kes == 0, like C problems without the file).
    /// COMPTONIZATION is a per-problem define the grey test problems do NOT
    /// set (unlike PUFFY), so the Compton term is off.
    pub fn grey(units: units_mod.Units, comp: thermo.Composition, kabs: f64, kes: f64) Params {
        return .{
            .consts = thermo.Consts.init(units, comp),
            .channels = .{ .comptonization = false },
            .kappa = .{ .grey = kabs },
            .kappaes = if (kes == 0.0) .none else .{ .grey = kes },
        };
    }

    fn kappaesAt(self: *const Params, rho: f64, trad: f64) f64 {
        return self.kappaesAtG(f64, rho, trad);
    }

    fn kappaesAtG(self: *const Params, comptime T: type, rho: T, trad: T) T {
        return switch (self.kappaes) {
            .none => simd.splat(T, 0),
            .puffy => opacities.kappaEsPuffyG(T, &self.consts, self.channels, rho, trad),
            .grey => |k| simd.splat(T, k) * rho,
        };
    }
};

/// The radiation-relevant subset of C's struct_of_state, over lane type T
/// (parallelization plan §2.2 — the `<name>G` functions are the generic
/// cores, the plain `<name>` scalar API delegates to T = f64).
pub fn RadStateOf(comptime T: type) type {
    return struct {
        rho: T,
        uint: T,
        tgas: T,
        te: T,
        ti: T,
        /// Sgas = kB/(μ_gas m_p) · calc_Sfromu (physics.c:41) — note the
        /// different normalization from the ENTR primitive
        sgas: T,
        ne: T,
        ucon: [4]T,
        ucov: [4]T,
        bsq: T,
        rij: [4][4]T,
        ehat: T,
        tradbb: T,
        trad: T,
        kappaes: T,
        kappa: T,
        opac: opacities.OpacOf(T),
    };
}
pub const RadState = RadStateOf(f64);

/// C: fill_struct_of_state (physics.c:12), the RADIATION branch without
/// photon number (Trad = TradBB) or electrons (Te = Ti = Tgas).
///
/// The error union is kept for API compatibility, but with VELPRIM == VELR
/// the fill is infallible (the velr→vel4 conversion cannot fail) — the
/// generic core below has no error path.
pub fn fillRadState(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma_adiab: f64,
    par: *const Params,
) relele.Error!RadState {
    return fillRadStateG(cfg, f64, pp, &geom.gg, &geom.GG, gamma_adiab, par);
}

/// fillRadState over lane type T.
pub fn fillRadStateG(
    comptime cfg: config.Config,
    comptime T: type,
    pp: [layout.VarLayout(cfg).count]T,
    gg: *const [4][5]T,
    GG: *const [4][5]T,
    gamma_adiab: f64,
    par: *const Params,
) RadStateOf(T) {
    const sp = simd.splat;
    const L = layout.VarLayout(cfg);
    const c = &par.consts;

    const rho = pp[L.index(.rho)];
    const uint = pp[L.index(.uu)];
    const temps = thermo.tempsFromUrhoG(T, c, uint, rho, gamma_adiab);
    const sgas = sp(T, c.kb_over_mugas_mp) * hydro.sFromUG(T, rho, uint, gamma_adiab);
    const ne = thermo.thermalNeG(T, c, rho);

    const u = relele.uconUcovFromPrimsG(
        T,
        .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
        gg,
        GG,
    );

    var bsq: T = sp(T, 0);
    if (comptime L.hasVar(.b1)) {
        const b = mhd.bconBcovBsqFrom4velG(
            T,
            .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
            u.con,
            u.cov,
            gg,
        );
        bsq = b.bsq;
    }

    const rij = radiation.calcRijG(cfg, T, pp, gg, GG);

    // C: calc_Ehat_from_Rij_ucov (rad.c:3054) — Ê = R^ab u_a u_b
    var ehat: T = sp(T, 0);
    for (0..4) |i| {
        for (0..4) |j| {
            ehat += rij[i][j] * u.cov[i] * u.cov[j];
        }
    }

    const tradbb = c.lteTfromEG(T, ehat);
    const trad = tradbb; // no EVOLVEPHOTONNUMBER

    const kappaes = par.kappaesAtG(T, rho, trad);

    const kappa_result = switch (par.kappa) {
        .default => opacities.calcKappaFromStateG(T, c, par.channels, .{
            .rho = rho,
            .tgas = temps.tgas,
            .te = temps.te,
            .trad = trad,
            .tradbb = tradbb,
            .ne = ne,
            .bsq = bsq,
        }),
        // PR_KAPPA grey: kappa = coeff·rho, all slots assigned by the
        // problem snippet, totEmissivity = kappaGasAbs·4πB
        // (opacities.c:80 with B = sigma_rad_over_pi·Te⁴).
        .grey => |coeff| blk: {
            const kap = sp(T, coeff) * rho;
            const b = sp(T, c.sigma_rad_over_pi) * temps.te * temps.te * temps.te * temps.te;
            break :blk opacities.KappaResultOf(T){
                .kappa = kap,
                .opac = .{
                    .gas_abs = kap,
                    .rad_abs = kap,
                    .gas_num = kap,
                    .rad_num = kap,
                    .gas_ross = kap,
                    .rad_ross = kap,
                    .tot_emissivity = kap * sp(T, c.fourpi) * b,
                },
            };
        },
    };

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
        .kappa = kappa_result.kappa,
        .opac = kappa_result.opac,
    };
}

pub fn GiOf(comptime T: type) type {
    return struct {
        /// fluid-frame thermal four-force (C: type 0/2)
        ff: [4]T,
        /// lab-frame thermal four-force (C: type 1/3)
        lab: [4]T,
    };
}
pub const Gi = GiOf(f64);

/// C: calc_Compt_Gi_with_state (rad.c:2907) — thermal Comptonization,
/// no RELELECTRONS correction. Returns coeff so both frames reuse it.
pub fn comptonGiCoeff(c: *const thermo.Consts, st: *const RadState) f64 {
    return comptonGiCoeffG(f64, c, st);
}

/// comptonGiCoeff over lane type T.
pub fn comptonGiCoeffG(comptime T: type, c: *const thermo.Consts, st: *const RadStateOf(T)) T {
    const sp = simd.splat;
    const thetae = sp(T, c.kb_over_me) * st.te;
    return st.kappaes * st.ehat * (sp(T, 4.0 * c.kb_over_me) * (st.trad - st.te)) *
        (sp(T, 1.0) + sp(T, 3.683) * thetae + sp(T, 4.0) * thetae * thetae) / (sp(T, 1.0) + sp(T, 4.0) * thetae);
}

/// C: calc_all_Gi_with_state (rad.c:2653), thermal part (no RELELECTRONS,
/// so total == thermal). `vprim` are the gas velocity primitives used for
/// the lab→ff boost.
///
/// The error union is kept for API compatibility; with VELPRIM == VELR the
/// boost cannot fail (see the generic core).
pub fn calcGiFromState(
    st: *const RadState,
    vprim: [3]f64,
    geom: *const Geometry,
    par: *const Params,
) relele.Error!Gi {
    return calcGiFromStateG(f64, st, vprim, &geom.gg, &geom.GG, par);
}

/// calcGiFromState over lane type T.
pub fn calcGiFromStateG(
    comptime T: type,
    st: *const RadStateOf(T),
    vprim: [3]T,
    gg: *const [4][5]T,
    GG: *const [4][5]T,
    par: *const Params,
) GiOf(T) {
    const sp = simd.splat;
    const c = &par.consts;
    const b = sp(T, c.sigma_rad_over_pi) * st.te * st.te * st.te * st.te;

    const k_rad_abs = st.opac.rad_abs;
    const k_gas_abs = st.opac.gas_abs;
    const k_rad_ross = st.opac.rad_ross;
    const kappaes = st.kappaes;
    const ruu = st.ehat;

    var gi_lab: [4]T = undefined;
    for (0..4) |i| {
        var ru: T = sp(T, 0);
        for (0..4) |j| {
            ru += st.rij[i][j] * st.ucov[j];
        }
        gi_lab[i] = -(k_rad_ross + kappaes) * ru -
            ((k_rad_ross + kappaes - k_rad_abs) * ruu + k_gas_abs * sp(T, c.fourpi) * b) * st.ucon[i];
    }

    // boost to the fluid frame, then rewrite the time component directly
    var gi_ff = frames.boost2Lab2FfG(T, gi_lab, vprim, gg, GG);
    gi_ff[0] = -k_gas_abs * sp(T, c.fourpi) * b + k_rad_abs * st.ehat;

    if (par.channels.comptonization) {
        const coeff = comptonGiCoeffG(T, c, st);
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
    gamma_adiab: f64,
    par: *const Params,
) relele.Error!Gi {
    const L = layout.VarLayout(cfg);
    const st = try fillRadState(cfg, pp, geom, gamma_adiab, par);
    return calcGiFromState(&st, .{
        pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)],
    }, geom, par);
}

/// C: calc_kappa (opacities.c:22) — state fill + default opacity code.
pub fn calcKappa(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma_adiab: f64,
    par: *const Params,
) relele.Error!f64 {
    const st = try fillRadState(cfg, pp, geom, gamma_adiab, par);
    return st.kappa;
}

/// C: calc_kappaes (opacities.c:126) — the standalone flavor with
/// Trad = Te (NOT TradBB).
pub fn calcKappaes(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    gamma_adiab: f64,
    par: *const Params,
) f64 {
    const L = layout.VarLayout(cfg);
    const temps = thermo.tempsFromUrho(&par.consts, pp[L.index(.uu)], pp[L.index(.rho)], gamma_adiab);
    const trad = temps.te;
    return par.kappaesAt(pp[L.index(.rho)], trad);
}

/// C: calc_chi (opacities.c:148) — κ + κ_es for the wavespeed τ-limiter.
pub fn calcChi(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma_adiab: f64,
    par: *const Params,
) relele.Error!f64 {
    const kappa = try calcKappa(cfg, pp, geom, gamma_adiab, par);
    return kappa + calcKappaes(cfg, pp, gamma_adiab, par);
}
