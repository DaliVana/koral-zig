//! Radiation-gas coupling: the state fill (the radiation-relevant subset
//! of C's fill_struct_of_state, physics.c:12), the thermal four-force
//! calc_Gi (rad.c:2653 calc_all_Gi_with_state) with its Comptonization
//! term (rad.c:2907), and the pp-level opacity wrappers calc_kappa /
//! calc_kappaes / calc_chi (opacities.c).
//!
//! Physics: G^μ is the gas-radiation energy-momentum exchange rate,
//! entering the gas and radiation equations with opposite signs. In the
//! fluid frame the time component is transparent:
//!   Ĝ⁰ = −κ_gasAbs·4πB(T_e) + κ_radAbs·Ê
//! (emission minus absorption, zero in equilibrium). The lab-frame vector
//!   G^μ = −(κ_radRoss+κ_es) R^μν u_ν
//!         − [(κ_radRoss+κ_es−κ_radAbs) Ê + κ_gasAbs·4πB] u^μ
//! uses the flux-weighted (Rosseland + scattering) opacity for the
//! momentum drag and the Planck-type means for the energy exchange; the
//! standard mixed-mean grey formulation. Thermal Comptonization adds
//! ∝ κ_es Ê·4k(T_rad−T_e)/m_ec² (with a relativistic Θ_e correction): net
//! scattering energy exchange driving T_rad → T_e; Compton heating of the
//! gas when the radiation is hotter, inverse-Compton cooling when the
//! electrons are. G^μ is stiff (∝ opacity), which is why the implicit
//! solver iterates on it. Hence the Slim variants below.
//!
//! C-fidelity notes:
//!  * Two kappaes flavors coexist in C: fill_struct_of_state calls
//!    calc_kappaes_with_temperatures with Trad = TradBB (used by calc_Gi),
//!    while the standalone calc_kappaes(pp,ggg); the one calc_chi and the
//!    wavespeed limiter use; sets Trad = Te. Both preserved here.
//!  * calc_all_Gi_with_state's return value is 1/(Tgas²·kappa) with a
//!    NEVER-ASSIGNED local kappa; uninitialized stack garbage. Every call
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

/// C: PR_KAPPA. The per-problem absorption-opacity hook textually included
/// into calc_kappa_from_state (opacities.c:72). `.default` is the no-PR_KAPPA
/// branch (calc_opacities_from_state). `.grey` mirrors a problem kappa.c of
/// the shape `kappa = COEFF*rho;` that also assigns all six opac slots (the
/// C wrapper otherwise reads UNINITIALIZED stack in its `kappaGasAbs >= 0.`
/// broadcast test; LRTORUS/kappa.c really does that; our oracle problems
/// set the slots explicitly to stay deterministic).
pub const KappaMode = union(enum) {
    default,
    /// κ = coeff·ρ (both in geometrical units), all slots equal,
    /// totEmissivity = κ·4πB.
    grey: f64,
};

/// C: PR_KAPPAES. The scattering hook (opacities.c:114/138). Without it C
/// returns 0; PUFFY's kappaes.c is the Klein-Nishina-corrected Thomson value.
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
    /// PUFFY's Klein-Nishina scattering is opt-in via `Params.puffy()`, so a
    /// non-PUFFY problem never silently inherits PUFFY scattering physics.
    kappaes: KappaesMode = .none,
    /// C: OPDAMPINIMPLICIT opacity damping (rad.c:1488). Scales the thermal
    /// four-force Ĝ (both frames, incl. the Comptonization term) inside the
    /// implicit solver's residual. 1.0 = no damping (the value everywhere
    /// except the implicit ladder's damped retries). Multiplying by 1.0 is
    /// exact, so every non-damped path stays bit-identical.
    opdamp: f64 = 1.0,

    pub fn init(units: units_mod.Units, comp: thermo.Composition) Params {
        return .{ .consts = thermo.Consts.init(units, comp) };
    }

    /// The PUFFY build: MASS = 10 M☉, HFRAC = 1 with the direct
    /// MU_GAS/MU_I/MU_E = 1/2/2 overrides, all channels on, and the
    /// Klein-Nishina scattering hook (PR_KAPPAES) enabled explicitly.
    /// This is the validated-golden entry point; every oracle comparison
    /// holds the mass at 10, so leave it here.
    pub fn puffy() Params {
        return puffyMass(10.0);
    }

    /// The PUFFY build at an arbitrary black-hole mass (solar masses); same
    /// composition, channels and Klein-Nishina scattering as `puffy()`; only
    /// the CGS↔GU unit scale differs (opacities, LTE temperatures, radiation
    /// floors). Used by the Sagittarius A* preset (MASS ≈ 4.3e6). The goldens
    /// stay on `puffy()` at MASS = 10.
    pub fn puffyMass(mass_msun: f64) Params {
        return puffyMassChan(mass_msun, thermo.Composition.puffy, .{});
    }

    /// The PUFFY build at an arbitrary mass with an explicit gas composition
    /// and opacity channel set; the entry point the `puffy_agn.toml` preset
    /// uses to retarget to the koral_lite_puffy configuration (HFRAC/HEFRAC/
    /// MFRAC metallicity, bremsstrahlung/Klein-Nishina toggled). Klein-Nishina
    /// scattering (`kappaes = .puffy`) is still the PUFFY hook; its KN
    /// correction is itself gated by `chan.kleinnishina`, so switching that
    /// channel off falls back to plain Thomson. `puffyMassChan(m,
    /// Composition.puffy, .{})` is bit-identical to `puffyMass(m)`.
    pub fn puffyMassChan(mass_msun: f64, comp: thermo.Composition, chan: opacities.Channels) Params {
        var p = init(units_mod.Units.init(mass_msun), comp);
        p.channels = chan;
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
/// (parallelization plan §2.2; the `<name>G` functions are the generic
/// cores, the plain `<name>` scalar API delegates to T = f64).
pub fn RadStateOf(comptime T: type) type {
    return struct {
        rho: T,
        uint: T,
        tgas: T,
        te: T,
        ti: T,
        /// Sgas = kB/(μ_gas m_p) · calc_Sfromu (physics.c:41); note the
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
/// the fill is infallible (the velr→vel4 conversion cannot fail); the
/// generic core below has no error path.
pub fn fillRadState(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma_adiab: f64,
    par: *const Params,
) relele.Error!RadState {
    return fillRadStateG(cfg, f64, pp, geom.cov(), geom.con(), gamma_adiab, par);
}

/// Slim scalar fillRadState; the solver hot path (see fillRadStateSlimG).
pub fn fillRadStateSlim(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma_adiab: f64,
    par: *const Params,
) relele.Error!RadState {
    return fillRadStateSlimG(cfg, f64, pp, geom.cov(), geom.con(), gamma_adiab, par);
}

/// Slim fillRadState over lane type T: the implicit-solver path. Skips the
/// entropy `sgas` (only the once-per-solve reference state's Sgas is read;
/// implicit.zig:377) and the number-averaged / gas-Rosseland opacity channels
/// (radforce.zig:264), all of which the residual and f1dErr never consume.
/// Every consumed field is bit-identical to fillRadStateG (simd_tests.zig).
pub fn fillRadStateSlimG(
    comptime cfg: config.Config,
    comptime T: type,
    pp: [layout.VarLayout(cfg).count]T,
    gg: relele.MetricCovOf(T),
    GG: relele.MetricConOf(T),
    gamma_adiab: f64,
    par: *const Params,
) RadStateOf(T) {
    return fillRadStateCoreG(cfg, T, pp, gg, GG, gamma_adiab, par, false);
}

/// fillRadState over lane type T (the full struct_of_state; every field).
pub fn fillRadStateG(
    comptime cfg: config.Config,
    comptime T: type,
    pp: [layout.VarLayout(cfg).count]T,
    gg: relele.MetricCovOf(T),
    GG: relele.MetricConOf(T),
    gamma_adiab: f64,
    par: *const Params,
) RadStateOf(T) {
    return fillRadStateCoreG(cfg, T, pp, gg, GG, gamma_adiab, par, true);
}

/// The shared core. `full` (comptime): when false, `sgas` and the dead
/// opacity channels are skipped for the solver hot path; when true, every
/// field is populated for the golden/opacity/state tests.
fn fillRadStateCoreG(
    comptime cfg: config.Config,
    comptime T: type,
    pp: [layout.VarLayout(cfg).count]T,
    gg: relele.MetricCovOf(T),
    GG: relele.MetricConOf(T),
    gamma_adiab: f64,
    par: *const Params,
    comptime full: bool,
) RadStateOf(T) {
    const sp = simd.splat;
    const L = layout.VarLayout(cfg);
    const c = &par.consts;

    const rho = pp[L.index(.rho)];
    const uint = pp[L.index(.uu)];
    const temps = thermo.tempsFromUrhoG(T, c, uint, rho, gamma_adiab);
    const sgas = if (comptime full)
        sp(T, c.kb_over_mugas_mp) * hydro.sFromUG(T, rho, uint, gamma_adiab)
    else
        sp(T, 0);
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

    // the residual/f1dErr read gas_abs/rad_abs/rad_ross only; the full state
    // fills every channel for the golden/opacity tests.
    const op_level: opacities.OpacLevel = if (full) .full else .residual;
    const kappa_result = switch (par.kappa) {
        .default => opacities.calcKappaFromStateG(T, c, par.channels, .{
            .rho = rho,
            .tgas = temps.tgas,
            .te = temps.te,
            .trad = trad,
            .tradbb = tradbb,
            .ne = ne,
            .bsq = bsq,
        }, op_level),
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

/// C: calc_Compt_Gi_with_state (rad.c:2907). Thermal Comptonization,
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
    return calcGiFromStateG(f64, st, vprim, geom.cov(), geom.con(), par);
}

/// Slim scalar calcGiFromState; the solver hot path (see calcGiFromStateSlimG).
pub fn calcGiFromStateSlim(
    st: *const RadState,
    vprim: [3]f64,
    geom: *const Geometry,
    par: *const Params,
) relele.Error!Gi {
    return calcGiFromStateSlimG(f64, st, vprim, geom.cov(), geom.con(), par);
}

/// Slim calcGiFromState over lane type T: the implicit-residual path. Skips
/// the lab→ff Lorentz boost; the residual and f1dErr read only `gi.ff[0]`
/// (implicit.zig:366/424), which the direct fluid-frame expression below
/// overwrites regardless, and `gi.lab` (boost-independent). Bit-identical to
/// calcGiFromStateG on `ff[0]` and all of `lab` (simd_tests.zig).
pub fn calcGiFromStateSlimG(
    comptime T: type,
    st: *const RadStateOf(T),
    vprim: [3]T,
    gg: relele.MetricCovOf(T),
    GG: relele.MetricConOf(T),
    par: *const Params,
) GiOf(T) {
    return calcGiFromStateCoreG(T, st, vprim, gg, GG, par, false);
}

/// calcGiFromState over lane type T (both frames fully populated).
pub fn calcGiFromStateG(
    comptime T: type,
    st: *const RadStateOf(T),
    vprim: [3]T,
    gg: relele.MetricCovOf(T),
    GG: relele.MetricConOf(T),
    par: *const Params,
) GiOf(T) {
    return calcGiFromStateCoreG(T, st, vprim, gg, GG, par, true);
}

/// The shared core. `full` (comptime): when false, `gi.ff[1..3]`; read only
/// by the golden/opacity tests; are left zero and the boost is skipped.
fn calcGiFromStateCoreG(
    comptime T: type,
    st: *const RadStateOf(T),
    vprim: [3]T,
    gg: relele.MetricCovOf(T),
    GG: relele.MetricConOf(T),
    par: *const Params,
    comptime full: bool,
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
    var gi_ff: [4]T = undefined;
    if (comptime full) {
        gi_ff = frames.boost2Lab2FfG(T, gi_lab, vprim, gg, GG);
    } else {
        // ff[1..3] are consumed only by tests; the residual reads ff[0] only,
        // which the direct expression below supplies bit-identically.
        gi_ff = @splat(sp(T, 0));
        _ = &vprim;
        _ = &gg;
        _ = &GG;
    }
    gi_ff[0] = -k_gas_abs * sp(T, c.fourpi) * b + k_rad_abs * st.ehat;

    if (par.channels.comptonization) {
        const coeff = comptonGiCoeffG(T, c, st);
        // lab: coeff·u^μ; ff: coeff·(1,0,0,0)
        for (0..4) |i| gi_lab[i] += coeff * st.ucon[i];
        gi_ff[0] += coeff;
    }

    // C: OPDAMPINIMPLICIT — scale the whole four-force (both frames, after the
    // Comptonization add) by opdamp (rad.c:1488-1494). 1.0 elsewhere ⇒ ×1.0 is
    // exact and every non-damped path is bit-identical.
    if (par.opdamp != 1.0) {
        const od = sp(T, par.opdamp);
        for (0..4) |i| {
            gi_ff[i] *= od;
            gi_lab[i] *= od;
        }
    }

    return .{ .ff = gi_ff, .lab = gi_lab };
}

/// C: calc_Gi (rad.c:2588). The C return value is uninitialized garbage
/// discarded by all callers; not reproduced.
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

/// C: calc_kappa (opacities.c:22). State fill + default opacity code.
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

/// C: calc_kappaes (opacities.c:126). The standalone flavor with
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

/// C: calc_chi (opacities.c:148); κ + κ_es for the wavespeed τ-limiter.
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

/// Slim calc_chi for the wavespeed τ-limiter (sim.zig) and the radviscosity
/// mean-free-path (sim/rijvisc.zig); finding #5. χ = κ + κ_es needs neither the
/// radiation frame (Rij / Ê / TradBB) nor the four Trad-dependent opacity
/// channels: κ = opac.gas_abs depends only on (ρ, Te, ne, b²) and the
/// standalone κ_es uses Trad = Te. So this skips calcRij, the Ehat
/// contraction, lteTfromE, sgas, the state-path kappaes, and the
/// rad_abs / rad_ross / rad_num / gas_ross channels (a dozen transcendentals
/// per cell), yet is bit-identical to calcChi (gated in simd_tests.zig). Keep
/// calcChi as the C-parity golden entry point.
///
/// The error union mirrors calcChi's signature so the callers' `try` is a pure
/// name swap; with VELPRIM == VELR the fill below cannot actually fail.
pub fn calcChiSlim(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma_adiab: f64,
    par: *const Params,
) relele.Error!f64 {
    const L = layout.VarLayout(cfg);
    const c = &par.consts;
    const rho = pp[L.index(.rho)];
    const uint = pp[L.index(.uu)];
    const temps = thermo.tempsFromUrhoG(f64, c, uint, rho, gamma_adiab);
    const ne = thermo.thermalNeG(f64, c, rho);

    // ucon/ucov + b² exactly as fillRadStateCoreG derives them (kappagassyn's
    // bmagcgs is the only radiation-independent term that needs b²).
    const u = relele.uconUcovFromPrimsG(f64, .{
        pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)],
    }, geom.cov(), geom.con());

    var bsq: f64 = 0;
    if (comptime L.hasVar(.b1)) {
        const bf = mhd.bconBcovBsqFrom4velG(f64, .{
            pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)],
        }, u.con, u.cov, geom.cov());
        bsq = bf.bsq;
    }

    // κ = opac.gas_abs (Trad-independent); tgas/trad/tradbb are unread on the
    // `.chi` opacity level, passed for the StateIn shape only.
    const kappa = switch (par.kappa) {
        .default => opacities.calcKappaFromStateG(f64, c, par.channels, .{
            .rho = rho,
            .tgas = temps.tgas,
            .te = temps.te,
            .trad = temps.te,
            .tradbb = temps.te,
            .ne = ne,
            .bsq = bsq,
        }, .chi).kappa,
        .grey => |coeff| coeff * rho,
    };

    // standalone κ_es with Trad = Te (calc_kappaes, opacities.c:126) — the
    // same term calcKappaes adds, minus the second tempsFromUrho recompute.
    const kappaes = par.kappaesAt(rho, temps.te);
    return kappa + kappaes;
}
