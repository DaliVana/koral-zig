//! Opacities (C: opacities.c) — the channels the PUFFY build compiles:
//! BREMSSTRAHLUNG + SYNCHROTRON absorption (calc_opacities_from_state) and
//! the PUFFY scattering hook (PROBLEMS/PUFFY/kappaes.c) with its
//! KLEINNISHINA correction. Everything here is scalar-in/scalar-out; the
//! pp/geometry-level wrappers live in physics/radforce.zig.
//!
//! C-fidelity notes:
//!  * PUFFY does NOT wire PR_KAPPA — PROBLEMS/PUFFY/kappa.c is dead code;
//!    absorption comes from the default calc_opacities_from_state.
//!  * USE_SYNCHROTRON_BRIDGE_FUNCTIONS is commented out of choices.h
//!    entirely (PUFFY's NO_SYNCHROTRON_BRIDGE_FUNCTIONS is vestigial), so
//!    the bridge blocks are off and the Terelfactor suppression is active.
//!  * The C body computes Trad_lim/nph_lim from pp[NF] even without
//!    EVOLVEPHOTONNUMBER — unused locals reading a stale slot; skipped.
//!  * calc_opacities_from_state uses the misc.c *globals* (kappacgs2gu,
//!    rhogu2cgs, numdensgu2cgs) while the PR_KAPPAES hook uses the ko.h
//!    *macros* (kappaCGS2GU) — both shapes preserved via thermo.Consts
//!    vs units.Units.
//!  * BOUNDELECTRON / BOUNDFREE / DOUBLECOMPTON / SUTHERLAND-DOPITA /
//!    CHIANTI are off in every target build and are not implemented.

const std = @import("std");
const simd = @import("../math/simd.zig");
const thermo = @import("thermo.zig");
const units_mod = @import("../units.zig");

/// C: struct opacities (the fields the M8 path fills), over lane type T —
/// the `<name>G` functions below are the comptime-T-generic cores of the
/// opacity chain (parallelization plan §2.2); the plain `<name>` scalar
/// API delegates to T = f64.
pub fn OpacOf(comptime T: type) type {
    return struct {
        gas_abs: T, // kappaGasAbs — Planck emission at T_e
        rad_abs: T, // kappaRadAbs — Planck absorption at T_r, T_e
        gas_num: T, // kappaGasNum
        rad_num: T, // kappaRadNum
        gas_ross: T, // kappaGasRoss
        rad_ross: T, // kappaRadRoss
        tot_emissivity: T,
    };
}
pub const Opac = OpacOf(f64);

/// Channel switches (comptime macros in C; runtime here — they only gate
/// whole terms). Defaults = the PUFFY build.
pub const Channels = struct {
    bremsstrahlung: bool = true,
    synchrotron: bool = true,
    kleinnishina: bool = true,
    comptonization: bool = true, // used by radforce.calcGi

    pub const puffy = Channels{};
};

/// Scalar inputs of calc_opacities_from_state, over lane type T.
pub fn StateInOf(comptime T: type) type {
    return struct {
        rho: T,
        tgas: T,
        te: T,
        trad: T,
        tradbb: T,
        ne: T,
        bsq: T,
    };
}
pub const StateIn = StateInOf(f64);

/// C: calc_opacities_from_state (opacities.c:186), BREMSSTRAHLUNG +
/// SYNCHROTRON channels, no SKIPFANCYOPACITIES. Returns kappa=kappaGasAbs
/// with all six opacity fields (tot_emissivity is filled by the caller,
/// calc_kappa_from_state — see calcKappaFromState).
pub fn calcOpacitiesFromState(c: *const thermo.Consts, ch: Channels, s: StateIn) Opac {
    return calcOpacitiesFromStateG(f64, c, ch, s);
}

/// calcOpacitiesFromState over lane type T. Scalar constants and the
/// all-scalar parens are broadcast whole (same f64 product as the scalar
/// code); the B = 0 Rosseland guard becomes a select with pow(0, −0.463)
/// computed-and-discarded on the guarded lanes.
pub fn calcOpacitiesFromStateG(comptime T: type, c: *const thermo.Consts, ch: Channels, s: StateInOf(T)) OpacOf(T) {
    const sp = simd.splat;
    const rho = s.rho;
    const te = s.te;
    const trad = s.trad;
    const rt_te = @sqrt(te);

    const rhocgs = sp(T, c.rhogu2cgs) * rho;
    const nethcgs = s.ne * sp(T, c.numdensgu2cgs);

    const zeta = trad / te;
    const zeta_inv = sp(T, 1.0) / zeta;
    const zeta_inv_3 = zeta_inv * zeta_inv * zeta_inv;
    const zeta_root5 = simd.pow(T, zeta, 0.2);
    const zeta_root5_inv_4 = sp(T, 1.0) / (zeta_root5 * zeta_root5 * zeta_root5 * zeta_root5);
    const zeta_root5_inv_3 = zeta_root5 * zeta_root5_inv_4;

    // C: bsqcgs = fourmpi*endenGU2CGS(bsq) — exact M_PI and the ko.h macro
    const bsqcgs = sp(T, c.fourmpi) * c.units.endenGu2CgsG(T, s.bsq);
    const bmagcgs = @sqrt(bsqcgs);

    const bbenergy = sp(T, 4.0 * c.sigma_rad_cgs) * te * te * te * te;

    var kappagasff: T = sp(T, 0);
    var kapparadff: T = sp(T, 0);
    var kappagasffross: T = sp(T, 0);
    var kapparadffross: T = sp(T, 0);
    var kappagassyn: T = sp(T, 0);
    var kapparadsyn: T = sp(T, 0);
    var kapparadnumsyn: T = sp(T, 0);
    var kapparadsynross: T = sp(T, 0);
    var kappagassynross: T = sp(T, 0);

    if (ch.bremsstrahlung) {
        // free-free emissivity with Gaunt factor 1.2 and a relativistic
        // correction; n_avg = (X + Y + <Z²/A>·Z)·ρ/mp
        const emis_ff = sp(T, 1.4e-27) * nethcgs *
            sp(T, (c.comp.hfrac + c.comp.hefrac + c.comp.z2diva_mean * c.comp.mfrac()) / c.m_proton_cgs) *
            rt_te * sp(T, 1.2) * (sp(T, 1.0) + sp(T, 4.4e-10) * te);
        kappagasff = sp(T, c.kappacgs2gu) * (emis_ff / bbenergy) * rho;

        const scale_pla_ros_ff = sp(T, 14.12) / (sp(T, 432.7) - sp(T, 106.8) * zeta_root5_inv_3 +
            sp(T, 43.17) * zeta_root5_inv_4 + sp(T, 57.88) * zeta_inv);
        kappagasffross = kappagasff * sp(T, 0.0330);

        // no GRAY_BREMSS
        kapparadff = kappagasff * simd.log1p(T, sp(T, 1.6) * zeta) * sp(T, c.one_over_log_2p6) * zeta_inv_3;
        kapparadffross = kappagasff * scale_pla_ros_ff * zeta_inv_3;
    }

    if (ch.synchrotron) {
        const emis_synchro = sp(T, 3.61e-34) * (nethcgs / rhocgs) * te * te * bmagcgs * bmagcgs;
        const nu_mbsyn = sp(T, 1.19e-13) * te * te;
        const zbr = sp(T, c.k_boltz_cgs) * trad / sp(T, c.h_cgs) / nu_mbsyn;
        const zbg = sp(T, c.k_boltz_cgs) * te / sp(T, c.h_cgs) / nu_mbsyn;

        const zeta_a_denom_rad =
            sp(T, 1.79) * simd.cbrt(T, zbr * zbr * zbr * zbr * zbr * bmagcgs * bmagcgs * bmagcgs * bmagcgs) +
            sp(T, 1.35) * simd.cbrt(T, zbr * zbr * zbr * zbr * zbr * zbr * zbr * bmagcgs * bmagcgs) +
            sp(T, 0.248) * zbr * zbr * zbr;
        const ia_by_b_rad = bmagcgs * bmagcgs / zeta_a_denom_rad;

        const te5 = te * te * te * te * te;
        kapparadsyn = sp(T, c.kappacgs2gu) * (sp(T, 2.13e39) * (nethcgs / rhocgs) / te5 * ia_by_b_rad) * rho;
        kappagassyn = sp(T, c.kappacgs2gu) * (emis_synchro / bbenergy) * rho;

        // number-of-photons averaged opacity
        const zeta_a_denom_num =
            (sp(T, 0.025) * simd.cbrt(T, zbr * zbr * zbr * zbr * bmagcgs * bmagcgs) +
                sp(T, 0.169) * simd.cbrt(T, zbr * zbr * zbr * zbr * zbr * bmagcgs) +
                sp(T, 0.287) * zbr * zbr);
        const ia_by_b_num = bmagcgs / zeta_a_denom_num;
        kapparadnumsyn = sp(T, c.kappacgs2gu) * (sp(T, 2.13e39) * (nethcgs / rhocgs) / te5 * ia_by_b_num) * rho;

        // Rosseland; Bmb guards B = 0
        const bmb: T = simd.select(T, bmagcgs > sp(T, 0.0), simd.pow(T, bmagcgs, -0.463), sp(T, 0.0));
        const ia_by_b_ross_rad = sp(T, 0.13) * simd.pow(T, bmagcgs, 0.69) /
            (simd.pow(T, zbr, 1.69) * simd.exp(T, sp(T, 1.6) * simd.pow(T, zbr, 0.463) * bmb));
        kapparadsynross = sp(T, c.kappacgs2gu) * (sp(T, 2.13e39) * (nethcgs / rhocgs) / te5 * ia_by_b_ross_rad) * rho;

        const ia_by_b_ross_gas = sp(T, 0.13) * simd.pow(T, bmagcgs, 0.69) /
            (simd.pow(T, zbg, 1.69) * simd.exp(T, sp(T, 1.6) * simd.pow(T, zbg, 0.463) * bmb));
        kappagassynross = sp(T, c.kappacgs2gu) * (sp(T, 2.13e39) * (nethcgs / rhocgs) / te5 * ia_by_b_ross_gas) * rho;

        // suppress synchrotron at nonrelativistic temperatures
        // (no USE_SYNCHROTRON_BRIDGE_FUNCTIONS)
        const terel = te * sp(T, c.k_over_mecsq);
        const terelfactor = (terel * terel) / (sp(T, 1.0) + terel * terel);
        kapparadsyn *= terelfactor;
        kappagassyn *= terelfactor;
        kapparadnumsyn *= terelfactor;
        kapparadsynross *= terelfactor;
        kappagassynross *= terelfactor;
    }

    // sums (opacities.c:661-666); be/bf channels are off
    return .{
        .gas_abs = kappagasff + kappagassyn,
        .rad_abs = kapparadff + kapparadsyn,
        .gas_num = kappagasff, // synchrotron number opacity applied separately
        .rad_num = kapparadff + kapparadnumsyn,
        .gas_ross = kappagasffross + kappagassynross,
        .rad_ross = kapparadffross + kapparadsynross,
        .tot_emissivity = sp(T, 0), // filled by calcKappaFromState
    };
}

pub fn KappaResultOf(comptime T: type) type {
    return struct { kappa: T, opac: OpacOf(T) };
}
pub const KappaResult = KappaResultOf(f64);

/// C: calc_kappa_from_state (opacities.c:37) — the default (no PR_KAPPA)
/// path plus the totEmissivity bookkeeping. Returns kappa = kappaGasAbs.
pub fn calcKappaFromState(c: *const thermo.Consts, ch: Channels, s: StateIn) KappaResult {
    return calcKappaFromStateG(f64, c, ch, s);
}

/// calcKappaFromState over lane type T.
pub fn calcKappaFromStateG(comptime T: type, c: *const thermo.Consts, ch: Channels, s: StateInOf(T)) KappaResultOf(T) {
    const sp = simd.splat;
    const b = sp(T, c.sigma_rad_over_pi) * s.te * s.te * s.te * s.te;
    var opac = calcOpacitiesFromStateG(T, c, ch, s);
    // kappaGasAbs >= 0 always holds here, so totEmissivity uses it and no
    // field rewriting happens (opacities.c:80-99)
    opac.tot_emissivity = opac.gas_abs * sp(T, c.fourpi) * b;
    return .{ .kappa = opac.gas_abs, .opac = opac };
}

/// C: PROBLEMS/PUFFY/kappaes.c via calc_kappaes_with_temperatures — the
/// KLEINNISHINA correction uses Tkn = Trad (which is Te on the standalone
/// calc_kappaes path and TradBB on the struct_of_state path). Uses the
/// ko.h kappaCGS2GU *macro* shape.
pub fn kappaEsPuffy(c: *const thermo.Consts, ch: Channels, rho: f64, trad: f64) f64 {
    return kappaEsPuffyG(f64, c, ch, rho, trad);
}

/// kappaEsPuffy over lane type T.
pub fn kappaEsPuffyG(comptime T: type, c: *const thermo.Consts, ch: Channels, rho: T, trad: T) T {
    const sp = simd.splat;
    var opac_correction: T = sp(T, 1.0);
    if (ch.kleinnishina) {
        const tkn = trad;
        opac_correction = sp(T, 1.0) / (sp(T, 1.0) + simd.pow(T, tkn / sp(T, 4.5e8), 0.86));
    }
    return c.units.kappaCgs2GuG(T, sp(T, 0.2 * (1.0 + c.comp.hfrac)) * opac_correction) * rho;
}

/// C: calc_tautot (opacities.c:159) — τ = χ·dx per direction.
pub fn tautot(chi: f64, dx: [3]f64) [3]f64 {
    return .{ chi * dx[0], chi * dx[1], chi * dx[2] };
}

/// C: calc_tauabs (opacities.c:173).
pub fn tauabs(kappa: f64, dx: [3]f64) [3]f64 {
    return .{ kappa * dx[0], kappa * dx[1], kappa * dx[2] };
}
