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
const thermo = @import("thermo.zig");
const units_mod = @import("../units.zig");

/// C: struct opacities (the fields the M8 path fills).
pub const Opac = struct {
    gas_abs: f64, // kappaGasAbs — Planck emission at T_e
    rad_abs: f64, // kappaRadAbs — Planck absorption at T_r, T_e
    gas_num: f64, // kappaGasNum
    rad_num: f64, // kappaRadNum
    gas_ross: f64, // kappaGasRoss
    rad_ross: f64, // kappaRadRoss
    tot_emissivity: f64,
};

/// Channel switches (comptime macros in C; runtime here — they only gate
/// whole terms). Defaults = the PUFFY build.
pub const Channels = struct {
    bremsstrahlung: bool = true,
    synchrotron: bool = true,
    kleinnishina: bool = true,
    comptonization: bool = true, // used by radforce.calcGi

    pub const puffy = Channels{};
};

/// Scalar inputs of calc_opacities_from_state.
pub const StateIn = struct {
    rho: f64,
    tgas: f64,
    te: f64,
    trad: f64,
    tradbb: f64,
    ne: f64,
    bsq: f64,
};

/// C: calc_opacities_from_state (opacities.c:186), BREMSSTRAHLUNG +
/// SYNCHROTRON channels, no SKIPFANCYOPACITIES. Returns kappa=kappaGasAbs
/// with all six opacity fields (tot_emissivity is filled by the caller,
/// calc_kappa_from_state — see kappaFromState).
pub fn calcOpacitiesFromState(c: *const thermo.Consts, ch: Channels, s: StateIn) Opac {
    const rho = s.rho;
    const te = s.te;
    const trad = s.trad;
    const rt_te = @sqrt(te);

    const rhocgs = c.rhogu2cgs * rho;
    const nethcgs = s.ne * c.numdensgu2cgs;

    const zeta = trad / te;
    const zeta_inv = 1.0 / zeta;
    const zeta_inv_3 = zeta_inv * zeta_inv * zeta_inv;
    const zeta_root5 = std.math.pow(f64, zeta, 0.2);
    const zeta_root5_inv_4 = 1.0 / (zeta_root5 * zeta_root5 * zeta_root5 * zeta_root5);
    const zeta_root5_inv_3 = zeta_root5 * zeta_root5_inv_4;

    // C: bsqcgs = fourmpi*endenGU2CGS(bsq) — exact M_PI and the ko.h macro
    const bsqcgs = c.fourmpi * c.units.endenGu2Cgs(s.bsq);
    const bmagcgs = @sqrt(bsqcgs);

    const bbenergy = 4.0 * c.sigma_rad_cgs * te * te * te * te;

    var kappagasff: f64 = 0;
    var kapparadff: f64 = 0;
    var kappagasffross: f64 = 0;
    var kapparadffross: f64 = 0;
    var kappagassyn: f64 = 0;
    var kapparadsyn: f64 = 0;
    var kapparadnumsyn: f64 = 0;
    var kapparadsynross: f64 = 0;
    var kappagassynross: f64 = 0;

    if (ch.bremsstrahlung) {
        // free-free emissivity with Gaunt factor 1.2 and a relativistic
        // correction; n_avg = (X + Y + <Z²/A>·Z)·ρ/mp
        const emis_ff = 1.4e-27 * nethcgs *
            ((c.comp.hfrac + c.comp.hefrac + c.comp.z2diva_mean * c.comp.mfrac()) / c.m_proton_cgs) *
            rt_te * 1.2 * (1.0 + 4.4e-10 * te);
        kappagasff = c.kappacgs2gu * (emis_ff / bbenergy) * rho;

        const scale_pla_ros_ff = 14.12 / (432.7 - 106.8 * zeta_root5_inv_3 +
            43.17 * zeta_root5_inv_4 + 57.88 * zeta_inv);
        kappagasffross = kappagasff * 0.0330;

        // no GRAY_BREMSS
        kapparadff = kappagasff * std.math.log1p(1.6 * zeta) * c.one_over_log_2p6 * zeta_inv_3;
        kapparadffross = kappagasff * scale_pla_ros_ff * zeta_inv_3;
    }

    if (ch.synchrotron) {
        const emis_synchro = 3.61e-34 * (nethcgs / rhocgs) * te * te * bmagcgs * bmagcgs;
        const nu_mbsyn = 1.19e-13 * te * te;
        const zbr = c.k_boltz_cgs * trad / c.h_cgs / nu_mbsyn;
        const zbg = c.k_boltz_cgs * te / c.h_cgs / nu_mbsyn;

        const zeta_a_denom_rad =
            1.79 * std.math.cbrt(zbr * zbr * zbr * zbr * zbr * bmagcgs * bmagcgs * bmagcgs * bmagcgs) +
            1.35 * std.math.cbrt(zbr * zbr * zbr * zbr * zbr * zbr * zbr * bmagcgs * bmagcgs) +
            0.248 * zbr * zbr * zbr;
        const ia_by_b_rad = bmagcgs * bmagcgs / zeta_a_denom_rad;

        const te5 = te * te * te * te * te;
        kapparadsyn = c.kappacgs2gu * (2.13e39 * (nethcgs / rhocgs) / te5 * ia_by_b_rad) * rho;
        kappagassyn = c.kappacgs2gu * (emis_synchro / bbenergy) * rho;

        // number-of-photons averaged opacity
        const zeta_a_denom_num =
            (0.025 * std.math.cbrt(zbr * zbr * zbr * zbr * bmagcgs * bmagcgs) +
                0.169 * std.math.cbrt(zbr * zbr * zbr * zbr * zbr * bmagcgs) +
                0.287 * zbr * zbr);
        const ia_by_b_num = bmagcgs / zeta_a_denom_num;
        kapparadnumsyn = c.kappacgs2gu * (2.13e39 * (nethcgs / rhocgs) / te5 * ia_by_b_num) * rho;

        // Rosseland; Bmb guards B = 0
        const bmb: f64 = if (bmagcgs > 0.0) std.math.pow(f64, bmagcgs, -0.463) else 0.0;
        const ia_by_b_ross_rad = 0.13 * std.math.pow(f64, bmagcgs, 0.69) /
            (std.math.pow(f64, zbr, 1.69) * @exp(1.6 * std.math.pow(f64, zbr, 0.463) * bmb));
        kapparadsynross = c.kappacgs2gu * (2.13e39 * (nethcgs / rhocgs) / te5 * ia_by_b_ross_rad) * rho;

        const ia_by_b_ross_gas = 0.13 * std.math.pow(f64, bmagcgs, 0.69) /
            (std.math.pow(f64, zbg, 1.69) * @exp(1.6 * std.math.pow(f64, zbg, 0.463) * bmb));
        kappagassynross = c.kappacgs2gu * (2.13e39 * (nethcgs / rhocgs) / te5 * ia_by_b_ross_gas) * rho;

        // suppress synchrotron at nonrelativistic temperatures
        // (no USE_SYNCHROTRON_BRIDGE_FUNCTIONS)
        const terel = te * c.k_over_mecsq;
        const terelfactor = (terel * terel) / (1.0 + terel * terel);
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
        .tot_emissivity = 0, // filled by kappaFromState
    };
}

/// C: calc_kappa_from_state (opacities.c:37) — the default (no PR_KAPPA)
/// path plus the totEmissivity bookkeeping. Returns kappa = kappaGasAbs.
pub fn kappaFromState(c: *const thermo.Consts, ch: Channels, s: StateIn) struct { kappa: f64, opac: Opac } {
    const b = c.sigma_rad_over_pi * s.te * s.te * s.te * s.te;
    var opac = calcOpacitiesFromState(c, ch, s);
    // kappaGasAbs >= 0 always holds here, so totEmissivity uses it and no
    // field rewriting happens (opacities.c:80-99)
    opac.tot_emissivity = opac.gas_abs * c.fourpi * b;
    return .{ .kappa = opac.gas_abs, .opac = opac };
}

/// C: PROBLEMS/PUFFY/kappaes.c via calc_kappaes_with_temperatures — the
/// KLEINNISHINA correction uses Tkn = Trad (which is Te on the standalone
/// calc_kappaes path and TradBB on the struct_of_state path). Uses the
/// ko.h kappaCGS2GU *macro* shape.
pub fn kappaEsPuffy(c: *const thermo.Consts, ch: Channels, rho: f64, trad: f64) f64 {
    var opac_correction: f64 = 1.0;
    if (ch.kleinnishina) {
        const tkn = trad;
        opac_correction = 1.0 / (1.0 + std.math.pow(f64, tkn / 4.5e8, 0.86));
    }
    return c.units.kappaCgs2Gu(0.2 * (1.0 + c.comp.hfrac) * opac_correction) * rho;
}

/// C: calc_tautot (opacities.c:159) — τ = χ·dx per direction.
pub fn tautot(chi: f64, dx: [3]f64) [3]f64 {
    return .{ chi * dx[0], chi * dx[1], chi * dx[2] };
}

/// C: calc_tauabs (opacities.c:173).
pub fn tauabs(kappa: f64, dx: [3]f64) [3]f64 {
    return .{ kappa * dx[0], kappa * dx[1], kappa * dx[2] };
}
