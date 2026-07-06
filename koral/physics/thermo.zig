//! Gas composition, precomputed physical constants, and temperature
//! relations (C: choices.h:14-51, misc.c initialize_constants,
//! physics.c calc_PEQ_* / calc_thermal_ne).
//!
//! C-fidelity notes:
//!  * Every constant is computed with the C global's exact expression shape
//!    (initialize_constants uses left-to-right chains like
//!    `1./GGG*CCC*CCC/MASSCM/MASSCM`, while the ko.h *macros* parenthesize
//!    differently — both shapes exist in the same C build and they differ
//!    by ulps; opacities use the globals, the PR_KAPPAES hook the macros).
//!  * fourpi and fourmpi both use the same exact π, so the four-force
//!    emission term is consistent: kappaGasAbs·fourpi·B with
//!    B = SIGMA_RAD/Pi·Te⁴.
//!  * Single-temperature gas (no EVOLVEELECTRONS): Te = Ti = Tgas, floored
//!    at TEMPEMINIMAL = TEMPIMINIMAL = 1e2 (Tgas itself is NOT floored).

const std = @import("std");
const units_mod = @import("../units.zig");
const Units = units_mod.Units;

/// Exact π used throughout the port for a single consistent constant.
const pi = std.math.pi;

/// C: choices.h:441-457 (no PUFFY override).
pub const tempeminimal: f64 = 1.0e2;
pub const tempiminimal: f64 = 1.0e2;

/// Mass fractions and mean weights (C: choices.h:14-51). Defaults are
/// choices.h's (HFRAC=1, pure hydrogen). Problems may #define MU_GAS /
/// MU_I / MU_E directly, bypassing the composition formulas — PUFFY does
/// (define.h:2-4: 1, 2, 2) while keeping HFRAC=1 for the opacity formulas.
pub const Composition = struct {
    hfrac: f64 = 1.0,
    hefrac: f64 = 0.0,
    /// weighted mean of 1/A (solar), C: A_INV_MEAN
    a_inv_mean: f64 = 0.0570490404519,
    /// weighted mean of Z²/A (solar), C: Z2divA_MEAN
    z2diva_mean: f64 = 5.14445150955,
    mu_gas_override: ?f64 = null,
    mu_i_override: ?f64 = null,
    mu_e_override: ?f64 = null,

    pub const cdefault = Composition{};

    /// PROBLEMS/PUFFY/define.h:2-4.
    pub const puffy = Composition{
        .mu_gas_override = 1.0,
        .mu_i_override = 2.0,
        .mu_e_override = 2.0,
    };

    pub fn mfrac(c: Composition) f64 {
        return 1.0 - c.hfrac - c.hefrac;
    }
    pub fn muGas(c: Composition) f64 {
        return c.mu_gas_override orelse
            1.0 / (0.5 + 1.5 * c.hfrac + 0.25 * c.hefrac + c.a_inv_mean * c.mfrac());
    }
    pub fn muI(c: Composition) f64 {
        return c.mu_i_override orelse
            1.0 / (c.hfrac + 0.25 * c.hefrac + c.a_inv_mean * c.mfrac());
    }
    pub fn muE(c: Composition) f64 {
        return c.mu_e_override orelse 2.0 / (1.0 + c.hfrac);
    }
};

/// The precomputed C globals this milestone needs, with C's exact
/// expression shapes (misc.c:12-92 + the ko.h GU-constant macros).
pub const Consts = struct {
    units: Units,
    comp: Composition,

    // conversion factors (misc.c global shapes)
    lengu2cgs: f64,
    numdensgu2cgs: f64,
    rhogu2cgs: f64,
    kappacgs2gu: f64,

    // physical constants
    k_boltz_cgs: f64,
    h_cgs: f64,
    sigma_rad_cgs: f64,
    m_proton_cgs: f64,
    m_electr_cgs: f64,

    // misc coefficients
    fourpi: f64, // 4·Pi (truncated)
    fourmpi: f64, // 4·M_PI (exact)
    sigma_rad_over_pi: f64, // SIGMA_RAD/Pi (truncated)
    four_sigmarad: f64,
    one_over_four_sigmarad: f64,
    k_over_mecsq: f64,
    kb_over_me: f64,
    kb_over_mugas_mp: f64,
    mugas_mp_over_kb: f64,
    one_over_mue_mp: f64,
    one_over_mui_mp: f64,
    one_over_log_2p6: f64,

    pub fn init(units: Units, comp: Composition) Consts {
        const masscm = units.masscm;
        const GGG = units_mod.GGG;
        const CCC = units_mod.CCC;

        // ko.h GU constants (macro shapes)
        const k_boltz = units_mod.K_BOLTZ_CGS * GGG / CCC / CCC / CCC / CCC / masscm;
        const m_proton = units_mod.M_PROTON_CGS * (GGG / (masscm * CCC * CCC));
        const m_electr = units_mod.M_ELECTR_CGS * (GGG / (masscm * CCC * CCC));
        const sigma_rad = units_mod.SIGMA_RAD_CGS * GGG / CCC / CCC / CCC / CCC / CCC * masscm * masscm;

        const four_sigmarad = 4.0 * sigma_rad;

        return .{
            .units = units,
            .comp = comp,
            .lengu2cgs = masscm,
            .numdensgu2cgs = 1.0 / masscm / masscm / masscm,
            .rhogu2cgs = 1.0 / GGG * CCC * CCC / masscm / masscm,
            .kappacgs2gu = 1.0 / GGG * CCC * CCC / masscm,
            .k_boltz_cgs = units_mod.K_BOLTZ_CGS,
            .h_cgs = units_mod.H_CGS,
            .sigma_rad_cgs = units_mod.SIGMA_RAD_CGS,
            .m_proton_cgs = units_mod.M_PROTON_CGS,
            .m_electr_cgs = units_mod.M_ELECTR_CGS,
            .fourpi = 4.0 * pi,
            .fourmpi = 4.0 * pi,
            .sigma_rad_over_pi = sigma_rad / pi,
            .four_sigmarad = four_sigmarad,
            .one_over_four_sigmarad = 1.0 / four_sigmarad,
            .k_over_mecsq = 1.69e-10,
            .kb_over_me = k_boltz / m_electr,
            .kb_over_mugas_mp = k_boltz / comp.muGas() / m_proton,
            .mugas_mp_over_kb = 1.0 / (k_boltz / comp.muGas() / m_proton),
            .one_over_mue_mp = 1.0 / comp.muE() / m_proton,
            .one_over_mui_mp = 1.0 / comp.muI() / m_proton,
            .one_over_log_2p6 = 1.0 / @log(2.6),
        };
    }

    /// C: calc_LTE_EfromT (rad.c:3140) — E = 4σT⁴ in GU.
    pub fn lteEfromT(c: *const Consts, t: f64) f64 {
        return c.four_sigmarad * t * t * t * t;
    }
    /// C: calc_LTE_TfromE (rad.c:3147).
    pub fn lteTfromE(c: *const Consts, e: f64) f64 {
        return @sqrt(@sqrt(c.one_over_four_sigmarad * e));
    }
};

/// C: calc_PEQ_Tfromurho (physics.c) — Tgas from u and ρ.
pub fn tFromUrho(c: *const Consts, uint: f64, rho: f64, gamma_adiab: f64) f64 {
    const p = uint * (gamma_adiab - 1.0);
    return c.mugas_mp_over_kb * p / rho;
}

/// C: calc_PEQ_ufromTrho (physics.c:1912) — u from T and ρ
/// (no CONSISTENTGAMMA).
pub fn uFromTrho(c: *const Consts, t: f64, rho: f64, gamma_adiab: f64) f64 {
    const p = c.kb_over_mugas_mp * rho * t;
    return p / (gamma_adiab - 1.0);
}

pub const GasTemps = struct { tgas: f64, te: f64, ti: f64 };

/// C: calc_PEQ_Teifrompp (physics.c:1869), single-temperature branch:
/// Te = Ti = Tgas, non-finite → BIG, floored at TEMPE/IMINIMAL.
pub fn tempsFromUrho(c: *const Consts, uint: f64, rho: f64, gamma_adiab: f64) GasTemps {
    const tgas = tFromUrho(c, uint, rho, gamma_adiab);
    var te = tgas;
    var ti = tgas;
    if (!std.math.isFinite(te)) te = units_mod.BIG;
    if (!std.math.isFinite(ti)) ti = units_mod.BIG;
    if (te < tempeminimal) te = tempeminimal;
    if (ti < tempiminimal) ti = tempiminimal;
    return .{ .tgas = tgas, .te = te, .ti = ti };
}

/// C: calc_thermal_ne (physics.c:341), no RELELECTRONS.
pub fn thermalNe(c: *const Consts, rho: f64) f64 {
    return c.one_over_mue_mp * rho;
}

test "thermo: default composition is pure hydrogen; PUFFY overrides mu directly" {
    const comp = Composition.cdefault;
    try std.testing.expectApproxEqRel(@as(f64, 0.5), comp.muGas(), 1e-15);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), comp.muE(), 1e-15);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), comp.muI(), 1e-15);

    const puffy = Composition.puffy;
    try std.testing.expectEqual(@as(f64, 1.0), puffy.muGas());
    try std.testing.expectEqual(@as(f64, 2.0), puffy.muI());
    try std.testing.expectEqual(@as(f64, 2.0), puffy.muE());
    try std.testing.expectEqual(@as(f64, 1.0), puffy.hfrac); // opacities keep HFRAC=1
}

test "thermo: ideal-gas temperature roundtrip and floor" {
    const c = Consts.init(Units.init(10.0), Composition.cdefault);
    const gam: f64 = 5.0 / 3.0;
    // T(u(T)) identity
    const t0: f64 = 3.7e8;
    const uint0 = t0 / (c.mugas_mp_over_kb * (gam - 1.0));
    try std.testing.expectApproxEqRel(t0, tFromUrho(&c, uint0, 1.0, gam), 1e-14);
    // Te/Ti floored, Tgas not
    const temps = tempsFromUrho(&c, 1.0e-30, 1.0, gam);
    try std.testing.expectEqual(tempeminimal, temps.te);
    try std.testing.expect(temps.tgas < tempeminimal);
}
