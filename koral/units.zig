//! Units and physical constants — the single source of truth for
//! geometrized-unit (GU) ↔ CGS conversions (C: ko.h lines 6–78).
//!
//! KORAL's geometrized units: G = c = 1, the black-hole mass sets the length
//! scale MASSCM = MASS·MSUNCM (cm per code length unit). Temperatures stay
//! in Kelvin in both systems. Expression structure deliberately mirrors the
//! ko.h macros (same parenthesization/association) so results agree with the
//! C oracle at the few-ulp level.

const std = @import("std");

// Physical constants, CGS (ko.h:12-31 — exact values, do not "improve").
pub const GGG0: f64 = 6.674e-8;
pub const CCC0: f64 = 2.9979246e10;
pub const g_tilda: f64 = 1.0;
pub const c_tilda: f64 = 1.0;
pub const GGG: f64 = GGG0 / g_tilda;
pub const CCC: f64 = CCC0 / c_tilda;
/// GM_sun/c^2 in cm (C: MSUNCM).
pub const MSUNCM: f64 = 147700.0;
pub const PARSEC_CGS: f64 = 3.086e18;
pub const K_BOLTZ_CGS: f64 = 1.3806488e-16;
pub const M_ELECTR_CGS: f64 = 9.1094e-28;
pub const M_PROTON_CGS: f64 = 1.67262158e-24;
pub const SIGMA_RAD_CGS: f64 = 5.670367e-5;
pub const H_CGS: f64 = 6.6260755e-27;
pub const SIGMATH_CGS: f64 = 6.6524587158e-25;
pub const HFRAC_SUN: f64 = 0.7381;
pub const HEFRAC_SUN: f64 = 0.2485;
pub const MFRAC_SUN: f64 = 1.0 - HFRAC_SUN - HEFRAC_SUN;

pub const SMALL: f64 = 1.0e-80;
pub const BIG: f64 = 1.0 / SMALL;

/// All conversions for one choice of black-hole mass.
pub const Units = struct {
    /// BH mass in solar masses (C: MASS). PUFFY: 10.
    mass: f64,
    /// cm per code length unit (C: MASSCM = MASS*MSUNCM).
    masscm: f64,

    pub fn init(mass_msun: f64) Units {
        return .{ .mass = mass_msun, .masscm = mass_msun * MSUNCM };
    }

    // --- conversions, mirroring ko.h:37-66 ---

    pub fn tempCgs2Gu(_: Units, x: f64) f64 {
        return x;
    }
    pub fn tempGu2Cgs(_: Units, x: f64) f64 {
        return x;
    }
    pub fn lenCgs2Gu(u: Units, x: f64) f64 {
        return x / u.masscm;
    }
    pub fn lenGu2Cgs(u: Units, x: f64) f64 {
        return x * u.masscm;
    }
    pub fn numdensCgs2Gu(u: Units, x: f64) f64 {
        return x * (u.masscm * u.masscm * u.masscm);
    }
    pub fn numdensGu2Cgs(u: Units, x: f64) f64 {
        return x / (u.masscm * u.masscm * u.masscm);
    }
    pub fn timeCgs2Gu(u: Units, x: f64) f64 {
        return x * (CCC / u.masscm);
    }
    pub fn timeGu2Cgs(u: Units, x: f64) f64 {
        return x * (u.masscm / CCC);
    }
    pub fn velCgs2Gu(_: Units, x: f64) f64 {
        return x / CCC;
    }
    pub fn velGu2Cgs(_: Units, x: f64) f64 {
        return x * CCC;
    }
    pub fn rhoCgs2Gu(u: Units, x: f64) f64 {
        return x * ((GGG * u.masscm * u.masscm) / (CCC * CCC));
    }
    pub fn rhoGu2Cgs(u: Units, x: f64) f64 {
        return x * ((CCC * CCC) / (GGG * u.masscm * u.masscm));
    }
    pub fn surfdensCgs2Gu(u: Units, x: f64) f64 {
        return x * ((GGG * u.masscm) / (CCC * CCC));
    }
    pub fn surfdensGu2Cgs(u: Units, x: f64) f64 {
        return x * ((CCC * CCC) / (GGG * u.masscm));
    }
    pub fn massCgs2Gu(u: Units, x: f64) f64 {
        return x * (GGG / (u.masscm * CCC * CCC));
    }
    pub fn massGu2Cgs(u: Units, x: f64) f64 {
        return x * ((u.masscm * CCC * CCC) / GGG);
    }
    pub fn kappaCgs2Gu(u: Units, x: f64) f64 {
        return x * ((CCC * CCC) / (GGG * u.masscm));
    }
    pub fn kappaGu2Cgs(u: Units, x: f64) f64 {
        return x * ((GGG * u.masscm) / (CCC * CCC));
    }
    pub fn endenCgs2Gu(u: Units, x: f64) f64 {
        return x * ((GGG * u.masscm * u.masscm) / (CCC * CCC * CCC * CCC));
    }
    pub fn endenGu2Cgs(u: Units, x: f64) f64 {
        return x * ((CCC * CCC * CCC * CCC) / (GGG * u.masscm * u.masscm));
    }
    pub fn fluxCgs2Gu(u: Units, x: f64) f64 {
        return x * ((GGG * u.masscm * u.masscm) / (CCC * CCC * CCC * CCC * CCC));
    }
    pub fn fluxGu2Cgs(u: Units, x: f64) f64 {
        return x * ((CCC * CCC * CCC * CCC * CCC) / (GGG * u.masscm * u.masscm));
    }
    pub fn heatcoolCgs2Gu(u: Units, x: f64) f64 {
        return u.endenCgs2Gu(u.timeGu2Cgs(x));
    }
    pub fn heatcoolGu2Cgs(u: Units, x: f64) f64 {
        return u.endenGu2Cgs(u.timeCgs2Gu(x));
    }
    pub fn ergCgs2Gu(u: Units, x: f64) f64 {
        return x * (GGG / (u.masscm * CCC * CCC * CCC * CCC));
    }
    pub fn ergGu2Cgs(u: Units, x: f64) f64 {
        return x * ((u.masscm * CCC * CCC * CCC * CCC) / GGG);
    }
    pub fn chargeCgs2Gu(u: Units, x: f64) f64 {
        return x * @sqrt(GGG / u.masscm / CCC / CCC / CCC / CCC) * @sqrt(1.0 / u.masscm);
    }
    pub fn chargeGu2Cgs(u: Units, x: f64) f64 {
        return x / @sqrt(GGG / u.masscm / CCC / CCC / CCC / CCC) / @sqrt(1.0 / u.masscm);
    }
    pub fn crossCgs2Gu(u: Units, x: f64) f64 {
        return x / (u.masscm * u.masscm);
    }
    pub fn crossGu2Cgs(u: Units, x: f64) f64 {
        return x * (u.masscm * u.masscm);
    }

    // --- constants in GU, mirroring ko.h:68-78 ---

    pub fn kBoltz(u: Units) f64 {
        return K_BOLTZ_CGS * GGG / CCC / CCC / CCC / CCC / u.masscm;
    }
    pub fn mProton(u: Units) f64 {
        return u.massCgs2Gu(M_PROTON_CGS);
    }
    pub fn mElectr(u: Units) f64 {
        return u.massCgs2Gu(M_ELECTR_CGS);
    }
    pub fn mpeRatio(u: Units) f64 {
        return u.mProton() / u.mElectr();
    }
    pub fn sigmaRad(u: Units) f64 {
        return SIGMA_RAD_CGS * GGG / CCC / CCC / CCC / CCC / CCC * u.masscm * u.masscm;
    }
    pub fn aRad(u: Units) f64 {
        return 4.0 * u.sigmaRad();
    }
    pub fn kappaEsCoeff(u: Units, hfrac: f64) f64 {
        return u.kappaCgs2Gu(0.2 * (1.0 + hfrac));
    }
    pub fn sigmaThompson(u: Units) f64 {
        return u.crossCgs2Gu(SIGMATH_CGS);
    }

    /// GM/c^2 in cm for this mass (C: GMC2 = MSUNCM*MASS == masscm).
    pub fn gmc2(u: Units) f64 {
        return u.masscm;
    }
    /// GM/c^3 in seconds (C: GMC3).
    pub fn gmc3(u: Units) f64 {
        return u.gmc2() / CCC0;
    }

    // --- LTE relations (C: rad.c calc_LTE_EfromT/calc_LTE_TfromE) ---

    /// Blackbody radiation energy density E = 4·σ·T⁴ (GU; T in Kelvin).
    pub fn lteEfromT(u: Units, t: f64) f64 {
        return 4.0 * u.sigmaRad() * t * t * t * t;
    }
    /// Inverse of lteEfromT.
    pub fn lteTfromE(u: Units, e: f64) f64 {
        return @sqrt(@sqrt(e / (4.0 * u.sigmaRad())));
    }
};

//
// ---- M0 gate tests -------------------------------------------------------
//

fn expectRel(expected: f64, actual: f64, rtol: f64) !void {
    const denom = @max(@abs(expected), @abs(actual));
    if (denom == 0) return;
    const rel = @abs(expected - actual) / denom;
    if (rel > rtol) {
        std.debug.print("expected {e}, got {e}, rel err {e} > rtol {e}\n", .{ expected, actual, rel, rtol });
        return error.ToleranceExceeded;
    }
}

test "units: sigmaThompson uses the standard Thomson cross section" {
    const u = Units.init(1.0);
    const expected = u.crossCgs2Gu(SIGMATH_CGS);
    try expectRel(expected, u.sigmaThompson(), 1e-15);
}

test "units: CGS→GU→CGS round-trips at 1e-15 across magnitudes and masses" {
    const masses = [_]f64{ 1.0, 10.0, 1.0 / MSUNCM, 4.3e6 };
    const xs = [_]f64{ 1.0e-24, 1.0e-8, 1.0, 3.7, 1.0e10, 1.0e24 };
    for (masses) |m| {
        const u = Units.init(m);
        for (xs) |x| {
            try expectRel(x, u.lenGu2Cgs(u.lenCgs2Gu(x)), 1e-15);
            try expectRel(x, u.timeGu2Cgs(u.timeCgs2Gu(x)), 1e-15);
            try expectRel(x, u.velGu2Cgs(u.velCgs2Gu(x)), 1e-15);
            try expectRel(x, u.rhoGu2Cgs(u.rhoCgs2Gu(x)), 1e-15);
            try expectRel(x, u.surfdensGu2Cgs(u.surfdensCgs2Gu(x)), 1e-15);
            try expectRel(x, u.massGu2Cgs(u.massCgs2Gu(x)), 1e-15);
            try expectRel(x, u.kappaGu2Cgs(u.kappaCgs2Gu(x)), 1e-15);
            try expectRel(x, u.endenGu2Cgs(u.endenCgs2Gu(x)), 1e-15);
            try expectRel(x, u.fluxGu2Cgs(u.fluxCgs2Gu(x)), 1e-15);
            try expectRel(x, u.ergGu2Cgs(u.ergCgs2Gu(x)), 1e-15);
            try expectRel(x, u.chargeGu2Cgs(u.chargeCgs2Gu(x)), 1e-15);
            try expectRel(x, u.crossGu2Cgs(u.crossCgs2Gu(x)), 1e-15);
            try expectRel(x, u.numdensGu2Cgs(u.numdensCgs2Gu(x)), 1e-15);
            try expectRel(x, u.heatcoolGu2Cgs(u.heatcoolCgs2Gu(x)), 1e-15);
        }
    }
}

test "units: E = aT⁴ round-trip at 1e-14 over E ∈ [1e-30, 1e10]" {
    const u = Units.init(10.0); // PUFFY mass
    var e: f64 = 1.0e-30;
    while (e <= 1.0e10) : (e *= 10.0) {
        const t = u.lteTfromE(e);
        try expectRel(e, u.lteEfromT(t), 1e-14);
    }
    // and T→E→T
    var t: f64 = 1.0e2;
    while (t <= 1.0e12) : (t *= 10.0) {
        try expectRel(t, u.lteTfromE(u.lteEfromT(t)), 1e-14);
    }
}

test "units: unit scales carry the exact MASS powers" {
    const ua = Units.init(3.0);
    const ub = Units.init(6.0); // ×2
    // length, time, mass units ∝ M
    try expectRel(2.0, ub.lenGu2Cgs(1.0) / ua.lenGu2Cgs(1.0), 1e-15);
    try expectRel(2.0, ub.timeGu2Cgs(1.0) / ua.timeGu2Cgs(1.0), 1e-15);
    try expectRel(2.0, ub.massGu2Cgs(1.0) / ua.massGu2Cgs(1.0), 1e-15);
    // density and energy-density units ∝ M⁻²
    try expectRel(0.25, ub.rhoGu2Cgs(1.0) / ua.rhoGu2Cgs(1.0), 1e-15);
    try expectRel(0.25, ub.endenGu2Cgs(1.0) / ua.endenGu2Cgs(1.0), 1e-15);
    // opacity unit ∝ M
    try expectRel(2.0, ub.kappaGu2Cgs(1.0) / ua.kappaGu2Cgs(1.0), 1e-15);
}

test "units: default C mass (MASS = 1/MSUNCM) makes MASSCM = 1 → length identity" {
    const u = Units.init(1.0 / MSUNCM);
    try expectRel(1.0, u.masscm, 1e-15);
    try expectRel(123.456, u.lenGu2Cgs(123.456), 1e-15);
}

test "units: GU constants sanity (dimensionless ratios independent of mass)" {
    const ua = Units.init(1.0);
    const ub = Units.init(1.0e6);
    // m_p/m_e is a pure number — must not depend on the unit system
    try expectRel(ua.mpeRatio(), ub.mpeRatio(), 1e-15);
    try expectRel(M_PROTON_CGS / M_ELECTR_CGS, ua.mpeRatio(), 1e-15);
}
