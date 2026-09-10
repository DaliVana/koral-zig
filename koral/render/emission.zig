//! Frequency-dependent (monochromatic) emissivities for the renderer's
//! synthetic-observation mode: thermal synchrotron (the Leung, Tchekhovskoy
//! & Gammie 2011 fit used by grmonty/ipole), thermal bremsstrahlung
//! (Rybicki & Lightman 5.18b with the Born-approximation thermal Gaunt
//! factor ḡ_ff(hν/kT_e) and the gray opacity's relativistic correction; the
//! spectral shape j_ν ∝ ḡ e^{−u} IS the Gaunt factor), and the
//! electron-scattering source: the M1 field re-emitted as the diluted
//! (colour-corrected) blackbody f⁻⁴B_ν(fT_rad), exact in normalization by
//! the definition of T_rad (Ê = 4σT_rad⁴) for ANY f, with the hardening
//! factor f from Done et al. (2012) blended by the cell's scattering
//! fraction (fcolDone12/blendFcol; f = 1 is the plain blackbody), times the
//! M1 dipole factor the caller supplies.
//!
//! Everything here is CGS and per-Hz in the FLUID frame: j [erg/s/cm³/sr/Hz],
//! χ [1/cm], ν [Hz], T [K], B [Gauss]. Absorption obeys Kirchhoff's law
//! (α_ν = j_ν/B_ν(T_e)) for both thermal channels, so LTE source functions
//! are exactly Planckian; pinned by a test.
//!
//! Constants match units.zig/thermo.zig where they exist there.

const std = @import("std");

pub const c_cgs: f64 = 2.9979246e10; // units.CCC0
pub const h_cgs: f64 = 6.6260755e-27; // units.H_CGS
pub const k_cgs: f64 = 1.3806488e-16; // units.K_BOLTZ_CGS
pub const m_e_cgs: f64 = 9.1094e-28; // units.M_ELECTR_CGS
pub const e_esu: f64 = 4.80320425e-10; // electron charge (statC; not in units.zig)

// ---- free-free Gaunt factor ----------------------------------------------

/// √3/π, the prefactor of the thermal free-free Gaunt factor.
const sqrt3_over_pi: f64 = 0.5513288954217921;

/// Thermally averaged free-free Gaunt factor in the Born approximation,
///     ḡ_ff(u) = (√3/π) e^{u/2} K₀(u/2),   u = hν/kT_e,
/// the γ² = Z²Ry/kT → 0 limit of the exact Karzas & Latter (1961) result
/// (Rybicki & Lightman §5.3; van Hoof et al. 2014 tabulate the exact one).
/// Limits: (√3/π) ln(4kT/(ζhν)) for u ≪ 1 (ζ = e^{γ_E}) and √(3kT/(πhν))
/// for u ≫ 1, so one expression covers the mm band (u ~ 10⁻⁹, ḡ ~ 12) and
/// the X-ray band (u ~ 1, ḡ ~ 0.8) the renderer images; a constant is
/// right only near u ≈ 1 and turns X-ray slopes into factor-2 errors at
/// the band edges. Its emission-weighted average ∫ḡ e^{−u} du = 2√3/π =
/// 1.10, next to the 1.2 the gray opacity uses for the total power.
/// Elwert–Sommerfeld corrections grow with γ² and matter below ~10⁶ K
/// (γ² ≳ 0.2), where X-ray free-free is Wien-suppressed anyway; at mm
/// frequencies the Born value stays within tens of percent there.
pub fn gauntFF(u: f64) f64 {
    return sqrt3_over_pi * besselK0e(0.5 * @max(u, 1e-300));
}

/// Relativistic correction to the thermal free-free power, 1 + 4.4×10⁻¹⁰ T_e
/// (Rybicki & Lightman 5.25b), applied frequency-independently: the same
/// factor physics/opacities.zig puts on the gray emissivity, so the
/// frequency integral of j_ν tracks the cooling the sim actually evolved.
pub fn relFF(te: f64) f64 {
    return 1.0 + 4.4e-10 * te;
}

// ---- Planck --------------------------------------------------------------

/// B_ν(T) [erg/s/cm²/sr/Hz].
pub fn planckNu(nu: f64, temp: f64) f64 {
    if (!(nu > 0) or !(temp > 0)) return 0;
    const x = h_cgs * nu / (k_cgs * temp);
    if (x > 700.0) return 0; // Wien underflow
    const pref = 2.0 * h_cgs * nu * nu * nu / (c_cgs * c_cgs);
    return pref / std.math.expm1(x);
}

// ---- colour temperature correction ----------------------------------------

/// kT in keV per kelvin (1 keV = 1.602176634×10⁻⁹ erg).
pub const kev_per_kelvin: f64 = k_cgs / 1.602176634e-9;

/// Spectral hardening (colour correction) factor of a scattering-dominated
/// atmosphere as a function of its effective temperature: the local
/// prescription of Done, Davis, Jin, Blaes & Ward (2012, MNRAS 420, 1848,
/// eqs. 1–2; eq. 1 is Davis et al. 2006's A13, the electron-scattering
/// saturation value):
///     f = (72 keV / kT)^{1/9}     electron scattering saturated (T ≳ 10⁵ K)
///     f = (T / 3×10⁴ K)^{0.82}    3×10⁴ K < T < 10⁵ K (H/He ionizing)
///     f = 1                       below 3×10⁴ K (no free electrons)
/// Written as max(1, min(eq. 2, eq. 1)) so the branches join continuously
/// (published as-is they differ by 2% at 10⁵ K). Reference values quoted in
/// the paper: 1.6 at kT = 1 keV, 2.34 at 4×10⁵ K, 2.4 at 3×10⁵ K, 2.7 at
/// 10⁵ K; inside the f ≈ 1.4–2 range of Davis & El-Abd (2019) for 1–100%
/// Eddington and next to Shimura & Takahara's (1995) canonical 1.7.
/// Magnetically supported atmospheres harden further (Blaes et al. 2006),
/// so for puffy discs treat this as a floor.
pub fn fcolDone12(t_kelvin: f64) f64 {
    if (!(t_kelvin > 3.0e4)) return 1.0;
    const low = std.math.pow(f64, t_kelvin / 3.0e4, 0.82);
    const high = std.math.pow(f64, 72.0 / (kev_per_kelvin * t_kelvin), 1.0 / 9.0);
    return @max(1.0, @min(low, high));
}

/// Diluted (colour-corrected) blackbody f⁻⁴ B_ν(fT): the Shimura & Takahara
/// (1995) form of the emergent spectrum of a scattering-dominated
/// atmosphere. f⁻⁴ keeps the frequency integral at σT⁴/π for any f, so the
/// gray energy budget of the scattered light is unchanged; only the shape
/// hardens (peak at f × the blackbody peak, Rayleigh–Jeans tail × f⁻³).
/// f = 1 is planckNu bit-for-bit.
pub fn dilutedPlanck(nu: f64, temp: f64, f: f64) f64 {
    const f2 = f * f;
    return planckNu(nu, f * temp) / (f2 * f2);
}

/// Blend the prescription with the cell's scattering fraction
/// w = κ_es/(κ_es + κ_abs) (gray single-scattering albedo):
/// f_eff = 1 + (f − 1)·w. Done et al. state eq. 1 holds where scattering
/// dominates and f → 1 where absorption does (the field is then Planckian
/// at the gas temperature and Kirchhoff's law is recovered); the linear
/// blend is the simplest interpolation between those two limits.
pub fn blendFcol(f: f64, albedo: f64) f64 {
    const w = std.math.clamp(albedo, 0.0, 1.0);
    return 1.0 + (f - 1.0) * w;
}

// ---- modified Bessel functions (A&S 9.8, |rel err| ~ 1e-7) ---------------

pub fn besselK0(x: f64) f64 {
    std.debug.assert(x > 0);
    if (x <= 2.0) {
        const t = x * x / 4.0;
        const bi0 = blk: {
            const s = x * x / (3.75 * 3.75);
            break :blk 1.0 + s * (3.5156229 + s * (3.0899424 + s * (1.2067492 + s * (0.2659732 + s * (0.0360768 + s * 0.0045813)))));
        };
        return -@log(x / 2.0) * bi0 +
            (-0.57721566 + t * (0.42278420 + t * (0.23069756 + t * (0.03488590 + t * (0.00262698 + t * (0.00010750 + t * 0.00000740))))));
    }
    const u = 2.0 / x;
    return @exp(-x) / @sqrt(x) * k0AsymPoly(u);
}

/// A&S 9.8.6 polynomial in u = 2/x: e^x √x K₀(x) for x ≥ 2.
inline fn k0AsymPoly(u: f64) f64 {
    return 1.25331414 + u * (-0.07832358 + u * (0.02189568 + u * (-0.01062446 + u * (0.00587872 + u * (-0.00251540 + u * 0.00053208)))));
}

/// e^x·K₀(x), the exponentially scaled K₀: finite for every x > 0 (K₀
/// itself underflows past x ≈ 700 while the product tends to √(π/2x)).
pub fn besselK0e(x: f64) f64 {
    std.debug.assert(x > 0);
    if (x <= 2.0) return @exp(x) * besselK0(x);
    return k0AsymPoly(2.0 / x) / @sqrt(x);
}

pub fn besselK1(x: f64) f64 {
    std.debug.assert(x > 0);
    if (x <= 2.0) {
        const t = x * x / 4.0;
        const bi1 = blk: {
            const s = x * x / (3.75 * 3.75);
            break :blk x * (0.5 + s * (0.87890594 + s * (0.51498869 + s * (0.15084934 + s * (0.02658733 + s * (0.00301532 + s * 0.00032411))))));
        };
        return @log(x / 2.0) * bi1 + (1.0 / x) *
            (1.0 + t * (0.15443144 + t * (-0.67278579 + t * (-0.18156897 + t * (-0.01919402 + t * (-0.00110404 + t * -0.00004686))))));
    }
    const u = 2.0 / x;
    return @exp(-x) / @sqrt(x) *
        (1.25331414 + u * (0.23498619 + u * (-0.03655620 + u * (0.01504268 + u * (-0.00780353 + u * (0.00325614 + u * -0.00068245))))));
}

/// K₂ via the standard recurrence K_{n+1} = K_{n−1} + 2n·K_n/x.
pub fn besselK2(x: f64) f64 {
    return besselK0(x) + 2.0 * besselK1(x) / x;
}

// ---- assembled monochromatic emissivity/extinction ------------------------

pub const MonoIn = struct {
    /// fluid-frame frequency [Hz]
    nu: f64,
    ne_cgs: f64,
    ni_cgs: f64,
    /// electron temperature [K]
    te: f64,
    /// radiation (blackbody) temperature [K] for the scattering source
    trad: f64,
    b_gauss: f64,
    /// sine of the photon-B pitch angle in the fluid frame
    sin_pitch: f64,
    /// electron-scattering extinction [1/cm]
    chi_es_cgs: f64,
    /// M1 dipole factor 1 + 3n̂·F̂/Ê, clamped ≥ 0 by the caller
    dip: f64,
    /// effective colour correction of the scattered field (blendFcol of
    /// the prescription with the cell's albedo); 1 = plain blackbody at T_rad
    fcol: f64 = 1.0,
};

pub const MonoOut = struct {
    /// total emissivity toward the ray [erg/s/cm³/sr/Hz]
    j: f64,
    /// total extinction [1/cm]
    chi: f64,
};

/// Thermal synchrotron emissivity, Leung+2011 approximation (their eq. 72,
/// the grmonty/ipole form): valid for θ_e ≳ 0.5 but harmless down to the
/// θ_e = 0.05 gate below, under which the emission is utterly negligible
/// and K₂(1/θ_e) underflows.
fn synchrotronJ(nu: f64, ne: f64, theta_e: f64, b: f64, sin_pitch: f64) f64 {
    const nu_c = e_esu * b / (2.0 * std.math.pi * m_e_cgs * c_cgs);
    const nu_s = (2.0 / 9.0) * nu_c * theta_e * theta_e * sin_pitch;
    if (!(nu_s > 0)) return 0;
    const big_x = nu / nu_s;
    const cbrt_x = std.math.cbrt(big_x);
    if (cbrt_x > 700.0) return 0; // exponential cutoff underflow
    const root = @sqrt(big_x) + 1.8871249 * std.math.pow(f64, big_x, 1.0 / 6.0); // 2^(11/12)
    const pref = ne * std.math.sqrt2 * std.math.pi * e_esu * e_esu * nu_s /
        (3.0 * c_cgs * besselK2(1.0 / theta_e));
    return pref * root * root * @exp(-cbrt_x);
}

/// j_ν and χ_ν at one point, fluid frame, CGS.
pub fn monoJChi(in: MonoIn) MonoOut {
    var j: f64 = 0;
    var chi: f64 = in.chi_es_cgs;

    const bnu_e = planckNu(in.nu, in.te);

    // free-free: α from RL 5.18b with the Born thermal Gaunt factor and the
    // gray opacity's relativistic correction; j via Kirchhoff. With the ν³
    // factors cancelling, j_ν = α_ν B_ν ∝ ḡ(u) e^{−u} at fixed T_e — the
    // spectral shape is the Gaunt factor itself (pinned by test).
    if (in.te > 0 and in.ne_cgs > 0 and in.nu > 0) {
        const x = h_cgs * in.nu / (k_cgs * in.te);
        const one_m_emx = if (x > 1e-6) -std.math.expm1(-x) else x;
        const aff = 3.7e8 * in.ne_cgs * in.ni_cgs * gauntFF(x) * relFF(in.te) * one_m_emx /
            (@sqrt(in.te) * in.nu * in.nu * in.nu);
        j += aff * bnu_e;
        chi += aff;
    }

    // thermal synchrotron, j from the fit, α via Kirchhoff
    const theta_e = k_cgs * in.te / (m_e_cgs * c_cgs * c_cgs);
    if (theta_e > 0.05 and in.b_gauss > 1e-8 and in.sin_pitch > 0.02 and bnu_e > 1e-300) {
        const js = synchrotronJ(in.nu, in.ne_cgs, theta_e, in.b_gauss, in.sin_pitch);
        j += js;
        chi += js / bnu_e;
    }

    // scattering source: the M1 field at T_rad, colour-corrected
    // (f⁻⁴B_ν(fT_rad); f = 1 → plain blackbody), times the dipole
    j += in.chi_es_cgs * dilutedPlanck(in.nu, in.trad, in.fcol) * in.dip;

    return .{ .j = j, .chi = chi };
}
