//! Frequency-dependent (monochromatic) emissivities for the renderer's
//! synthetic-observation mode: thermal synchrotron (the Leung, Tchekhovskoy
//! & Gammie 2011 fit used by grmonty/ipole), thermal bremsstrahlung
//! (Rybicki & Lightman 5.18b with a constant mean Gaunt factor), and the
//! electron-scattering source with the radiation field taken blackbody-
//! shaped at T_rad, which is exact in normalization by the definition of
//! T_rad (Ê = 4σT_rad⁴), times the M1 dipole factor the caller supplies.
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

/// Mean free-free Gaunt factor. A constant ~1.2 is good to ~20% across the
/// mm-X-ray range we image; upgrade to a ν,T-dependent fit if spectra ever
/// need better.
pub const gaunt_ff: f64 = 1.2;

// ---- Planck --------------------------------------------------------------

/// B_ν(T) [erg/s/cm²/sr/Hz].
pub fn planckNu(nu: f64, temp: f64) f64 {
    if (!(nu > 0) or !(temp > 0)) return 0;
    const x = h_cgs * nu / (k_cgs * temp);
    if (x > 700.0) return 0; // Wien underflow
    const pref = 2.0 * h_cgs * nu * nu * nu / (c_cgs * c_cgs);
    return pref / std.math.expm1(x);
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
    return @exp(-x) / @sqrt(x) *
        (1.25331414 + u * (-0.07832358 + u * (0.02189568 + u * (-0.01062446 + u * (0.00587872 + u * (-0.00251540 + u * 0.00053208))))));
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

    // free-free: α from RL 5.18b, j via Kirchhoff
    if (in.te > 0 and in.ne_cgs > 0 and in.nu > 0) {
        const x = h_cgs * in.nu / (k_cgs * in.te);
        const one_m_emx = if (x > 1e-6) -std.math.expm1(-x) else x;
        const aff = 3.7e8 * in.ne_cgs * in.ni_cgs * gaunt_ff * one_m_emx /
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

    // scattering source: blackbody-shaped M1 field at T_rad times the dipole
    j += in.chi_es_cgs * planckNu(in.nu, in.trad) * in.dip;

    return .{ .j = j, .chi = chi };
}
