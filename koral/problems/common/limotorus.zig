//! The "limotorus" equilibrium torus (Penna et al. 2013 §2; Lančová et al.
//! 2019), transcribed from KORAL's problem-side tools.c (PROBLEMS/PUFFY,
//! shared by every torus problem that uses init_dsandvels_limotorus):
//!
//!   compute_gd       tools.c:1    BL g_tt, g_tφ, g_φφ
//!   lK / l3d         tools.c:14/22 Keplerian ℓ and the broken-power-law ℓ(λ)
//!   lamBL            tools.c:68   von Zeipel cylinder radius (bisection)
//!   omega3d / Agrav  tools.c:89/93
//!   rmidlam          tools.c:97   midplane radius of a given λ (bisection)
//!   lnf_integrand    tools.c:146  the ln f quadrature integrand
//!   init_dsandvels   tools.c:195  ρ, u, ℓ at (r, θ)
//!
//! The angular-momentum window (LT_XI, LT_R1, LT_R2), the polytrope
//! (LT_GAMMA, LT_KAPPA) and the inner edge (LT_RIN) are C `define.h`
//! constants; here they are a `Params` value so a second problem can pass
//! its own. gsl_integration_qags (epsrel 1e-8) is replaced by
//! math/quad.zig adaptive GK21 at rtol 1e-12, so the C diff is bounded by
//! C's own epsrel (pinned by the epsrel-1e-12 oracle variant).
//!
//! C recomputes the angular-momentum breaks and inner-edge constants for
//! every cell (tools.c:239-253); they are cell-independent, so
//! `torusConsts` computes them once — bitwise the same values.
//!
//! Moved verbatim out of problems/puffy/puffy.zig (redesign step 5,
//! 2026-09-04); expression shapes are unchanged.

const std = @import("std");
const quad = @import("../../math/quad.zig");
const misc = @import("../../math/misc.zig");

/// The torus constants a problem chooses (PUFFY: define.h LT_*).
pub const Params = struct {
    /// LT_XI: ℓ = ξ·ℓ_K(λ) inside the window.
    xi: f64,
    /// LT_R1 / LT_R2: radii of the angular-momentum breaks.
    r1: f64,
    r2: f64,
    /// LT_GAMMA: the torus polytropic index.
    gamma: f64,
    /// LT_RIN: the inner edge (cylindrical radius R = r sinθ).
    rin: f64,
    /// LT_KAPPA: the polytropic constant, P = κ ρ^γ.
    kappa: f64,
};

const xacc: f64 = 5.0 * std.math.floatEps(f64); // C: 5.*DBL_EPSILON
const pi_2: f64 = std.math.pi / 2.0; // C: M_PI_2

/// HARM's nrutil.c bisection (tools.c:26), shared through math/misc.zig.
pub const rtbis = misc.rtbis;

pub const Gd = struct { gdtt: f64, gdtp: f64, gdpp: f64 };

/// tools.c:1 compute_gd; BL g_tt, g_tφ, g_φφ (limotorus4.nb expressions).
pub fn computeGd(r: f64, th: f64, a: f64) Gd {
    const ac = a * @cos(th);
    const sigma = ac * ac + r * r;
    const s2 = @sin(th) * @sin(th);
    const tmp = 2.0 * r * s2 / sigma;
    return .{
        .gdtt = -1.0 + 2.0 * r / sigma,
        .gdtp = -a * tmp,
        .gdpp = (r * r + a * a * (1.0 + tmp)) * s2,
    };
}

/// tools.c:14 lK; Keplerian equatorial ℓ = u_φ/u_t.
pub fn lK(r: f64, a: f64) f64 {
    const curly_f = 1.0 - 2.0 * a / std.math.pow(f64, r, 1.5) + (a / r) * (a / r);
    const curly_g = 1.0 - 2.0 / r + a / std.math.pow(f64, r, 1.5);
    return @sqrt(r) * curly_f / curly_g;
}

/// tools.c:22 l3d; ξ·lK clamped to the broken-power-law window.
pub fn l3d(lam: f64, a: f64, lambreak1: f64, lambreak2: f64, xi: f64) f64 {
    const arg = if (lam <= lambreak1) lambreak1 else if (lam >= lambreak2) lambreak2 else lam;
    return xi * lK(arg, a);
}

/// tools.c:51 lamBL_func.
pub const LamF = struct {
    gdtt: f64,
    gdtp: f64,
    gdpp: f64,
    a: f64,
    lambreak1: f64,
    lambreak2: f64,
    xi: f64,
    pub fn eval(s: LamF, lam: f64) f64 {
        const l = l3d(lam, s.a, s.lambreak1, s.lambreak2, s.xi);
        return lam * lam + l * (l * s.gdtp + s.gdpp) / (l * s.gdtt + s.gdtp);
    }
};

/// tools.c:68 lamBL; von Zeipel cylinder radius, bracket (R, 10R).
pub fn lamBL(R: f64, gd: Gd, a: f64, lambreak1: f64, lambreak2: f64, xi: f64) f64 {
    return rtbis(LamF{
        .gdtt = gd.gdtt,
        .gdtp = gd.gdtp,
        .gdpp = gd.gdpp,
        .a = a,
        .lambreak1 = lambreak1,
        .lambreak2 = lambreak2,
        .xi = xi,
    }, R, 10.0 * R, xacc);
}

/// tools.c:89 omega3d (pow(x,-1) folds to a reciprocal under clang -O2).
pub fn omega3d(l: f64, gd: Gd) f64 {
    return -(gd.gdtt * l + gd.gdtp) * (1.0 / (gd.gdpp + gd.gdtp * l));
}

/// tools.c:93 compute_Agrav.
pub fn computeAgrav(om: f64, gd: Gd) f64 {
    return @sqrt(@abs(1.0 / (gd.gdtt + 2.0 * om * gd.gdtp + om * om * gd.gdpp)));
}

/// tools.c:97 rmidlam; the passed-in gd values are dead in C (immediately
/// overwritten with midplane values at x); only lam², a, breaks, ξ matter.
pub const RmidF = struct {
    lamsq: f64,
    a: f64,
    lambreak1: f64,
    lambreak2: f64,
    xi: f64,
    pub fn eval(s: RmidF, x: f64) f64 {
        const gd = computeGd(x, pi_2, s.a);
        const lam_x = lamBL(x, gd, s.a, s.lambreak1, s.lambreak2, s.xi);
        return s.lamsq - lam_x * lam_x;
    }
};

/// tools.c:118 limotorus_findrml; bracket (6, 10·λ²) as written in C.
pub fn findRml(lam: f64, a: f64, lambreak1: f64, lambreak2: f64, xi: f64) f64 {
    const lamsq = lam * lam;
    return rtbis(RmidF{
        .lamsq = lamsq,
        .a = a,
        .lambreak1 = lambreak1,
        .lambreak2 = lambreak2,
        .xi = xi,
    }, 6.0, 10.0 * lamsq, xacc);
}

/// tools.c:146 lnf_integrand.
pub const LnfIntegrand = struct {
    a: f64,
    lambreak1: f64,
    lambreak2: f64,
    xi: f64,
    pub fn eval(s: LnfIntegrand, r: f64) f64 {
        const r2 = r * r;
        const r3 = r2 * r;
        const a2 = s.a * s.a;
        const gd = computeGd(r, pi_2, s.a);
        const lam = lamBL(r, gd, s.a, s.lambreak1, s.lambreak2, s.xi);
        const lam2 = lam * lam;
        const l = l3d(lam, s.a, s.lambreak1, s.lambreak2, s.xi);

        const term1 = (r - 2.0) * l;
        const term2 = 2.0 * s.a;
        const term3 = r3 + a2 * (r + 2.0);
        const term4 = term2 * l;
        const om_numerator = term1 + term2;
        const om_denominator = term3 - term4;
        const om = om_numerator / om_denominator;

        var dl_dr: f64 = 0.0;
        if (lam <= s.lambreak1 or lam >= s.lambreak2) {
            dl_dr = 0.0;
        } else {
            const oneplusx = 1.0 - 2.0 / lam + s.a * std.math.pow(f64, lam, -1.5);
            const dx_dlam = 2.0 * std.math.pow(f64, lam, -2.0) - 1.5 * s.a * std.math.pow(f64, lam, -2.5);
            const lamroot = @sqrt(lam);
            const dlk_dlam = (oneplusx * 0.5 * (1.0 / lamroot) + (s.a - lamroot) * dx_dlam) /
                (oneplusx * oneplusx);
            const dd = term3 - 2.0 * term4 - lam2 * (r - 2.0);
            const ee = l * (3.0 * r2 + a2 - lam2);
            dl_dr = ee / (2.0 * lam * om_numerator / (s.xi * dlk_dlam) - dd);
        }

        const dom_dr = (om_denominator * (l + (r - 2.0) * dl_dr) -
            om_numerator * (3.0 * r2 + a2 + 2.0 * s.a * dl_dr)) /
            (om_denominator * om_denominator);

        return -l / (1.0 - om * l) * dom_dr;
    }
};

/// The cell-independent part of init_dsandvels_limotorus (tools.c:239-253):
/// break λ's, and ℓ/ω/Agrav at the torus inner edge. C recomputes these per
/// cell: the values are identical.
pub const TorusConsts = struct {
    lambreak1: f64,
    lambreak2: f64,
    lamin: f64,
    lin: f64,
    omin: f64,
    agravin: f64,
};

pub fn torusConsts(a: f64, lt: Params) TorusConsts {
    var gd = computeGd(lt.r1, pi_2, a);
    const lambreak1 = lamBL(lt.r1, gd, a, 0.0, 200000.0, lt.xi);
    gd = computeGd(lt.r2, pi_2, a);
    const lambreak2 = lamBL(lt.r2, gd, a, lambreak1, 200000.0, lt.xi);

    gd = computeGd(lt.rin, pi_2, a);
    const lamin = lamBL(lt.rin, gd, a, lambreak1, lambreak2, lt.xi);
    const lin = l3d(lamin, a, lambreak1, lambreak2, lt.xi);
    const omin = omega3d(lin, gd);
    const agravin = computeAgrav(omin, gd);

    return .{
        .lambreak1 = lambreak1,
        .lambreak2 = lambreak2,
        .lamin = lamin,
        .lin = lin,
        .omin = omin,
        .agravin = agravin,
    };
}

pub const DsVels = struct { rho: f64, uu: f64, ell: f64 };

/// tools.c:195 init_dsandvels_limotorus at (r, θ) in BL. rho = −1 flags
/// "outside the torus" (including quadrature failure, mirroring the
/// GSL_NEGINF branch).
pub fn initDsandvels(r: f64, th: f64, a: f64, tc: *const TorusConsts, lt: Params) DsVels {
    const kappa = lt.kappa;
    const R = r * @sin(th);
    if (R < lt.rin) return .{ .rho = -1.0, .uu = 0.0, .ell = 0.0 };
    const gd = computeGd(r, th, a);
    const lam = lamBL(R, gd, a, tc.lambreak1, tc.lambreak2, lt.xi);
    const l = l3d(lam, a, tc.lambreak1, tc.lambreak2, lt.xi);
    const om = omega3d(l, gd);
    const agrav = computeAgrav(om, gd);
    const rml = findRml(lam, a, tc.lambreak1, tc.lambreak2, lt.xi);

    const integrand = LnfIntegrand{
        .a = a,
        .lambreak1 = tc.lambreak1,
        .lambreak2 = tc.lambreak2,
        .xi = lt.xi,
    };
    // C: gsl_integration_qags(rin → rml, epsabs 0, epsrel 1e-8); failure sets
    // lnf3d = GSL_NEGINF so the density comes out 0/negative.
    const q = quad.integrate(integrand, lt.rin, rml, 0.0, 1.0e-12);
    const lnf3d: f64 = if (q.converged) q.value else -std.math.inf(f64);
    const f3d = @exp(-lnf3d);

    const f3din: f64 = 1.0; // ≡ 1 at r = rin by construction
    const hh = f3din * agrav / (f3d * tc.agravin);
    const eps = (-1.0 + hh) * (1.0 / lt.gamma);

    // C: pow(kappa,-1) and pow(-1+LT_GAMMA,-1) fold to reciprocals.
    const rho: f64 = if (eps < 0.0)
        -1.0
    else
        std.math.pow(f64, (-1.0 + lt.gamma) * eps * (1.0 / kappa), 1.0 / (-1.0 + lt.gamma));

    return .{
        .rho = rho,
        .ell = l,
        .uu = kappa * std.math.pow(f64, rho, lt.gamma) / (lt.gamma - 1.0),
    };
}
