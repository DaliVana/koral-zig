//! The Farris et al. (2008) radiative shock tube battery, in the form used
//! by Sądowski et al. (2013, MNRAS 429, 3533) Table 3 — tests 1, 2, 3a, 3b,
//! 4a, 4b (3b/4b are the higher-opacity variants of Roedig et al. 2012).
//! Shared between the M10 theory battery and the C step tests.
//!
//! Unit convention: the papers set T = p/ρ (code units) and tune the
//! radiation constant a so both asymptotic states are in LTE, Ê = a·T⁴.
//! KORAL instead computes T in Kelvin with physical constants:
//! T = mugas_mp_over_kb·(p/ρ) and Ê_LTE = 4σ_gu·T⁴, with 4σ_gu ∝ MASSCM².
//! So a_code(MASS) = four_sigmarad·mugas_mp_over_kb⁴ ∝ MASS², and choosing
//!   MASS = √(a_paper / a_code(1 M☉))
//! makes KORAL's LTE exactly the paper's — identically on the C side, which
//! shares the same constant shapes (ZIGRADTUBE's define.h carries the MASS
//! printed by the `tube MASS values` test below).

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const units_mod = @import("../units.zig");
const thermo = @import("../physics/thermo.zig");
const hydro = @import("../physics/hydro.zig");
const radforce = @import("../physics/radforce.zig");

/// Hydro + M1 radiation, no B — C: RADIATION without MAGNFIELD, NV = 10.
pub const cfg_rad = config.Config{
    .modules = &.{ .hydro, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};

// the C-diffability contract for the hydro+rad layout: EE0 = NVMHD = 6
comptime {
    const L = layout.VarLayout(cfg_rad);
    std.debug.assert(L.count == 10);
    std.debug.assert(L.index(.entr) == 5);
    std.debug.assert(L.index(.ee) == 6);
    std.debug.assert(L.index(.fz) == 9);
}

/// MU_GAS = MU_I = MU_E = 1 (ZIGRADTUBE/ZIGRADPULSE define.h), HFRAC = 1.
pub const comp = thermo.Composition{
    .mu_gas_override = 1.0,
    .mu_i_override = 1.0,
    .mu_e_override = 1.0,
};

pub const Side = struct {
    rho: f64,
    p: f64,
    ux: f64, // proper velocity u^x (= VELR primitive in MINK)
    erad: f64, // fluid-frame Ê; radiation comoving with gas (F̂ = 0)
};

pub const Tube = struct {
    name: []const u8,
    gam: f64,
    kappa: f64, // grey absorption per unit mass, χ = κ·ρ (code units)
    left: Side,
    right: Side,
    smooth: bool, // continuous stationary profile (no embedded gas shock)

    /// paper radiation constant a = Ê_L/T_L⁴ with T = p/ρ
    pub fn aPaper(t: *const Tube) f64 {
        const tl = t.left.p / t.left.rho;
        return t.left.erad / (tl * tl * tl * tl);
    }

    pub fn mass(t: *const Tube) f64 {
        return massForA(t.aPaper());
    }

    pub fn params(t: *const Tube) radforce.Params {
        return radforce.Params.grey(units_mod.Units.init(t.mass()), comp, t.kappa, 0.0);
    }
};

/// Sądowski et al. (2013) Table 3 (= Farris et al. 2008 §6.3, plus the
/// Roedig et al. 2012 high-opacity variants).
pub const tubes = [_]Tube{
    .{ .name = "1", .gam = 5.0 / 3.0, .kappa = 0.4, .smooth = false,
        .left = .{ .rho = 1.0, .p = 3.0e-5, .ux = 0.015, .erad = 1.0e-8 },
        .right = .{ .rho = 2.4, .p = 1.61e-4, .ux = 6.25e-3, .erad = 2.51e-7 } },
    .{ .name = "2", .gam = 5.0 / 3.0, .kappa = 0.2, .smooth = false,
        .left = .{ .rho = 1.0, .p = 4.0e-3, .ux = 0.25, .erad = 2.0e-5 },
        .right = .{ .rho = 3.11, .p = 4.512e-2, .ux = 8.04e-2, .erad = 3.46e-3 } },
    .{ .name = "3a", .gam = 2.0, .kappa = 0.3, .smooth = true,
        .left = .{ .rho = 1.0, .p = 6.0e1, .ux = 10.0, .erad = 2.0 },
        .right = .{ .rho = 8.0, .p = 2.34e3, .ux = 1.25, .erad = 1.14e3 } },
    .{ .name = "3b", .gam = 2.0, .kappa = 25.0, .smooth = true,
        .left = .{ .rho = 1.0, .p = 6.0e1, .ux = 10.0, .erad = 2.0 },
        .right = .{ .rho = 8.0, .p = 2.34e3, .ux = 1.25, .erad = 1.14e3 } },
    .{ .name = "4a", .gam = 5.0 / 3.0, .kappa = 0.08, .smooth = true,
        .left = .{ .rho = 1.0, .p = 6.0e-3, .ux = 0.69, .erad = 0.18 },
        .right = .{ .rho = 3.65, .p = 3.59e-2, .ux = 0.189, .erad = 1.30 } },
    .{ .name = "4b", .gam = 5.0 / 3.0, .kappa = 0.7, .smooth = true,
        .left = .{ .rho = 1.0, .p = 6.0e-3, .ux = 0.69, .erad = 0.18 },
        .right = .{ .rho = 3.65, .p = 3.59e-2, .ux = 0.189, .erad = 1.30 } },
};

/// ZIGRADPULSE: an optically thick LTE temperature bump (gas and radiation
/// consistent everywhere — a cold pulse on a non-LTE background collapses:
/// wing cells go M1-inconsistent and pin the γ_rad ceiling) diffusing under
/// grey absorption. Paper units: T = p/ρ, E = a·T⁴ with a via MASS.
pub const pulse = struct {
    pub const rho: f64 = 1.0;
    pub const t0: f64 = 0.01; // background T = p/ρ
    pub const amp: f64 = 0.5; // T(x) = t0·(1 + amp·exp(−x²/w²))
    pub const w: f64 = 10.0;
    pub const kappa: f64 = 10.0; // τ_cell ≈ 16 at 64 cells over [-50,50]
    pub const a_paper: f64 = 5.0e5; // E0 = a·t0⁴ = 5e-3, E0/u0 = 1/3

    pub fn mass() f64 {
        return massForA(a_paper);
    }

    pub fn params() radforce.Params {
        return radforce.Params.grey(units_mod.Units.init(mass()), comp, kappa, 0.0);
    }

    /// primitives at position x (LTE, at rest)
    pub fn pp(comptime cfg: config.Config, x: f64, gam: f64) [layout.VarLayout(cfg).count]f64 {
        const L = layout.VarLayout(cfg);
        const t = t0 * (1.0 + amp * @exp(-x * x / (w * w)));
        var p: [L.count]f64 = @splat(0);
        p[L.index(.rho)] = rho;
        p[L.index(.uu)] = rho * t / (gam - 1.0);
        p[L.index(.entr)] = hydro.sFromU(rho, rho * t / (gam - 1.0), gam);
        p[L.index(.ee)] = a_paper * t * t * t * t;
        return p;
    }
};

/// a_code(mass) = four_sigmarad·mugas_mp_over_kb⁴ for MU_GAS = 1.
pub fn aCode(mass: f64) f64 {
    const c = thermo.Consts.init(units_mod.Units.init(mass), comp);
    const m1 = c.mugas_mp_over_kb;
    return c.four_sigmarad * m1 * m1 * m1 * m1;
}

/// Solve a_code(MASS) = a_paper; a_code ∝ MASS² exactly.
pub fn massForA(a_paper: f64) f64 {
    return @sqrt(a_paper / aCode(1.0));
}

/// Primitives for one asymptotic state: gas + comoving radiation.
pub fn ppSide(comptime cfg: config.Config, s: Side, gam: f64) [layout.VarLayout(cfg).count]f64 {
    const L = layout.VarLayout(cfg);
    var pp: [L.count]f64 = @splat(0);
    pp[L.index(.rho)] = s.rho;
    pp[L.index(.uu)] = s.p / (gam - 1.0);
    pp[L.index(.vx)] = s.ux;
    pp[L.index(.entr)] = hydro.sFromU(s.rho, s.p / (gam - 1.0), gam);
    pp[L.index(.ee)] = s.erad;
    pp[L.index(.fx)] = s.ux; // comoving: rad velocity = gas velocity
    return pp;
}

test "tube masses make code LTE the paper LTE on both sides" {
    for (&tubes) |*t| {
        const m = t.mass();
        try std.testing.expect(m > 0 and std.math.isFinite(m));
        // a_code(mass) == a_paper to rounding
        try std.testing.expectApproxEqRel(t.aPaper(), aCode(m), 1e-12);
        // right state is in LTE too — to the papers' 3-significant-digit
        // rounding (tube 3's Ê_R = 1.14e3 is ~0.9% off exact)
        const tr = t.right.p / t.right.rho;
        const er = aCode(m) * tr * tr * tr * tr;
        try std.testing.expectApproxEqRel(t.right.erad, er, 1.5e-2);
    }
}

test "tube MASS values for the C define.h (printed for regeneration)" {
    // ZIGRADTUBE uses tube 2; ZIGRADPULSE uses pulse.mass()
    for (&tubes) |*t| {
        std.debug.print("tube {s}: MASS = {e:.17}\n", .{ t.name, t.mass() });
    }
    std.debug.print("pulse: MASS = {e:.17}\n", .{pulse.mass()});
}
