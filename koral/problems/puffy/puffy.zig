//! The PUFFY problem (C: PROBLEMS/PUFFY, PROBLEM 147): 2D axisymmetric
//! radiation-MHD limotorus around a 10 M☉ Schwarzschild hole in MKS2.
//!
//! This module transcribes the problem-side C files:
//!   tools.c     — limotorus solver (lamBL/rmidlam bisections at 5ε, the
//!                 ln f quadrature; gsl_integration_qags epsrel 1e-8 is
//!                 replaced by math/quad.zig adaptive GK21 at rtol 1e-12,
//!                 so the C diff is bounded by C's own epsrel — pinned by
//!                 the epsrel-1e-12 oracle variant)
//!   prepinit.c  — per-cell initial primitives (torus + atmospheres, LTE
//!                 pressure split P = bT + aT⁴, prad_ff2lab, BL→MKS2,
//!                 QUADLOOPS vector potential in the B slots)
//!   init.c      — ENTR from calc_Sfromu + p2u
//!   postinit.c  — global β normalization: fac = √(MAXBETA/maxβ),
//!                 BETANORMFULL (max over the whole domain)
//!   bc.c        — XBCHI outflow with r-rescaling + no-inflow, XBCLO copy
//!                 (RMIN=1.85 < r_horizon=2 skips the inflow check),
//!                 YBC polar reflection (VY/B2/FY sign flip)
//! and the ko.c init sequence (ko.c:140-263): prepinit+init → set_bc →
//! calc_BfromA(p,1) → set_bc → postinit. No BC refresh after postinit —
//! ghost B keeps the pre-scaling values, exactly as in C.
//!
//! C recomputes the angular-momentum breaks and inner-edge constants for
//! every cell (tools.c:239-253); they are cell-independent, so we compute
//! them once (TorusConsts) — bitwise the same values.

const std = @import("std");
const config = @import("../../config.zig");
const layout = @import("../../layout.zig");
const grid_mod = @import("../../grid.zig");
const geometry = @import("../../geometry.zig");
const relele = @import("../../relele.zig");
const frames = @import("../../frames.zig");
const p2u_mod = @import("../../p2u.zig");
const hydro = @import("../../physics/hydro.zig");
const mhd = @import("../../physics/bfield.zig");
const thermo = @import("../../physics/thermo.zig");
const radiation = @import("../../physics/radiation.zig");
const invert_rad = @import("../../solve/invert_rad.zig");
const invert = @import("../../solve/invert.zig");
const threading = @import("../../threading.zig");
const implicit = @import("../../solve/implicit.zig");
const opacities = @import("../../physics/opacities.zig");
const mesa = @import("../../physics/mesa.zig");
const radforce = @import("../../physics/radforce.zig");
const units_mod = @import("../../units.zig");
const metric = @import("../../metric/metric.zig");
const coco = @import("../../metric/coco.zig");
const precompute = @import("../../metric/precompute.zig");
const quad = @import("../../math/quad.zig");
const ct = @import("../../sim/ct.zig");
const sim_mod = @import("../../sim.zig");
const params_mod = @import("../../params.zig");

const Grid = grid_mod.Grid;
const Geometry = geometry.Geometry;

// ---------------------------------------------------------------------------
// problem constants (PROBLEMS/PUFFY/define.h)

/// MASS (define.h) in solar masses. A module global like C's `#define MASS`,
/// but settable once at startup so a preset (e.g. Sagittarius A*, ~4.3e6) can
/// retarget the torus thermo/opacity/radiation-floor unit scale. Tests and
/// every golden leave this at 10 — only `PROBLEMS/puffy/main.zig` writes it,
/// from the params file, before `initAll`. It feeds `consts()` and
/// `atmConsts()` (init); the stepping opacity mass is set in parallel via
/// `radforce.Params.puffyMass`. The torus geometry (`torusConsts`) is
/// dimensionless in GM units and does not depend on mass.
pub var mass: f64 = 10.0; // MASS
pub const gam: f64 = 5.0 / 3.0; // GAMMA
/// Metric parameters (BHSPIN, MKSR0, MKSH0). Like `mass`, a module global set
/// once at startup: `PROBLEMS/puffy/main.zig` may overwrite `mp.a` from the
/// params file's `bhspin` for a spinning preset. Tests and every golden leave
/// `a = 0` (Schwarzschild, matching C PUFFY's BHSPIN = 0), so they are
/// bit-identical. The full torus/dynamo/metric chain is already general in `a`
/// (limotorus `computeGd`/`lK`, `rHorizonBL`/`rIscoBL`, MKS2 Kerr–Schild).
/// MKSR0/MKSH0 do not depend on spin, but RMIN does: it is recomputed from `a`
/// (see `rminForSpin`) so the inner boundary stays inside the shrinking Kerr
/// horizon and the plain-copy XBCLO excision remains causally clean.
pub var mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
/// RMIN — the active inner radial boundary read by `makeGridNz`. A module
/// global (like `mass`/`mp`): the driver overwrites it once at startup from the
/// spin via `rminForSpin` (or an explicit params override) so the excision
/// tracks the Kerr horizon. Tests/goldens leave it at the fiducial 1.85.
pub var rmin: f64 = 1.85; // RMIN

/// The fiducial (Schwarzschild) RMIN — PUFFY's define.h value, immutable. Used
/// as the reference depth for `rminForSpin`; also the value tests/goldens use.
pub const rmin_ref: f64 = 1.85;

/// Inner boundary placed the same fraction inside the (Kerr) horizon as the
/// fiducial a = 0 setup: RMIN(a) = rmin_ref · r_h(a)/r_h(0), i.e. 0.925·r_h(a)
/// (r_h(0) = 2). This is the standard KORAL idiom — other problems define
/// `RMIN = 0.7…0.825·RH` — and it is bit-exactly rmin_ref at a = 0 (·2/2 is
/// exact), so a Schwarzschild run is unchanged. For a = 0.9375, r_h = 1.348 →
/// RMIN ≈ 1.247 (~3.5 cells inside the horizon at nx = 256), restoring the
/// clean, causally-disconnected plain-copy excision at high spin.
pub fn rminForSpin(a: f64) f64 {
    return rmin_ref * metric.rHorizonBL(a) / metric.rHorizonBL(0.0);
}

/// RMAX — the outer radial boundary read by `makeGridNz`. A module global (like
/// `rmin`): the driver may overwrite it from a params `rmax > 0` override, e.g.
/// to extend the domain for a long outflow/wind run. Default 500 M snugly
/// contains the initial torus (midplane edge ≈ 499 M at a = 0; the torus
/// shrinks inward with spin), which is the C-validated PUFFY value. Unlike
/// `rmin` it is NOT derived from mass/spin — the torus extent is dimensionless
/// in GM units and only weakly (inward) spin-dependent, so 500 always contains
/// it; enlarging RMAX is a modeling choice, not an auto-tracked scale.
pub var rmax: f64 = 500.0; // RMAX

// ---------------------------------------------------------------------------
// AGN-preset-overridable physics state.
//
// These module globals default to the VALIDATED koral_lite PUFFY values
// (10 M☉ torus, PROBLEM 147) so tests and every golden — which never write
// them — are bit-for-bit unchanged. `PROBLEMS/puffy/main.zig` overwrites them
// once at startup from the params file (like `mass`/`mp`/`rmin`/`rmax` above),
// which is how the `puffy_agn.toml` preset retargets the run to the
// koral_lite_puffy AGN configuration without disturbing the const `.puffy`
// defaults the tests pin against. See docs/PUFFY_AGN_DIVERGENCES.md for what
// this can and cannot match.
pub var rhoatmmin: f64 = 1.0e-24; // RHOATMMIN
pub var maxbeta: f64 = 1.0 / 20.0; // MAXBETA (after #undef, BETANORMFULL)
/// Fractional init perturbation of the torus internal energy (see
/// params.perturb). 0 = off — the validated init, bit-identical for every
/// golden. Set from the params file via applyPhysicsOverrides.
pub var perturb: f64 = 0.0;
pub var lt_kappa: f64 = 6.0e1; // LT_KAPPA

/// TGASATMMIN — gas temperature the atmosphere floor UINTATMMIN is built at.
pub var atm_tgas: f64 = 1.0e10;
/// ATMTRADINIT — radiation temperature the atmosphere floor ERADATMMIN uses.
pub var atm_trad_init: f64 = 3.0e5;
/// ERADATMMIN form selector. `null` = the validated Brandon collapsing-sim
/// form `LTE(atm_trad_init)/10·6.62/MASS`; when set (AGN preset), the
/// koral_lite_puffy form `atm_erad_factor·LTE(atm_trad_init)`.
pub var atm_erad_factor: ?f64 = null;

/// Floor/ceiling set (C: choices.h + PROBLEMS/PUFFY/define.h). Default is the
/// validated `.puffy`; the AGN preset widens it via main.zig.
pub var floor_params: invert.FloorParams = invert.FloorParams.puffy;
/// Radiation caps/floors (C: define.h EE*/GAMMAMAXRAD). Used by both the init
/// `pradFf2Lab` and the stepping u2p_rad, so it lives here (one source).
pub var rad_params: invert_rad.RadParams = invert_rad.RadParams.puffy;
/// Implicit rad–gas solver knobs (C: define.h RADIMP*).
pub var impl_params: implicit.ImplicitParams = implicit.ImplicitParams.puffy;
/// Gas composition (C: MU_* or HFRAC/HEFRAC/MFRAC). Feeds `consts()` and the
/// stepping opacity params — one source keeps them consistent.
pub var composition: thermo.Composition = thermo.Composition.puffy;
/// Opacity channels (C: BREMSSTRAHLUNG/SYNCHROTRON/KLEINNISHINA/COMPTONIZATION
/// + USE_SYNCHROTRON_BRIDGE_FUNCTIONS + the MESA table pointer).
pub var channels: opacities.Channels = opacities.Channels.puffy;
/// Electron-scattering opacity on/off. Default on = the validated PUFFY
/// Klein–Nishina hook; the AGN preset turns it OFF (koral_lite_puffy leaves
/// PR_KAPPAES undefined, so calc_kappaes ≡ 0 — no scattering, and the Compton
/// four-force term, which is ∝ κ_es, also vanishes). Consumed by `options()`.
pub var scattering: bool = true;
/// MESA Rosseland opacity table (C: MESA_KAPPA). null = off. Owned by the
/// driver (`main.zig` loads it and points `channels.mesa` at it).
pub var mesa_table: ?mesa.MesaTable = null;
/// koral_lite_puffy fluid-frame floors inside the horizon (u2p.c, 2026-08-11).
/// Consumed by `simOptions()`, which turns it into floors.horizon_x1 — the
/// spin (mp.a) is only final there.
pub var fluid_floor_inside_horizon: bool = false;
/// Radiative-viscosity coefficients (C: ALPHARADVISC / MAXRADVISCVEL) and the
/// dynamo's assumed H/R (C: EXPECTEDHR) — defaults are the validated build's;
/// overridable from the params file (koral_lite_puffy runs 0.3 / 0.7).
pub var alpharadvisc: f64 = 0.1;
pub var maxradviscvel: f64 = 0.1;
pub var expectedhr: f64 = 0.3;

pub const lt = struct {
    pub const xi: f64 = 0.995; // LT_XI
    pub const r1: f64 = 20.0; // LT_R1
    pub const r2: f64 = 350.0; // LT_R2
    pub const gamma: f64 = 4.0 / 3.0; // LT_GAMMA
    pub const rin: f64 = 35.0; // LT_RIN
};

/// The production grid (define.h: TNX=384, TNY=360, TNZ=1, NG=3); nx/ny can
/// be reduced for cheaper tests — the coordinate extents stay PUFFY's. The
/// 2D axisymmetric slice (nz=1).
pub fn makeGrid(nx: usize, ny: usize) Grid {
    return makeGridNz(nx, ny, 1);
}

/// Grid with an explicit azimuthal resolution (define.h TNZ). nz=1 is the 2D
/// axisymmetric slice and reproduces `makeGrid` byte-for-byte; nz>1 subdivides
/// the fixed PHIWEDGE=π/2 wedge with periodic z (φ) boundaries. Only the
/// resolution is tunable — the extents (RMIN/RMAX, MINY/MAXY, ±PHIWEDGE/2)
/// stay PUFFY's problem constants, exactly as nx/ny do.
pub fn makeGridNz(nx: usize, ny: usize, nz: usize) Grid {
    return Grid.init(.{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .ng = 3,
        .minx = @log(rmin - mp.mksr0), // MINX = log(RMIN-MKSR0)
        .maxx = @log(rmax - mp.mksr0),
        .miny = 0.001, // MINY
        .maxy = 1.0 - 0.001, // MAXY
        .minz = -std.math.pi / 4.0, // MINZ = -PHIWEDGE/2, PHIWEDGE = π/2
        .maxz = std.math.pi / 4.0,
    });
}

/// Apply the optional params-file physics overrides onto this module's
/// overridable state. Every override is `null` (→ keep the validated `.puffy`
/// value) unless the file sets it, so a plain `puffy.toml` run is unchanged and
/// the `puffy_agn.toml` preset retargets to the koral_lite_puffy configuration.
/// Call once at startup BEFORE `simOptions()`/`makeGridNz`/`initAll` so the
/// whole chain (grid extents, torus, opacities, floors, solver) agrees. Lives
/// here (not in the driver) so tools that replay checkpoints — kdmp2silo —
/// reconstruct the identical configuration from the same params file. Note:
/// several koral_lite_puffy settings are NOT ports and cannot be matched here —
/// see docs/PUFFY_AGN_DIVERGENCES.md.
pub fn applyPhysicsOverrides(p: *const params_mod.Params) void {
    // MKS2 coordinate shape
    if (p.mksr0) |v| mp.mksr0 = v;
    if (p.mksh0) |v| mp.mksh0 = v;
    // torus entropy constant / β-norm target / atmosphere floors
    if (p.lt_kappa) |v| lt_kappa = v;
    if (p.maxbeta) |v| maxbeta = v;
    if (p.perturb) |v| perturb = v;
    if (p.rhoatmmin) |v| rhoatmmin = v;
    if (p.atm_tgas) |v| atm_tgas = v;
    if (p.atm_trad_init) |v| atm_trad_init = v;
    if (p.atm_erad_factor) |v| atm_erad_factor = v;
    // gas composition: giving hfrac switches to the formula-based μ's (no MU_*
    // override), matching koral_lite_puffy's HFRAC/HEFRAC/MFRAC path.
    if (p.hfrac) |h| composition = .{ .hfrac = h, .hefrac = p.hefrac orelse 0.0 };
    // opacity channels
    if (p.bremsstrahlung) |b| channels.bremsstrahlung = b;
    if (p.kleinnishina) |b| channels.kleinnishina = b;
    if (p.synchrotron_bridge) |b| channels.synchrotron_bridge = b;
    if (p.scattering) |s| scattering = s;
    // magnetic floor frame (C: B2RHOFLOORFRAME)
    if (p.zamo_floor_frame) |z| floor_params.b2rhofloorframe = if (z) .zamoframe else .driftframe;
    // koral_lite_puffy drift-floor policy (2026-08-11): isentropic scaling,
    // b²/u trigger off, fluid-frame floors inside the horizon. The horizon x1
    // itself is computed in simOptions() — mp.a is only final there.
    if (p.isentropic_b2rhofloors) |b| floor_params.isentropic_b2rhofloors = b;
    if (p.b2uufloor) |b| floor_params.b2uufloor = b;
    if (p.fluid_floor_inside_horizon) |b| fluid_floor_inside_horizon = b;
    // implicit opacity-damping ladder (C: OPDAMPINIMPLICIT / OPDAMPMAXLEVELS / OPDAMPFACTOR)
    if (p.opdamp_maxlevels) |n| impl_params.opdamp_maxlevels = n;
    if (p.opdamp_factor) |v| impl_params.opdamp_factor = v;
    // koral_lite_puffy implicit-solver options (2026-08-11)
    if (p.radimp_lag_opac) |b| impl_params.lag_opac = b;
    if (p.scale_jacobian) |b| impl_params.scale_jacobian = b;
    if (p.radimp_max_en_change_down) |v| impl_params.max_en_change_down = v;
    if (p.radimp_max_en_change_up) |v| impl_params.max_en_change_up = v;
    if (p.radimp_max_damping) |v| impl_params.max_damping = v;
    // radiative viscosity / dynamo coefficients
    if (p.alpharadvisc) |v| alpharadvisc = v;
    if (p.maxradviscvel) |v| maxradviscvel = v;
    if (p.expectedhr) |v| expectedhr = v;
    // rmhd floors / ceilings
    if (p.rhofloor) |v| floor_params.rhofloor = v;
    if (p.uurhoratiomin) |v| floor_params.uurhoratiomin = v;
    if (p.uurhoratiomax) |v| floor_params.uurhoratiomax = v;
    if (p.b2rhoratiomax) |v| floor_params.b2rhoratiomax = v;
    if (p.b2uuratiomax) |v| floor_params.b2uuratiomax = v;
    if (p.gammamaxhd) |v| floor_params.gammamaxhd = v;
    // radiation caps / floors
    if (p.gammamaxrad) |v| rad_params.gammamaxrad = v;
    if (p.eerhoratiomin) |v| rad_params.eerhoratiomin = v;
    if (p.eerhoratiomax) |v| rad_params.eerhoratiomax = v;
    if (p.eeuuratiomin) |v| rad_params.eeuuratiomin = v;
    if (p.eeuuratiomax) |v| rad_params.eeuuratiomax = v;
    // implicit rad–gas solver
    if (p.radimpeps) |v| impl_params.eps = v;
    if (p.radimpmaxiter) |v| impl_params.maxiter = v;
}

/// The Sim.Options for a PUFFY run of `SimT` under runtime params `p`
/// (PROBLEMS/PUFFY/define.h choices; the comm/decomp fields are the
/// caller's to fill). Shared by the driver and kdmp2silo so a replayed
/// checkpoint sees the exact configuration the run had.
pub fn simOptions(comptime SimT: type, p: *const params_mod.Params) SimT.Options {
    // The physics param sets come from this module's overridable state
    // (defaults = the validated `.puffy` constants; `applyPhysicsOverrides`
    // may retarget them from the params file — the puffy_agn.toml preset).
    // `channels` already carries the synchrotron-bridge flag and the MESA
    // table pointer (set by applyPhysicsOverrides / the driver).
    var opac = radforce.Params.puffyMassChan(p.mass, composition, channels);
    // koral_lite_puffy leaves PR_KAPPAES undefined ⇒ calc_kappaes ≡ 0. The AGN
    // preset turns scattering off (scattering=false), zeroing both the
    // scattering opacity and the Compton four-force term (∝ κ_es).
    if (!scattering) opac.kappaes = .none;
    var floors = floor_params;
    // koral_lite_puffy horizon fluid-frame floors: precompute the MKS2 x1 of
    // the BL horizon (x1 = ln(r − r0) is monotone in r), so checkFloorsMhd
    // compares coordinate-locally. Left at −inf (dead branch) when off.
    if (fluid_floor_inside_horizon)
        floors.horizon_x1 = @log(metric.rHorizonBL(mp.a) - mp.mksr0);
    return .{
        .coords = .mks2,
        .mp = mp,
        .gam = gam,
        .tsteplim = p.tsteplim,
        .floors = floors,
        .rad = rad_params,
        .opac = opac,
        .implicit = impl_params,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .radviscosity = true,
        .radvisc = .{ .alpha = alpharadvisc, .maxvel = maxradviscvel },
        .dynamo = true,
        .dynamo_params = .{ .expectedhr = expectedhr },
        // C: DORADIMPFIXUPS / REDUCEORDERATBH / DAMPRADWAVESPEEDNEARAXIS — off
        // in the validated build, retargeted by the AGN preset (null = off).
        .do_radimp_fixups = p.doradimpfixups orelse false,
        .reduceorderatbh = p.reduceorderatbh orelse false,
        .reduceorderafterfixup = p.reduceorderafterfixup orelse false,
        .dampradwavespeednearaxis = p.dampradwavespeednearaxis orelse 0,
        .bc_x = .specific,
        .bc_y = .specific,
        // C: PERIODIC_ZBC (PUFFY define.h:188). Irrelevant in 2D (nz==1 has
        // no z-ghosts); for 3D wedges this was previously left at the .copy
        // default — a latent pre-MPI bug (MPI plan §11.1-3), fixed in P4a.
        .bc_z = .periodic,
        .specific_bc = &Bc(SimT).calc,
        .nthreads = p.nthreads,
        .pin_threads = p.pin_threads,
    };
}

pub fn consts() thermo.Consts {
    return thermo.Consts.init(units_mod.Units.init(mass), composition);
}

/// UINTATMMIN / ERADATMMIN (define.h:221/225), computed once.
pub const Atm = struct { uintatmmin: f64, eradatmmin: f64 };

pub fn atmConsts(con: *const thermo.Consts) Atm {
    return .{
        .uintatmmin = thermo.uFromTrho(con, atm_tgas, rhoatmmin, gam),
        // Default (validated): calc_LTE_EfromT(3.e5)/10*6.62/MASS — Brandon's
        // collapsing-sim value. With `atm_erad_factor` set (AGN preset), the
        // koral_lite_puffy form  ERADATMMIN = factor·calc_LTE_EfromT(ATMTRADINIT).
        .eradatmmin = if (atm_erad_factor) |f|
            f * con.lteEfromT(atm_trad_init)
        else
            con.lteEfromT(atm_trad_init) / 10.0 * 6.62 / mass,
    };
}

// ---------------------------------------------------------------------------
// limotorus (PROBLEMS/PUFFY/tools.c)

const xacc: f64 = 5.0 * std.math.floatEps(f64); // C: 5.*DBL_EPSILON
const pi_2: f64 = std.math.pi / 2.0; // C: M_PI_2

pub const Gd = struct { gdtt: f64, gdtp: f64, gdpp: f64 };

/// tools.c:1 compute_gd — BL g_tt, g_tφ, g_φφ (limotorus4.nb expressions).
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

/// tools.c:14 lK — Keplerian equatorial ℓ = u_φ/u_t.
pub fn lK(r: f64, a: f64) f64 {
    const curly_f = 1.0 - 2.0 * a / std.math.pow(f64, r, 1.5) + (a / r) * (a / r);
    const curly_g = 1.0 - 2.0 / r + a / std.math.pow(f64, r, 1.5);
    return @sqrt(r) * curly_f / curly_g;
}

/// tools.c:22 l3d — ξ·lK clamped to the broken-power-law window.
pub fn l3d(lam: f64, a: f64, lambreak1: f64, lambreak2: f64, xi: f64) f64 {
    const arg = if (lam <= lambreak1) lambreak1 else if (lam >= lambreak2) lambreak2 else lam;
    return xi * lK(arg, a);
}

/// tools.c:26 rtbis (HARM's nrutil.c bisection): 100 halvings max, exits at
/// |dx| < xacc. On an unbracketed root C prints a warning and marches on;
/// we do the same (silently).
pub fn rtbis(func: anytype, x1: f64, x2: f64, xacc_: f64) f64 {
    const f = func.eval(x1);
    var fmid = func.eval(x2);
    var dx: f64 = undefined;
    var rtb: f64 = undefined;
    if (f < 0.0) {
        dx = x2 - x1;
        rtb = x1;
    } else {
        dx = x1 - x2;
        rtb = x2;
    }
    var j: usize = 1;
    while (j <= 100) : (j += 1) {
        dx *= 0.5;
        const xmid = rtb + dx;
        fmid = func.eval(xmid);
        if (fmid <= 0.0) rtb = xmid;
        if (@abs(dx) < xacc_ or fmid == 0.0) return rtb;
    }
    return 0.0; // C: "Too many bisections in rtbis"
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

/// tools.c:68 lamBL — von Zeipel cylinder radius, bracket (R, 10R).
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

/// tools.c:97 rmidlam — the passed-in gd values are dead in C (immediately
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

/// tools.c:118 limotorus_findrml — bracket (6, 10·λ²) as written in C.
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
/// cell; the values are identical.
pub const TorusConsts = struct {
    lambreak1: f64,
    lambreak2: f64,
    lamin: f64,
    lin: f64,
    omin: f64,
    agravin: f64,
};

pub fn torusConsts(a: f64) TorusConsts {
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
pub fn initDsandvels(r: f64, th: f64, a: f64, tc: *const TorusConsts) DsVels {
    const R = r * @sin(th);
    if (R < lt.rin) return .{ .rho = -1.0, .uu = 0.0, .ell = 0.0 };

    const kappa = lt_kappa;
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

// ---------------------------------------------------------------------------
// atmospheres (relele.c:518 / :718, atmtype 0 — normal observer)

pub fn setHdAtmosphere(
    comptime cfg: config.Config,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    atm: *const Atm,
) void {
    const L = layout.VarLayout(cfg);
    // normal observer in VELR ≡ VELPRIM — no conversion
    const ucon = relele.normalObsVelr(geom);
    pp[L.index(.vx)] = ucon[1];
    pp[L.index(.vy)] = ucon[2];
    pp[L.index(.vz)] = ucon[3];

    // Bondi-like profile, normalized at r_BL = 2
    const xx2 = coco.cocoN(geom.xxvec, geom.coords, .bl, mp);
    const r = xx2[1];
    const rout: f64 = 2.0;
    pp[L.index(.rho)] = rhoatmmin * std.math.pow(f64, r / rout, -1.5);
    pp[L.index(.uu)] = atm.uintatmmin * std.math.pow(f64, r / rout, -2.5);

    if (comptime L.hasVar(.b1)) {
        pp[L.index(.b1)] = 0.0;
        pp[L.index(.b2)] = 0.0;
        pp[L.index(.b3)] = 0.0;
    }
}

pub fn setRadAtmosphere(
    comptime cfg: config.Config,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    atm: *const Atm,
) void {
    const L = layout.VarLayout(cfg);
    pp[L.index(.ee)] = atm.eradatmmin;
    const ucon = relele.normalObsVelr(geom); // VELR ≡ VELPRIMRAD
    pp[L.index(.fx)] = ucon[1];
    pp[L.index(.fy)] = ucon[2];
    pp[L.index(.fz)] = ucon[3];
}

// ---------------------------------------------------------------------------
// prepinit.c — per-cell initial primitives (vector potential in the B slots)

/// BL-coords geometry at the cell center (C: fill_geometry_arb KERRCOORDS),
/// supplying this problem's fixed metric params. The reduction lives once in
/// precompute.geometryBLat.
pub fn fillGeometryBL(g: *const Grid, coords: config.Coords, ix: i64, iy: i64, iz: i64) Geometry {
    return precompute.geometryBLat(g, coords, mp, ix, iy, iz);
}

/// prepinit.c:92-93 — the T > 0 root of P = bbb·T + aaa·T⁴ (Mathematica
/// closed form, transcribed with its literal decimal exponents).
pub fn tFromPtot(P: f64, aaa: f64, bbb: f64) f64 {
    const third = 0.3333333333333333;
    const twothird = 0.6666666666666666;
    const naw1 = std.math.cbrt(9.0 * aaa * (bbb * bbb) -
        @sqrt(3.0) * @sqrt(27.0 * (aaa * aaa) * std.math.pow(f64, bbb, 4.0) +
            256.0 * std.math.pow(f64, aaa, 3.0) * std.math.pow(f64, P, 3.0)));
    const c23 = std.math.pow(f64, twothird, third);
    const c2 = std.math.pow(f64, 2.0, third);
    const c3 = std.math.pow(f64, 3.0, twothird);
    return -@sqrt((-4.0 * c23 * P) / naw1 + naw1 / (c2 * c3 * aaa)) / 2.0 +
        @sqrt((4.0 * c23 * P) / naw1 - naw1 / (c2 * c3 * aaa) +
            (2.0 * bbb) / (aaa * @sqrt((-4.0 * c23 * P) / naw1 + naw1 / (c2 * c3 * aaa)))) / 2.0;
}

/// splitmix64 finalizer — self-contained (no std.hash dependency, so the
/// noise field is reproducible across Zig versions).
fn mix64(z0: u64) u64 {
    var z = z0 +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Deterministic per-cell noise ξ ∈ [−1, 1) for the init perturbation,
/// hashed from the cell-center INTERNAL coordinates. The grid contract
/// (MPI plan gate 1) makes those coordinates bit-identical for the same
/// physical cell on every rank and thread count, so the noise field is
/// decomposition-invariant by construction.
pub fn perturbXi(x: [4]f64) f64 {
    const b1: u64 = @bitCast(x[1]);
    const b2: u64 = @bitCast(x[2]);
    const b3: u64 = @bitCast(x[3]);
    const u = mix64(mix64(mix64(b1) ^ b2) ^ b3);
    return 2.0 * (@as(f64, @floatFromInt(u >> 11)) * 0x1.0p-53) - 1.0;
}

pub fn prepInitCell(
    comptime cfg: config.Config,
    geom: *const Geometry,
    geomBL: *const Geometry,
    tc: *const TorusConsts,
    con: *const thermo.Consts,
    atm: *const Atm,
) relele.Error![layout.VarLayout(cfg).count]f64 {
    const L = layout.VarLayout(cfg);
    const has_rad = comptime cfg.has(.radiation);
    const has_b = comptime L.hasVar(.b1);

    const r = geomBL.xxvec[1];
    const th = geomBL.xxvec[2];

    const dv = initDsandvels(r, th, mp.a, tc);
    const rho = dv.rho;
    var uint = dv.uu;
    var ell = dv.ell;

    var pp = [_]f64{0} ** L.count;

    if (rho < 0.0) { // outside the donut
        setHdAtmosphere(cfg, &pp, geom, atm);
        if (has_rad) setRadAtmosphere(cfg, &pp, geom, atm);
        pp[L.index(.entr)] = -1.0; // marker only; init overwrites
        return pp;
    }

    // inside the donut
    pp[L.index(.entr)] = 1.0;

    var ppback = [_]f64{0} ** L.count;
    setHdAtmosphere(cfg, &ppback, geom, atm);
    if (has_rad) setRadAtmosphere(cfg, &ppback, geom, atm);

    uint = lt_kappa * std.math.pow(f64, rho, lt.gamma) / (lt.gamma - 1.0);
    // optional MRI-seeding noise, torus interior only — applied BEFORE the
    // gas/radiation pressure split so the split stays LTE-consistent
    if (perturb != 0.0) uint *= 1.0 + perturb * perturbXi(geom.xxvec);
    ell *= -1.0;

    const GGBL = &geomBL.GG;
    const ulph = @sqrt(-1.0 / (GGBL[0][0] / ell / ell + 2.0 / ell * GGBL[0][3] + GGBL[3][3]));
    const ult = ulph / ell;
    const ucov = [4]f64{ ult, 0.0, 0.0, ulph };
    var ucon = relele.raiseVec(ucov, geomBL);
    ucon = try relele.convert(ucon, .vel4, .velr, geomBL, .trust_ut);

    pp[L.index(.rho)] = @max(rho, ppback[L.index(.rho)]);
    pp[L.index(.uu)] = @max(uint, ppback[L.index(.uu)]);
    pp[L.index(.vx)] = ucon[1];
    pp[L.index(.vy)] = ucon[2];
    pp[L.index(.vz)] = ucon[3];
    // B slots zeroed so the coordinate transformation below stays sane

    if (has_rad) {
        // distribute P = GAMMAM1·uint between gas and radiation by solving
        // P = bbb·T + aaa·T⁴ (prepinit.c:86-97, Mathematica closed form)
        const P = (gam - 1.0) * uint;
        const aaa = con.four_sigmarad / 3.0; // C: 4.*SIGMA_RAD/3.
        const bbb = con.kb_over_mugas_mp * rho; // C: K_BOLTZ*rho/MU_GAS/M_PROTON
        const t4 = tFromPtot(P, aaa, bbb);

        const E = con.lteEfromT(t4);
        uint = thermo.uFromTrho(con, t4, rho, gam);

        pp[L.index(.uu)] = @max(uint, ppback[L.index(.uu)]);
        pp[L.index(.ee)] = @max(E, ppback[L.index(.ee)]);
        pp[L.index(.fx)] = 0.0;
        pp[L.index(.fy)] = 0.0;
        pp[L.index(.fz)] = 0.0;

        // BL fluid-frame radiative primitives → BL lab
        pp = try radiation.pradFf2Lab(cfg, pp, geomBL, rad_params);
    }

    // BL → MYCOORDS
    pp = try frames.transPallCoco(cfg, pp, geomBL, geom, mp);

    if (has_b) {
        // MYCOORDS vector potential (only A_φ), stored in the B slots
        const base = pp[L.index(.rho)] * geomBL.xxvec[1] / 4.0e-22;
        var acov3 = @max(base * base - 0.02, 0.0) * @sqrt(1.0e-23);
        // QUADLOOPS
        acov3 *= @sin((std.math.pi / 2.0 - geomBL.xxvec[2]) / 0.1);
        pp[L.index(.b1)] = 0.0;
        pp[L.index(.b2)] = 0.0;
        pp[L.index(.b3)] = acov3;
    }

    return pp;
}

// ---------------------------------------------------------------------------
// the full init sequence (ko.c:140-263 with ifinit == 1)

/// prepinit + init over the domain (leaves the vector potential A_φ in the
/// B slots, ENTR = calc_Sfromu, p2u). Does NOT fill ghosts — the caller
/// runs set_bc next, matching ko.c's set_initial_profile → set_bc.
/// Band-parallel over iy: every cell writes only its own p/u slots and reads
/// only the (already-built, const) metric cache, so the initialized state is
/// bit-identical at any thread count. This is the dominant startup cost —
/// the limotorus solve per cell is heavy in pow/log/exp — and left serial it
/// ignored `nthreads` entirely.
pub fn prepInitDomain(comptime SimT: type, sim: *SimT) !void {
    const W = struct {
        fn rows(s: *SimT, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            prepInitRows(SimT, s, iy0, iy1) catch |e| {
                res.err = e;
            };
        }
    };
    const res = threading.parallelRange(SimT, sim, sim.team, 0, @intCast(sim.grid.ny), W.rows);
    if (res.err) |e| return e;
}

fn prepInitRows(comptime SimT: type, sim: *SimT, iy0: i64, iy1: i64) !void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;

    const con = consts();
    const atm = atmConsts(&con);
    const tc = torusConsts(mp.a);

    const nx: i64 = @intCast(sim.grid.nx);
    const nz: i64 = @intCast(sim.grid.nz);

    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                const geomBL = fillGeometryBL(&sim.grid, cfg.coords, ix, iy, iz);
                var pp = try prepInitCell(cfg, &geom, &geomBL, &tc, &con, &atm);
                pp[L.index(.entr)] = hydro.sFromU(pp[L.index(.rho)], pp[L.index(.uu)], sim.opt.gam);
                const uu = try p2u_mod.p2u(cfg, pp, &geom, sim.opt.gam);
                sim.p.store(ix, iy, iz, &pp);
                sim.u.store(ix, iy, iz, &uu);
            }
        }
    }
}

/// The full ko.c:140-263 init sequence: prepinit + init → set_bc →
/// calc_BfromA → set_bc → postinit. Returns the β-normalization factor
/// fac (postinit.c:78).
pub fn initAll(comptime SimT: type, sim: *SimT) !f64 {
    try prepInitDomain(SimT, sim);
    // MPI: each setBc needs exchanged z-ghosts first (its interior-face path
    // is p2u-of-exchanged-p; before the first exchange the ghost planes are
    // still the Field.init zeros). No-ops serially.
    sim.exchangeHalos();
    try sim.setBc(0.0, true);
    try ct.calcBfromA(SimT, sim, true);
    sim.exchangeHalos(); // domain B changed — refresh ghosts before the re-fill
    try sim.setBc(0.0, true);
    return try postinit(SimT, sim);
}

/// postinit.c — global β normalization. Returns fac; ghost cells keep their
/// unscaled B (C does not refresh BCs after postinit).
pub fn postinit(comptime SimT: type, sim: *SimT) !f64 {
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return 1.0;

    const ny: i64 = @intCast(sim.grid.ny);

    // Pass 1 — the BETANORMFULL max, band-parallel over iy. `tsd_max` is
    // ChunkResult's max-merged slot, so combining bands is order-insensitive
    // and therefore bit-identical to the serial scan; clamping at 0 below
    // reproduces the serial `maxb = 0.0` starting value exactly (the slot's
    // neutral element is -inf).
    const MaxW = struct {
        fn rows(s: *SimT, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            betaMaxRows(SimT, s, iy0, iy1, res) catch |e| {
                res.err = e;
            };
        }
    };
    const mres = threading.parallelRange(SimT, sim, sim.team, 0, ny, MaxW.rows);
    if (mres.err) |e| return e;
    var maxb: f64 = @max(0.0, mres.tsd_max);

    // BETANORMFULL is a GLOBAL max (MPI plan §1.1-5): without the fold each
    // rank normalizes B differently and the field is discontinuous from
    // step 0. Identity serially; max is exact, so the folded value is
    // bit-identical to a serial run's.
    maxb = sim.globalMax(maxb);
    const fac = @sqrt(maxbeta / maxb);

    // Pass 2 — apply the factor. Per-cell writes only; band-parallel.
    const ScaleCtx = struct { sim: *SimT, fac: f64 };
    var sctx = ScaleCtx{ .sim = sim, .fac = fac };
    const ScaleW = struct {
        fn rows(c: *ScaleCtx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            betaScaleRows(SimT, c.sim, c.fac, iy0, iy1) catch |e| {
                res.err = e;
            };
        }
    };
    const sres = threading.parallelRange(ScaleCtx, &sctx, sim.team, 0, ny, ScaleW.rows);
    if (sres.err) |e| return e;

    return fac;
}

/// postinit pass 1 body for iy ∈ [iy0, iy1): the per-cell β = p_mag/p_tot,
/// accumulated into the chunk's max-merged slot.
fn betaMaxRows(comptime SimT: type, sim: *SimT, iy0: i64, iy1: i64, res: *threading.ChunkResult) !void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const nx: i64 = @intCast(sim.grid.nx);
    const nz: i64 = @intCast(sim.grid.nz);

    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                var pp: [SimT.nv]f64 = undefined;
                sim.p.load(ix, iy, iz, &pp);
                const geom = sim.cache.fillGeometry(ix, iy, iz);

                const ug = try relele.uconUcovFromPrims(
                    .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
                    &geom,
                );
                const bb = mhd.bconBcovBsqFrom4vel(
                    .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
                    ug.con,
                    ug.cov,
                    &geom,
                );
                const pmag = bb.bsq / 2.0;

                const pgas = (sim.opt.gam - 1.0) * pp[L.index(.uu)];
                var ptot = pgas;
                if (comptime cfg.has(.radiation)) {
                    const rt = try radiation.calcFfRtt(cfg, pp, &geom);
                    const ehat = -rt.rtt;
                    ptot += ehat / 3.0;
                }

                // BETANORMFULL: max over the whole domain
                if (pmag / ptot > res.tsd_max) res.tsd_max = pmag / ptot;
            }
        }
    }
}

/// postinit pass 2 body for iy ∈ [iy0, iy1): scale B by `fac` and refresh
/// the conserveds.
fn betaScaleRows(comptime SimT: type, sim: *SimT, fac: f64, iy0: i64, iy1: i64) !void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const nx: i64 = @intCast(sim.grid.nx);
    const nz: i64 = @intCast(sim.grid.nz);

    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                var pp: [SimT.nv]f64 = undefined;
                sim.p.load(ix, iy, iz, &pp);
                pp[L.index(.b1)] *= fac;
                pp[L.index(.b2)] *= fac;
                pp[L.index(.b3)] *= fac;
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                const uu = try p2u_mod.p2u(cfg, pp, &geom, sim.opt.gam);
                sim.p.store(ix, iy, iz, &pp);
                sim.u.store(ix, iy, iz, &uu);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bc.c — problem-specific boundary conditions

// ---------------------------------------------------------------------------
// seed MRI-quality report (campaign notes 2026-08-08)

/// Mass-weighted seed MRI-quality sums over this rank's slab. The caller
/// folds them across ranks (globalSum) and divides — every field is
/// sum-reducible on purpose. Same disk mask as tools/qmri.zig (ρ > 10³×
/// the atmosphere floor profile, r < 100) so the startup numbers are
/// directly comparable to qmri on later dumps. Q_i = 2π|b^i|/(√(ρh+b²)·
/// |Ω|·Δx^i) with radiation-inclusive inertia; Q_φ ≡ 0 for the A_φ-only
/// seed and is omitted. Serial pass over the interior — the geometry is
/// cached, so this costs well under a second even on campaign grids.
/// Assumes MKS2 internal coordinates (r = e^{x1} + mksr0), like the rest
/// of this problem.
pub const SeedQ = struct { mass: f64 = 0, qr_m: f64 = 0, qth_m: f64 = 0 };

pub fn seedQuality(comptime SimT: type, s: *SimT) !SeedQ {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return .{};
    var out = SeedQ{};
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                var pp: [SimT.nv]f64 = undefined;
                s.p.load(ix, iy, iz, &pp);
                const rho = pp[L.index(.rho)];
                if (!(rho > 0)) continue;
                const geom = s.cache.fillGeometry(ix, iy, iz);
                const r = @exp(geom.xxvec[1]) + mp.mksr0;
                if (r > 100.0) continue;
                if (rho < 1.0e3 * rhoatmmin * std.math.pow(f64, r / 2.0, -1.5)) continue;
                const ug = try relele.uconUcovFromPrims(
                    .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
                    &geom,
                );
                const bb = mhd.bconBcovBsqFrom4vel(
                    .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
                    ug.con,
                    ug.cov,
                    &geom,
                );
                const omega = @abs(ug.con[3] / ug.con[0]);
                if (!(omega > 1e-12)) continue;
                var w = rho + gam * pp[L.index(.uu)];
                if (comptime cfg.has(.radiation)) {
                    const rt = try radiation.calcFfRtt(cfg, pp, &geom);
                    const ehat = -rt.rtt;
                    if (ehat > 0 and std.math.isFinite(ehat)) w += (4.0 / 3.0) * ehat;
                }
                const den = @sqrt(w + bb.bsq) * omega;
                const wgt = rho * geom.gdet;
                out.mass += wgt;
                out.qr_m += wgt * 2.0 * std.math.pi * @abs(bb.bcon[1]) / (den * s.grid.dx);
                out.qth_m += wgt * 2.0 * std.math.pi * @abs(bb.bcon[2]) / (den * s.grid.dy);
            }
        }
    }
    return out;
}

pub fn Bc(comptime SimT: type) type {
    return struct {
        const cfg = SimT.Cfg;
        const L = SimT.Layout;
        const NV = SimT.nv;
        const has_rad = cfg.has(.radiation);
        const has_b = L.hasVar(.b1);

        pub fn calc(
            ctx: ?*const anyopaque,
            sim: *const SimT,
            ix: i64,
            iy: i64,
            iz: i64,
            t: f64,
            ifinit: bool,
            face: sim_mod.BcFace,
        ) relele.Error![NV]f64 {
            _ = ctx; // comptime-bound BC — no runtime context
            _ = t;
            _ = ifinit;
            const nx: i64 = @intCast(sim.grid.nx);
            const ny: i64 = @intCast(sim.grid.ny);
            var pp: [NV]f64 = undefined;

            switch (face) {
                .xhi => {
                    // outflow with r-rescaling (bc.c:23-121)
                    sim.p.load(nx - 1, iy, iz, &pp);

                    const geom = sim.cache.fillGeometry(ix, iy, iz);
                    const geomBL = fillGeometryBL(&sim.grid, cfg.coords, ix, iy, iz);

                    // MHD prims to BL (ghost-cell geometries, as in C)
                    pp = try frames.transPmhdCoco(cfg, pp, &geom, &geomBL, mp);

                    const geombdBL = fillGeometryBL(&sim.grid, cfg.coords, nx - 1, iy, iz);
                    const rghost = geomBL.xxvec[1];
                    const rbound = geombdBL.xxvec[1];
                    const scale1 = rbound * rbound / rghost / rghost;
                    const scale2 = rbound / rghost;

                    pp[L.index(.rho)] *= scale1;
                    pp[L.index(.uu)] *= scale1;
                    if (comptime has_b) {
                        pp[L.index(.b1)] *= scale1;
                        pp[L.index(.b2)] *= scale2;
                        pp[L.index(.b3)] *= scale2;
                    }
                    pp[L.index(.vy)] *= scale1;
                    pp[L.index(.vz)] *= scale1;
                    if (comptime has_rad) {
                        // note: rad prims scaled while the MHD block sits in
                        // BL — they never left MYCOORDS (C does the same)
                        pp[L.index(.ee)] *= scale1;
                        pp[L.index(.fy)] *= scale1;
                        pp[L.index(.fz)] *= scale1;
                    }

                    pp = try frames.transPmhdCoco(cfg, pp, &geomBL, &geom, mp);

                    // no-inflow: gas
                    var ucon = [4]f64{ 0.0, pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] };
                    ucon = try relele.convert(ucon, .velr, .vel4, &geom, .recompute_ut);
                    ucon = frames.trans2Coco(geom.xxvec, ucon, cfg.coords, .bl, mp);
                    if (ucon[1] < 0.0) {
                        ucon[1] = 0.0;
                        ucon = frames.trans2Coco(geomBL.xxvec, ucon, .bl, cfg.coords, mp);
                        ucon = try relele.convert(ucon, .vel4, .velr, &geom, .recompute_ut);
                        pp[L.index(.vx)] = ucon[1];
                        pp[L.index(.vy)] = ucon[2];
                        pp[L.index(.vz)] = ucon[3];
                    }
                    if (comptime has_rad) {
                        var urf = [4]f64{ 0.0, pp[L.index(.fx)], pp[L.index(.fy)], pp[L.index(.fz)] };
                        urf = try relele.convert(urf, .velr, .vel4, &geom, .recompute_ut);
                        urf = frames.trans2Coco(geom.xxvec, urf, cfg.coords, .bl, mp);
                        if (urf[1] < 0.0) {
                            urf[1] = 0.0;
                            urf = frames.trans2Coco(geomBL.xxvec, urf, .bl, cfg.coords, mp);
                            urf = try relele.convert(urf, .vel4, .velr, &geom, .recompute_ut);
                            pp[L.index(.fx)] = urf[1];
                            pp[L.index(.fy)] = urf[2];
                            pp[L.index(.fz)] = urf[3];
                        }
                    }
                },
                .xlo => {
                    // outflow near the BH; RMIN=1.85 < r_horizon=2 in PUFFY,
                    // so the C inflow check (bc.c:133) never runs
                    sim.p.load(0, iy, iz, &pp);
                },
                .ylo => {
                    // polar reflection, upper axis
                    const iiy = -iy - 1;
                    sim.p.load(ix, iiy, iz, &pp);
                    pp[L.index(.vy)] = -pp[L.index(.vy)];
                    if (comptime has_b) pp[L.index(.b2)] = -pp[L.index(.b2)];
                    if (comptime has_rad) pp[L.index(.fy)] = -pp[L.index(.fy)];
                },
                .yhi => {
                    // polar reflection, lower axis
                    const iiy = ny - (iy - ny) - 1;
                    sim.p.load(ix, iiy, iz, &pp);
                    pp[L.index(.vy)] = -pp[L.index(.vy)];
                    if (comptime has_b) pp[L.index(.b2)] = -pp[L.index(.b2)];
                    if (comptime has_rad) pp[L.index(.fy)] = -pp[L.index(.fy)];
                },
                // TNZ == 1: no z ghost cells exist
                .zlo, .zhi => unreachable,
            }
            return pp;
        }
    };
}
