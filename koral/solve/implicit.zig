//! The implicit radiation–gas coupling solver (C: rad.c):
//!
//!   rad.c:73    solve_implicit_lab — the rung ladder (energy-LAB →
//!               energy-FF → RAD/MHD swap ×2 → entropy-FF ×2)
//!   rad.c:341   solve_implicit_lab_4dprim — 4-prim Newton with one-sided
//!               FD Jacobian (ε = RADIMPEPS), Jacobian scaling
//!               (SCALE_JACOBIAN, on in PUFFY), damping ladder, per-step
//!               energy/temperature change limiters
//!   rad.c:1279  implicit_apply_constraints — the conservation constraint
//!               tying the non-iterated fluid to the iterated one
//!   rad.c:1460  f_implicit_lab_4dprim_with_state — the residual
//!   rad.c:2001  solve_implicit_lab_1dprim — bisection starting guess
//!               (RADIMP_START_WITH_BISECT, on in PUFFY)
//!
//! C-fidelity notes:
//!  * The 4D ENTROPYEQ residual reads `S = state->Tgas` where Sgas is
//!    obviously intended (rad.c:1749/1767) — the entropy rungs solve
//!    Tgas·(Tgas − Sgas0) = dτ·Ĝ⁰. Transcribed as-is; the 1D bisection's
//!    entropy branch uses Sgas correctly (and is never reached anyway:
//!    the bisect always runs the energy equation).
//!  * The 1D bisection's RAD-branch residual uses pp0[EE]/ratio where
//!    ·ratio would reproduce Ehat0 (rad.c:1929) — transcribed as-is (it
//!    only shapes the starting guess).
//!  * The wrapper picks RAD/MHD by Ehat < RADIMPLICITTHRESHOLD·u (1e-2);
//!    the bisection uses its own hardwired 1e-3.
//!  * inverse_matrix (misc.c) is GSL LU without error checking — a
//!    singular Jacobian silently yields inf/nan which the damping ladder
//!    then rejects. invert4 below mirrors that (no error path).
//!  * xxxbest/errbest bookkeeping in C is dead (never restored); the
//!    `isinf` flag in the residual checks f[0] repeatedly and is unused;
//!    u2pret in the 4D solver is never set after init — all skipped.
//!  * C's my_err abort on NaN mid-bisection is replaced by a clean −1
//!    (the caller treats the bisect result as optional anyway).
//!  * PUFFY switches: RADIMP_START_WITH_BISECT, SCALE_JACOBIAN,
//!    ALLOWRADCEILINGINIMPLICIT, ALLOWFORENTRINF4DPRIM all on;
//!    OPDAMPINIMPLICIT 0, BASICRADIMPLICIT off (full ladder),
//!    DORADIMPFIXUPS 0. PUFFY's "RADIMPLICIMAXENCHANGEUP 10." define is
//!    a typo (code reads RADIMPLICITMAXENCHANGEUP → choices.h default
//!    10., coincidentally equal).
//!
//! SIMD (Zig-only, parallelization plan §2.3 rank 1): with
//! ImplicitParams.simd_jacobian (default on) the FD Jacobian's 4 perturbed
//! residual evaluations run as lanes of one @Vector(4, f64) batch through
//! the comptime-T-generic chain (fillRadStateG → calcGiFromStateG →
//! residualG). The batch is bit-identical to the scalar loop — same
//! per-lane arithmetic, per-lane libm transcendentals, selects of
//! both-sides-computed values — gated in simd_tests.zig, so all C goldens
//! see identical results either way.

const std = @import("std");
const simd = @import("../math/simd.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const relele = @import("../relele.zig");
const hydro = @import("../physics/hydro.zig");
const radiation = @import("../physics/radiation.zig");
const radforce = @import("../physics/radforce.zig");
const thermo = @import("../physics/thermo.zig");
const invert = @import("invert.zig");
const invert_rad = @import("invert_rad.zig");
const p2u_mod = @import("../p2u.zig");
const Geometry = @import("../geometry.zig").Geometry;

const small: f64 = 1.0e-80; // C: SMALL

/// One lane per FD-Jacobian perturbation (the batch is the 4 columns).
const V4 = @Vector(4, f64);

pub const WhichPrim = enum(i32) { rad = 1, mhd = 2 }; // mnemonics.h:7-8
pub const WhichEq = enum(i32) { energy = 0, entropy = 1 };
pub const WhichFrame = enum(i32) { lab = 0, ff = 1 };

/// The RADIMP* build macros (C: choices.h defaults / PROBLEMS overrides).
pub const ImplicitParams = struct {
    conv: f64 = 1.0e-10, // RADIMPCONV
    entrconv: f64 = 1.0e-5, // RADIMPENTRCONV
    eps: f64 = 1.0e-6, // RADIMPEPS
    maxiter: usize = 50, // RADIMPMAXITER
    convrel: f64 = 1.0e-6, // RADIMPCONVREL
    convrelerr: f64 = 1.0e-5, // RADIMPCONVRELERR
    convrelentr: f64 = 1.0e-6, // RADIMPCONVRELENTR
    convrelentrerr: f64 = 1.0e-4, // RADIMPCONVRELENTRERR
    damping_factor: f64 = 3.0, // RADIMPLICITDAMPINGFACTOR
    max_en_change_down: f64 = 10.0, // RADIMPLICITMAXENCHANGEDOWN
    max_en_change_up: f64 = 10.0, // RADIMPLICITMAXENCHANGEUP
    max_damping: f64 = 1.0e-5, // MAXRADIMPDAMPING
    max_tgas_change: f64 = 10.0, // IMPLICITMAXTGASCHANGE
    max_trad_change: f64 = 10.0, // RADIMPLICITMAXTRADCHANGE
    threshold: f64 = 1.0e-2, // RADIMPLICITTHRESHOLD
    start_with_bisect: bool = false, // RADIMP_START_WITH_BISECT
    scale_jacobian: bool = false, // SCALE_JACOBIAN
    allow_rad_ceiling: bool = false, // ALLOWRADCEILINGINIMPLICIT
    allow_entr_in_4dprim: bool = false, // ALLOWFORENTRINF4DPRIM

    /// Zig-only, no C counterpart (parallelization plan §2.3 rank 1):
    /// evaluate the FD Jacobian's 4 perturbed residuals as lanes of one
    /// @Vector(4, f64) batch. Bit-identical to the scalar loop (gated in
    /// simd_tests.zig), so goldens are unaffected; flip off to A/B against
    /// the scalar reference path.
    simd_jacobian: bool = true,

    pub const cdefault = ImplicitParams{};

    /// PROBLEMS/PUFFY/define.h:26-61.
    pub const puffy = ImplicitParams{
        .maxiter = 40,
        .convrel = 1.0e-8,
        .convrelerr = 1.0e-4,
        .convrelentr = 1.0e-4,
        .convrelentrerr = 1.0e-2,
        .max_damping = 1.0e-3,
        .start_with_bisect = true,
        .scale_jacobian = true,
        .allow_rad_ceiling = true,
        .allow_entr_in_4dprim = true,
    };
};

pub const Result = struct {
    ok: bool,
    /// which ladder rung succeeded (0-based; C's attempts 1, 1.5, 2, 2.5,
    /// 3, 4); −1 on failure
    rung: i32,
    /// Newton iterations of the successful 4dprim call
    iters: usize,
    /// the successful rung's configuration (valid when ok — comparable
    /// against C's GLOBALINTSLOT_NIMP* success counters)
    whichprim: WhichPrim = .rad,
    whicheq: WhichEq = .energy,
    whichframe: WhichFrame = .lab,
};

/// C: my_sign (misc.c:1285) — sign(0) = +1.
fn mySign(x: f64) f64 {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 1.0;
}

/// 4×4 LU inversion with partial pivoting (C: inverse_matrix, GSL LU).
/// No error path — a singular matrix propagates inf/nan into the update,
/// which the damping ladder then rejects.
fn invert4(a_in: [4][4]f64) [4][4]f64 {
    var lu = a_in;
    var perm = [4]usize{ 0, 1, 2, 3 };

    for (0..4) |k| {
        var piv = k;
        var pmax = @abs(lu[k][k]);
        for (k + 1..4) |r| {
            if (@abs(lu[r][k]) > pmax) {
                pmax = @abs(lu[r][k]);
                piv = r;
            }
        }
        if (piv != k) {
            const tr = lu[k];
            lu[k] = lu[piv];
            lu[piv] = tr;
            const tp = perm[k];
            perm[k] = perm[piv];
            perm[piv] = tp;
        }
        for (k + 1..4) |r| {
            lu[r][k] /= lu[k][k];
            for (k + 1..4) |cc| {
                lu[r][cc] -= lu[r][k] * lu[k][cc];
            }
        }
    }

    var inv: [4][4]f64 = undefined;
    for (0..4) |col| {
        // solve A x = e_col
        var y: [4]f64 = undefined;
        for (0..4) |r| {
            y[r] = if (perm[r] == col) 1.0 else 0.0;
            for (0..r) |cc| y[r] -= lu[r][cc] * y[cc];
        }
        var x: [4]f64 = undefined;
        var r: usize = 4;
        while (r > 0) {
            r -= 1;
            x[r] = y[r];
            for (r + 1..4) |cc| x[r] -= lu[r][cc] * x[cc];
            x[r] /= lu[r][r];
        }
        for (0..4) |rr| inv[rr][col] = x[rr];
    }
    return inv;
}

pub fn Solver(comptime cfg: config.Config) type {
    const L = layout.VarLayout(cfg);
    const NV = L.count;
    comptime std.debug.assert(L.hasVar(.ee));

    const i_rho = L.index(.rho);
    const i_uu = L.index(.uu);
    const i_entr = L.index(.entr);
    const i_ee = L.index(.ee);

    return struct {
        /// C: implicit_apply_constraints (rad.c:1279) — make the
        /// non-iterated fluid consistent with total conservation.
        /// Returns C's u2pret: 0 ok, −1 tolerable correction, −2 fatal.
        pub fn applyConstraints(
            pp: *[NV]f64,
            uu: *[NV]f64,
            uu0: *const [NV]f64,
            geom: *const Geometry,
            gamma_adiab: f64,
            rad: invert_rad.RadParams,
            ip: *const ImplicitParams,
            whichprim: WhichPrim,
        ) i32 {
            const gdetu = geom.gdet; // GDETIN == 1
            const gdetu_inv = 1.0 / gdetu;

            if (whichprim == .mhd) {
                const u = relele.uconUcovFromPrims(
                    .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
                    geom,
                ) catch return -2;

                // rho may be inconsistent when iterating MHD primitives
                pp[i_rho] = uu0[i_rho] * gdetu_inv / u.con[0];
                pp[i_entr] = hydro.sFromU(pp[i_rho], pp[i_uu], gamma_adiab);

                p2u_mod.p2uMhd(cfg, pp.*, uu, geom, gamma_adiab) catch return -2;

                // opposite changes in the RAD conserveds
                inline for (0..4) |k| {
                    uu[i_ee + k] = uu0[i_ee + k] - (uu[i_uu + k] - uu0[i_uu + k]);
                }

                const rr = invert_rad.u2pRad(cfg, uu.*, pp, geom, rad);
                var u2pret: i32 = rr.ret;
                if (rr.corrected) {
                    // don't allow hitting the radiation ceiling mid-iteration
                    // unless ALLOWRADCEILINGINIMPLICIT
                    u2pret = if (ip.allow_rad_ceiling) -1 else -2;
                }
                return u2pret;
            } else {
                p2u_mod.p2uRad(cfg, pp.*, uu, geom) catch return -2;

                // opposite changes in the MHD conserveds
                uu[i_rho] = uu0[i_rho];
                inline for (0..4) |k| {
                    uu[i_uu + k] = uu0[i_uu + k] - (uu[i_ee + k] - uu0[i_ee + k]);
                }
                if (comptime L.hasVar(.b1)) {
                    inline for (0..3) |k| {
                        uu[L.index(.b1) + k] = gdetu * pp[L.index(.b1) + k];
                    }
                }

                var rettemp = invert.u2pSolverW(cfg, uu.*, pp, geom, .hot, gamma_adiab);
                if (rettemp < 0 and ip.allow_entr_in_4dprim) {
                    rettemp = invert.u2pSolverW(cfg, uu.*, pp, geom, .entropy, gamma_adiab);
                }
                const u2pret: i32 = if (rettemp < 0) -2 else 0;

                const u = relele.uconUcovFromPrims(
                    .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
                    geom,
                ) catch return -2;
                pp[i_entr] = hydro.sFromU(pp[i_rho], pp[i_uu], gamma_adiab);
                uu[i_entr] = pp[i_entr] * gdetu * u.con[0];
                return u2pret;
            }
        }

        /// C: f_implicit_lab_4dprim_with_state (rad.c:1460), 4 equations
        /// (no photons/electrons). Fills f[0..4]; returns the max
        /// normalized error. Public so tests can measure how well a
        /// "converged" solution actually satisfies the equations (the
        /// relative-change exit can accept heavily-damped near-stalls,
        /// in C exactly as here).
        ///
        /// The error union is kept for API compatibility; with
        /// VELPRIM == VELR the evaluation is infallible (see residualG).
        pub fn residual(
            uu: *const [NV]f64,
            pp: *const [NV]f64,
            st: *const radforce.RadState,
            uu0: *const [NV]f64,
            st0: *const radforce.RadState,
            dt: f64,
            geom: *const Geometry,
            opac: *const radforce.Params,
            whichprim: WhichPrim,
            whicheq: WhichEq,
            whichframe: WhichFrame,
            f: *[4]f64,
        ) relele.Error!f64 {
            return residualG(f64, uu, pp, st, uu0, st0, dt, &geom.gg, &geom.GG, geom.gdet, opac, whichprim, whicheq, whichframe, f);
        }

        /// residual over lane type T — the batched Jacobian evaluates the
        /// 4 perturbed states as lanes of one @Vector(4, f64) call. uu0/st0
        /// stay scalar (the base state is shared by all lanes).
        pub fn residualG(
            comptime T: type,
            uu: *const [NV]T,
            pp: *const [NV]T,
            st: *const radforce.RadStateOf(T),
            uu0: *const [NV]f64,
            st0: *const radforce.RadState,
            dt: f64,
            gg: *const [4][5]T,
            GG: *const [4][5]T,
            gdet: T,
            opac: *const radforce.Params,
            whichprim: WhichPrim,
            whicheq: WhichEq,
            whichframe: WhichFrame,
            f: *[4]T,
        ) T {
            const sp = simd.splat;
            const gdetu = gdet;
            const tgas = st.tgas;
            const ehat0 = sp(T, st0.ehat);
            const ehat = st.ehat;
            const dtau = sp(T, dt) / st.ucon[0];

            const gi_pair = radforce.calcGiFromStateG(T, st, .{
                pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)],
            }, gg, GG, opac);
            const giff = gi_pair.ff;
            const gi = relele.indices21G(T, gi_pair.lab, gg); // G_μ

            var err: [4]T = @splat(sp(T, 0));

            // momenta — always in the lab frame
            if (whichprim == .mhd) {
                inline for (1..4) |k| {
                    f[k] = uu[i_uu + k] - sp(T, uu0[i_uu + k]) - sp(T, dt) * gdetu * gi[k];
                    err[k] = simd.select(T, @abs(f[k]) > sp(T, small), @abs(f[k]) /
                        (sp(T, 1.0e-20) * uu[i_uu] + @abs(uu[i_uu + k]) + sp(T, @abs(uu0[i_uu + k])) + @abs(sp(T, dt) * gdetu * gi[k])), sp(T, 0.0));
                }
            } else {
                inline for (1..4) |k| {
                    f[k] = uu[i_ee + k] - sp(T, uu0[i_ee + k]) + sp(T, dt) * gdetu * gi[k];
                    err[k] = simd.select(T, @abs(f[k]) > sp(T, small), @abs(f[k]) /
                        (sp(T, 1.0e-20) * uu[i_ee] + @abs(uu[i_ee + k]) + sp(T, @abs(uu0[i_ee + k])) + @abs(sp(T, dt) * gdetu * gi[k])), sp(T, 0.0));
                }
            }

            // energy/entropy
            if (whichframe == .lab) {
                std.debug.assert(whicheq == .energy); // ENTROPYEQ+LAB is my_err in C
                if (whichprim == .rad) {
                    f[0] = uu[i_ee] - sp(T, uu0[i_ee]) + sp(T, dt) * gdetu * gi[0];
                    err[0] = simd.select(T, @abs(f[0]) > sp(T, small), @abs(f[0]) /
                        (@abs(uu[i_ee]) + sp(T, @abs(uu0[i_ee])) + @abs(sp(T, dt) * gdetu * gi[0])), sp(T, 0.0));
                } else {
                    f[0] = uu[i_uu] - sp(T, uu0[i_uu]) - sp(T, dt) * gdetu * gi[0];
                    err[0] = simd.select(T, @abs(f[0]) > sp(T, small), @abs(f[0]) /
                        (@abs(uu[i_uu]) + sp(T, @abs(uu0[i_uu])) + @abs(sp(T, dt) * gdetu * gi[0])), sp(T, 0.0));
                }
            } else {
                if (whicheq == .energy) {
                    if (whichprim == .rad) {
                        f[0] = ehat - ehat0 + dtau * giff[0];
                        err[0] = @abs(f[0]) / (@abs(ehat) + @abs(ehat0) + @abs(dtau * giff[0]));
                    } else {
                        // C: pp[UU] − pp0[UU] − dτ·Ĝ⁰ (pp0 = post-explicit)
                        f[0] = pp[i_uu] - sp(T, st0.uint) - dtau * giff[0];
                        err[0] = @abs(f[0]) / (@abs(pp[i_uu]) + sp(T, @abs(st0.uint)) + @abs(dtau * giff[0]));
                    }
                } else {
                    // C BUG transcribed (rad.c:1749/1767): S = state->Tgas,
                    // S0 = state0->Sgas — the entropy equation compares the
                    // gas TEMPERATURE against the initial entropy
                    const s0 = sp(T, st0.sgas);
                    const s = st.tgas;
                    f[0] = tgas * (s - s0) - dtau * giff[0];
                    err[0] = @abs(f[0]) / (@abs(tgas * s) + @abs(tgas * s0) + @abs(dtau * giff[0]));
                }
            }

            var err0: T = sp(T, 0);
            for (0..4) |k| {
                err0 = simd.select(T, err[k] > err0, err[k], err0);
            }
            return err0;
        }

        /// C: f_implicit_1dprim_err (rad.c:1872), energy equation only
        /// (the bisect never requests entropy). NaN aborts the bracket.
        fn f1dErr(
            xxx: f64,
            pp0: *const [NV]f64,
            dtau: f64,
            geom: *const Geometry,
            gamma_adiab: f64,
            opac: *const radforce.Params,
            whichprim: WhichPrim,
            totenergy: f64,
            ratio: f64,
        ) f64 {
            var ugas: f64 = undefined;
            var ehat: f64 = undefined;
            if (whichprim == .rad) {
                ehat = xxx * ratio;
                ugas = totenergy - ehat;
            } else {
                ugas = xxx;
                ehat = totenergy - ugas;
            }

            var pp = pp0.*;
            pp[i_uu] = ugas;
            pp[i_ee] = ehat / ratio;
            const st = radforce.fillRadState(cfg, pp, geom, gamma_adiab, opac) catch return std.math.nan(f64);
            const gi = radforce.calcGiFromState(&st, .{
                pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)],
            }, geom, opac) catch return std.math.nan(f64);

            if (whichprim == .rad) {
                // C quirk: pp0[EE]/ratio (·ratio would give Ehat0)
                return ehat - pp0[i_ee] / ratio + dtau * gi.ff[0];
            } else {
                return ugas - pp0[i_uu] - dtau * gi.ff[0];
            }
        }

        /// C: solve_implicit_lab_1dprim (rad.c:2001) — bisection on the
        /// fluid-frame energy equation, writing the improved guess into
        /// pp0out. Returns −1 (pp0out untouched beyond the pp00 copy) if
        /// no bracket is found.
        pub fn solve1dPrim(
            pp00: *const [NV]f64,
            st00: *const radforce.RadState,
            geom: *const Geometry,
            dt: f64,
            gamma_adiab: f64,
            opac: *const radforce.Params,
            pp0out: *[NV]f64,
        ) i32 {
            const ehat = st00.ehat;
            const dtau = dt / st00.ucon[0];
            const totenergy = ehat + pp00[i_uu];

            // the bisect's own threshold is hardwired 1e-3 (not
            // RADIMPLICITTHRESHOLD)
            const whichprim: WhichPrim = if (ehat < 1.0e-3 * pp00[i_uu]) .rad else .mhd;

            const tgas = st00.tgas;
            const ratio = ehat / pp00[i_ee];
            const fac: f64 = 3.0;
            const fac1 = fac + 1.0;

            var pp_uu = pp00[i_uu];
            var pp_ee = pp00[i_ee];
            if (tgas > 1.0e14) {
                pp_uu = pp00[i_uu] * 1.0e14 / tgas;
                pp_ee = (totenergy - pp_uu) / ratio;
            }

            var xlo: f64 = undefined;
            var xhi: f64 = undefined;
            if (whichprim == .mhd) {
                xlo = pp_uu / 1.05;
                xhi = @min(pp_uu * 1.05, (pp_uu + fac * totenergy) / fac1);
            } else {
                xlo = pp_ee / 1.05;
                xhi = @min(pp_ee * 1.05, (pp_ee + fac * totenergy / ratio) / fac1);
            }

            var errlo = f1dErr(xlo, pp00, dtau, geom, gamma_adiab, opac, whichprim, totenergy, ratio);
            var errhi = f1dErr(xhi, pp00, dtau, geom, gamma_adiab, opac, whichprim, totenergy, ratio);

            const maxiter: usize = 50;
            const conv: f64 = 1.0e-3; // no need for much precision here

            var iter: usize = 0;
            while (errlo * errhi > 0.0) {
                iter += 1;
                if (errlo > 0.0) { // extend the low end
                    xhi = xlo;
                    errhi = errlo;
                    xlo /= fac;
                    errlo = f1dErr(xlo, pp00, dtau, geom, gamma_adiab, opac, whichprim, totenergy, ratio);
                } else { // expand the high end, staying below totenergy
                    xlo = xhi;
                    errlo = errhi;
                    if (whichprim == .rad) {
                        xhi = @min(xhi * fac, (xhi + fac * totenergy / ratio) / fac1);
                    } else {
                        xhi = @min(xhi * fac, (xhi + fac * totenergy) / fac1);
                    }
                    errhi = f1dErr(xhi, pp00, dtau, geom, gamma_adiab, opac, whichprim, totenergy, ratio);
                }
                if (std.math.isNan(errlo) or std.math.isNan(errhi)) iter = maxiter;
                if (iter >= maxiter) break;
            }

            if (iter >= maxiter) {
                pp0out.* = pp00.*;
                return -1;
            }

            var xmid: f64 = undefined;
            while (true) {
                iter += 1;
                xmid = 0.5 * (xlo + xhi);
                const errmid = f1dErr(xmid, pp00, dtau, geom, gamma_adiab, opac, whichprim, totenergy, ratio);
                if (errmid * errlo > 0.0) {
                    xlo = xmid;
                    errlo = errmid;
                } else {
                    xhi = xmid;
                    errhi = errmid;
                }
                if (std.math.isNan(xmid) or std.math.isNan(errmid)) {
                    // C: my_err("nan in 1dprim") aborts; fail cleanly instead
                    pp0out.* = pp00.*;
                    return -1;
                }
                if (!(@abs((xhi - xlo) / xmid) > conv)) break;
            }

            pp0out.* = pp00.*;
            if (whichprim == .rad) {
                pp0out[i_ee] = xmid;
                pp0out[i_uu] = totenergy - xmid * ratio;
            } else {
                pp0out[i_uu] = xmid;
                pp0out[i_ee] = (totenergy - xmid) / ratio;
            }
            return 0;
        }

        pub const Solve4dResult = struct {
            ok: bool,
            iters: usize,
            /// the solver's internal conserveds — exactly conservation-
            /// constrained (before the wrapper's final p2u regenerates uu)
            uu: [NV]f64,
        };

        /// The one-sided FD step of Jacobian perturbation j (rad.c:600-621),
        /// shared by the scalar and the lane-batched paths.
        fn fdDel(j: usize, sign: f64, ppp: *const [NV]f64, sh: usize, eps: f64, geom: *const Geometry) f64 {
            if (j == 0) return sign * eps * ppp[sh];
            const veleps = eps * mySign(ppp[j + sh]) *
                @max(1.0e-8 / @sqrt(geom.gg[j][j]), @abs(ppp[j + sh]));
            return if (ppp[j + sh] >= 0.0) sign * veleps else -sign * veleps;
        }

        /// C: solve_implicit_lab_4dprim (rad.c:341), 4 equations.
        pub fn solve4dPrim(
            uu00: *const [NV]f64,
            pp00: *const [NV]f64,
            geom: *const Geometry,
            dt: f64,
            gamma_adiab: f64,
            rad: invert_rad.RadParams,
            opac: *const radforce.Params,
            ip: *const ImplicitParams,
            whichprim: WhichPrim,
            whicheq: WhichEq,
            whichframe: WhichFrame,
            pp_out: *[NV]f64,
        ) Solve4dResult {
            const fail = Solve4dResult{ .ok = false, .iters = 0, .uu = uu00.* };

            var pp0 = pp00.*;
            var uu = uu00.*;

            const state00 = radforce.fillRadState(cfg, pp00.*, geom, gamma_adiab, opac) catch return fail;

            if (ip.start_with_bisect) {
                _ = solve1dPrim(pp00, &state00, geom, dt, gamma_adiab, opac, &pp0);
            }

            const sh: usize = if (whichprim == .mhd) i_uu else i_ee;
            const conv = if (whicheq == .entropy) ip.entrconv else ip.conv;

            var pp = pp0;
            var fret = applyConstraints(&pp, &uu, uu00, geom, gamma_adiab, rad, ip, whichprim);
            var state = radforce.fillRadState(cfg, pp, geom, gamma_adiab, opac) catch return fail;

            var f1: [4]f64 = undefined;
            var f2: [4]f64 = undefined;
            var jac: [4][4]f64 = undefined;

            // lane-broadcast geometry for the batched Jacobian (used only
            // when ip.simd_jacobian)
            const ggv = simd.splatBlock(V4, &geom.gg);
            const GGv = simd.splatBlock(V4, &geom.GG);
            const gdetv = simd.splat(V4, geom.gdet);

            var iter: usize = 0;
            var failed = false;

            main: while (true) {
                iter += 1;

                // Step 0: base state
                const ppp = pp;
                var xxx: [4]f64 = undefined;
                for (0..4) |k| xxx[k] = ppp[k + sh];

                if (fret < -1) return fail;

                const errbase = residual(&uu, &pp, &state, uu00, &state00, dt, geom, opac, whichprim, whicheq, whichframe, &f1) catch return fail;
                if (errbase < conv) break; // success (error)

                // Step 1: one-sided FD Jacobian, retrying the other
                // perturbation sign on constraint failure
                if (ip.simd_jacobian) {
                    // (a) scalar per lane: perturbation + constraints with
                    // the sign retry — exactly the scalar path below minus
                    // the state fill and residual. The retry decision only
                    // reads applyConstraints' fret: fillRadState is
                    // infallible for VELR primitives, so the scalar path's
                    // `catch → fret = -2` can never fire and the fill can
                    // move behind the retry without changing any decision.
                    var lane_pp: [4][NV]f64 = undefined;
                    var lane_uu: [4][NV]f64 = undefined;
                    var lane_del: [4]f64 = undefined;
                    for (0..4) |j| {
                        var sign: f64 = -1.0; // minus avoids u2p_mhd errors on rad
                        while (true) {
                            const del = fdDel(j, sign, &ppp, sh, ip.eps, geom);
                            pp[j + sh] = ppp[j + sh] + del;

                            fret = applyConstraints(&pp, &uu, uu00, geom, gamma_adiab, rad, ip, whichprim);

                            if (fret < -1) {
                                if (sign > 0.0) { // both signs failed
                                    failed = true;
                                    break;
                                } else {
                                    pp[j + sh] = ppp[j + sh];
                                    sign *= -1.0;
                                    continue;
                                }
                            }

                            lane_pp[j] = pp;
                            lane_uu[j] = uu;
                            lane_del[j] = del;
                            pp[j + sh] = ppp[j + sh];
                            break;
                        }
                        if (failed) break;
                    }
                    if (failed) break;

                    // (b) the 4 perturbed residuals as one 4-lane batch
                    var ppv: [NV]V4 = undefined;
                    var uuv: [NV]V4 = undefined;
                    for (0..NV) |iv| {
                        ppv[iv] = .{ lane_pp[0][iv], lane_pp[1][iv], lane_pp[2][iv], lane_pp[3][iv] };
                        uuv[iv] = .{ lane_uu[0][iv], lane_uu[1][iv], lane_uu[2][iv], lane_uu[3][iv] };
                    }
                    const stv = radforce.fillRadStateG(cfg, V4, ppv, &ggv, &GGv, gamma_adiab, opac);
                    var f2v: [4]V4 = undefined;
                    _ = residualG(V4, &uuv, &ppv, &stv, uu00, &state00, dt, &ggv, &GGv, gdetv, opac, whichprim, whicheq, whichframe, &f2v);
                    for (0..4) |i| {
                        const fi: [4]f64 = f2v[i]; // lane j = perturbation j
                        for (0..4) |j| {
                            jac[i][j] = (fi[j] - f1[i]) / lane_del[j];
                        }
                    }
                    // the scalar loop leaves `state` holding the 4th
                    // perturbation's fill — preserved (it is overwritten in
                    // Step 3 before any read; kept for exactness)
                    state = simd.extractLane(radforce.RadState, stv, 3);
                } else {
                    for (0..4) |j| {
                        var sign: f64 = -1.0; // minus avoids u2p_mhd errors on rad
                        while (true) {
                            const del = fdDel(j, sign, &ppp, sh, ip.eps, geom);
                            pp[j + sh] = ppp[j + sh] + del;

                            fret = applyConstraints(&pp, &uu, uu00, geom, gamma_adiab, rad, ip, whichprim);
                            state = radforce.fillRadState(cfg, pp, geom, gamma_adiab, opac) catch blk: {
                                fret = -2;
                                break :blk state;
                            };

                            if (fret < -1) {
                                if (sign > 0.0) { // both signs failed
                                    failed = true;
                                    break;
                                } else {
                                    pp[j + sh] = ppp[j + sh];
                                    sign *= -1.0;
                                    continue;
                                }
                            }

                            _ = residual(&uu, &pp, &state, uu00, &state00, dt, geom, opac, whichprim, whicheq, whichframe, &f2) catch {
                                failed = true;
                                break;
                            };
                            for (0..4) |i| {
                                jac[i][j] = (f2[i] - f1[i]) / del;
                            }
                            pp[j + sh] = ppp[j + sh];
                            break;
                        }
                        if (failed) break;
                    }
                    if (failed) break;
                }

                // Jacobian scaling (SCALE_JACOBIAN, on in PUFFY):
                // scalevec = {pp[sh], 1, 1, 1}
                var scalevec = [4]f64{ pp[sh], 1.0, 1.0, 1.0 };
                var invscalevec: [4]f64 = undefined;
                if (ip.scale_jacobian) {
                    for (0..4) |j| {
                        if (scalevec[j] == 0.0) scalevec[j] = 1.0;
                        invscalevec[j] = 1.0 / scalevec[j];
                    }
                    for (0..4) |i| {
                        for (0..4) |j| {
                            if (i != j) jac[i][j] = jac[i][j] * (scalevec[j] * invscalevec[i]);
                        }
                    }
                }

                // Step 2: invert
                const ijac = invert4(jac);

                // Step 3: apply with damping
                var xiapp: f64 = 1.0;
                while (true) {
                    for (0..4) |k| xxx[k] = ppp[k + sh];
                    for (0..4) |i| {
                        for (0..4) |j| {
                            if (ip.scale_jacobian) {
                                xxx[i] -= xiapp * (ijac[i][j] * scalevec[i] * invscalevec[j]) * f1[j];
                            } else {
                                xxx[i] -= xiapp * ijac[i][j] * f1[j];
                            }
                        }
                    }
                    for (0..4) |k| pp[k + sh] = xxx[k];

                    fret = applyConstraints(&pp, &uu, uu00, geom, gamma_adiab, rad, ip, whichprim);
                    state = radforce.fillRadState(cfg, pp, geom, gamma_adiab, opac) catch blk: {
                        fret = -2;
                        break :blk state;
                    };

                    // per-step change limiters
                    var okcheck = false;
                    const edenmin = ppp[sh] / ip.max_en_change_down;
                    const edenmax = ppp[sh] * ip.max_en_change_up;
                    if (xxx[0] > edenmin and xxx[0] < edenmax and fret >= -1) okcheck = true;

                    const c = &opac.consts;
                    const tgasold = thermo.tFromUrho(c, ppp[i_uu], ppp[i_rho], gamma_adiab);
                    const tgasnew = thermo.tFromUrho(c, pp[i_uu], pp[i_rho], gamma_adiab);
                    if (okcheck and (tgasnew > tgasold * ip.max_tgas_change or tgasnew < tgasold / ip.max_tgas_change)) {
                        okcheck = false;
                    }

                    const tradnew = c.lteTfromE(pp[i_ee]);
                    const tradold = c.lteTfromE(ppp[i_ee]);
                    if (okcheck and (tradnew > tradold * ip.max_trad_change or tradnew < tradold / ip.max_trad_change)) {
                        okcheck = false;
                    }

                    if (okcheck) break;

                    if (xxx[0] <= 0.0) {
                        xiapp *= ppp[sh] / (ppp[sh] + @abs(xxx[0]));
                        xiapp *= 1.0e-1; // not to land too close to zero
                    } else {
                        xiapp /= ip.damping_factor;
                    }

                    if (xiapp < ip.max_damping) {
                        failed = true;
                        break;
                    }
                }

                // Step 5: relative-change convergence
                if (!failed) {
                    var f3: [4]f64 = undefined;
                    f3[0] = @abs((pp[sh] - ppp[sh]) / ppp[sh]);
                    for (1..4) |i| {
                        f3[i] = pp[i + sh] - ppp[i + sh];
                        f3[i] = @abs(f3[i] / @max(ip.eps, @abs(ppp[i + sh])));
                    }

                    const convrel = if (whicheq == .entropy) ip.convrelentr else ip.convrel;
                    const convrelerr = if (whicheq == .entropy) ip.convrelentrerr else ip.convrelerr;
                    var convrelcheck = true;
                    for (0..4) |ib| {
                        if (f3[ib] > convrel) {
                            convrelcheck = false;
                            break;
                        }
                    }
                    if (convrelcheck or errbase <= convrelerr) break :main; // success (rel.change)
                }

                if (iter > ip.maxiter or failed) break;
            }

            if (iter > ip.maxiter or failed) return fail;

            pp_out.* = pp;
            return .{ .ok = true, .iters = iter, .uu = uu };
        }

        /// C: solve_implicit_lab (rad.c:73) — the rung ladder. Reads pp
        /// (uu is regenerated: FORCEUEQPINIMPLICIT == 1); on success both
        /// are overwritten with the coupled state, on failure untouched.
        pub fn solveImplicitLab(
            uu: *[NV]f64,
            pp: *[NV]f64,
            geom: *const Geometry,
            dt: f64,
            gamma_adiab: f64,
            rad: invert_rad.RadParams,
            opac: *const radforce.Params,
            ip: *const ImplicitParams,
        ) Result {
            const failres = Result{ .ok = false, .rung = -1, .iters = 0 };

            const pp00 = pp.*;
            const uu00 = p2u_mod.p2u(cfg, pp00, geom, gamma_adiab) catch return failres;

            const ff = radiation.calcFfRtt(cfg, pp00, geom) catch return failres;
            const ehat = -ff.rtt;

            const startwith: WhichPrim = if (ehat < ip.threshold * pp00[i_uu]) .rad else .mhd;
            const swapped: WhichPrim = if (startwith == .rad) .mhd else .rad;

            // BASICRADIMPLICIT off, OPDAMPINIMPLICIT 0: the full 6-rung
            // ladder at a single opacity damping level
            const rungs = [6]struct { p: WhichPrim, e: WhichEq, fr: WhichFrame }{
                .{ .p = startwith, .e = .energy, .fr = .lab },
                .{ .p = startwith, .e = .energy, .fr = .ff },
                .{ .p = swapped, .e = .energy, .fr = .lab },
                .{ .p = swapped, .e = .energy, .fr = .ff },
                .{ .p = startwith, .e = .entropy, .fr = .ff },
                .{ .p = swapped, .e = .entropy, .fr = .ff },
            };

            for (rungs, 0..) |rung, ir| {
                var ppw = pp.*;
                const r4 = solve4dPrim(&uu00, &pp00, geom, dt, gamma_adiab, rad, opac, ip, rung.p, rung.e, rung.fr, &ppw);
                if (r4.ok) {
                    pp.* = ppw;
                    uu.* = p2u_mod.p2u(cfg, ppw, geom, gamma_adiab) catch return failres;
                    return .{
                        .ok = true,
                        .rung = @intCast(ir),
                        .iters = r4.iters,
                        .whichprim = rung.p,
                        .whicheq = rung.e,
                        .whichframe = rung.fr,
                    };
                }
            }
            return failres;
        }
    };
}
