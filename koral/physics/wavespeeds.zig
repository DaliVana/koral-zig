//! Characteristic MHD wavespeeds in the coordinate frame
//! (C: calc_wavespeeds_lr_pure physics.c:399 + calc_wavespeeds_lr_core :653).
//!
//! The fluid-frame dispersion is isotropic with v² = c_s² + v_A² − c_s²v_A²;
//! the core routine boosts it to the lab frame per direction by solving the
//! quadratic for the characteristic dx^i/dt (HARM's algorithm).
//!
//! Physics: these speeds set the Riemann solver's dissipation and the CFL
//! step. c_s² = Γp/(ρ+u+p) is the relativistic sound speed and
//! v_A² = b²/(b²+ρ+u+p) the Alfvén speed; their relativistic combination
//! bounds the anisotropic fast-magnetosonic cone from above — an
//! overestimate except along/perpendicular to B, which only adds numerical
//! diffusion, never instability. The lab-frame quadratic carries the flow
//! velocity, lapse, shift and frame dragging, so the left/right roots are
//! asymmetric; the co-going clamps (left ≤ 0 ≤ right) are what HLL
//! requires of its bounding speeds.
//!
//! Radiation wavespeeds (τ-limited) arrive with M7; this module is the gas
//! part only. `active_dims` marks active directions (C: TNX/TNY/TNZ > 1) — C
//! leaves the entries of *inactive* directions uninitialized, we return 0.

const std = @import("std");
const relele = @import("../relele.zig");
const mhd = @import("bfield.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const Geometry = @import("../geometry.zig").Geometry;

/// Left/right lab-frame speeds per direction, ordered like C's aret[6]:
/// {axl, axr, ayl, ayr, azl, azr}. `wspeed2` is per direction (C passes
/// wspeed2x/y/z — the gas uses one value, the τ-damped rad speeds differ).
pub fn lrCore(
    ucon: [4]f64,
    geom: *const Geometry,
    wspeed2s: [3]f64,
    active_dims: [3]bool,
) [6]f64 {
    var aret: [6]f64 = @splat(0);

    // C: Bcov/Bcon/Bsq/Bu (physics.c:657) — the time/normal-observer basis
    // vector, kept capitalized to avoid colliding with the magnetic
    // four-vector bcon/bcov/bsq used elsewhere in the codebase.
    const Bcov = [4]f64{ 1, 0, 0, 0 };
    const Bcon = relele.raiseVec(Bcov, geom);
    const Bsq = relele.dot(Bcon, Bcov);
    const Bu = relele.dot(Bcov, ucon);
    const Bu2 = Bu * Bu;

    for (0..3) |dim| {
        if (active_dims[dim]) {
            const wspeed2 = wspeed2s[dim];
            var Acov: [4]f64 = @splat(0);
            Acov[dim + 1] = 1;
            const Acon = relele.raiseVec(Acov, geom);

            const Asq = relele.dot(Acon, Acov);
            const Au = relele.dot(Acov, ucon);
            const AB = relele.dot(Acon, Bcov);
            const Au2 = Au * Au;
            const AuBu = Au * Bu;

            const b = -2.0 * (AuBu * (1.0 - wspeed2) - AB * wspeed2);
            const a = Bu2 * (1.0 - wspeed2) - Bsq * wspeed2;
            var discr = 4.0 * wspeed2 * ((AB * AB - Asq * Bsq) * wspeed2 +
                (2.0 * AB * Au * Bu - Asq * Bu2 - Bsq * Au2) * (wspeed2 - 1.0));
            // C prints on discr < 0 and proceeds into sqrt (NaN); mirror the
            // arithmetic without the print.
            discr = @sqrt(discr);
            const cst1 = (-b + discr) / (2.0 * a);
            const cst2 = (-b - discr) / (2.0 * a);
            if (cst2 > cst1) {
                aret[2 * dim] = cst1;
                aret[2 * dim + 1] = cst2;
            } else {
                aret[2 * dim] = cst2;
                aret[2 * dim + 1] = cst1;
            }
        }
    }
    return aret;
}

/// Fluid-frame wavespeed² of the gas: c_s² + v_A² − c_s²v_A², with C's
/// c_s² ceiling of 0.95 (physics.c:476-498; no GASRADCOUPLEDWAVESPEEDS).
pub fn gasWspeed2(rho: f64, uu: f64, bsq: f64, gamma_adiab: f64) f64 {
    const pre = (gamma_adiab - 1.0) * uu;
    var cs2 = gamma_adiab * pre / (rho + uu + pre);
    if (cs2 > 0.95) cs2 = 0.95;

    var va2 = bsq / (bsq + (rho + uu + pre));
    if (va2 < 0.0) va2 = 0.0;

    return cs2 + va2 - cs2 * va2;
}

/// Gas wavespeeds of C's calc_wavespeeds_lr_pure — aaa[0..6] with the
/// co-going clamps applied (left ≤ 0 ≤ right).
pub fn gasWavespeedsLr(
    comptime cfg: config.Config,
    pp: [layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    gamma_adiab: f64,
    active_dims: [3]bool,
) relele.Error![6]f64 {
    const L = layout.VarLayout(cfg);

    const u = try relele.uconUcovFromPrims(
        .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
        geom,
    );

    var bsq: f64 = 0;
    if (comptime L.hasVar(.b1)) {
        const b = mhd.bconBcovBsqFrom4vel(
            .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
            u.con,
            u.cov,
            geom,
        );
        bsq = b.bsq;
    }

    const vtot2 = gasWspeed2(pp[L.index(.rho)], pp[L.index(.uu)], bsq, gamma_adiab);
    var a = lrCore(u.con, geom, .{ vtot2, vtot2, vtot2 }, active_dims);

    // zero 'co-going' velocities (physics.c:602-607)
    for (0..3) |dim| {
        if (a[2 * dim] > 0.0) a[2 * dim] = 0.0;
        if (a[2 * dim + 1] < 0.0) a[2 * dim + 1] = 0.0;
    }
    return a;
}
