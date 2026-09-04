//! The floor atmospheres every torus problem fills outside the disk
//! (C: relele.c:518 set_hdatmosphere / relele.c:718 set_radatmosphere,
//! atmtype 0 — normal observer, Bondi-like ρ ∝ r^-1.5, u ∝ r^-2.5
//! normalized at r_BL = 2), and the LTE pressure split prepinit.c uses to
//! divide a torus cell's pressure between gas and radiation.
//!
//! Moved verbatim out of problems/puffy/puffy.zig (redesign step 5,
//! 2026-09-04); expression shapes are unchanged.

const std = @import("std");
const config = @import("../../config.zig");
const layout = @import("../../layout.zig");
const relele = @import("../../relele.zig");
const coco = @import("../../metric/coco.zig");
const metric = @import("../../metric/metric.zig");
const Geometry = @import("../../geometry.zig").Geometry;

/// UINTATMMIN / ERADATMMIN (PUFFY define.h:221/225), computed once by the
/// problem from its atmosphere temperatures.
pub const Atm = struct { uintatmmin: f64, eradatmmin: f64 };

/// relele.c:518 set_hdatmosphere, atmtype 0. `rhoatmmin` is the density at
/// r_BL = 2 (RHOATMMIN); `mp` the metric the cell's coordinates live in.
pub fn setHdAtmosphere(
    comptime cfg: config.Config,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    atm: *const Atm,
    rhoatmmin: f64,
    mp: metric.MetricParams,
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

/// relele.c:718 set_radatmosphere, atmtype 0: E = ERADATMMIN at rest in the
/// normal-observer frame.
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

/// prepinit.c:92-93; the T > 0 root of P = bbb·T + aaa·T⁴ (Mathematica
/// closed form, transcribed with its literal decimal exponents). Used to
/// split a torus cell's total pressure into gas (bT) and radiation (aT⁴)
/// in LTE.
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
