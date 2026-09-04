//! Seed MRI-quality diagnostic for a magnetised torus (campaign notes
//! 2026-08-08). Mass-weighted sums over this rank's slab; the caller folds
//! them across ranks (globalSum) and divides — every field is
//! sum-reducible on purpose. Same disk mask as tools/qmri.zig (ρ > 10³× the
//! atmosphere floor profile, r < 100) so the startup numbers are directly
//! comparable to qmri on later dumps. Q_i = 2π|b^i|/(√(ρh+b²)·|Ω|·Δx^i)
//! with radiation-inclusive inertia; Q_φ ≡ 0 for the A_φ-only seed and is
//! omitted. Serial pass over the interior; the geometry is cached, so this
//! costs well under a second even on campaign grids.
//!
//! Assumes MKS2 internal coordinates (r = e^{x1} + mksr0) — a Track C item
//! for JETCOORDS along with the renderer.
//!
//! Moved verbatim out of problems/puffy/puffy.zig (redesign step 5,
//! 2026-09-04).

const std = @import("std");
const relele = @import("../../relele.zig");
const mhd = @import("../../physics/bfield.zig");
const radiation = @import("../../physics/radiation.zig");

pub const SeedQ = struct { mass: f64 = 0, qr_m: f64 = 0, qth_m: f64 = 0 };

/// The disk mask and inertia inputs the problem supplies.
pub const Mask = struct {
    /// MKS2 radial offset (r = e^{x1} + mksr0).
    mksr0: f64,
    /// The atmosphere floor density at r = 2 (RHOATMMIN).
    rhoatmmin: f64,
    /// Adiabatic index for the enthalpy.
    gam: f64,
};

pub fn seedQuality(comptime CoreT: type, s: *CoreT, m: Mask) !SeedQ {
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    if (comptime !L.hasVar(.b1)) return .{};
    var out = SeedQ{};
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                var pp: [CoreT.nv]f64 = undefined;
                s.p.load(ix, iy, iz, &pp);
                const rho = pp[L.index(.rho)];
                if (!(rho > 0)) continue;
                const geom = s.cache.fillGeometry(ix, iy, iz);
                const r = @exp(geom.xxvec[1]) + m.mksr0;
                if (r > 100.0) continue;
                if (rho < 1.0e3 * m.rhoatmmin * std.math.pow(f64, r / 2.0, -1.5)) continue;
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
                var w = rho + m.gam * pp[L.index(.uu)];
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
