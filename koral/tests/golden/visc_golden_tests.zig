//! M12 keystone (radiative viscosity): the shear tensor σ^ij, the viscosity
//! coefficient ν, and the assembled viscous R^i_j on the PUFFY t=0 state,
//! cell-by-cell against the C oracle (oracle/harness_visc.c →
//! tests/golden/init/puffy_{shear,visc}.kini.gz over the domain + 1 ghost
//! ring, with global_dt = 1).
//!
//! What the numbers say (field-scale = max|C−z| / max|C| over the class):
//!
//!  * σ^ij (calc_rad_shearviscosity) depends only on the radiation-frame
//!    velocities: which are analytic (ℓ(λ) + metric, LTE F̂=0) and match C
//!    to ~1e-13; plus the FD stencil and the Christoffels. It matches to
//!    ~1e-8, the MKS2 near-axis metric spread (the M1 "two-π" floor) that the
//!    Christoffel/velocity-lowering chain inherits. This isolates the *new*
//!    M12 shear code from the quadrature story.
//!  * ν (= ALPHARADVISC·mfp, capped) goes through calc_chi → opacities of
//!    ρ,T,bsq, which carry C's ~1e-3 qags error (M11 keystone), so ν matches
//!    at field-scale ~1e-3.
//!  * R^i_j = indices_2221(−2 ν Ê σ^ij): ν and Ê both carry the qags ~1e-3,
//!    so the assembled tensor matches at ~1e-3; the same C-qags-limited
//!    story as the M11 keystone, NOT a Zig error (a shear bug would show far
//!    above this floor).

const std = @import("std");
const build_options = @import("build_options");
const config = @import("../../config.zig");
const grid_mod = @import("../../grid.zig");
const sim_mod = @import("../../sim.zig");
const golden = @import("../../testing/golden.zig");
const puffy = @import("../../problems/puffy/puffy.zig");
const rijvisc = @import("../../sim/rijvisc.zig");
const invert = @import("../../solve/invert.zig");
const invert_rad = @import("../../solve/invert_rad.zig");

const SimP = sim_mod.Sim(config.puffy);
const LP = SimP.Layout;

const global_dt_golden: f64 = 1.0; // C: harness_visc global_dt

/// Field-scale deviation aggregated over a tensor class (all components, all
/// cells): max|c−z| / max|c|.
const Dev = golden.FieldDev;

fn viscOptions() SimP.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .opac = puffy_opac,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .radviscosity = true,
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimP).calc,
    };
}
const puffy_opac = @import("../../physics/radforce.zig").Params.puffy();

fn outsideGc(nx: i64, ny: i64, ix: i64, iy: i64) bool {
    var o: u2 = 0;
    if (ix < 0 or ix >= nx) o += 1;
    if (iy < 0 or iy >= ny) o += 1;
    return o >= 2;
}

test "M12 radviscosity: shear + nu + R^i_j vs C (PUFFY t=0, 384×360)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;

    var s = try SimP.init(a, puffy.makeGrid(384, 360), viscOptions());
    defer s.deinit();

    // full init leaves the "stale ghost" state the first-step
    // calc_Rij_visc_total sees (no set_bc after postinit) — matches harness
    _ = try puffy.initAll(SimP, &s);
    try rijvisc.calcRijViscTotal(SimP, &s, global_dt_golden);

    const nx = s.nxi();
    const ny = s.nyi();

    // -- shear σ^ij + ν (isolates the FD/Christoffel + calc_chi code) ------
    var kshear = try golden.readKini(a, "init/puffy_shear.kini.gz");
    defer kshear.deinit();
    var dev_shear: Dev = .{};
    var dev_nu: Dev = .{};
    {
        var jy: usize = 0;
        while (jy < kshear.nSamplesY()) : (jy += 1) {
            var jx: usize = 0;
            while (jx < kshear.nSamplesX()) : (jx += 1) {
                const ix = kshear.cellX(jx);
                const iy = kshear.cellY(jy);
                if (outsideGc(nx, ny, ix, iy)) continue;
                const base = kshear.base(jx, jy, 0);
                var pp: [SimP.nv]f64 = undefined;
                s.p.load(ix, iy, 0, &pp);
                const geom = s.cache.fillGeometry(ix, iy, 0);
                const rv = try rijvisc.calcRadShearViscosity(SimP, &s, ix, iy, 0, &pp, &geom, global_dt_golden);
                for (0..4) |i| {
                    for (0..4) |j| dev_shear.add(kshear.data[base + i * 4 + j], rv.shear[i][j]);
                }
                dev_nu.add(kshear.data[base + 16], rv.nu);
            }
        }
    }

    // -- assembled viscous R^i_j (Rijviscglobal) --------------------------
    var kvisc = try golden.readKini(a, "init/puffy_visc.kini.gz");
    defer kvisc.deinit();
    var dev_rij: Dev = .{};
    {
        var jy: usize = 0;
        while (jy < kvisc.nSamplesY()) : (jy += 1) {
            var jx: usize = 0;
            while (jx < kvisc.nSamplesX()) : (jx += 1) {
                const ix = kvisc.cellX(jx);
                const iy = kvisc.cellY(jy);
                const base = kvisc.base(jx, jy, 0);
                var t: [16]f64 = undefined;
                s.visc.?.rijvisc.load(ix, iy, 0, &t);
                for (0..16) |k| dev_rij.add(kvisc.data[base + k], t[k]);
            }
        }
    }

    std.debug.print("M12 radviscosity field-scale deviations:\n", .{});
    std.debug.print("  shear^ij {e:.3}  (|Δ|max {e:.3} / |σ|max {e:.3})\n", .{ dev_shear.dev(), dev_shear.max_diff, dev_shear.max_mag });
    std.debug.print("  nu       {e:.3}  (|Δ|max {e:.3} / |ν|max {e:.3})\n", .{ dev_nu.dev(), dev_nu.max_diff, dev_nu.max_mag });
    std.debug.print("  R^i_j    {e:.3}  (|Δ|max {e:.3} / |R|max {e:.3})\n", .{ dev_rij.dev(), dev_rij.max_diff, dev_rij.max_mag });

    // measured: shear^ij 6.4e-8, nu 3.1e-5, R^i_j 2.3e-4.
    // σ^ij isolates the new FD/Christoffel code (velocities analytic to
    // 1e-13); floored by the MKS2 near-axis metric spread (~1e-8)
    try std.testing.expect(dev_shear.dev() <= 1e-6);
    // ν (calc_chi ρ,T,bsq) and R^i_j (× Ê) inherit C's qags kink error
    try std.testing.expect(dev_nu.dev() <= 3e-4);
    try std.testing.expect(dev_rij.dev() <= 2e-3);
}
